#!/usr/bin/env bash

# Read-only incremental backup of live Nebius experiment artifacts.
#
# Commands:
#   sync   refresh the local Mac mirror and its checksum manifest
#   delta  refresh the mirror and package changes since the last capsule
#   full   refresh the mirror and package a self-contained recovery baseline
#
# The script never writes to the VM. Drive upload is intentionally performed by
# the Codex automation so that it uses the authenticated Google Drive connector.

set -Eeuo pipefail

command_name=${1:-sync}
case "$command_name" in
  sync|delta|full) ;;
  *)
    printf 'usage: %s {sync|delta|full}\n' "$0" >&2
    exit 2
    ;;
esac

script_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_root/.." && pwd)
remote_host=${REMOTE_HOST:-artemvorozhtsov@66.201.4.161}
remote_root=${REMOTE_ROOT:-/srv/braid/artifacts}
backup_root=${BACKUP_ROOT:-$repo_root/artifacts/nebius-semantic-v2-live-backup}
mirror_root=$backup_root/mirror
results_root=$mirror_root/results
resume_root=$mirror_root/resume
provenance_root=$mirror_root/provenance
state_root=$backup_root/state
outbox_root=$backup_root/drive-outbox
stamp=${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
minimum_free_gib=${MINIMUM_FREE_GIB:-8}
active_progress_days=${ACTIVE_PROGRESS_DAYS:-2}

lock_dir=$backup_root/.sync-lock
mkdir -p "$backup_root"
if ! mkdir "$lock_dir" 2>/dev/null; then
  printf 'backup already running: %s\n' "$lock_dir" >&2
  exit 75
fi
work_root=$(mktemp -d "$backup_root/.work.XXXXXX")
cleanup() {
  rm -rf "$work_root"
  rmdir "$lock_dir" 2>/dev/null || true
}
trap cleanup EXIT

mkdir -p "$results_root" "$resume_root" "$provenance_root/repos" \
  "$state_root" "$outbox_root"

available_kib=$(df -Pk "$backup_root" | awk 'NR == 2 {print $4}')
required_kib=$((minimum_free_gib * 1024 * 1024))
if (( available_kib < required_kib )); then
  printf 'refusing backup: only %s KiB free; require %s KiB\n' \
    "$available_kib" "$required_kib" >&2
  exit 73
fi

ssh_options=(-o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30)
ssh "${ssh_options[@]}" "$remote_host" \
  "test -d '$remote_root' && test -r '$remote_root'"

# Pass 1: mirror every result, event log, manifest, report, figure and launcher,
# excluding only model/checkpoint tensors and disposable cache files.
rsync -a --delete --prune-empty-dirs \
  --exclude='*.pt' \
  --exclude='*.pt.gz' \
  --exclude='*.tmp' \
  --exclude='*.tmp.*' \
  --exclude='*.pyc' \
  --exclude='__pycache__/' \
  --exclude='.pytest_cache/' \
  --exclude='.ruff_cache/' \
  -e "ssh ${ssh_options[*]}" \
  "$remote_host:$remote_root/" \
  "$results_root/"

# Pass 2: mirror restart states, final/selected checkpoints, and only recently
# active progress.pt files. Historical progress tensors are intentionally not
# retained; they are neither final scientists nor needed to resume active work.
resume_files=$work_root/resume-files.txt
ssh "${ssh_options[@]}" "$remote_host" \
  "cd '$remote_root' && find . -type f \
    \( -name 'state.pt.gz' -o -path '*/checkpoints/*.pt' -o -path '*/frozen-checkpoints/*.pt' \) \
    ! -path '*smoke*' ! -path '*invalidated*' ! -path '*archive*' -print" \
  | sed 's#^\./##' \
  | while IFS= read -r relative; do
      if [[ ${relative##*/} == progress.pt ]]; then
        # -n prevents this nested SSH call from consuming the selection
        # pipeline on stdin. It must not be added to rsync's SSH transport.
        if ssh -n "${ssh_options[@]}" "$remote_host" \
          "find '$remote_root/$relative' -mtime -$active_progress_days -print -quit" \
          | grep -q .; then
          printf '%s\n' "$relative"
        fi
      else
        printf '%s\n' "$relative"
      fi
    done \
  | sort -u > "$resume_files"

if [[ -s "$resume_files" ]]; then
  rsync -a --files-from="$resume_files" \
    -e "ssh ${ssh_options[*]}" \
    "$remote_host:$remote_root/" \
    "$resume_root/"
fi
cp "$resume_files" "$state_root/current-resume-files.txt"

# Capture the exact execution environment required to reconstruct another VM.
# Every command below is read-only on Nebius.
ssh "${ssh_options[@]}" "$remote_host" '
  set -u
  printf "captured_at=%s\n" "$(date -Is)"
  printf "hostname=%s\n" "$(hostname)"
  uname -a
  printf "\n--- os-release ---\n"
  cat /etc/os-release 2>/dev/null || true
  printf "\n--- cpu ---\n"
  lscpu 2>/dev/null || true
  printf "\n--- memory ---\n"
  free -h 2>/dev/null || true
  printf "\n--- disks ---\n"
  df -h / /srv 2>/dev/null || true
' > "$provenance_root/host.txt"

ssh "${ssh_options[@]}" "$remote_host" '
  printf "%s\n" "--- running experiment services ---"
  systemctl list-units --type=service --state=running --no-legend 2>/dev/null \
    | grep -E "(braid|semantic|sv2|r200|r24|invariant|l1000|raster)" || true
  printf "%s\n" "--- timers ---"
  systemctl list-timers --all --no-pager 2>/dev/null || true
  printf "%s\n" "--- matching unit definitions ---"
  systemctl list-unit-files --type=service --no-legend 2>/dev/null \
    | awk "{print \$1}" \
    | grep -E "(braid|semantic|sv2|r200|r24|invariant|l1000|raster)" \
    | while IFS= read -r unit; do
        printf "\n===== %s =====\n" "$unit"
        systemctl cat "$unit" 2>/dev/null || true
      done
' > "$provenance_root/systemd.txt"

remote_repos=$work_root/remote-repos.txt
ssh "${ssh_options[@]}" "$remote_host" \
  "find /srv/braid -maxdepth 3 \( -type d -o -type f \) -name .git -print 2>/dev/null" \
  | sed 's#/.git$##' \
  | sort -u > "$remote_repos"

repo_index=$provenance_root/repos/index.tsv
: > "$repo_index"
while IFS= read -r remote_repo; do
  [[ -n "$remote_repo" ]] || continue
  label=$(printf '%s' "$remote_repo" | sed 's#^/##; s#[^A-Za-z0-9._-]#_#g')
  printf '%s\t%s\n' "$label" "$remote_repo" >> "$repo_index"
  ssh "${ssh_options[@]}" "$remote_host" \
    "git -C '$remote_repo' rev-parse HEAD; git -C '$remote_repo' status --short --branch" \
    > "$provenance_root/repos/$label.status.txt" 2>&1 || true
  ssh "${ssh_options[@]}" "$remote_host" \
    "git -C '$remote_repo' diff --binary" \
    > "$provenance_root/repos/$label.patch" 2>/dev/null || true
  ssh "${ssh_options[@]}" "$remote_host" \
    "cd '$remote_repo' && git ls-files --others --exclude-standard -z \
      | tar --null -T - -czf -" \
    > "$provenance_root/repos/$label.untracked.tar.gz" 2>/dev/null || true
done < "$remote_repos"

printf '%s\n' "$remote_host:$remote_root" > "$provenance_root/source.txt"
printf '%s\n' "$stamp" > "$provenance_root/captured-at-utc.txt"

current_manifest=$state_root/current.manifest.tsv
previous_manifest=$state_root/previous.manifest.tsv
if [[ -f "$current_manifest" ]]; then
  cp "$current_manifest" "$previous_manifest"
fi

find "$mirror_root" -type f -print0 \
  | sort -z \
  | while IFS= read -r -d '' path; do
      relative=${path#"$mirror_root"/}
      size=$(stat -f %z "$path")
      modified=$(stat -f %m "$path")
      digest=$(shasum -a 256 "$path" | awk '{print $1}')
      printf '%s\t%s\t%s\t%s\n' "$digest" "$size" "$modified" "$relative"
    done > "$current_manifest"

file_count=$(wc -l < "$current_manifest" | tr -d ' ')
mirror_bytes=$(du -sk "$mirror_root" | awk '{print $1 * 1024}')
printf 'snapshot=%s\nfiles=%s\nbytes=%s\nsource=%s:%s\n' \
  "$stamp" "$file_count" "$mirror_bytes" "$remote_host" "$remote_root" \
  > "$state_root/current-summary.txt"

printf 'mirror_root=%s\nmanifest=%s\nfiles=%s\nbytes=%s\n' \
  "$mirror_root" "$current_manifest" "$file_count" "$mirror_bytes"

if [[ "$command_name" == sync ]]; then
  exit 0
fi

last_packaged=$state_root/last-packaged.manifest.tsv
changed_files=$work_root/changed-files.txt
deleted_files=$work_root/deleted-files.txt

if [[ "$command_name" == full || ! -f "$last_packaged" ]]; then
  capsule_kind=full
  cut -f4- "$current_manifest" > "$changed_files"
  : > "$deleted_files"
else
  capsule_kind=delta
  awk -F '\t' '
    NR == FNR {old[$4] = $1; next}
    !($4 in old) || old[$4] != $1 {print $4}
  ' "$last_packaged" "$current_manifest" > "$changed_files"
  awk -F '\t' '
    NR == FNR {current[$4] = 1; next}
    !($4 in current) {print $4}
  ' "$current_manifest" "$last_packaged" > "$deleted_files"
fi

changed_count=$(wc -l < "$changed_files" | tr -d ' ')
deleted_count=$(wc -l < "$deleted_files" | tr -d ' ')
if (( changed_count == 0 && deleted_count == 0 )); then
  printf 'capsule=none\n'
  exit 0
fi

capsule_name=nebius-semantic-v2-${capsule_kind}-${stamp}
capsule_dir=$outbox_root/$capsule_name
mkdir -p "$capsule_dir"
archive=$capsule_dir/$capsule_name.tar.gz

if (( changed_count > 0 )); then
  tar -czf "$archive" -C "$mirror_root" -T "$changed_files"
else
  tar -czf "$archive" --files-from /dev/null
fi

cp "$current_manifest" "$capsule_dir/$capsule_name.manifest.tsv"
cp "$deleted_files" "$capsule_dir/$capsule_name.deleted-paths.txt"
if [[ ! -s "$capsule_dir/$capsule_name.deleted-paths.txt" ]]; then
  printf '# no deleted paths\n' > "$capsule_dir/$capsule_name.deleted-paths.txt"
fi
archive_digest=$(shasum -a 256 "$archive" | awk '{print $1}')
archive_bytes=$(stat -f %z "$archive")
printf '%s  %s\n' "$archive_digest" "$(basename "$archive")" \
  > "$capsule_dir/$capsule_name.tar.gz.sha256"

cat > "$capsule_dir/$capsule_name.README.txt" <<EOF
Nebius semantic-v2 recovery capsule
kind=$capsule_kind
created_at_utc=$stamp
source=$remote_host:$remote_root
changed_files=$changed_count
deleted_files=$deleted_count
archive_bytes=$archive_bytes

Restore capsules in timestamp order into an explicit, empty recovery directory.
For every capsule: concatenate .part-* files when present, verify SHA-256,
extract the tar.gz, then apply deleted-paths.txt only inside that recovery root.
The newest manifest is the authoritative final file/checksum inventory.
EOF

if (( archive_bytes > 90 * 1024 * 1024 )); then
  split -b 90m -a 3 "$archive" "$archive.part-"
  (
    cd "$capsule_dir"
    shasum -a 256 "$capsule_name.tar.gz".part-* \
      > "$capsule_name.parts.sha256"
  )
  rm -f "$archive"
fi

find "$capsule_dir" -maxdepth 1 -type f ! -name UPLOAD_FILES.txt -print \
  | sort > "$capsule_dir/UPLOAD_FILES.txt"
printf '%s\n' "$capsule_kind" > "$capsule_dir/PENDING"
cp "$current_manifest" "$last_packaged"

printf 'capsule=%s\nkind=%s\nchanged_files=%s\ndeleted_files=%s\nupload_list=%s\n' \
  "$capsule_dir" "$capsule_kind" "$changed_count" "$deleted_count" \
  "$capsule_dir/UPLOAD_FILES.txt"
