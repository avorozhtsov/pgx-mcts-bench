#!/usr/bin/env bash

# Incrementally mirror the live Nebius rung-18 artifact root and package only
# files that are new or changed since the previous local sync.

set -Eeuo pipefail

remote_host=${REMOTE_HOST:-ubuntu@195.242.13.78}
remote_root=${REMOTE_ROOT:-/srv/braid/rung18}
local_root=${LOCAL_ROOT:-artifacts/nebius-rung18-20260801-current}
state_root=${STATE_ROOT:-artifacts/nebius-rung18-20260801-sync}
delta_root=${DELTA_ROOT:-artifacts/nebius-rung18-20260801-deltas}
stamp=${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}

mkdir -p "$local_root" "$state_root" "$delta_root"
changes=$(mktemp)
changed_files=$(mktemp)
trap 'rm -f "$changes" "$changed_files"' EXIT

rsync -a \
  --itemize-changes \
  --out-format='%i|%n' \
  --exclude='*.tmp' \
  --exclude='*.tmp.*' \
  "$remote_host:$remote_root/" \
  "$local_root/" \
  | tee "$changes"

awk -F '|' '$1 ~ /^>f/ {print $2}' "$changes" > "$changed_files"

archive=""
if [[ -s "$changed_files" ]]; then
  archive="$delta_root/nebius-rung18-delta-$stamp.tar.gz"
  tar -czf "$archive" -C "$local_root" -T "$changed_files"
  (
    cd "$(dirname "$archive")"
    shasum -a 256 "$(basename "$archive")" > "$(basename "$archive").sha256"
  )
  if (( $(stat -f %z "$archive") > 100 * 1024 * 1024 )); then
    split -b 90m "$archive" "$archive.part-"
    (
      cd "$(dirname "$archive")"
      shasum -a 256 "$(basename "$archive")".part-* \
        > "$(basename "$archive").parts.sha256"
    )
  fi
fi

manifest="$state_root/current.manifest.tsv"
if [[ -f "$manifest" ]]; then
  cp "$manifest" "$state_root/previous.manifest.tsv"
fi
find "$local_root" -type f -print0 \
  | sort -z \
  | while IFS= read -r -d '' path; do
      relative=${path#"$local_root"/}
      size=$(stat -f %z "$path")
      modified=$(stat -f %m "$path")
      digest=$(shasum -a 256 "$path" | awk '{print $1}')
      printf '%s\t%s\t%s\t%s\n' "$digest" "$size" "$modified" "$relative"
    done > "$manifest"

printf 'changed_files=%s\n' "$(wc -l < "$changed_files" | tr -d ' ')"
printf 'local_root=%s\n' "$local_root"
printf 'manifest=%s\n' "$manifest"
if [[ -n "$archive" ]]; then
  printf 'delta_archive=%s\n' "$archive"
  if compgen -G "$archive.part-*" >/dev/null; then
    printf 'delta_parts=%s.part-*\n' "$archive"
  fi
else
  printf 'delta_archive=none\n'
fi
