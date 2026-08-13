#!/usr/bin/env bash
set -Eeuo pipefail

pretrain_unit=${PRETRAIN_UNIT:-invariant-oracle-pretrain-run2.service}
foundation=${FOUNDATION:-/srv/braid/artifacts/invariant-oracle-pretrain-20260812}
selection=${SELECTION:-$foundation/selection.json}
r24_output=${R24_OUTPUT:-/srv/braid/artifacts/invariant-oracle-r24-static-no-sharing-20260812}

while [[ "$(systemctl show "$pretrain_unit" -p ActiveState --value)" == active ]]; do
  sleep 60
done
result=$(systemctl show "$pretrain_unit" -p Result --value)
[[ "$result" == success ]] || {
  echo "$pretrain_unit ended with Result=$result; R24 remains gated" >&2
  exit 1
}

python scripts/select_invariant_oracle.py "$foundation" "$selection"
exec scripts/run_invariant_oracle_r24_static_no_sharing_cpu32.sh \
  "$selection" "$r24_output"
