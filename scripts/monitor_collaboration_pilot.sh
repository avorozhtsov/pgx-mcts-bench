#!/usr/bin/env bash
set -euo pipefail

artifact_root=${1:-artifacts/collaboration-pilot-200}

echo "host load:"
uptime
echo
echo "artifact storage:"
du -sh "$artifact_root" 2>/dev/null || true
df -h "$artifact_root" 2>/dev/null || true
echo
echo "collaboration processes:"
ps -axo pid,etime,%cpu,%mem,command | grep '[b]raid-collaborative-scientists' || true
echo
echo "committed rounds:"
find "$artifact_root" -path '*/rounds/[0-9][0-9][0-9][0-9][0-9][0-9]' -type d 2>/dev/null \
  | sed 's#/rounds/[0-9][0-9][0-9][0-9][0-9][0-9]$##' \
  | sort | uniq -c | sort -k2
echo
echo "recent failures:"
grep -H -E 'Traceback|Error:|failed' "$artifact_root"/logs/*.log 2>/dev/null | tail -20 || true
