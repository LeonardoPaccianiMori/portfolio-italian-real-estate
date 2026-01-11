#!/usr/bin/env bash
set -euo pipefail

cols="$(tput cols 2>/dev/null || true)"
lines="$(tput lines 2>/dev/null || true)"
if [[ -z "$cols" || -z "$lines" ]]; then
  if stty_size="$(stty size 2>/dev/null || true)"; then
    lines="${lines:-${stty_size%% *}}"
    cols="${cols:-${stty_size##* }}"
  fi
fi
cols="${cols:-0}"
lines="${lines:-0}"
term="${TERM:-xterm-256color}"

exec script -q /dev/null -c "docker compose exec -e TERM=${term} -e COLUMNS=${cols} -e LINES=${lines} app python scripts/pipeline_tui.py"
