#!/usr/bin/env bash
# Daily run: new arXiv submissions + newly published accepted-paper lists
#   -> venue verification -> Codex screening -> Codex deep reads -> validation -> digest -> push.
# Usage: run_daily.sh [YYYY-MM-DD]     (cron: see deploy/crontab.example)
set -uo pipefail
DAY="${1:-$(date -u +%F)}"
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
exec 9>"$RAFEED_STATE/run.lock"; flock -n 9 || { log "another run is active"; exit 0; }

log "daily $DAY model=$MODEL"
pull_code_and_data
rf harvest            || warn "harvest failed"
rf enrich --max 300   || warn "enrich failed"
rf verify             || warn "verify failed"
screen_all
deep_read "${MAX_DEEP_READS:-4}"
rf render             || warn "render failed"
push_data "daily $DAY"
