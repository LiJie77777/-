#!/usr/bin/env bash
# Weekly run: citation snowballing from seeds and relevant papers, Codex expansion with live
# web search (queries + titles only; titles are resolved by Semantic Scholar or dropped),
# venue re-checks, then the same screening / deep reads as the daily run.
# Usage: run_weekly.sh [YYYY-MM-DD]    Backfill: run_weekly.sh --backfill (first run)
set -uo pipefail
BACKFILL=""
if [ "${1:-}" = "--backfill" ]; then BACKFILL="--backfill"; shift; fi
DAY="${1:-$(date -u +%F)}"
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
exec 9>"$RAFEED_STATE/run.lock"; flock -n 9 || { log "another run is active"; exit 0; }

log "weekly $DAY model=$MODEL $BACKFILL"
pull_code_and_data
if [ -n "$BACKFILL" ]; then rf harvest --backfill --skip-arxiv || warn "backfill harvest failed"; fi
rf snowball || warn "snowball failed"
if rf prompt --kind expand \
     | codex_run "$AGENT_DIR/schemas/expand.schema.json" "$INBOX/expand.json" "$INBOX/codex_expand.jsonl" \
         -c 'web_search="live"' \
   && [ -s "$INBOX/expand.json" ]; then
  rf resolve --file "$INBOX/expand.json" || warn "resolve failed"
else
  warn "codex expansion failed"
fi
rf enrich --max "${ENRICH_MAX:-600}" || warn "enrich failed"
rf verify || warn "verify failed"
screen_all
deep_read "${MAX_DEEP_READS_WEEKLY:-12}"
rf render || warn "render failed"
push_data "weekly $DAY"
