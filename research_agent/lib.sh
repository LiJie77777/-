# Shared helpers for run_daily.sh / run_weekly.sh (sourced, not executed).
# Layout on the server (created by deploy/setup.sh):
#   $RAFEED_CODE  checkout of the code branch (this directory lives inside it)
#   $RAFEED_DATA  checkout of the orphan branch research-feed (data only: research/...)
#   $RAFEED_STATE local cache: listings, PDFs, texts, inbox, Codex logs (never committed)

AGENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HOME/.config/rafeed/env" ] && . "$HOME/.config/rafeed/env"   # S2_API_KEY, CODEX_API_KEY, overrides
RAFEED_CODE="${RAFEED_CODE:-$(cd "$AGENT_DIR/.." && pwd)}"
RAFEED_DATA="${RAFEED_DATA:-$HOME/rafeed-data}"
RAFEED_STATE="${RAFEED_STATE:-$HOME/.cache/rafeed}"
DAY="${DAY:-$(date -u +%F)}"
CONFIG="$AGENT_DIR/config.toml"
MODEL="${CODEX_MODEL:-$(python3 - "$CONFIG" <<'EOF'
import sys
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
print(tomllib.load(open(sys.argv[1], "rb"))["codex"]["model"])
EOF
)}"
INBOX="$RAFEED_STATE/inbox/$DAY"
mkdir -p "$INBOX"

log() { echo "[$(date -u +%FT%TZ)] $*"; }
warn() { log "WARN $*"; echo "$*" >> "$INBOX/errors.txt"; }
safe() { echo "${1//[^A-Za-z0-9._-]/_}"; }

rf() {  # deterministic steps; S2_API_KEY is only visible here
  (cd "$AGENT_DIR" && python3 -m rafeed.cli --config "$CONFIG" --data "$RAFEED_DATA/research" \
     --state "$RAFEED_STATE" --date "$DAY" "$@")
}

codex_run() {  # $1 schema, $2 output file, $3 event log; prompt on stdin; extra codex args after
  local schema="$1" out="$2" events="$3"; shift 3
  # read-only sandbox: the model can read files but cannot write or reach the network from
  # shell commands; no git/S2 credentials in its environment. The output is the final
  # message, validated against the schema, written by Codex itself (-o).
  (cd "$AGENT_DIR" && env -u GH_TOKEN -u GITHUB_TOKEN -u S2_API_KEY -u GIT_SSH_COMMAND \
     codex exec -m "$MODEL" --sandbox read-only --output-schema "$schema" -o "$out" --json "$@" -) \
     > "$events" 2>> "$INBOX/codex.stderr"
}

screen_all() {
  local chunk out
  for chunk in $(rf screen-input --chunk "${SCREEN_CHUNK:-120}"); do
    out="${chunk%.jsonl}.out.json"
    if rf prompt --kind screen --input "$chunk" \
         | codex_run "$AGENT_DIR/schemas/screen.schema.json" "$out" "$INBOX/codex_$(basename "${chunk%.jsonl}").jsonl" \
       && [ -s "$out" ]; then
      rf apply-screen --file "$out" --input "$chunk" || warn "apply-screen failed: $chunk"
    else
      warn "codex screening failed: $chunk"
    fi
  done
}

deep_read() {  # $1 = how many papers
  local id txt out
  for id in $(rf select-deep --k "$1"); do
    txt="$(rf fetch-text --id "$id")" || { warn "fetch-text failed: $id"; continue; }
    out="$INBOX/card_$(safe "$id").json"
    if rf prompt --kind extract --id "$id" --input "$txt" \
         | codex_run "$AGENT_DIR/schemas/card.schema.json" "$out" "$INBOX/codex_card_$(safe "$id").jsonl" \
       && [ -s "$out" ]; then
      rf apply-card --id "$id" --file "$out" --model "$MODEL" || warn "apply-card failed: $id"
    else
      warn "codex extraction failed: $id (retried once more, then dropped from the queue)"
      rf mark-failed --id "$id" --what extract
    fi
  done
}

pull_code_and_data() {
  git -C "$RAFEED_CODE" pull --ff-only -q || warn "code pull failed (running the current checkout)"
  git -C "$RAFEED_DATA" pull --rebase -q origin research-feed || warn "data pull failed"
}

push_data() {  # $1 = commit message suffix
  git -C "$RAFEED_DATA" add research
  if git -C "$RAFEED_DATA" diff --cached --quiet; then log "nothing to push"; return 0; fi
  git -C "$RAFEED_DATA" commit -q -m "feed: $1"
  local i
  for i in 1 2 3 4; do
    git -C "$RAFEED_DATA" pull --rebase -q origin research-feed \
      && git -C "$RAFEED_DATA" push -q origin HEAD:research-feed && { log "pushed"; return 0; }
    sleep $((2 ** i))
  done
  warn "push failed"; return 1
}
