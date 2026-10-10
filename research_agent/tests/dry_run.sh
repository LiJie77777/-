#!/usr/bin/env bash
# Plumbing test of run_daily.sh / run_weekly.sh with no network and no real model:
# HTTP is served from tests/fixtures (RAFEED_FIXTURES), `codex` is a stub that answers
# with schema-shaped JSON, and the data branch is pushed to a local bare repository.
# Usage: bash research_agent/tests/dry_run.sh
set -euo pipefail
AGENT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT

mkdir -p "$T/bin" "$T/home"
cat > "$T/bin/codex" <<'EOF'
#!/usr/bin/env python3
import json, re, sys
args = sys.argv[1:]
open(sys.argv[0] + ".args", "a").write(json.dumps(args) + "\n")
schema, out = args[args.index("--output-schema") + 1], args[args.index("-o") + 1]
assert args[0] == "exec" and args[-1] == "-" and "--json" in args
assert args[args.index("--sandbox") + 1] == "read-only"
prompt = sys.stdin.read()
if schema.endswith("screen.schema.json"):
    path = re.search(r"`([^`]+screen_in_\d+\.jsonl)`", prompt).group(1)
    rows = [json.loads(l) for l in open(path)]
    res = {"items": [{"id": r["id"], "relevance": 2, "topics": ["geometry"], "evidence_type": "method",
                      "deep_read": False, "reason": "stub"} for r in rows]}
else:
    res = {"queries": [], "seed_titles": [], "gaps": ["stub"], "notes": ""}
json.dump(res, open(out, "w"))
print(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 1000, "cached_input_tokens": 200, "output_tokens": 50}}))
EOF
chmod +x "$T/bin/codex"

git init -q --bare "$T/remote.git"
git clone -q "$T/remote.git" "$T/data" 2>/dev/null
git -C "$T/data" checkout -q --orphan research-feed
git -C "$T/data" -c user.name=t -c user.email=t@t commit -q --allow-empty -m init
git -C "$T/data" push -q -u origin research-feed 2>/dev/null
git -C "$T/data" config user.name rafeed-bot && git -C "$T/data" config user.email bot@example.com

export HOME="$T/home" PATH="$T/bin:$PATH" RAFEED_FIXTURES="$AGENT/tests/fixtures"
export RAFEED_CODE="$T/nocode" RAFEED_DATA="$T/data" RAFEED_STATE="$T/state" MAX_DEEP_READS=0 MAX_DEEP_READS_WEEKLY=0
bash "$AGENT/run_daily.sh" 2026-10-10 > "$T/daily.log" 2>&1 || { cat "$T/daily.log"; exit 1; }
bash "$AGENT/run_weekly.sh" 2026-10-11 > "$T/weekly.log" 2>&1 || { cat "$T/weekly.log"; exit 1; }

git -C "$T/remote.git" ls-tree -r --name-only research-feed > "$T/files"
grep -q '^research/registry.jsonl$' "$T/files"
grep -q '^research/digests/2026-10-10.md$' "$T/files"
grep -q '^research/runs/2026-10-10.json$' "$T/files"
grep -q '^research/digests/2026-10-11.md$' "$T/files"
test "$(git -C "$T/remote.git" rev-list --count research-feed)" = 3
python3 - "$T" <<'EOF'
import json, sys
t = sys.argv[1]
calls = [json.loads(l) for l in open(f"{t}/bin/codex.args")]
assert any("web_search=\"live\"" in c for c in calls), calls      # only the weekly expansion
assert sum("web_search=\"live\"" in c for c in calls) == 1
run = json.load(open(f"{t}/data/research/runs/2026-10-10.json"))
assert run["screened"] == 4 and run["usage"]["input_tokens"] == 1000, run
EOF
echo "dry run OK: $(wc -l < "$T/files") files on research-feed, $(wc -l < "$T/bin/codex.args") codex calls"
