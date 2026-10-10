#!/usr/bin/env bash
# One-time setup on the server (run as the user that will run cron).
#   bash setup.sh git@github.com:LiJie77777/-.git <code-branch>
# Step 1 creates a deploy key and stops; add it to GitHub (repo Settings -> Deploy keys,
# tick "Allow write access"), then run the same command again.
set -euo pipefail
REPO="${1:?usage: setup.sh <repo ssh url> [code branch]}"
CODE_BRANCH="${2:-main}"
KEY="$HOME/.ssh/rafeed_deploy"
CODE="${RAFEED_CODE:-$HOME/rafeed-code}"
DATA="${RAFEED_DATA:-$HOME/rafeed-data}"
STATE="${RAFEED_STATE:-$HOME/.cache/rafeed}"

need() { command -v "$1" >/dev/null || { echo "missing: $1 ($2)"; exit 1; }; }
need git "apt install git"
need python3 "Python >= 3.11, or pip install tomli"
need pdftotext "apt install poppler-utils"
need flock "util-linux"
need codex "npm i -g @openai/codex, then: printenv OPENAI_API_KEY | codex login --with-api-key"
python3 -c 'import sys; import tomllib' 2>/dev/null || python3 -c 'import tomli' \
  || { echo "Python < 3.11: pip install tomli"; exit 1; }

if [ ! -f "$KEY" ]; then
  mkdir -p "$HOME/.ssh" && ssh-keygen -t ed25519 -N "" -f "$KEY" -C "rafeed-deploy"
  echo; echo "== Add this public key to GitHub: repo Settings -> Deploy keys -> Add, tick 'Allow write access':"
  cat "$KEY.pub"; echo; echo "Then run this script again."; exit 0
fi
SSH="ssh -i $KEY -o IdentitiesOnly=yes"

if [ ! -d "$CODE/.git" ]; then GIT_SSH_COMMAND="$SSH" git clone -q -b "$CODE_BRANCH" "$REPO" "$CODE"; fi
git -C "$CODE" config core.sshCommand "$SSH"

if [ ! -d "$DATA/.git" ]; then
  GIT_SSH_COMMAND="$SSH" git clone -q "$REPO" "$DATA"
  git -C "$DATA" config core.sshCommand "$SSH"
  git -C "$DATA" config user.name "rafeed-bot"
  git -C "$DATA" config user.email "rafeed-bot@users.noreply.github.com"
  if git -C "$DATA" ls-remote --exit-code --heads origin research-feed >/dev/null 2>&1; then
    git -C "$DATA" fetch -q origin research-feed
    git -C "$DATA" checkout -q -B research-feed origin/research-feed
  else  # orphan data branch: no code, no shared history, so it never conflicts with code branches
    git -C "$DATA" checkout -q --orphan research-feed
    git -C "$DATA" rm -rqf . && git -C "$DATA" clean -fdq
    mkdir -p "$DATA/research"
    cp "$CODE/research_agent/deploy/data_README.md" "$DATA/research/README.md"
    printf '# 给调研 agent 的反馈（由 Claude 维护）\n\n（暂无）\n' > "$DATA/research/feedback.md"
    printf '# 整合记录（由 Claude 维护）\n\n' > "$DATA/research/integration_log.md"
    git -C "$DATA" add research && git -C "$DATA" commit -qm "init research-feed"
    git -C "$DATA" push -q -u origin research-feed
  fi
fi

mkdir -p "$STATE" "$HOME/.config/rafeed"
if [ ! -f "$HOME/.config/rafeed/env" ]; then
  cat > "$HOME/.config/rafeed/env" <<EOF
# rafeed environment (chmod 600). Leave a line commented to use the default.
export RAFEED_CODE="$CODE"
export RAFEED_DATA="$DATA"
export RAFEED_STATE="$STATE"
# export S2_API_KEY=...        # Semantic Scholar key (free; 1 request/s); strongly recommended
# export CODEX_MODEL=gpt-6.1-sol
# export MAX_DEEP_READS=4
EOF
  chmod 600 "$HOME/.config/rafeed/env"
fi
chmod +x "$CODE/research_agent/"*.sh
echo "OK. Smoke test:  $CODE/research_agent/run_daily.sh"
echo "First backfill:  ENRICH_MAX=3000 $CODE/research_agent/run_weekly.sh --backfill"
echo "Cron:            see $CODE/research_agent/deploy/crontab.example"
