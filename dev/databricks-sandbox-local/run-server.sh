#!/bin/bash
# Local Omnigent server for #906: Databricks Sandbox hosts over REST exec.
# Usage: ~/.omnigent-og906/run-server.sh   (run inside tmux session og906-server)
set -euo pipefail
D="$HOME/.omnigent-og906"
WT="$HOME/projects/omnigent-worktrees/feat-databricks-sandbox-rest"
PORT=8906
URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' /tmp/og906/cloudflared.log | head -1)
[ -n "$URL" ] || { echo "no tunnel URL; start tmux session og906-tunnel first"; exit 1; }
SHA=$(git -C "$WT" rev-parse --short=8 HEAD)
git -C "$WT" branch -r --contains "$SHA" | grep -q "fork/feat/databricks-sandbox-rest" || { echo "HEAD $SHA not pushed to fork"; exit 1; }
SPEC="omnigent @ git+https://github.com/randypitcherii/omnigent@$SHA"
cat > "$D/config.yaml" <<YAML
database_uri: sqlite:///$D/chat.db
sandbox:
  provider: databricks
  server_url: $URL
  databricks:
    profile: DEFAULT
    sandbox_id_prefix: og906
    inactivity_timeout_s: 3600
    # Keep the in-sandbox host on the server's exact commit (idempotent:
    # the install persists on the Sandbox's home disk across stop/start).
    bootstrap_command: >-
      "\$HOME/.local/bin/omnigent" --version 2>/dev/null | grep -q "$SHA" ||
      OMNIGENT_SKIP_WEB_UI=true uv tool install --force --quiet "$SPEC"
YAML
export OMNIGENT_DATA_DIR="$D"
export OMNIGENT_CONFIG_HOME="$D/config-home"
export OMNIGENT_AUTH_ENABLED=1
export OMNIGENT_ACCOUNTS_INIT_ADMIN_USERNAME=randy
export OMNIGENT_ACCOUNTS_INIT_ADMIN_PASSWORD="$(cat "$D/admin-password")"
echo "tunnel: $URL  local: http://127.0.0.1:$PORT"
exec "$WT/.venv/bin/omnigent" server --host 127.0.0.1 --port $PORT -c "$D/config.yaml" --no-open 2>&1 | tee -a "$D/server.log"
