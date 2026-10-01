# Local Omnigent server with Databricks Sandbox hosts (REST exec, no SSH)

Fork-only dev harness for the `databricks` sandbox provider
(`omnigent/onboarding/sandboxes/databricks_sandbox.py`). It runs a local
server whose managed sessions provision real Databricks Sandboxes and
bootstrap `omnigent host` inside them through `execute_command_sync`.

State lives in `~/.omnigent-og906/` (SQLite, logs, generated config,
accounts admin password). Nothing here touches a deployed App.

```bash
# 1. public URL the in-sandbox host dials back to (QUIC is blocked here → http2)
tmux new -d -s og906-tunnel "cloudflared tunnel --no-autoupdate --protocol http2 \
  --url http://127.0.0.1:8906 2>&1 | tee /tmp/og906/cloudflared.log"
# 2. the server (re-generates config.yaml from the tunnel URL + pushed HEAD)
cp run-server.sh drive.py ~/.omnigent-og906/
tmux new -d -s og906-server ~/.omnigent-og906/run-server.sh
# 3. proof, HTTP API only: create → host online → shell → turn → stop → wake turn
.venv/bin/python ~/.omnigent-og906/drive.py e2e codex-native-ui
```

Needs `databricks-sdk>=0.143` in the venv (`uv pip install --exclude-newer
2026-10-01T00:00:00Z databricks-sdk==0.143.0`) and a `DEFAULT` profile.
The bootstrap step `uv tool install`s this branch's pushed HEAD into the
Sandbox so host and server versions match (the image bakes an older omnigent).
