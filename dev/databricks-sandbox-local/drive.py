"""Drive the local #906 Omnigent server through its HTTP API only.

  drive.py create [--agent NAME] [--message TEXT]  → new managed (Databricks Sandbox) session
  drive.py status SESSION                           → session + host snapshot
  drive.py shell SESSION CMD                        → run CMD in the session environment
  drive.py send SESSION TEXT                        → post a user turn and wait for the reply
  drive.py stop SESSION                             → stop the session's Sandbox (real API)
  drive.py resume-test SESSION                      → stop Sandbox, wait host offline, send turn, time host online
  drive.py hosts                                    → list hosts

Prints timings. Auth: the local accounts admin (password in ~/.omnigent-og906/admin-password).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:8906"
D = Path.home() / ".omnigent-og906"


def client() -> httpx.Client:
    pw = (D / "admin-password").read_text().strip()
    r = httpx.post(f"{BASE}/auth/login", json={"username": "randy", "password": pw}, timeout=30)
    r.raise_for_status()
    return httpx.Client(
        base_url=BASE, headers={"Authorization": f"Bearer {r.json()['token']}"}, timeout=120
    )


def agent_id(c: httpx.Client, name: str) -> str:
    for a in c.get("/v1/agents", params={"limit": 200}).json()["data"]:
        if a["name"] == name:
            return a["id"]
    raise SystemExit(f"no agent {name}")


def session(c: httpx.Client, sid: str) -> dict:
    r = c.get(f"/v1/sessions/{sid}")
    r.raise_for_status()
    return r.json()


def host(c: httpx.Client, host_id: str) -> dict | None:
    for h in c.get("/v1/hosts").json().get("hosts", []):
        if h.get("host_id") == host_id:
            return h
    return None


def host_status(h: dict | None) -> str:
    if h is None:
        return "absent"
    return str(h.get("status"))


def wait_host(c: httpx.Client, sid: str, want: str, timeout: float = 600) -> tuple[float, dict]:
    t0 = time.monotonic()
    last = None
    while time.monotonic() - t0 < timeout:
        s = session(c, sid)
        hid = s.get("host_id")
        h = host(c, hid) if hid else None
        st = host_status(h)
        line = (s.get("managed_launch_status") or s.get("launch_status"), hid, st)
        if line != last:
            print(f"  [{time.monotonic() - t0:6.1f}s] launch={line[0]} host={hid} status={st}", flush=True)
            last = line
        err = s.get("managed_launch_error") or s.get("launch_error")
        if err:
            raise SystemExit(f"launch error: {err}")
        if hid and st == want:
            return time.monotonic() - t0, s
        time.sleep(2)
    raise SystemExit(f"timed out waiting for host {want}")


def environment_id(c: httpx.Client, sid: str) -> str:
    r = c.get(f"/v1/sessions/{sid}/resources")
    r.raise_for_status()
    for res in r.json().get("data", []):
        if res.get("type") == "environment" or str(res.get("id", "")).startswith("env"):
            return res["id"]
    raise SystemExit(f"no environment resource: {r.text[:500]}")


def shell(c: httpx.Client, sid: str, cmd: str) -> dict:
    env = environment_id(c, sid)
    r = c.post(
        f"/v1/sessions/{sid}/resources/environments/{env}/shell",
        json={"command": cmd, "timeout": 60},
    )
    print(r.status_code, json.dumps(r.json(), indent=1)[:3000])
    return r.json()


def items(c: httpx.Client, sid: str) -> list[dict]:
    r = c.get(f"/v1/sessions/{sid}/items", params={"limit": 200})
    r.raise_for_status()
    return r.json().get("data", [])


def send(c: httpx.Client, sid: str, text: str, timeout: float = 600) -> float:
    before = len(items(c, sid))
    t0 = time.monotonic()
    r = c.post(
        f"/v1/sessions/{sid}/events",
        json={"type": "message", "data": {"role": "user", "content": [{"type": "input_text", "text": text}]}},
    )
    print("send", r.status_code, r.text[:300])
    r.raise_for_status()
    while time.monotonic() - t0 < timeout:
        s = session(c, sid)
        its = items(c, sid)
        replies = [
            i
            for i in its[before:]
            if i.get("type") == "message" and (i.get("role") or i.get("data", {}).get("role")) == "assistant"
        ]
        if replies and s.get("status") in ("idle", "completed", "waiting"):
            print(f"reply after {time.monotonic() - t0:.1f}s:")
            print(json.dumps(replies[-1], indent=1)[:2000])
            return time.monotonic() - t0
        time.sleep(2)
    print(json.dumps(items(c, sid)[before:], indent=1)[:4000])
    raise SystemExit("no assistant reply in time")


def sandbox_api():
    from databricks.sdk import WorkspaceClient

    return WorkspaceClient(profile="DEFAULT").sandbox


def sandbox_of(c: httpx.Client, sid: str) -> str:
    # /v1/hosts does not expose sandbox ids; the launcher labels each
    # Sandbox with the host name (display_name), so resolve through the API.
    s = session(c, sid)
    h = host(c, s["host_id"]) or {}
    for sb in sandbox_api().list_sandboxes():
        if sb.display_name == h.get("name"):
            return sb.name.split("/", 1)[1]
    raise SystemExit(f"no Sandbox labelled {h.get('name')!r}")


def main() -> None:
    cmd, *args = sys.argv[1:] or ["hosts"]
    c = client()
    if cmd == "hosts":
        print(json.dumps(c.get("/v1/hosts").json(), indent=1)[:4000])
    elif cmd == "create":
        agent = args[args.index("--agent") + 1] if "--agent" in args else "codex-native-ui"
        body = {"agent_id": agent_id(c, agent), "host_type": "managed", "sandbox_provider": "databricks"}
        t0 = time.monotonic()
        r = c.post("/v1/sessions", json=body)
        print("create", r.status_code, r.text[:400])
        r.raise_for_status()
        sid = r.json()["id"]
        print("session", sid)
        took, s = wait_host(c, sid, "online")
        print(f"HOST ONLINE: create→online {time.monotonic() - t0:.1f}s host={s['host_id']} workspace={s.get('workspace')}")
    elif cmd == "status":
        s = session(c, args[0])
        print(json.dumps(s, indent=1)[:3000])
        print(json.dumps(host(c, s.get("host_id") or ""), indent=1))
    elif cmd == "shell":
        shell(c, args[0], args[1])
    elif cmd == "send":
        send(c, args[0], args[1])
    elif cmd == "stop":
        sb = sandbox_of(c, args[0])
        sandbox_api().stop_sandbox(f"sandboxes/{sb}")
        print("stop requested", sb)
    elif cmd == "resume-test":
        sid = args[0]
        sb = sandbox_of(c, sid)
        api = sandbox_api()
        t0 = time.monotonic()
        api.stop_sandbox(f"sandboxes/{sb}")
        print("stop requested", sb)
        took, _ = wait_host(c, sid, "offline", timeout=900)
        print(f"HOST OFFLINE after stop: {took:.1f}s")
        while str(api.get_sandbox(f"sandboxes/{sb}").status.state.value) != "SANDBOX_STATE_STOPPED":
            time.sleep(2)
        print(f"sandbox STOPPED after {time.monotonic() - t0:.1f}s")
        t1 = time.monotonic()
        took = send(c, sid, args[1] if len(args) > 1 else "Reply with exactly: RESUMED OK")
        print(f"RESUME: turn-after-stop answered in {took:.1f}s; host={host_status(host(c, session(c, sid)['host_id']))} ({time.monotonic() - t1:.1f}s)")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__" and sys.argv[1:2] != ["e2e"]:
    main()


def create_on_host(agent: str, host_id: str, workspace: str) -> None:
    c = client()
    body = {"agent_id": agent_id(c, agent), "host_id": host_id, "workspace": workspace}
    r = c.post("/v1/sessions", json=body)
    print("create", r.status_code, r.text[:300])
    r.raise_for_status()
    print("session", r.json()["id"])


def e2e(agent: str = "claude-native-ui", sid: str | None = None) -> None:
    """Full proof: create → host online → shell → turn → stop → offline → turn (wake)."""
    c = client()
    t0 = time.monotonic()
    if sid is None:
        r = c.post("/v1/sessions", json={"agent_id": agent_id(c, agent), "host_type": "managed", "sandbox_provider": "databricks"})
        r.raise_for_status()
        sid = r.json()["id"]
    print("session", sid, flush=True)
    _, s = wait_host(c, sid, "online")
    t_online = time.monotonic() - t0
    print(f"T create→host online: {t_online:.1f}s host={s['host_id']}", flush=True)
    while not (session(c, sid).get("runner_id") and c.get(f"/v1/sessions/{sid}/resources").status_code == 200):
        time.sleep(2)
    print(f"T create→runner ready: {time.monotonic() - t0:.1f}s", flush=True)
    out = shell(c, sid, 'id -un; hostname; omnigent --version; ps -eo args | grep -c "[O]MNIGENT_HOST_TOKEN=" || true')
    assert out["exit_code"] == 0 and out["stdout"].strip().endswith("0"), "token visible in argv"
    tool = agent.startswith("claude")
    t_turn = send(c, sid, "Use your Bash tool to run `hostname && id -un`, then reply with exactly: SANDBOX TURN OK plus the output." if tool else "Reply with exactly: SANDBOX TURN OK")
    sb = sandbox_of(c, sid)
    api = sandbox_api()
    t1 = time.monotonic()
    api.stop_sandbox(f"sandboxes/{sb}")
    took_off, _ = wait_host(c, sid, "offline", timeout=900)
    while str(api.get_sandbox(f"sandboxes/{sb}").status.state.value) != "SANDBOX_STATE_STOPPED":
        time.sleep(2)
    t_stopped = time.monotonic() - t1
    t_wake = send(c, sid, "Reply with exactly: RESUMED OK")
    print(json.dumps({
        "session": sid, "host_id": s["host_id"], "sandbox": sb,
        "create_to_online_s": round(t_online, 1), "first_turn_s": round(t_turn, 1),
        "stop_to_host_offline_s": round(took_off, 1), "stop_to_STOPPED_s": round(t_stopped, 1),
        "turn_after_stop_s": round(t_wake, 1),
    }, indent=1))


if __name__ == "__main__" and sys.argv[1:2] == ["e2e"]:
    e2e(*sys.argv[2:])
