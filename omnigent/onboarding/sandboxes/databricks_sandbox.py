"""
Databricks Sandbox launcher.

Implements :class:`~omnigent.onboarding.sandboxes.base.SandboxLauncher`
for `Databricks Sandbox
<https://docs.databricks.com/aws/en/compute/serverless/sandbox>`_ —
SSH-accessible, microVM-isolated personal development environments.

Unlike every other provider in this package, the transport is not a
Python SDK: it is the **public** ``databricks sandbox`` CLI surface
(``create`` / ``delete`` / ``list`` / ``start`` / ``stop`` / ``status`` /
``ssh`` / ``config``), which went GA when the Databricks CLI renamed
``databricks lakebox`` to ``databricks sandbox``
(https://github.com/databricks/cli/pull/5487). There is therefore no
optional Python extra to install and no internal dependency — an
operator who can run ``databricks sandbox ssh`` by hand can run this
provider. Commands are built as argv lists and executed WITHOUT a shell
locally, so a sandbox id or path never reaches a local shell.

Platform notes that shape this launcher:

- **Real SSH means real port forwarding.** ``databricks sandbox ssh <id>
  -- -L <port>:localhost:<port>`` passes flags straight through to
  ``ssh``, so this is the one exec-model provider in the tree with
  :attr:`supports_local_port_forward` ``True``. Modal, Daytona, and Islo
  all lack a local→sandbox path and must skip the in-sandbox App OAuth
  flow; this provider does not.
- **But the transport allocates no remote PTY.** ``-- -tt <cmd>`` was
  verified against CLI v1.11.0 to still run the remote command with
  ``tty`` reporting "not a tty". :meth:`stream_exec` therefore ignores
  its ``pty`` argument, and :meth:`exec_foreground` runs without a
  controlling terminal. CLIs that suppress output when not attached to a
  terminal — including the ``databricks auth login`` step of the
  in-sandbox App OAuth flow — may degrade. See the module's open
  questions in the accompanying issue.
- **Stop/start with persistent storage.** Sandboxes auto-stop on idle and
  restart with their disk intact, so a dormant managed host is resumed in
  place under the SAME sandbox id (:attr:`can_resume` ``True``) rather
  than reprovisioned. That matters more here than elsewhere: Databricks
  Sandbox enforces a per-user cap on sandbox count, so a
  provision-per-session deployment would eventually hit the ceiling.
- **No image selection and no create-time env injection.** The CLI's
  ``create`` takes only ``--name``; the sandbox image is Databricks'
  own (Ubuntu 24.04, user ``sandbox-agent``) and already ships git,
  tmux, python3, uv, node, and the coding-harness CLIs, so the generic
  :meth:`~omnigent.onboarding.sandboxes.base.ExecModelHostLauncher.start_host`
  bootstrap works unmodified. There is deliberately no ``env``
  passthrough knob: with no create-time env API, the only way to inject
  credentials would be to prefix them onto the remote argv of every
  exec, which exposes them in the sandbox's process table.
- **PEP 668 image.** The sandbox's Python is externally managed
  (``/usr/lib/python3.12/EXTERNALLY-MANAGED``) and its site-packages are
  not writable by ``sandbox-agent``, so :meth:`wheel_install_command`
  passes ``--break-system-packages`` and lets pip fall back to a user
  install, which precedes ``dist-packages`` on ``sys.path``.
"""

from __future__ import annotations

import json
import logging
import re
import shlex
import shutil
import subprocess
import threading
import time
import uuid
from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import ClassVar

import click

from omnigent.onboarding.sandboxes.base import (
    RemoteCommandResult,
    RemoteProcess,
    SandboxLauncher,
    render_host_config_write_command,
    supervise_host_command,
)
from omnigent.onboarding.sandboxes.types import SandboxCapabilities

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────

DEFAULT_CLI_BINARY: str = "databricks"
"""Name of the Databricks CLI executable resolved on ``PATH``."""

INSTALL_HINT: str = (
    "The Databricks CLI is required for the 'databricks' sandbox provider. "
    "Install it (https://docs.databricks.com/aws/en/dev-tools/cli/install) — "
    "v0.270.0 or newer, which is where `databricks sandbox` landed — then run "
    "`databricks auth login --host https://<workspace>` and "
    "`databricks sandbox register`."
)

AUTH_HINT: str = (
    "The Databricks CLI is installed but could not list sandboxes. "
    "Authenticate with `databricks auth login --host https://<workspace>` "
    "(or set DATABRICKS_HOST/DATABRICKS_TOKEN), then register this machine "
    "for sandbox SSH with `databricks sandbox register`."
)

# `databricks sandbox create` blocks until the microVM reports Running.
# A cold create takes tens of seconds; the ceiling covers a slow region
# without hanging a managed launch forever.
_CREATE_TIMEOUT_S: float = 600.0

# `databricks sandbox start` documents its own 10-minute ceiling for a
# stopped sandbox to reach Running; match it rather than cutting it short.
_START_TIMEOUT_S: float = 660.0

# Blocking remote commands (`run`, `put`). The bootstrap's longest single
# command is the wheel install, which is seconds on a warm sandbox; the
# ceiling covers a slow git clone of a large repository without letting a
# wedged SSH session hang a managed launch indefinitely.
_REMOTE_COMMAND_TIMEOUT_S: float = 1800.0

# Short control-plane calls (list / status / config / delete). Generous
# enough for a slow workspace round-trip, short enough that a wedged
# control plane surfaces as an error instead of a stalled launch.
_CONTROL_TIMEOUT_S: float = 120.0

# How long `forward_local_port` waits for ssh to bind the local port
# before giving up. Live measurement was ~2 s; the ceiling covers a
# stopped sandbox that `ssh` auto-starts on connection.
_FORWARD_BIND_TIMEOUT_S: float = 120.0
_FORWARD_POLL_INTERVAL_S: float = 0.25

# Grace period for the port-forward child to exit on SIGTERM before it
# is killed outright.
_FORWARD_TERMINATE_TIMEOUT_S: float = 5.0

# The line `ssh -v` prints once a `-L` listener is actually bound, e.g.
# "debug1: Local forwarding listening on 127.0.0.1 port 8022.". Stable
# across OpenSSH releases and the only non-invasive readiness signal
# available — see `DatabricksSandboxLauncher._await_forward` for why
# probing the port instead is wrong in both possible directions. Built
# per-port so a transcript can never match a different forward.
_FORWARD_READY_TEMPLATE: str = "Local forwarding listening on 127.0.0.1 port {port}"

# Substrings that mean the forward will never come up, so the wait can
# fail immediately instead of burning the full readiness timeout.
#
# `ExitOnForwardFailure=yes` would be the tidier mechanism, but the
# Databricks CLI re-quotes the arguments it forwards to ssh and mangles
# `-o Key=Value` in BOTH spellings — verified live on v1.11.0, where
# `-o ExitOnForwardFailure=yes` yields "Bad configuration option:
# 'exitonforwardfailure / invalid quotes" and the single-token
# `-oExitOnForwardFailure=yes` silently loses `-N`. Watching ssh's own
# failure lines gets the same fail-fast without an `-o` flag.
_FORWARD_FAILURE_MARKERS: tuple[str, ...] = (
    "cannot listen to port",
    "bind: ",
    "could not request local forwarding",
)

# Substrings that mark an `ssh -v` line as explaining a failure, used to
# lift the useful lines out of verbose handshake chatter.
_FORWARD_ERROR_MARKERS: tuple[str, ...] = (
    "address already in use",
    "bind:",
    "cannot listen",
    "permission denied",
    "connection refused",
    "connection closed",
    "could not request local forwarding",
    "ssh: ",
)

# Status strings the control plane reports, lower-cased.
_RUNNING_STATUS: str = "running"
_STOPPED_STATUSES: frozenset[str] = frozenset({"stopped", "stopping", "starting", "pending"})

# Substrings that mark a control-plane response as "this sandbox does not
# exist". Matched case-insensitively against combined output so a delete
# of an already-gone sandbox is idempotent.
_NOT_FOUND_MARKERS: tuple[str, ...] = ("not found", "does not exist", "no such sandbox")

# Line the composed bootstrap script prints as its last stdout line, so the
# job-run output (fetched after the run reaches a terminal state) can be
# parsed back into the workspace path `start_host` must return, without a
# second round trip into the sandbox to ask for it again.
_WORKSPACE_TAG: str = "OMNIGENT_JOB_BOOTSTRAP_WORKSPACE="

# Ceiling on classic-compute job-run completion: cold single-node cluster
# spin-up dominates (observed 3-6 minutes), then the bootstrap script itself
# (clone + host launch) is seconds. Generous enough to cover a slow cluster
# start without letting a wedged run hang a managed launch indefinitely.
_JOB_BOOTSTRAP_TIMEOUT_S: float = 900.0

# Ceiling on getting a WARM bootstrap cluster to Running. Sized like the
# throwaway path's ceiling because the worst case is the same work: a cold
# create, or a restart of a cluster that autoterminated. The point of the warm
# path is that only the first launch after a create/restart waits at all —
# every launch while the cluster stays up skips straight to submission.
_WARM_CLUSTER_READY_TIMEOUT_S: float = 900.0

# Stamped on a launcher-created warm cluster. A long-lived cluster nobody
# recognizes is a cluster someone deletes; this tells an operator scanning the
# workspace's compute list what created it and what deleting it would break.
_WARM_CLUSTER_TAG_KEY: str = "OmnigentSandboxBootstrap"
_WARM_CLUSTER_TAG_VALUE: str = "warm-compute"

# Databricks Apps hostname suffix. A server URL on this host is fronted by the
# Apps OAuth edge, which answers HTTP 302 to anything but a workspace OAuth
# token -- and a sandbox holds a workspace PAT, not an OAuth token, and cannot
# obtain one. See `_resolve_dial_back_url`.
_APPS_HOST_SUFFIX: str = ".databricksapps.com"

# Sandbox control-plane REST surface, as used by the bootstrap notebook.
# Mirrors `cmd/sandbox/api.go` in the Databricks CLI, whose `sandboxAPIRoot`
# still says "lakebox" because the server-side rename is pending.
_LAKEBOX_API_ROOT: str = "/api/2.0/lakebox"

# Port the sandbox SSH gateway listens on (CLI: `defaultGatewayPort`).
_SANDBOX_GATEWAY_PORT: str = "2222"

# Name the bootstrap's SSH key is registered under, in the *job principal's*
# key list. Purely a human label -- the gateway matches on key hash -- so a
# stable one keeps a redeployment from accumulating near-duplicate entries.
_KEY_REGISTRATION_NAME: str = "omnigent-job-bootstrap"

# Gap between control-plane status polls while waiting for a sandbox to
# reach Running. Matches the interval the bootstrap notebook uses.
_REST_POLL_INTERVAL_S: float = 5.0

# Driver notebook staged (and overwritten on every submission) at
# `JobBootstrapConfig.workspace_notebook_path`. Runs on the classic-compute
# cluster and deliberately does NOT shell out to the `databricks` CLI: the
# CLI shipped in DBR images is a stub that refuses non-interactive use
# ("only supported for interactive use from the web terminal ... we
# recommend using the Databricks Python SDK", exit 1), so the CLI's
# `sandbox ssh` behavior is reproduced here directly — resolve the gateway
# and start the sandbox over REST, then exec `ssh` with the argv shape from
# the CLI's `buildSSHArgs`.
#
# Both secrets it needs go through `dbutils.secrets.get`, which Databricks
# redacts from notebook output and logs automatically: the SSH private key
# (long-lived, operator-registered) and the connect payload (transient,
# holding the remote script — which embeds the armed host token, so it must
# never reach the job run JSON).
_JOB_BOOTSTRAP_NOTEBOOK_TEMPLATE: str = '''# Databricks notebook source
import json
import os
import shlex
import stat
import subprocess
import time
import urllib.error
import urllib.request

API_ROOT = "{api_root}"
GATEWAY_PORT = "{gateway_port}"

# Notebook-context credentials rather than a PAT in a secret: this token is
# ephemeral, scoped to the run, and never leaves the driver.
_ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
_token = _ctx.apiToken().get()

# On classic compute `apiUrl()` returns the REGIONAL control-plane host, not
# the workspace-specific one, so the workspace must be named explicitly via
# the org-id header or the gateway rejects the credential as "not sent or of
# an unsupported type for this API" (CLI: `orgIDHeader`).
_host = "{workspace_host}".rstrip("/")
_org_id = "{workspace_id}"


def api(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(_host + path, data=data, method=method)
    req.add_header("Authorization", "Bearer " + _token)
    req.add_header("Content-Type", "application/json")
    if _org_id:
        req.add_header("X-Databricks-Org-Id", _org_id)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(
            f"{{method}} {{path}} failed: HTTP {{exc.code}} {{detail}}"
        ) from None
    return json.loads(raw) if raw.strip() else {{}}


# --- SSH identity -----------------------------------------------------
# Only the private half is stored: the public half is derived here with
# `ssh-keygen -y`, so the operator has one secret to rotate instead of two.
key_material = dbutils.secrets.get(scope="{ssh_key_scope}", key="{ssh_key_key}")
key_path = os.path.expanduser("~/.ssh/sandbox_ed25519")
os.makedirs(os.path.dirname(key_path), exist_ok=True)
with open(key_path, "w") as fh:
    fh.write(key_material)
    if not key_material.endswith("\\n"):
        fh.write("\\n")
os.chmod(key_path, stat.S_IRUSR | stat.S_IWUSR)

# Sandboxes AND registered SSH keys are both per-identity, and this job runs
# as the same principal that created the sandbox. A key registered under a
# human's identity therefore does not open that principal's sandbox -- the
# gateway answers `Permission denied (publickey)`. Registration is keyed by
# key hash and re-POSTing an already-registered key returns the existing
# record unchanged, so doing this on every run is idempotent and cheap; it
# also means a fresh deployment needs no out-of-band registration step for
# the principal.
public_key = subprocess.run(
    ["ssh-keygen", "-y", "-f", key_path],
    capture_output=True,
    text=True,
    timeout=60,
)
if public_key.returncode != 0:
    raise RuntimeError(
        "could not derive the public half of the sandbox key: "
        + public_key.stderr.strip()
    )
api(
    "POST",
    API_ROOT + "/ssh-keys",
    {{"name": "{key_registration_name}", "publicKey": public_key.stdout.strip()}},
)

payload = json.loads(dbutils.secrets.get(scope="{argv_scope}", key="{argv_key}"))
sandbox_id = payload["sandbox_id"]
remote_command = payload["remote_command"]
# Anything the caller marks secret (the armed host token) must not survive
# into this notebook's output. `dbutils.secrets.get` is redacted by
# Databricks automatically, but the remote command's OWN stdout/stderr is
# not, and both `dbutils.notebook.exit` and an exception message land in
# durable job-run JSON that anyone with job-read access can read.
redact = [str(value) for value in payload.get("redact", []) if value]

# --- Dial-back credential (optional) ----------------------------------
# A sandbox's ambient credential is its owner's workspace PAT, and the
# Databricks Apps OAuth edge answers HTTP 302 to a PAT -- so a host dialling
# an Apps URL cannot authenticate at all. When the launcher names a service
# principal, its OAuth client credentials are fetched HERE rather than in the
# coordinator: `dbutils.secrets.get` is redacted from notebook output
# automatically, and the values never enter the job run JSON. Exported ahead
# of the script so the in-sandbox SDK mints an SP OAuth token the edge
# accepts instead of falling back to that PAT.
def read_host_auth_secret(scope, key):
    """
    Read one dial-back credential, naming the fix when the read is denied.

    This notebook runs as the *coordinator's* identity -- the App's
    platform-created service principal -- which is not the identity that
    created the scope. A missing ACL therefore surfaces as a bare Py4J
    ``PERMISSION_DENIED`` wrapped in the job's "Workload failed, see run
    output for details", which is precisely the opaque failure this whole
    path exists to remove. Name the scope, the key and the grant instead.
    """
    try:
        return dbutils.secrets.get(scope=scope, key=key)
    except Exception as error:
        raise RuntimeError(
            "cannot read the dial-back credential (secret scope "
            + repr(scope)
            + ", key "
            + repr(key)
            + "): "
            + str(error)
            + " -- this bootstrap job runs as the Omnigent App's service"
            + " principal, so THAT principal (not the operator who created"
            + " the scope) needs read access to it:"
            + " databricks secrets put-acl "
            + scope
            + " <app-service-principal-application-id> READ"
        ) from error


host_auth = payload.get("host_auth")
if host_auth:
    client_id = read_host_auth_secret(host_auth["scope"], host_auth["client_id_key"])
    client_secret = read_host_auth_secret(
        host_auth["scope"], host_auth["client_secret_key"]
    )
    # The remote script's own stdout is NOT auto-redacted, and it lands in
    # durable job-run JSON -- so the secret joins the caller's scrub list.
    redact.append(client_secret)
    remote_command = "\\n".join(
        [
            "DATABRICKS_HOST=" + shlex.quote(host_auth["workspace_host"]),
            "DATABRICKS_CLIENT_ID=" + shlex.quote(client_id),
            "DATABRICKS_CLIENT_SECRET=" + shlex.quote(client_secret),
            "export DATABRICKS_HOST DATABRICKS_CLIENT_ID DATABRICKS_CLIENT_SECRET",
            # The host filters its own env down to an allowlist before
            # spawning a runner, and bearer secrets are deliberately NOT on
            # it -- so without this the runner re-resolves auth from
            # ~/.databrickscfg (the sandbox owner's PAT), and the Apps edge
            # answers its tunnel upgrade with a login-page redirect:
            # "runner tunnel rejected by server". The passthrough env var is
            # the supported way to widen that allowlist; any value already
            # baked into the image is preserved.
            'OMNIGENT_RUNNER_ENV_PASSTHROUGH="${{OMNIGENT_RUNNER_ENV_PASSTHROUGH:+'
            + '$OMNIGENT_RUNNER_ENV_PASSTHROUGH,}}DATABRICKS_HOST,'
            + 'DATABRICKS_CLIENT_ID,DATABRICKS_CLIENT_SECRET"',
            "export OMNIGENT_RUNNER_ENV_PASSTHROUGH",
            # A profile name left in the env sends the SDK back to the config
            # file (and the PAT in it): `Config._known_file_config_loader`
            # skips that file only when NO profile is named.
            "unset DATABRICKS_CONFIG_PROFILE",
            remote_command,
        ]
    )


def scrub(text):
    """Replace every secret in *text* before it can reach job-run output."""
    for secret in redact:
        text = text.replace(secret, "***")
    return text


# --- Resolve the gateway, and start the sandbox if it is not up -------
sandbox_path = API_ROOT + "/sandboxes/" + sandbox_id
sandbox = api("GET", sandbox_path)
if str(sandbox.get("status", "")).lower() != "running":
    api("POST", sandbox_path + "/start")
    deadline = time.time() + 600
    while time.time() < deadline:
        sandbox = api("GET", sandbox_path)
        if str(sandbox.get("status", "")).lower() == "running":
            break
        time.sleep(5)
    else:
        raise RuntimeError(
            f"sandbox {{sandbox_id}} did not reach running: {{sandbox.get('status')}}"
        )

gateway_host = sandbox.get("gatewayHost") or ""
if not gateway_host:
    raise RuntimeError(f"sandbox {{sandbox_id}} reported no gatewayHost")

# --- The one SSH call -------------------------------------------------
# argv mirrors the CLI's `buildSSHArgs` single-extra-arg path: the remote
# command is handed through untouched for the remote shell to parse.
# `BatchMode=yes` is the one addition — the CLI assumes a human at a
# terminal, and in a job any auth prompt would hang until the run timeout
# instead of failing fast.
argv = [
    "ssh",
    "-i", key_path,
    "-p", GATEWAY_PORT,
    "-o", "IdentitiesOnly=yes",
    "-o", "PreferredAuthentications=publickey",
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "LogLevel=ERROR",
    "-o", "BatchMode=yes",
    f"{{sandbox_id}}@{{gateway_host}}",
    remote_command,
]
completed = subprocess.run(argv, capture_output=True, text=True, timeout=1800)
if completed.returncode != 0:
    raise RuntimeError(
        f"ssh to sandbox {{sandbox_id}} exited {{completed.returncode}}: "
        f"{{scrub(completed.stderr.strip())}}"
    )
# `dbutils.notebook.exit` (not `print`) is the reliable channel back to the
# caller: it lands in `run-output.notebook_output.result` deterministically,
# where plain stdout capture depends on cluster log-delivery configuration.
dbutils.notebook.exit(scrub(completed.stdout))
'''


@dataclass(frozen=True)
class JobBootstrapConfig:
    """
    Delegate the one-time SSH bootstrap to a classic-compute Databricks Job.

    Exists because a Databricks App container cannot route to the sandbox's
    SSH gateway on port 2222: inside the container the gateway name resolves
    to a private-link address (``192.168.200.10``) that only answers on 443.
    **Classic compute is the specific escape hatch, and it is not
    interchangeable with serverless** — measured from a serverless job the
    gateway resolves to that same private address and port 2222 returns "no
    route to host", exactly as in the App. Only classic compute resolves the
    gateway to its public addresses and completes an SSH handshake, so the
    throwaway single-node cluster below is load-bearing, not incidental.

    Egress there is selectively permitted rather than open (``github.com:22``
    and the PyPI mirror both time out from the same cluster); what this needs
    — the sandbox gateway on 2222 — is reachable.

    When set on :class:`DatabricksSandboxLauncher`, ``start_host`` composes
    every remote step it would normally run itself (probe ``$HOME``,
    ``mkdir``, clone, write host config, launch ``omnigent host``) into ONE
    script and submits ONE job run that opens a single SSH session to run it
    — one job-run round trip, not one per step.

    **Compute reuse is the difference between a usable and an unusable
    iteration loop.** By default each run spins a throwaway single-node
    cluster and pays that cold start (observed 3-6 minutes) on EVERY managed
    launch — so a bootstrap that fails for a fixable reason costs minutes per
    attempt. Set :attr:`cluster_name` (or :attr:`existing_cluster_id`) and
    runs submit against a long-lived cluster instead: the spin-up is paid
    once, and later runs start in seconds, which is what makes trying several
    configurations in a row practical. The cost side of the trade is real —
    the warm cluster is all-purpose compute that bills while it is up — and
    :attr:`autotermination_minutes` is the dial for it.

    Both secrets this needs (the armed host token and an SSH private key
    registered against the sandbox gateway) travel through Databricks
    Secrets rather than job parameters or cluster env vars, which land
    verbatim in job run JSON visible to anyone with job-read access.
    :attr:`ssh_key_secret_scope` / :attr:`ssh_key_secret_key` name a
    long-lived secret an operator registers out of band (``databricks
    sandbox register`` run once against a throwaway machine, then the
    resulting ``~/.ssh/sandbox_ed25519`` content pasted into the secret) —
    see the module docstring's job-bootstrap section for why this is a
    deliberate tradeoff, not an oversight.
    """

    ssh_key_secret_scope: str
    """Databricks Secrets scope holding the sandbox gateway's registered SSH
    private key, e.g. ``"omnigent-sandbox-bootstrap"``."""

    ssh_key_secret_key: str
    """Key within *ssh_key_secret_scope* holding the private key content
    (OpenSSH format, as written by ``databricks sandbox register``)."""

    workspace_notebook_path: str
    """Base workspace path for the one-shot bootstrap notebook, e.g.
    ``"/Shared/omnigent/sandbox-job-bootstrap"``. Each run uploads to a
    per-run path derived from this and deletes it afterwards, so two
    concurrent launches can never race over one artifact."""

    payload_secret_scope: str | None = None
    """Scope the per-run connect payload (ssh argv + remote command) is
    written to, defaulting to :attr:`ssh_key_secret_scope`.

    Set this to a SEPARATE scope in any deployment that cares about least
    privilege: writing the payload needs ``WRITE`` on whatever scope holds
    it, and granting the coordinator ``WRITE`` on the scope that also holds
    the long-lived private key lets the coordinator overwrite that key.
    Split into two scopes and the coordinator needs only ``READ`` on the key
    scope and ``WRITE`` on this one."""

    node_type_id: str = "m5d.large"
    """Classic-compute node type for the throwaway single-node cluster.
    Instance-store-backed on purpose: EBS-only families (``m4.large``) are
    rejected outright — "At least one EBS volume must be attached for
    clusters created with node type m4.large" — unless the cluster spec also
    declares EBS volumes, which this deliberately does not."""

    spark_version: str = "15.4.x-scala2.12"
    """Databricks Runtime version for the cluster (throwaway or warm)."""

    timeout_s: float = _JOB_BOOTSTRAP_TIMEOUT_S
    """Ceiling on job-run completion. Dominated by cluster spin-up on the
    throwaway path; on the warm path a run is just the SSH session, so the
    same ceiling is generous rather than tight."""

    existing_cluster_id: str | None = None
    """Long-lived classic cluster to submit bootstrap runs against, e.g.
    ``"0806-241456-abcd1234"`` — skipping the throwaway cluster's cold start.

    Mutually exclusive with :attr:`cluster_name` (see :meth:`__post_init__`):
    an id pins a cluster whose lifecycle an operator owns, a name hands the
    lifecycle to the launcher. A pinned cluster is still RESTARTED when it is
    found terminated, so pinning one does not turn "someone stopped it" into
    a permanently broken launch path."""

    cluster_name: str | None = None
    """Name of a launcher-managed long-lived cluster, e.g.
    ``"omnigent-sandbox-bootstrap"``.

    The launcher resolves it by name, CREATES it when absent (single-node
    classic, the same spec the throwaway path uses), and starts it when it is
    found terminated. So the first launch pays a cold start and every later
    one submits against a cluster that is already up. Prefer this over
    :attr:`existing_cluster_id` unless an operator wants to own the cluster
    themselves — it needs no out-of-band setup step to go wrong."""

    autotermination_minutes: int = 0
    """Idle minutes before a launcher-created warm cluster terminates itself;
    ``0`` (the default) means never.

    Never-terminate is the right default *for this path*: the whole point is
    that a launch never waits on a cluster start, and an idle window
    reintroduces exactly that wait at an unpredictable time. It also means
    the cluster bills until something stops it, so a deployment that launches
    rarely should set a real window and accept the occasional restart.
    Applied only at CREATE time (and therefore only with
    :attr:`cluster_name`) — the lifecycle of a cluster named by
    :attr:`existing_cluster_id` belongs to whoever created it."""

    def __post_init__(self) -> None:
        """
        Reject configurations whose intent cannot be recovered.

        Both cluster selectors at once is the ambiguous case: one of them
        would have to silently lose, and which one is not something a config
        author should have to guess from reading the implementation.

        :raises ValueError: When both warm-cluster selectors are set, or
            *autotermination_minutes* is negative.
        """
        if self.existing_cluster_id is not None and self.cluster_name is not None:
            raise ValueError(
                "JobBootstrapConfig takes existing_cluster_id OR cluster_name, not both "
                "— an id pins an operator-managed cluster, a name lets the launcher "
                "manage one."
            )
        if self.autotermination_minutes < 0:
            raise ValueError(
                "JobBootstrapConfig.autotermination_minutes must be >= 0 (0 = never "
                f"terminate), got {self.autotermination_minutes}."
            )

    @property
    def warm_compute(self) -> bool:
        """
        Whether runs submit against a long-lived cluster.

        ``False`` selects the throwaway ``new_cluster`` per run — the
        unchanged default, and the only behavior available before either
        cluster selector existed.
        """
        return self.existing_cluster_id is not None or self.cluster_name is not None

    @property
    def payload_scope(self) -> str:
        """
        The scope the transient connect payload is written to and deleted from.

        Falls back to :attr:`ssh_key_secret_scope` so a config that predates
        the split keeps working — at the cost of the coordinator needing
        ``WRITE`` on the scope holding the private key.
        """
        return self.payload_secret_scope or self.ssh_key_secret_scope


@dataclass(frozen=True)
class HostAuthConfig:
    """
    Service-principal OAuth credentials the in-sandbox host dials back with.

    Exists because a Databricks Sandbox cannot authenticate to a Databricks
    Apps URL on its own. The only credential the platform mounts in a sandbox
    is its owner's workspace PAT (``/run/lakebox/databrickscfg``), and the
    Apps OAuth edge answers HTTP 302 to a PAT under every header spelling --
    so ``omnigent host`` pointed at a ``*.databricksapps.com`` server dies on
    its ``/v1/me`` pre-flight and the launch fails as an opaque registration
    timeout. A sandbox also cannot mint an OAuth token (the in-sandbox
    ``databricks auth login`` needs a browser) nor exchange the PAT for one
    (workspace token exchange requires a federation policy it has none of).

    Naming a scope here closes that gap: the bootstrap notebook reads the
    named client id/secret and exports them as ``DATABRICKS_CLIENT_ID`` /
    ``DATABRICKS_CLIENT_SECRET`` / ``DATABRICKS_HOST`` ahead of the host
    launch, so the in-sandbox SDK mints an OAuth token for that principal
    instead of falling back to the PAT. The credential is read on the job
    cluster via ``dbutils.secrets.get`` -- never in the coordinator, and
    never through job-run JSON.

    **The service principal must hold nothing beyond ``CAN_USE`` on the
    app.** Environment variables are inherited, so the host process, the
    runner it spawns, and every agent command below them all authenticate as
    this principal. That is acceptable exactly to the degree the principal is
    powerless elsewhere: a dedicated SP whose only grant is ``CAN_USE`` on
    the Omnigent app (plus, necessarily, whatever the agent's own work
    legitimately needs) turns a sandbox compromise into app access, where
    reusing a broadly-privileged identity -- the coordinator app's own SP,
    say -- would hand over that identity's full workspace reach.
    """

    secret_scope: str
    """Databricks Secrets scope holding the service principal's OAuth client
    credentials, e.g. ``"omnigent-sandbox-host-auth"``.

    Worth keeping separate from the job-bootstrap scopes: this one holds a
    long-lived credential and the coordinator needs no access to it at all
    (the job cluster reads it), where the payload scope needs coordinator
    ``WRITE``."""

    client_id_key: str = "client-id"
    """Key within :attr:`secret_scope` holding the OAuth client id -- the
    service principal's ``applicationId``."""

    client_secret_key: str = "client-secret"
    """Key within :attr:`secret_scope` holding the OAuth client secret, as
    minted by ``databricks service-principal-secrets-proxy create``."""

    workspace_host: str | None = None
    """Workspace the client credentials authenticate against, e.g.
    ``"https://example.cloud.databricks.com"``.

    ``None`` (the default) uses the coordinator's own resolved workspace
    host, which is the right answer whenever the app and the sandbox live in
    the same workspace -- i.e. always, since a sandbox is provisioned by this
    coordinator."""

    def __post_init__(self) -> None:
        """
        Reject an unusable configuration.

        :raises ValueError: When any named field is blank -- a blank scope or
            key would fail deep inside the bootstrap notebook, minutes later,
            as a secrets lookup error with no hint that config is the cause.
        """
        for field_name in ("secret_scope", "client_id_key", "client_secret_key"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"HostAuthConfig.{field_name} must be a non-empty string")


def _drain_forward_output(
    process: subprocess.Popen[str],
    transcript: list[str],
    settled: threading.Event,
    state: dict[str, bool],
    port: int,
) -> None:
    """
    Drain an ``ssh -v`` forward child's output for its whole lifetime.

    Runs on a daemon thread. Draining is not optional: ``ssh -v`` keeps
    logging as channels open and close, and an unread pipe eventually
    fills and wedges the tunnel. Readiness and hard failure are both
    recognized here, as a side effect of the drain.

    :param process: The forwarding child, text-mode with stderr merged
        into ``stdout``.
    :param transcript: Appended to, so the caller can quote ``ssh``'s
        own words in an error.
    :param settled: Set once the outcome is known either way, so the
        waiter wakes immediately instead of polling to its deadline.
    :param state: Mutated with ``{"listening": True}`` on success or
        ``{"failed": True}`` when ssh reports it cannot listen.
    :param port: The local port whose readiness line to match, so a
        transcript can never satisfy a different forward.
    """
    stream = process.stdout
    if stream is None:
        return
    ready_marker = _FORWARD_READY_TEMPLATE.format(port=port)
    try:
        for line in stream:
            transcript.append(line)
            if ready_marker in line:
                state["listening"] = True
                settled.set()
            elif any(marker in line.lower() for marker in _FORWARD_FAILURE_MARKERS):
                state["failed"] = True
                settled.set()
    finally:
        stream.close()


def _forward_failure_detail(transcript: list[str]) -> str:
    """
    Summarize an ``ssh -v`` transcript for a forward error message.

    Prefers the lines that explain a failure (``ssh``'s own errors and
    bind complaints) over the verbose handshake chatter, falling back to
    the tail of the transcript when nothing matches.

    :param transcript: Collected output lines.
    :returns: A short, single-line explanation.
    """
    interesting = [
        line.strip()
        for line in transcript
        if any(marker in line.lower() for marker in _FORWARD_ERROR_MARKERS)
    ]
    lines = interesting or [line.strip() for line in transcript[-3:] if line.strip()]
    return " / ".join(lines) or "<no output>"


def _as_api_duration(value: str) -> str:
    """
    Normalize an idle-timeout string into the seconds form the API takes.

    The CLI accepts Go durations (``"4h"``, ``"90m"``, ``"30s"``) and puts
    seconds on the wire — ``--idle-timeout 4h`` sends ``"14400s"``. Config
    files carry the human spelling, so convert here rather than making
    operators do arithmetic.

    :param value: A Go-style duration, or already-normalized seconds.
    :returns: The duration as ``"<seconds>s"``.
    :raises click.ClickException: If the spelling is not understood, since
        silently sending it through would set an unintended timeout.
    """
    text = value.strip().lower()
    match = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", text)
    if not text or match is None or not any(match.groups()):
        raise click.ClickException(
            f"Could not read idle timeout {value!r}: expected a duration like "
            "'4h', '90m', or '30s'."
        )
    hours, minutes, seconds = (int(group or 0) for group in match.groups())
    return f"{hours * 3600 + minutes * 60 + seconds}s"


def _looks_missing(output: str) -> bool:
    """
    Return whether CLI output reports the sandbox as nonexistent.

    :param output: Combined stdout/stderr from a failed control-plane call.
    :returns: ``True`` when the failure is a not-found, e.g. a delete
        racing another cleanup path.
    """
    lowered = output.lower()
    return any(marker in lowered for marker in _NOT_FOUND_MARKERS)


class _DatabricksRemoteProcess(RemoteProcess):
    """
    :class:`RemoteProcess` over a ``databricks sandbox ssh`` child.

    stderr is merged into stdout because the caller consumes one line
    stream; the blocking :meth:`DatabricksSandboxLauncher.run` keeps the
    two streams separate instead.
    """

    def __init__(self, process: subprocess.Popen[str], command: str) -> None:
        """
        Wrap an already-spawned child.

        :param process: The ``Popen`` handle, text-mode, ``stdout=PIPE``
            with stderr merged in.
        :param command: The remote command, for error messages.
        """
        self._process = process
        self._command = command
        self._lines: Iterator[str] | None = None

    @property
    def lines(self) -> Iterator[str]:
        """
        Line iterator over the child's combined output.

        Cached so a caller can consume a few lines, do other work, and
        resume where it left off — the contract
        :class:`~omnigent.onboarding.sandboxes.base.RemoteProcess`
        requires for the OAuth flow.
        """
        if self._lines is None:
            stream = self._process.stdout
            self._lines = iter(()) if stream is None else iter(stream)
        return self._lines

    def wait(self) -> int:
        """
        Block until the remote command exits.

        :returns: The remote command's exit code, which
            ``databricks sandbox ssh`` propagates verbatim.
        """
        return self._process.wait()

    def close(self) -> None:
        """Terminate the child if it is still running, then reap it."""
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=_FORWARD_TERMINATE_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        if self._process.stdout is not None:
            self._process.stdout.close()


class DatabricksSandboxLauncher(SandboxLauncher):
    """
    :class:`SandboxLauncher` for Databricks Sandbox environments.

    Every primitive shells out to the public ``databricks sandbox`` CLI
    with an argv list (never ``shell=True``): ``create`` / ``delete`` /
    ``start`` / ``status`` / ``config`` for lifecycle, and ``ssh <id> --
    <cmd>`` for all transport. The remote command rides as a single argv
    element, which the CLI hands to ``ssh`` and the sandbox's login shell
    interprets — so ``run("…", "a; b")`` behaves like a shell line, and
    callers quote remote paths themselves as the base contract requires.
    """

    provider: ClassVar[str] = "databricks"
    # `databricks sandbox ssh <id> -- -L …` forwards flags to ssh, so a
    # local→sandbox bridge for the App OAuth callback genuinely works.
    supports_local_port_forward: ClassVar[bool] = True
    # Sandboxes stop on idle and restart with their disk intact.
    can_resume: ClassVar[bool] = True

    @property
    def capabilities(self) -> SandboxCapabilities:
        """
        Feature flags, declared against what was verified on CLI v1.11.0.

        ``streaming_exec`` is ``True`` but cannot honor a PTY request —
        see the module docstring — so the in-sandbox App OAuth flow it
        backs is the least-proven path in this launcher.
        """
        return SandboxCapabilities(
            cli_bootstrap=True,
            managed_launch=True,
            local_port_forward=True,
            resume_stopped=True,
            programmatic_terminate=True,
            file_copy=True,
            streaming_exec=True,
            foreground_exec=True,
        )

    def __init__(
        self,
        *,
        cli_path: str | None = None,
        profile: str | None = None,
        idle_timeout: str | None = None,
        no_autostop: bool = True,
        bootstrap_command: str | None = None,
        job_bootstrap: JobBootstrapConfig | None = None,
        dial_back_url: str | None = None,
        host_auth: HostAuthConfig | None = None,
    ) -> None:
        """
        Initialize the launcher.

        :param cli_path: Databricks CLI executable to invoke, e.g.
            ``"/opt/databricks/bin/databricks"`` — the server's
            ``sandbox.databricks.cli_path`` config. ``None`` resolves
            :data:`DEFAULT_CLI_BINARY` on ``PATH``.
        :param profile: ``~/.databrickscfg`` profile passed as ``-p`` to
            every call, e.g. ``"sandbox-sp"`` for a service-principal
            M2M profile. ``None`` uses the CLI's own resolution
            (``DATABRICKS_*`` env, then the ``DEFAULT`` profile).
        :param idle_timeout: Per-sandbox idle timeout applied at
            provision, e.g. ``"4h"``. Ignored while *no_autostop* is
            ``True``, which is the default.
        :param no_autostop: Exempt provisioned sandboxes from idle
            auto-stop. Defaults to ``True`` because a managed host must
            survive arbitrary idle gaps between turns; set ``False`` to
            let *idle_timeout* (or the workspace default) apply.
        :param job_bootstrap: When set, ``start_host`` delegates the
            entire one-time SSH bootstrap to a classic-compute Databricks
            Job instead of exec'ing directly — see
            :class:`JobBootstrapConfig`. ``None`` (the default) preserves
            direct SSH exactly, so this launcher keeps working unmodified
            from a laptop that CAN reach the sandbox gateway.

            Setting it ALSO moves the whole control plane (create, start,
            status, config, delete) off the ``databricks`` CLI and onto the
            workspace REST API — see the "REST control plane" section. Both
            follow from the same fact: this mode exists for a Databricks
            Apps container, which can neither reach port 2222 nor run a Go
            CLI binary that is not installed in it.
        :param dial_back_url: Overrides the server URL the in-sandbox host
            dials, e.g. ``"https://omnigent.example.com"`` for a reverse proxy
            that fronts the same server on an address a sandbox can
            authenticate to. ``None`` (the default) means the host dials
            ``sandbox.server_url`` as configured -- which a Databricks Apps
            URL cannot be *unless* *host_auth* supplies a credential the
            Apps edge accepts, so that combination is rejected up front rather
            than left to time out. See :meth:`_resolve_dial_back_url`.
        :param host_auth: Service-principal OAuth credentials the in-sandbox
            host authenticates its dial-back with -- see
            :class:`HostAuthConfig`. Setting it is what makes a Databricks
            Apps ``server_url`` usable, since the sandbox's ambient PAT is
            not. ``None`` (the default) leaves the host on that ambient
            credential, which works for a server the PAT can reach.
        """
        self._cli_path = cli_path or DEFAULT_CLI_BINARY
        self._profile = profile
        self._idle_timeout = idle_timeout
        self._no_autostop = no_autostop
        self._bootstrap_command = bootstrap_command
        self._job_bootstrap = job_bootstrap
        self._dial_back_url = (dial_back_url or "").strip().rstrip("/") or None
        self._host_auth = host_auth
        # Memoized lazily by `_sdk()`; the SDK is an optional extra, so it is
        # never imported on the direct-SSH path.
        self._client: object | None = None
        # Resolved warm-cluster id, memoized so repeated launches skip the
        # list-by-name lookup. Its RUNNING state is re-checked every run
        # regardless -- see `_ensure_warm_cluster`.
        self._warm_cluster_id: str | None = None

    def start_host(self, sandbox_id: str, **kwargs):  # type: ignore[override]
        """Run the configured bootstrap, then start the host as usual.

        ``sandbox.databricks.bootstrap_command`` exists because the
        sandbox image's preinstalled omnigent can lag the server; the
        operator-supplied command (typically a self-update from git)
        runs on every host start — fresh provisions AND resumes — so a
        revived sandbox is refreshed before ``omnigent host`` launches.

        When ``job_bootstrap`` is configured, none of this launcher's own
        exec methods touch the sandbox at all: the whole sequence below
        (bootstrap command, workspace setup, host launch) is composed into
        one script and handed to a classic-compute job to run over SSH,
        because THIS process (typically inside a Databricks App) cannot
        reach the sandbox gateway on port 2222.

        ``server_url`` is resolved first, on both paths -- see
        :meth:`_resolve_dial_back_url`.
        """
        server_url = kwargs.get("server_url")
        if isinstance(server_url, str):
            kwargs["server_url"] = self._resolve_dial_back_url(server_url)
        if self._job_bootstrap is not None:
            return self._start_host_via_job(sandbox_id, self._job_bootstrap, **kwargs)
        if self._bootstrap_command:
            self.run(sandbox_id, self._bootstrap_command)
        return super().start_host(sandbox_id, **kwargs)

    def _resolve_dial_back_url(self, server_url: str) -> str:
        """
        Resolve the URL the in-sandbox host should dial back to.

        A sandbox authenticates as its owner with the ambient workspace PAT
        the platform mounts at ``/run/lakebox/databrickscfg``. That is the
        only credential it has: it holds no workspace OAuth token, cannot
        mint one (the in-sandbox ``databricks auth login`` flow needs a
        browser), and cannot exchange the PAT for one (workspace token
        exchange requires a federation policy the sandbox identity has none
        of). The Databricks Apps edge answers HTTP 302 to that PAT under
        every header spelling, so an ``omnigent host`` pointed at a
        ``*.databricksapps.com`` URL dies on its ``/v1/me`` pre-flight and
        the launch fails as an opaque registration timeout two minutes
        later.

        Two things lift that, and an Apps URL passes when either is
        configured. ``sandbox.databricks.host_auth`` gives the sandbox a
        service-principal OAuth credential the edge does accept (see
        :class:`HostAuthConfig`) -- measured: HTTP 200 on ``/v1/me`` where the
        PAT gets 302. ``sandbox.databricks.dial_back_url`` instead names a
        different address for the same server -- a reverse proxy or tunnel --
        and is used verbatim.

        With neither, an Apps ``server_url`` is rejected HERE, before a
        sandbox is provisioned and a bootstrap job submitted, naming the
        reason.

        Note this deliberately does NOT rewrite an Apps URL to the
        ``/api/2.0/omnigent`` workspace mount: that path is Databricks'
        own first-party host-sharded Omnigent deployment (see
        ``omnigent.cli_auth.WORKSPACE_API_PATH``), a DIFFERENT server from
        a self-deployed App. Sending a host there registers it with someone
        else's coordinator, which looks exactly like success from the
        outside and is not.

        :param server_url: The configured dial-back URL, e.g.
            ``"https://omnigent-dev-123.aws.databricksapps.com"``.
        :returns: The URL the in-sandbox host should dial.
        :raises click.ClickException: When *server_url* is a Databricks Apps
            URL and neither ``host_auth`` nor ``dial_back_url`` is configured.
        """
        from urllib.parse import urlsplit

        if self._dial_back_url is not None:
            logger.info("dialing back through the configured URL %s", self._dial_back_url)
            return self._dial_back_url
        hostname = urlsplit(server_url).hostname or ""
        if not hostname.endswith(_APPS_HOST_SUFFIX):
            return server_url
        if self._host_auth is not None:
            logger.info(
                "dialing back to the Apps URL %s with service-principal credentials "
                "from secret scope %s",
                server_url,
                self._host_auth.secret_scope,
            )
            return server_url
        raise click.ClickException(
            f"A Databricks Sandbox host cannot authenticate to the Databricks Apps URL "
            f"{server_url!r}: the Apps OAuth edge rejects the workspace PAT a sandbox "
            f"holds, and a sandbox can neither mint nor exchange for an OAuth token. "
            f"Give the sandbox a credential the edge accepts by setting "
            f"'sandbox.databricks.host_auth', or front this server on an address the "
            f"sandbox can reach and set 'sandbox.databricks.dial_back_url' to it."
        )

    # ── Job-delegated bootstrap ─────────────────────────

    def _compose_bootstrap_script(
        self,
        *,
        token: str,
        host_id: str,
        host_name: str,
        server_url: str,
        repo_url: str | None,
        repo_branch: str | None,
        repo_name: str | None,
        host_config: dict[str, object] | None,
    ) -> str:
        """
        Fold every step of the default ``start_host`` bootstrap into ONE
        POSIX ``sh`` script, so the job-delegated path costs one
        ``databricks sandbox ssh`` call (and therefore one job run) instead
        of one per step. Mirrors
        :meth:`~omnigent.onboarding.sandboxes.base.ExecModelHostLauncher.start_host`
        step for step — probe ``$HOME``, ``mkdir`` the workspace, clone if a
        repo was given, write host config, launch the supervised host in
        the background — plus this provider's own
        ``bootstrap_command`` hook up front, exactly where :meth:`start_host`
        runs it on the direct-SSH path.

        The resolved workspace path is ``printf``'d as the script's last
        line, tagged with :data:`_WORKSPACE_TAG`, so the caller can recover
        :meth:`start_host`'s return value from the job's captured output
        without a second trip into the sandbox.
        """
        lines: list[str] = ["set -eu"]
        if self._bootstrap_command:
            lines.append(self._bootstrap_command)
        lines.append('home="$(printf %s "$HOME")"')
        lines.append('workspace="$home/workspace"')
        lines.append('mkdir -p "$workspace"')
        if repo_url is not None:
            clone_dir_expr = f'"$workspace"/{shlex.quote(repo_name or "repo")}'
            branch_args = (
                f"--branch {shlex.quote(repo_branch)} --single-branch " if repo_branch else ""
            )
            lines.append(f"git clone {branch_args}-- {shlex.quote(repo_url)} {clone_dir_expr}")
            lines.append(f"workspace={clone_dir_expr}")
        if host_config is not None:
            lines.append(render_host_config_write_command(host_config))
        env_prefix = " ".join(
            f"{key}={shlex.quote(value)}"
            for key, value in (
                ("OMNIGENT_HOST_TOKEN", token),
                ("OMNIGENT_HOST_ID", host_id),
                ("OMNIGENT_HOST_NAME", host_name),
            )
        )
        host_command = f"{env_prefix} omnigent host --server {shlex.quote(server_url)}"
        supervised = supervise_host_command(host_command)
        lines.append(
            f"setsid nohup sh -c {shlex.quote(supervised)} "
            "> /tmp/omnigent-host.log 2>&1 < /dev/null & echo launched"
        )
        lines.append(f'printf "{_WORKSPACE_TAG}%s\\n" "$workspace"')
        return "\n".join(lines)

    def _start_host_via_job(
        self,
        sandbox_id: str,
        job_bootstrap: JobBootstrapConfig,
        *,
        token: str,
        host_id: str,
        host_name: str,
        server_url: str,
        repo_url: str | None = None,
        repo_branch: str | None = None,
        repo_name: str | None = None,
        host_config: dict[str, object] | None = None,
        on_stage: object | None = None,
    ) -> str:
        """
        Run the composed bootstrap script via a one-shot classic-compute Job.

        See :class:`JobBootstrapConfig` for why: this process cannot reach
        the sandbox gateway on port 2222, and classic compute can. The job's
        single task drops the SSH private key from Databricks Secrets,
        resolves the gateway over REST, and opens exactly one SSH session
        that runs the whole script — one job run, one SSH session,
        regardless of how many bootstrap steps the script contains.

        :raises click.ClickException: If the job run fails, times out, or
            its output does not carry the expected workspace-path tag.
        """
        if on_stage is not None and callable(on_stage):
            on_stage("starting")
        script = self._compose_bootstrap_script(
            token=token,
            host_id=host_id,
            host_name=host_name,
            server_url=server_url,
            repo_url=repo_url,
            repo_branch=repo_branch,
            repo_name=repo_name,
            host_config=host_config,
        )
        output = self._run_via_job(
            job_bootstrap, sandbox_id, script, redact=(token,), inject_host_auth=True
        )
        for line in output.splitlines():
            if line.startswith(_WORKSPACE_TAG):
                return line[len(_WORKSPACE_TAG) :].strip()
        raise click.ClickException(
            f"job-delegated bootstrap for Databricks Sandbox '{sandbox_id}' completed "
            f"but its output did not include the expected {_WORKSPACE_TAG!r} tag — "
            f"cannot report the workspace path. Output: {output!r}"
        )

    def _ensure_warm_cluster(self, job_bootstrap: JobBootstrapConfig) -> str | None:
        """
        Return the id of a RUNNING long-lived bootstrap cluster, or ``None``.

        ``None`` means warm compute is not configured, and the caller falls
        back to the throwaway per-run cluster — so every existing deployment
        keeps its current behavior until it opts in.

        The resolved id is memoized, but its state is re-checked on **every**
        run rather than trusted once. That is the difference between a warm
        path that self-heals and one that breaks quietly: a cluster can
        autoterminate, be stopped by an operator, or be restarted by the
        platform between launches, and a cached "it was running" would turn
        each of those into a confusing job-submission failure. When the
        cluster is already up the re-check costs one ``clusters.get``, which
        is nothing against the SSH session that follows.

        :returns: Cluster id when warm compute is configured, else ``None``.
        :raises click.ClickException: When the cluster cannot be brought to a
            RUNNING state.
        """
        if not job_bootstrap.warm_compute:
            return None
        cluster_id = self._warm_cluster_id or job_bootstrap.existing_cluster_id
        if cluster_id is None:
            cluster_id = self._resolve_named_cluster(job_bootstrap)
        self._warm_cluster_id = cluster_id
        self._await_cluster_running(cluster_id)
        return cluster_id

    def _resolve_named_cluster(self, job_bootstrap: JobBootstrapConfig) -> str:
        """
        Find the warm cluster by name, creating it when it does not exist.

        Resolving by name rather than id is what lets a deployment declare
        the warm cluster in config alone, with no out-of-band setup step that
        can be forgotten or done differently in another workspace.

        :returns: The cluster id.
        :raises click.ClickException: When the cluster cannot be listed or created.
        """
        name = job_bootstrap.cluster_name
        assert name is not None  # guaranteed by `warm_compute` + `_ensure_warm_cluster`
        client = self._sdk()
        try:
            for cluster in client.clusters.list():
                if cluster.cluster_name == name and cluster.cluster_id is not None:
                    logger.info("reusing warm bootstrap cluster %s (%s)", name, cluster.cluster_id)
                    return cluster.cluster_id
        except Exception as error:
            raise click.ClickException(
                f"Could not list Databricks clusters to find warm bootstrap "
                f"cluster {name!r}: {error}"
            ) from error
        return self._create_warm_cluster(job_bootstrap, name)

    def _create_warm_cluster(self, job_bootstrap: JobBootstrapConfig, name: str) -> str:
        """
        Create the long-lived single-node bootstrap cluster and wait for it.

        Same single-node classic spec the throwaway path submits — the point
        of this path is the cluster's lifetime, not a different shape of
        machine. It is tagged (:data:`_WARM_CLUSTER_TAG_KEY`) because a
        long-lived cluster nobody recognizes is a long-lived cluster somebody
        deletes.

        One thing DOES differ from the submitted job cluster: the access mode
        must be stated. An all-purpose cluster created with no
        ``data_security_mode`` lands on the legacy no-isolation mode, which
        many workspaces forbid outright ("NO_ISOLATION or custom access modes
        are not allowed in this workspace"), so creation fails and the whole
        managed launch fails with it. ``SINGLE_USER`` is the mode that matches
        what this cluster does — run one principal's bootstrap notebook — and
        the single user is the identity creating it (inside a Databricks App,
        the app's service principal).

        :returns: The new cluster id.
        :raises click.ClickException: When creation fails or times out.
        """
        from databricks.sdk.service.compute import DataSecurityMode

        client = self._sdk()
        logger.info("creating warm bootstrap cluster %s", name)
        try:
            created = client.clusters.create(
                cluster_name=name,
                spark_version=job_bootstrap.spark_version,
                node_type_id=job_bootstrap.node_type_id,
                num_workers=0,
                autotermination_minutes=job_bootstrap.autotermination_minutes,
                data_security_mode=DataSecurityMode.SINGLE_USER,
                single_user_name=self._current_principal(),
                spark_conf={
                    "spark.master": "local[*]",
                    "spark.databricks.cluster.profile": "singleNode",
                },
                custom_tags={
                    "ResourceClass": "SingleNode",
                    _WARM_CLUSTER_TAG_KEY: _WARM_CLUSTER_TAG_VALUE,
                },
            ).result(timeout=timedelta(seconds=_WARM_CLUSTER_READY_TIMEOUT_S))
        except Exception as error:
            raise click.ClickException(
                f"Could not create warm bootstrap cluster {name!r}: {error}"
            ) from error
        cluster_id = created.cluster_id
        if cluster_id is None:
            raise click.ClickException(
                f"Databricks reported no cluster id for created warm bootstrap cluster {name!r}."
            )
        return cluster_id

    def _current_principal(self) -> str | None:
        """
        Resolve the identity this launcher authenticates as, for
        ``single_user_name`` on the warm cluster.

        Inside a Databricks App this is the app's service principal (its
        application id); locally it is the operator's username. Returns
        ``None`` when the lookup fails, which lets the create fall back to
        whatever the API defaults the single user to rather than turning a
        best-effort identity lookup into a failed launch.
        """
        try:
            return self._sdk().current_user.me().user_name
        except Exception:  # pragma: no cover - best effort identity lookup
            logger.warning("could not resolve the current principal for the warm cluster")
            return None

    def _await_cluster_running(self, cluster_id: str) -> None:
        """
        Bring *cluster_id* to RUNNING, starting or waiting as its state requires.

        Every non-running state is handled explicitly rather than blanket
        "call start": starting an already-starting cluster is an error, and a
        TERMINATING cluster must finish terminating before a start is
        accepted. The one state worth failing fast on is ERROR — waiting out
        the full timeout on a cluster that cannot start just delays the
        message the operator needs.

        :raises click.ClickException: When the cluster ends up unusable or
            takes longer than :data:`_WARM_CLUSTER_READY_TIMEOUT_S`.
        """
        from databricks.sdk.service.compute import State

        client = self._sdk()
        timeout = timedelta(seconds=_WARM_CLUSTER_READY_TIMEOUT_S)
        unusable: str | None = None
        try:
            state = client.clusters.get(cluster_id).state
            if state == State.RUNNING:
                return
            if state == State.TERMINATING:
                # A start submitted mid-termination is rejected, so let the
                # termination land first and then start from a clean state.
                client.clusters.wait_get_cluster_terminated(cluster_id, timeout=timeout)
                state = State.TERMINATED
            if state == State.TERMINATED:
                logger.info("starting warm bootstrap cluster %s", cluster_id)
                client.clusters.start(cluster_id).result(timeout=timeout)
                return
            if state in (State.PENDING, State.RESTARTING, State.RESIZING):
                # Another launch (or an operator) already started it; joining
                # that wait is correct where a second start would error.
                client.clusters.wait_get_cluster_running(cluster_id, timeout=timeout)
                return
            unusable = str(getattr(state, "value", state))
        except Exception as error:
            raise click.ClickException(
                f"Warm bootstrap cluster {cluster_id} did not reach a running state: {error}"
            ) from error
        # Raised outside the `try` on purpose: inside, the `except` above
        # would catch it and rewrite the specific diagnosis into a generic one.
        raise click.ClickException(
            f"Warm bootstrap cluster {cluster_id} is in state {unusable}, which cannot be "
            "started. Inspect it in the workspace, then delete it (the launcher recreates a "
            "cluster it manages by name) or point job_bootstrap at a healthy cluster."
        )

    def _run_via_job(
        self,
        job_bootstrap: JobBootstrapConfig,
        sandbox_id: str,
        remote_command: str,
        *,
        redact: Sequence[str] = (),
        inject_host_auth: bool = False,
    ) -> str:
        """
        Submit and await the one-shot job that SSHes into *sandbox_id* on classic compute.

        The armed host token lives inside *remote_command* (folded into the
        bootstrap script by :meth:`_compose_bootstrap_script`), so it is
        never passed as a job parameter or cluster env var — both land
        verbatim in job run JSON visible to anyone with job-read access.
        Instead the connect payload is staged as a **transient** Databricks
        Secret, created immediately before submission and deleted in a
        ``finally`` block once the run reaches a terminal state. This is a
        deliberate tradeoff, not a closed problem: the secret is readable by
        anyone with read access to *job_bootstrap.payload_scope* for the
        run's lifetime (minutes), and a crash between "job submitted" and
        "job reached terminal state" leaves it stranded until the next
        cleanup pass.

        Writing it needs ``WRITE`` on that scope, which is why the payload
        scope is configurable separately from the key scope: point them at
        the same scope and whoever runs this can also overwrite the private
        key below. Splitting them keeps that grant down to ``READ`` on the
        key scope plus ``WRITE`` on a scope holding nothing durable.

        The SSH private key registered against the sandbox gateway
        (*job_bootstrap.ssh_key_secret_scope* / ``ssh_key_secret_key``) is
        NOT staged here — it is a long-lived secret an operator registers
        out of band once, so the job cluster can read it directly via
        ``dbutils.secrets.get``, which Databricks redacts from notebook
        output automatically (unlike a job parameter or cluster env var).

        The notebook itself is uploaded to a **per-run** path derived from
        *job_bootstrap.workspace_notebook_path* and removed afterwards. A
        single fixed path would be a correctness bug, not just untidiness:
        two concurrent launches would race between upload and run, and the
        loser would execute the winner's notebook — pointed at the winner's
        payload secret, and therefore at the wrong sandbox.

        Compute is either the throwaway per-run cluster or, when the config
        selects warm compute, an already-running long-lived one — see
        :meth:`_ensure_warm_cluster`. Nothing else about the run differs; the
        notebook, payload, and cleanup are identical either way.

        :param redact: Strings (the armed host token) scrubbed from anything
            the notebook returns, because notebook output is durable job-run
            JSON readable by anyone with job-read access.
        :param inject_host_auth: Have the notebook export the configured
            :class:`HostAuthConfig` credentials ahead of *remote_command*.
            Only the host bootstrap asks for this: the credential is what
            lets the host authenticate its dial-back, and every other remote
            command this launcher runs has no dial-back to authenticate, so
            handing it one would widen the credential's reach for nothing.
        :returns: The job task's captured stdout, with *redact* scrubbed.
        :raises click.ClickException: On job failure, timeout, or missing output.
        """
        from databricks.sdk.service.compute import ClusterSpec
        from databricks.sdk.service.jobs import NotebookTask, RunResultState, SubmitTask
        from databricks.sdk.service.workspace import ImportFormat, Language

        client = self._sdk()
        # Deliberately BEFORE the payload write. Getting compute ready can
        # take minutes (a cold create, or a restart of a cluster that
        # autoterminated), and the payload holds the armed host token --
        # staging it first would leave the token readable in the workspace
        # for that whole window for no benefit.
        warm_cluster_id = self._ensure_warm_cluster(job_bootstrap)
        run_key = uuid.uuid4().hex
        argv_secret_key = f"job-bootstrap-argv-{run_key}"
        notebook_path = f"{job_bootstrap.workspace_notebook_path}-{run_key}"
        client.secrets.put_secret(
            scope=job_bootstrap.payload_scope,
            key=argv_secret_key,
            string_value=json.dumps(
                {
                    "sandbox_id": sandbox_id,
                    "remote_command": remote_command,
                    "redact": list(redact),
                    # Scope and key NAMES only -- the values are read on the
                    # job cluster, so no credential passes through the
                    # coordinator or this payload.
                    **(
                        {
                            "host_auth": {
                                "scope": self._host_auth.secret_scope,
                                "client_id_key": self._host_auth.client_id_key,
                                "client_secret_key": self._host_auth.client_secret_key,
                                "workspace_host": (
                                    self._host_auth.workspace_host or client.config.host
                                ),
                            }
                        }
                        if inject_host_auth and self._host_auth is not None
                        else {}
                    ),
                }
            ),
        )
        run = None
        try:
            # The notebook cannot discover these itself: on classic compute
            # `apiUrl()` yields the regional control-plane host, which needs
            # the workspace's org id alongside it to route. Both are known
            # here. Everything from the payload write onwards lives inside
            # this `try` so that a failure between the write and the run --
            # an upload rejection, a submit error -- still reaches the
            # cleanup below; leaving the payload behind would leave the armed
            # host token readable in the workspace.
            notebook_source = _JOB_BOOTSTRAP_NOTEBOOK_TEMPLATE.format(
                ssh_key_scope=job_bootstrap.ssh_key_secret_scope,
                ssh_key_key=job_bootstrap.ssh_key_secret_key,
                argv_scope=job_bootstrap.payload_scope,
                argv_key=argv_secret_key,
                api_root=_LAKEBOX_API_ROOT,
                key_registration_name=_KEY_REGISTRATION_NAME,
                gateway_port=_SANDBOX_GATEWAY_PORT,
                workspace_host=client.config.host,
                workspace_id=client.get_workspace_id(),
            )
            # `format` and `language` are both load-bearing, despite the SDK
            # docstring implying SOURCE is the default: omit `format` and the
            # workspace import API treats the body as a DBC archive and rejects
            # it with "The zip archive contains no items", and the SDK only
            # infers `language` from a path suffix — the per-run notebook path
            # deliberately has none.
            client.workspace.upload(
                notebook_path,
                content=notebook_source.encode("utf-8"),
                format=ImportFormat.SOURCE,
                language=Language.PYTHON,
                overwrite=True,
            )
            # Warm path: submit against the already-running cluster
            # resolved above. Throwaway path: the unchanged per-run cluster,
            # which is what makes the warm fields purely additive.
            if warm_cluster_id is not None:
                task = SubmitTask(
                    task_key="bootstrap",
                    notebook_task=NotebookTask(notebook_path=notebook_path),
                    existing_cluster_id=warm_cluster_id,
                )
            else:
                task = SubmitTask(
                    task_key="bootstrap",
                    notebook_task=NotebookTask(notebook_path=notebook_path),
                    new_cluster=ClusterSpec(
                        spark_version=job_bootstrap.spark_version,
                        node_type_id=job_bootstrap.node_type_id,
                        num_workers=0,
                        spark_conf={
                            "spark.master": "local[*]",
                            "spark.databricks.cluster.profile": "singleNode",
                        },
                        custom_tags={"ResourceClass": "SingleNode"},
                    ),
                )
            waiter = client.jobs.submit(
                run_name="omnigent-sandbox-job-bootstrap",
                tasks=[task],
                timeout_seconds=int(job_bootstrap.timeout_s),
            )
            try:
                run = waiter.result()
            except Exception as wait_error:
                # A run that ends INTERNAL_ERROR (a notebook exception, most
                # often) makes the SDK waiter raise "failed to reach
                # TERMINATED or SKIPPED" -- which reports the run's LIFECYCLE
                # and nothing about the cause, while the actual traceback sits
                # in the run output fetched below. Fall through to the shared
                # failure path so the caller sees that instead.
                logger.debug("job-bootstrap wait failed: %s", wait_error)
                run = client.jobs.get_run(run_id=waiter.run_id)
        finally:
            # Both artifacts are per-run and carry (or point at) the armed
            # host token, so neither may outlive the run. Cleanup is
            # best-effort: a failure here must not mask the run's own
            # outcome, which is the thing the caller needs to see.
            for cleanup in (
                lambda: client.secrets.delete_secret(
                    scope=job_bootstrap.payload_scope, key=argv_secret_key
                ),
                lambda: client.workspace.delete(notebook_path),
            ):
                try:
                    cleanup()
                except Exception as error:  # see comment above
                    logger.warning("job-bootstrap cleanup failed: %s", error)
        state = run.state
        task_run_id = run.tasks[0].run_id if run.tasks else run.run_id
        run_output = client.jobs.get_run_output(task_run_id)
        notebook_result = (
            run_output.notebook_output.result if run_output.notebook_output is not None else None
        )
        combined_output = "\n".join(
            part for part in (notebook_result, run_output.logs, run_output.error) if part
        )
        if state is None or state.result_state != RunResultState.SUCCESS:
            raise click.ClickException(
                "job-delegated Databricks Sandbox bootstrap failed "
                f"(life_cycle_state={getattr(state, 'life_cycle_state', None)}, "
                f"result_state={getattr(state, 'result_state', None)}): {combined_output}"
            )
        return combined_output

    # ── CLI plumbing ────────────────────────────────────

    def _argv(self, *args: str) -> list[str]:
        """
        Build the argv for one ``databricks sandbox`` invocation.

        The profile flag is appended AFTER the subcommand's own
        arguments but is never allowed past a ``--`` separator, so
        ``ssh`` remote arguments stay untouched. Callers that need a
        ``--`` pass it themselves via :meth:`_ssh_argv`.

        :param args: Subcommand and its arguments, e.g.
            ``("status", "sb-1", "-o", "json")``.
        :returns: The full argv list, ready for :func:`subprocess.run`.
        """
        argv = [self._cli_path, "sandbox", *args]
        if self._profile is not None:
            argv += ["-p", self._profile]
        return argv

    def _ssh_argv(self, sandbox_id: str, *remote: str) -> list[str]:
        """
        Build the argv for ``databricks sandbox ssh <id> -- <remote…>``.

        :param sandbox_id: Target sandbox id, e.g.
            ``"spanking-pitta-1649"``.
        :param remote: Everything after ``--`` — ssh flags and/or the
            remote command as a single element.
        :returns: The full argv list.
        """
        return [*self._argv("ssh", sandbox_id), "--", *remote]

    def _cli(
        self,
        argv: list[str],
        *,
        timeout: float = _CONTROL_TIMEOUT_S,
        action: str,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        """
        Run one CLI invocation and normalize its failures.

        :param argv: Full argv from :meth:`_argv` / :meth:`_ssh_argv`.
        :param timeout: Seconds to wait before killing the child.
        :param action: Human phrase for error messages, e.g.
            ``"create a Databricks Sandbox"``.
        :param check: When ``True``, raise on a non-zero exit.
        :returns: The completed process, with text-mode output captured.
        :raises click.ClickException: When the CLI is missing, times
            out, or (with *check*) exits non-zero.
        """
        try:
            completed = subprocess.run(  # argv list, no shell
                argv,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            raise click.ClickException(INSTALL_HINT) from exc
        except subprocess.TimeoutExpired as exc:
            raise click.ClickException(
                f"Timed out after {timeout:.0f}s trying to {action}."
            ) from exc
        if check and completed.returncode != 0:
            raise click.ClickException(
                f"Could not {action} (exit {completed.returncode}): "
                f"{_combined(completed).strip() or '<no output>'}"
            )
        return completed

    def _cli_json(self, argv: list[str], *, timeout: float, action: str) -> dict[str, object]:
        """
        Run a CLI invocation whose stdout is a JSON object.

        :param argv: Full argv, which must already request JSON output.
        :param timeout: Seconds to wait before killing the child.
        :param action: Human phrase for error messages.
        :returns: The decoded object.
        :raises click.ClickException: On CLI failure or unparseable output.
        """
        completed = self._cli(argv, timeout=timeout, action=action)
        try:
            payload = json.loads(completed.stdout)
        except ValueError as exc:
            raise click.ClickException(
                f"Could not {action}: the Databricks CLI returned "
                f"unparseable JSON ({completed.stdout.strip()[:200]!r})."
            ) from exc
        if not isinstance(payload, dict):
            raise click.ClickException(
                f"Could not {action}: expected a JSON object from the "
                f"Databricks CLI, got {type(payload).__name__}."
            )
        return payload

    # ── REST control plane ──────────────────────────────
    #
    # Everything below reaches the sandbox control plane over the workspace
    # REST API through the Databricks SDK, with NO `databricks` CLI. That is
    # not a nicety: the CLI is a Go binary that does not exist inside a
    # Databricks Apps python container, so on the server-side path every
    # lifecycle call in this class would die with `INSTALL_HINT` long before
    # the job-delegated bootstrap ever ran.
    #
    # The switch is `job_bootstrap`: it is configured exactly when this
    # process cannot reach the sandbox gateway itself (an Apps container),
    # which is also exactly when the CLI is unavailable. One config key,
    # both consequences — see :meth:`_use_rest`.
    #
    # Request/response shapes were read off the CLI's own traffic
    # (`databricks sandbox … --debug`) rather than guessed. Note the
    # asymmetry: request bodies are snake_case, responses camelCase.

    def _use_rest(self) -> bool:
        """Whether the control plane goes over REST instead of the CLI."""
        return self._job_bootstrap is not None

    def _sdk(self):
        """
        Return a memoized ``WorkspaceClient``.

        Inside an App the environment supplies the credentials; locally the
        configured *profile* selects them. Memoized because a managed host's
        liveness polling calls the control plane repeatedly and each
        construction re-runs credential resolution.
        """
        if self._client is None:
            from databricks.sdk import WorkspaceClient

            self._client = (
                WorkspaceClient(profile=self._profile) if self._profile else WorkspaceClient()
            )
        return self._client

    def _api(
        self,
        method: str,
        path: str,
        *,
        action: str,
        body: dict[str, object] | None = None,
        query: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """
        Make one sandbox control-plane REST call.

        :param method: HTTP verb, e.g. ``"GET"``.
        :param path: Path under the workspace host, e.g.
            ``"/api/2.0/lakebox/sandboxes/sb-1"``.
        :param action: Human phrase for error messages.
        :param body: JSON request body (snake_case keys).
        :param query: Query-string parameters.
        :returns: The decoded response object, ``{}`` for an empty body.
        :raises click.ClickException: On any API failure.
        """
        try:
            payload = self._sdk().api_client.do(method, path, body=body, query=query)
        except Exception as error:  # SDK raises a wide family of DatabricksErrors
            raise click.ClickException(f"Could not {action}: {error}") from error
        return payload if isinstance(payload, dict) else {}

    def _rest_sandbox(self, sandbox_id: str) -> dict[str, object]:
        """Return one sandbox's control-plane record (camelCase keys)."""
        return self._api(
            "GET",
            f"{_LAKEBOX_API_ROOT}/sandboxes/{sandbox_id}",
            action=f"read Databricks Sandbox '{sandbox_id}'",
        )

    def _rest_wait_running(self, sandbox_id: str, *, timeout: float) -> None:
        """
        Poll until the sandbox reports Running.

        :raises click.ClickException: If it never does within *timeout*.
        """
        deadline = time.monotonic() + timeout
        status = ""
        while time.monotonic() < deadline:
            status = str(self._rest_sandbox(sandbox_id).get("status", "")).lower()
            if status == _RUNNING_STATUS:
                return
            time.sleep(_REST_POLL_INTERVAL_S)
        raise click.ClickException(
            f"Databricks Sandbox '{sandbox_id}' did not reach running within "
            f"{timeout:.0f}s (last status {status or 'unknown'!r})."
        )

    # ── Lifecycle ───────────────────────────────────────

    def prepare(self) -> None:
        """
        Local preflight: the ``databricks`` CLI must be on ``PATH`` and
        authenticated against a workspace.

        ``databricks sandbox list`` proves both in one call — it is
        read-only, it fails when no credentials resolve, and it fails
        when the CLI predates the ``sandbox`` command group.

        :raises click.ClickException: When the CLI is missing (with the
            install hint) or unauthenticated (with the login hint).
        """
        if self._use_rest():
            # No CLI to check: on this path there is none, and requiring one
            # would fail every launch from inside an App. Listing sandboxes
            # is the REST equivalent — read-only, and it fails when the
            # workspace credentials do not resolve.
            self._api(
                "GET",
                f"{_LAKEBOX_API_ROOT}/sandboxes",
                action="list Databricks Sandboxes",
                query={"page_size": 100},
            )
            return
        if shutil.which(self._cli_path) is None:
            raise click.ClickException(INSTALL_HINT)
        completed = self._cli(
            self._argv("list", "-o", "json"),
            action="list Databricks Sandboxes",
            check=False,
        )
        if completed.returncode != 0:
            raise click.ClickException(
                f"{AUTH_HINT}\n\nThe CLI reported: {_combined(completed).strip() or '<no output>'}"
            )

    def provision(self, name: str) -> str:
        """
        Create a new Databricks Sandbox and return its id.

        ``create`` blocks until the microVM reports Running, so the id
        this returns is immediately usable. The auto-stop policy is
        applied right after creation (see :meth:`keep_alive`) because
        ``create`` itself exposes no policy flag.

        :param name: Human-readable label, e.g. ``"managed-a1b2c3d4"``.
            Recorded as the sandbox's display name; the returned id is
            the canonical reference.
        :returns: The sandbox id, e.g. ``"spanking-pitta-1649"``.
        :raises click.ClickException: If provisioning fails or the CLI
            returns no id.
        """
        click.echo(f"▸ Creating Databricks Sandbox '{name}'")
        if self._use_rest():
            payload = self._api(
                "POST",
                f"{_LAKEBOX_API_ROOT}/sandboxes",
                action="create a Databricks Sandbox",
                body={"sandbox": {"name": name}},
            )
        else:
            payload = self._cli_json(
                self._argv("create", "--name", name, "--json"),
                timeout=_CREATE_TIMEOUT_S,
                action="create a Databricks Sandbox",
            )
        sandbox_id = payload.get("sandboxId")
        if not isinstance(sandbox_id, str) or not sandbox_id:
            raise click.ClickException(
                f"Databricks Sandbox creation returned no 'sandboxId' — got {payload!r}."
            )
        click.echo(f"  → created {sandbox_id}")
        if self._use_rest():
            # The CLI's `create` blocks until the microVM reports Running and
            # callers rely on the returned id being immediately usable, so the
            # REST path has to wait for the same condition itself.
            self._rest_wait_running(sandbox_id, timeout=_CREATE_TIMEOUT_S)
        self.keep_alive(sandbox_id)
        return sandbox_id

    def attach(self, sandbox_id: str) -> None:
        """
        Validate access to an existing sandbox, starting it if stopped.

        Databricks Sandboxes stop on idle and restart with their disk
        intact, so — like Daytona and unlike Modal — attach revives a
        stopped sandbox rather than rejecting it.

        :param sandbox_id: The sandbox to attach to.
        :raises click.ClickException: When the sandbox does not exist or
            cannot be started.
        """
        click.echo(f"▸ Reusing existing Databricks Sandbox '{sandbox_id}'")
        if self._status(sandbox_id) != _RUNNING_STATUS:
            self.resume(sandbox_id)

    def resume(self, sandbox_id: str) -> None:
        """
        Start a stopped sandbox in place, keeping its id and disk.

        Starting an already-running sandbox is a documented no-op, so
        this is safe to call unconditionally.

        :param sandbox_id: The sandbox to start.
        :raises click.ClickException: If the start fails or times out.
        """
        click.echo(f"▸ Starting Databricks Sandbox '{sandbox_id}'")
        if self._use_rest():
            self._api(
                "POST",
                f"{_LAKEBOX_API_ROOT}/sandboxes/{sandbox_id}/start",
                action=f"start Databricks Sandbox '{sandbox_id}'",
            )
            # `start` returns as soon as the request is accepted, so unlike the
            # CLI (which blocks) this path must poll for Running itself.
            self._rest_wait_running(sandbox_id, timeout=_START_TIMEOUT_S)
        else:
            self._cli(
                self._argv("start", sandbox_id),
                timeout=_START_TIMEOUT_S,
                action=f"start Databricks Sandbox '{sandbox_id}'",
            )
        click.echo(f"  → running {sandbox_id}")

    def is_running(self, sandbox_id: str) -> bool | None:
        """
        Return whether the control plane reports the sandbox as running.

        :param sandbox_id: The sandbox to inspect.
        :returns: ``True`` when Running, ``False`` for the known
            not-running states, ``None`` for anything unrecognized (so
            callers keep their existing liveness behavior).
        """
        status = self._status(sandbox_id)
        if status == _RUNNING_STATUS:
            return True
        if status in _STOPPED_STATUSES:
            return False
        return None

    def _status(self, sandbox_id: str) -> str:
        """
        Return the lower-cased status string for a sandbox.

        :param sandbox_id: The sandbox to inspect.
        :returns: e.g. ``"running"``, ``"stopped"``, or ``""`` when the
            control plane reported no status field.
        :raises click.ClickException: If the status call fails.
        """
        if self._use_rest():
            payload = self._rest_sandbox(sandbox_id)
        else:
            payload = self._cli_json(
                self._argv("status", sandbox_id, "-o", "json"),
                timeout=_CONTROL_TIMEOUT_S,
                action=f"read the status of Databricks Sandbox '{sandbox_id}'",
            )
        status = payload.get("status")
        return status.lower() if isinstance(status, str) else ""

    def keep_alive(self, sandbox_id: str) -> None:
        """
        Apply the configured auto-stop policy so the host survives idle
        gaps between turns.

        Soft-fail per the launcher contract: a rejected setting warns
        rather than aborting the bootstrap — the sandbox is usable, it
        just may reap itself on idle.

        :param sandbox_id: The sandbox to configure.
        """
        if self._no_autostop:
            flags = ["--no-autostop"]
            description = "idle auto-stop disabled"
        elif self._idle_timeout is not None:
            flags = ["--idle-timeout", self._idle_timeout]
            description = f"idle timeout set to {self._idle_timeout}"
        else:
            return
        if self._use_rest():
            # `sandbox_id` is repeated in the body, not just the path — the
            # CLI sends it that way and the API expects it.
            body: dict[str, object] = {"sandbox_id": sandbox_id}
            if self._no_autostop:
                body["no_autostop"] = True
            else:
                body["idle_timeout"] = _as_api_duration(self._idle_timeout or "")
            try:
                self._api(
                    "PATCH",
                    f"{_LAKEBOX_API_ROOT}/sandboxes/{sandbox_id}",
                    action=f"configure Databricks Sandbox '{sandbox_id}'",
                    body=body,
                )
            except click.ClickException as error:
                self._warn_autostop(sandbox_id, error.format_message())
                return
        else:
            completed = self._cli(
                self._argv("config", sandbox_id, *flags),
                action=f"configure Databricks Sandbox '{sandbox_id}'",
                check=False,
            )
            if completed.returncode != 0:
                self._warn_autostop(sandbox_id, _combined(completed).strip() or "no output")
                return
        click.echo(f"  → {description}")

    @staticmethod
    def _warn_autostop(sandbox_id: str, detail: str) -> None:
        """Report a soft-failed auto-stop change without aborting the launch."""
        click.echo(
            f"  → warning: could not configure auto-stop on '{sandbox_id}' "
            f"({detail}); the sandbox may stop after its idle timeout.",
            err=True,
        )

    def terminate(self, sandbox_id: str) -> None:
        """
        Delete a sandbox, releasing its compute and disk.

        ``--auto-approve`` is mandatory: ``delete`` prompts interactively
        otherwise, which would hang a server-side teardown. Idempotent
        from the caller's perspective — a sandbox that is already gone is
        treated as success, since the desired end state holds.

        :param sandbox_id: The sandbox to delete.
        :raises click.ClickException: When the delete fails for any
            reason other than the sandbox not existing.
        """
        if self._use_rest():
            try:
                self._api(
                    "DELETE",
                    f"{_LAKEBOX_API_ROOT}/sandboxes/{sandbox_id}",
                    action=f"delete Databricks Sandbox '{sandbox_id}'",
                )
            except click.ClickException as error:
                # Same idempotence as the CLI path: an already-gone sandbox
                # is the desired end state. The API says so with
                # `{"error_code":"NOT_FOUND","message":"sandbox … not found"}`.
                if not _looks_missing(error.format_message()):
                    raise
            return
        completed = self._cli(
            self._argv("delete", sandbox_id, "--auto-approve"),
            action=f"delete Databricks Sandbox '{sandbox_id}'",
            check=False,
        )
        if completed.returncode != 0 and not _looks_missing(_combined(completed)):
            raise click.ClickException(
                f"Could not delete Databricks Sandbox '{sandbox_id}' "
                f"(exit {completed.returncode}): "
                f"{_combined(completed).strip() or '<no output>'}"
            )

    # ── Transport ───────────────────────────────────────

    def run(self, sandbox_id: str, command: str, *, check: bool = True) -> RemoteCommandResult:
        """
        Run a shell command in the sandbox and capture its output.

        The remote transport keeps stdout and stderr separate, so both
        are reported faithfully rather than merged into ``stdout`` the
        way the SDK-backed providers must.

        When ``job_bootstrap`` is configured, the command is delegated to a
        one-shot classic-compute job exactly like the bootstrap is, because
        this process cannot reach the sandbox gateway on port 2222 — see
        :meth:`_run_via_job`. That path cannot separate the remote exit code
        from the job's own outcome (a non-zero remote exit fails the job), so
        it reports a remote failure as ``returncode=1`` with the job error in
        ``stderr`` rather than inventing a specific exit code.

        :param sandbox_id: Target sandbox.
        :param command: Shell command to execute remotely; quote remote
            paths yourself, per the base contract.
        :param check: When ``True``, raise on non-zero exit.
        :returns: Exit code plus captured stdout and stderr.
        :raises click.ClickException: If *check* is ``True`` and the
            command exits non-zero.
        """
        if self._job_bootstrap is not None:
            return self._run_via_job_as_command(
                self._job_bootstrap, sandbox_id, command, check=check
            )
        completed = self._cli(
            self._ssh_argv(sandbox_id, command),
            timeout=_REMOTE_COMMAND_TIMEOUT_S,
            action=f"run a command on Databricks Sandbox '{sandbox_id}'",
            check=False,
        )
        for line in completed.stdout.splitlines():
            if line.strip():
                click.echo(line)
        for line in completed.stderr.splitlines():
            if line.strip():
                click.echo(line, err=True)
        if check and completed.returncode != 0:
            raise click.ClickException(
                f"Remote command failed on Databricks Sandbox '{sandbox_id}' "
                f"(exit {completed.returncode}): {command}"
            )
        return RemoteCommandResult(
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    def _run_via_job_as_command(
        self,
        job_bootstrap: JobBootstrapConfig,
        sandbox_id: str,
        command: str,
        *,
        check: bool,
    ) -> RemoteCommandResult:
        """
        Adapt :meth:`_run_via_job` to the :meth:`run` contract.

        The job wrapper raises on any failure — remote non-zero exit, job
        error, timeout — so honoring ``check=False`` means catching that and
        mapping it onto a result. The mapping is deliberately coarse (see
        :meth:`run`): the remote command's own exit code is not recoverable
        from the job run, so callers that need it must not use this path.

        :param job_bootstrap: The configured job-delegation settings.
        :param sandbox_id: Target sandbox.
        :param command: Shell command to execute remotely.
        :param check: When ``True``, let the job failure propagate.
        :returns: The job task's captured stdout on success; a synthetic
            failure result carrying the job error when *check* is ``False``.
        :raises click.ClickException: If *check* is ``True`` and the job fails.
        """
        try:
            output = self._run_via_job(job_bootstrap, sandbox_id, command)
        except click.ClickException as exc:
            if check:
                raise
            return RemoteCommandResult(returncode=1, stdout="", stderr=exc.message)
        return RemoteCommandResult(returncode=0, stdout=output, stderr="")

    def put(self, sandbox_id: str, local_path: Path, remote_path: str) -> None:
        """
        Copy a local file into the sandbox.

        ``databricks sandbox ssh`` passes local stdin through to the
        remote process, so the transfer is a plain ``cat > <path>`` fed
        from the local file — no scp invocation and no dependency on the
        gateway exposing an sftp subsystem. Verified against CLI v1.11.0.

        :param sandbox_id: Target sandbox.
        :param local_path: Local file to read.
        :param remote_path: Absolute destination path on the sandbox,
            e.g. ``"/tmp/oa-wheels.tgz"``.
        :raises click.ClickException: If the file cannot be read or the
            transfer fails.
        """
        argv = self._ssh_argv(sandbox_id, f"cat > {shlex.quote(remote_path)}")
        try:
            with local_path.open("rb") as handle:
                completed = subprocess.run(  # argv list, no shell
                    argv,
                    stdin=handle,
                    capture_output=True,
                    text=True,
                    timeout=_REMOTE_COMMAND_TIMEOUT_S,
                    check=False,
                )
        except FileNotFoundError as exc:
            # Either the local file or the CLI itself is missing; the
            # local file is the likelier caller error, so name it.
            raise click.ClickException(
                f"Could not copy '{local_path}' into Databricks Sandbox '{sandbox_id}': {exc}"
            ) from exc
        except OSError as exc:
            raise click.ClickException(
                f"Could not read '{local_path}' to copy into Databricks "
                f"Sandbox '{sandbox_id}': {exc}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise click.ClickException(
                f"Timed out copying '{local_path}' into Databricks Sandbox '{sandbox_id}'."
            ) from exc
        if completed.returncode != 0:
            raise click.ClickException(
                f"File upload to Databricks Sandbox '{sandbox_id}' failed "
                f"(exit {completed.returncode}): "
                f"{_combined(completed).strip() or '<no output>'}"
            )

    def stream_exec(self, sandbox_id: str, command: str, *, pty: bool = False) -> RemoteProcess:
        """
        Spawn a command in the sandbox and stream its output line by line.

        ``pty`` is accepted for interface compatibility and **ignored**:
        passing ``-tt`` through ``databricks sandbox ssh`` was verified
        not to allocate a remote terminal on CLI v1.11.0. Commands that
        suppress output when not on a TTY will behave accordingly.

        :param sandbox_id: Target sandbox.
        :param command: Shell command to execute remotely.
        :param pty: Ignored — see above.
        :returns: A handle streaming the process's combined output.
        :raises click.ClickException: When the CLI is not installed.
        """
        del pty
        argv = self._ssh_argv(sandbox_id, command)
        try:
            process = subprocess.Popen(  # argv list, no shell
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError as exc:
            raise click.ClickException(INSTALL_HINT) from exc
        return _DatabricksRemoteProcess(process, command)

    def exec_foreground(self, sandbox_id: str, command: str) -> int:
        """
        Run *command* in the sandbox with local stdio inherited, blocking
        until it exits.

        No remote PTY is allocated (see the module docstring), so
        ``TERM`` is exported explicitly to give tmux-spawning harnesses a
        usable value rather than an empty one. Ctrl-C propagates to the
        child ``ssh``, which tears the remote command down.

        :param sandbox_id: Target sandbox.
        :param command: Shell command to execute remotely, e.g.
            ``"omnigent host --server https://…"``.
        :returns: The remote command's exit code.
        :raises click.ClickException: When the CLI is not installed.
        """
        argv = self._ssh_argv(sandbox_id, f"TERM=xterm-256color exec {command}")
        try:
            completed = subprocess.run(argv, check=False)  # argv list, no shell
        except FileNotFoundError as exc:
            raise click.ClickException(INSTALL_HINT) from exc
        except KeyboardInterrupt:
            click.echo("\n  → detaching; the remote command was signalled")
            raise
        return completed.returncode

    def forward_local_port(self, sandbox_id: str, port: int) -> AbstractContextManager[None]:
        """
        Forward ``127.0.0.1:<port>`` into the sandbox over SSH.

        ``databricks sandbox ssh <id> -- …`` hands the flags straight to
        ``ssh``. Two details are load-bearing, each fixing a failure seen
        live against CLI v1.11.0:

        - **Both endpoints are pinned to ``127.0.0.1``**, never
          ``localhost``. Unpinned, ``ssh`` binds the local port on BOTH
          ``::1`` and ``127.0.0.1``, and the sandbox side resolves
          ``localhost`` to whichever family it prefers — so a callback
          server bound only to IPv4 loopback can be missed entirely.
        - **Readiness comes from ``ssh -v`` itself**, not from probing the
          port. See :meth:`_await_forward`.

        Only bare flags are passed: the Databricks CLI re-quotes what it
        forwards to ``ssh`` and mangles ``-o Key=Value`` (see
        :data:`_FORWARD_FAILURE_MARKERS`), so the equivalent of
        ``ExitOnForwardFailure`` is done by watching ``ssh``'s own
        "cannot listen to port" output instead of setting the option.

        The context manager does not yield until ``ssh`` reports the
        forward listening, so a caller that immediately opens a browser
        at it cannot race the bind.

        :param sandbox_id: Target sandbox.
        :param port: Local + remote loopback port to bridge, e.g. ``8022``.
        :returns: Context manager holding the forward open.
        :raises click.ClickException: When the CLI is missing, the child
            exits early, or the forward is not established in time.
        """
        return self._forward_local_port(sandbox_id, port)

    @contextmanager
    def _forward_local_port(self, sandbox_id: str, port: int) -> Iterator[None]:
        """
        Implement :meth:`forward_local_port`.

        Split out so the public method can carry the documented
        ``AbstractContextManager`` return type rather than a generator.
        """
        argv = self._ssh_argv(
            sandbox_id,
            "-v",
            "-N",
            "-L",
            f"127.0.0.1:{port}:127.0.0.1:{port}",
        )
        try:
            process = subprocess.Popen(  # argv list, no shell
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError as exc:
            raise click.ClickException(INSTALL_HINT) from exc
        # ssh keeps logging for the life of the session; an unread pipe
        # would fill and wedge it, so one daemon thread drains stdout
        # for the whole lifetime and the readiness wait just watches
        # what it collects.
        transcript: list[str] = []
        settled = threading.Event()
        state: dict[str, bool] = {}
        reader = threading.Thread(
            target=_drain_forward_output,
            args=(process, transcript, settled, state, port),
            name="databricks-forward-reader",
            daemon=True,
        )
        reader.start()
        try:
            self._await_forward(process, settled, state, transcript, port, sandbox_id)
            yield
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=_FORWARD_TERMINATE_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            reader.join(timeout=_FORWARD_TERMINATE_TIMEOUT_S)

    @staticmethod
    def _await_forward(
        process: subprocess.Popen[str],
        settled: threading.Event,
        state: dict[str, bool],
        transcript: list[str],
        port: int,
        sandbox_id: str,
    ) -> None:
        """
        Block until ``ssh`` reports the local forward listening.

        Readiness is taken from ``ssh -v``'s own
        ``"Local forwarding listening on …"`` line rather than from
        probing the port, because BOTH probe styles are wrong here:

        - A **connect** probe traverses the tunnel and opens a real TCP
          session against whatever listens on the sandbox side. The
          in-sandbox App OAuth callback server accepts exactly one
          connection, so the probe consumes the very request the caller
          is waiting for. Observed live: the forward reported healthy and
          the caller's own connection then hung until timeout.
        - A **bind** probe races ``ssh`` for the port. When the probe
          happens to hold it at the moment ``ssh`` binds, ``ssh`` loses
          and the tunnel is dead while the port still looks taken.
          Observed live at roughly one run in three.

        Watching ``ssh``'s own output has neither failure mode: it
        touches nothing, and it reports the state ``ssh`` actually
        reached.

        :param process: The forwarding child, polled so an early exit is
            reported as itself rather than as a readiness timeout.
        :param settled: Set by the reader thread once the outcome is known.
        :param state: Carries ``listening`` / ``failed`` from the reader.
        :param transcript: Lines collected so far, for error messages.
        :param port: Local loopback port expected to bind.
        :param sandbox_id: Target sandbox, for error messages.
        :raises click.ClickException: When ``ssh`` reports it cannot
            listen, the child exits early, or the forward is not reported
            within :data:`_FORWARD_BIND_TIMEOUT_S`.
        """
        deadline = time.monotonic() + _FORWARD_BIND_TIMEOUT_S
        while time.monotonic() < deadline:
            if settled.wait(timeout=_FORWARD_POLL_INTERVAL_S):
                if state.get("listening"):
                    return
                raise click.ClickException(
                    f"Port forward to Databricks Sandbox '{sandbox_id}' could not "
                    f"listen on 127.0.0.1:{port}: {_forward_failure_detail(transcript)}"
                )
            if process.poll() is not None:
                raise click.ClickException(
                    f"Port forward to Databricks Sandbox '{sandbox_id}' exited "
                    f"before listening on 127.0.0.1:{port} "
                    f"(exit {process.returncode}): "
                    f"{_forward_failure_detail(transcript)}"
                )
        raise click.ClickException(
            f"Port forward to Databricks Sandbox '{sandbox_id}' did not report "
            f"listening on 127.0.0.1:{port} within "
            f"{_FORWARD_BIND_TIMEOUT_S:.0f}s: {_forward_failure_detail(transcript)}"
        )

    def wheel_install_command(self, remote_tgz_path: str) -> str:
        """
        Remote command that installs locally-built wheels over the
        sandbox's preinstalled omnigent.

        Diverges from
        :func:`~omnigent.onboarding.sandboxes.base.host_image_wheel_install_command`
        because the Databricks Sandbox image is not the Omnigent host
        image: its interpreter is PEP 668 externally-managed and its
        ``dist-packages`` are root-owned, so ``pip`` needs
        ``--break-system-packages`` and falls back to a user install.
        User site precedes ``dist-packages`` on ``sys.path``, so the
        overlay wins at import time.

        ``--force-reinstall`` is required for the same reason as the host
        image: the baked omnigent shares a version with the local build,
        so pip would otherwise consider it satisfied and silently skip.
        ``--no-deps`` keeps the overlay to the local wheels alone.

        :param remote_tgz_path: Sandbox path of the shipped tarball, e.g.
            ``"/tmp/oa-wheels.tgz"``.
        :returns: Shell command string for :meth:`run`.
        """
        quoted = shlex.quote(remote_tgz_path)
        return (
            "cd /tmp && rm -rf oa-wheels && mkdir oa-wheels && "
            f"tar xzf {quoted} -C oa-wheels --warning=no-unknown-keyword && "
            "pip install --quiet --break-system-packages --force-reinstall "
            "--no-deps --no-warn-script-location oa-wheels/*.whl"
        )


def _combined(completed: subprocess.CompletedProcess[str]) -> str:
    """
    Join a completed process's captured streams for error reporting.

    :param completed: The finished process.
    :returns: stdout and stderr concatenated, in that order.
    """
    return f"{completed.stdout or ''}{completed.stderr or ''}"
