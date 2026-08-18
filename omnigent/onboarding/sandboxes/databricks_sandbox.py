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

# Sandbox control-plane REST surface, as used by the bootstrap notebook.
# Mirrors `cmd/sandbox/api.go` in the Databricks CLI, whose `sandboxAPIRoot`
# still says "lakebox" because the server-side rename is pending.
_LAKEBOX_API_ROOT: str = "/api/2.0/lakebox"

# Port the sandbox SSH gateway listens on (CLI: `defaultGatewayPort`).
_SANDBOX_GATEWAY_PORT: str = "2222"

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
# Private key only: unlike the CLI, nothing here pre-verifies registration
# against `<key>.pub`, so no public half is needed on the cluster.
key_material = dbutils.secrets.get(scope="{ssh_key_scope}", key="{ssh_key_key}")
key_path = os.path.expanduser("~/.ssh/sandbox_ed25519")
os.makedirs(os.path.dirname(key_path), exist_ok=True)
with open(key_path, "w") as fh:
    fh.write(key_material)
    if not key_material.endswith("\\n"):
        fh.write("\\n")
os.chmod(key_path, stat.S_IRUSR | stat.S_IWUSR)

payload = json.loads(dbutils.secrets.get(scope="{argv_scope}", key="{argv_key}"))
sandbox_id = payload["sandbox_id"]
remote_command = payload["remote_command"]
# Anything the caller marks secret (the armed host token) must not survive
# into this notebook's output. `dbutils.secrets.get` is redacted by
# Databricks automatically, but the remote command's OWN stdout/stderr is
# not, and both `dbutils.notebook.exit` and an exception message land in
# durable job-run JSON that anyone with job-read access can read.
redact = [str(value) for value in payload.get("redact", []) if value]


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
    — one job-run round trip, not one per step. The job pays a cold
    single-node cluster start (observed 3-6 minutes) on every managed launch;
    that latency is the price of the routing gap.

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
    """Databricks Runtime version for the throwaway cluster."""

    timeout_s: float = _JOB_BOOTSTRAP_TIMEOUT_S
    """Ceiling on job-run completion, dominated by cluster spin-up."""

    @property
    def payload_scope(self) -> str:
        """
        The scope the transient connect payload is written to and deleted from.

        Falls back to :attr:`ssh_key_secret_scope` so a config that predates
        the split keeps working — at the cost of the coordinator needing
        ``WRITE`` on the scope holding the private key.
        """
        return self.payload_secret_scope or self.ssh_key_secret_scope


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
        """
        self._cli_path = cli_path or DEFAULT_CLI_BINARY
        self._profile = profile
        self._idle_timeout = idle_timeout
        self._no_autostop = no_autostop
        self._bootstrap_command = bootstrap_command
        self._job_bootstrap = job_bootstrap
        # Memoized lazily by `_sdk()`; the SDK is an optional extra, so it is
        # never imported on the direct-SSH path.
        self._client: object | None = None

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
        """
        if self._job_bootstrap is not None:
            return self._start_host_via_job(sandbox_id, self._job_bootstrap, **kwargs)
        if self._bootstrap_command:
            self.run(sandbox_id, self._bootstrap_command)
        return super().start_host(sandbox_id, **kwargs)

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
        output = self._run_via_job(job_bootstrap, sandbox_id, script, redact=(token,))
        for line in output.splitlines():
            if line.startswith(_WORKSPACE_TAG):
                return line[len(_WORKSPACE_TAG) :].strip()
        raise click.ClickException(
            f"job-delegated bootstrap for Databricks Sandbox '{sandbox_id}' completed "
            f"but its output did not include the expected {_WORKSPACE_TAG!r} tag — "
            f"cannot report the workspace path. Output: {output!r}"
        )

    def _run_via_job(
        self,
        job_bootstrap: JobBootstrapConfig,
        sandbox_id: str,
        remote_command: str,
        *,
        redact: Sequence[str] = (),
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

        :param redact: Strings (the armed host token) scrubbed from anything
            the notebook returns, because notebook output is durable job-run
            JSON readable by anyone with job-read access.
        :returns: The job task's captured stdout, with *redact* scrubbed.
        :raises click.ClickException: On job failure, timeout, or missing output.
        """
        from databricks.sdk.service.compute import ClusterSpec
        from databricks.sdk.service.jobs import NotebookTask, RunResultState, SubmitTask
        from databricks.sdk.service.workspace import ImportFormat, Language

        client = self._sdk()
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
            waiter = client.jobs.submit(
                run_name="omnigent-sandbox-job-bootstrap",
                tasks=[
                    SubmitTask(
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
                ],
                timeout_seconds=int(job_bootstrap.timeout_s),
            )
            run = waiter.result()
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

        :param sandbox_id: Target sandbox.
        :param command: Shell command to execute remotely; quote remote
            paths yourself, per the base contract.
        :param check: When ``True``, raise on non-zero exit.
        :returns: Exit code plus captured stdout and stderr.
        :raises click.ClickException: If *check* is ``True`` and the
            command exits non-zero.
        """
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
