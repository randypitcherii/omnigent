"""Tests for :mod:`omnigent.onboarding.sandboxes.databricks_sandbox`."""

from __future__ import annotations

import io
import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import click
import pytest

from omnigent.onboarding.sandboxes import databricks_sandbox as dbx
from omnigent.onboarding.sandboxes.databricks_sandbox import (
    AUTH_HINT,
    INSTALL_HINT,
    DatabricksSandboxLauncher,
    JobBootstrapConfig,
)

# ── Fake `databricks` CLI ───────────────────────────────────
#
# The launcher's whole transport is `subprocess.run` / `subprocess.Popen`
# against the real Databricks CLI, which the test environment neither has
# nor should reach. These are hand-rolled recorders (never MagicMock: the
# launcher's argv must hit an explicit stub, not silently succeed), swapped
# in with `monkeypatch.setattr(dbx.subprocess, …)` — the same pattern
# tests/onboarding/sandboxes/test_bootstrap.py uses.


@dataclass
class _Reply:
    """
    One canned CLI response.

    :param returncode: Exit code the fake CLI reports.
    :param stdout: Captured standard output.
    :param stderr: Captured standard error.
    """

    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


@dataclass
class _Call:
    """
    One recorded ``subprocess.run`` invocation.

    :param argv: The argv list the launcher built.
    :param stdin: Whatever was passed as ``stdin``, or ``None``.
    """

    argv: list[str]
    stdin: Any = None


@dataclass
class _FakeCLI:
    """
    Recorder + canned-reply table for the fake Databricks CLI.

    :param replies: Reply queue keyed by subcommand (``"create"``,
        ``"ssh"``, …). Each call pops the next reply for its subcommand,
        falling back to a bare success once the queue empties, so a test
        only has to script the responses it cares about.
    :param calls: Every invocation, in order.
    """

    replies: dict[str, list[_Reply]] = field(default_factory=dict)
    calls: list[_Call] = field(default_factory=list)

    def reply_with(self, subcommand: str, *replies: _Reply) -> None:
        """
        Queue responses for one subcommand.

        :param subcommand: e.g. ``"status"``.
        :param replies: Responses, returned in order.
        """
        self.replies.setdefault(subcommand, []).extend(replies)

    def argvs(self, subcommand: str) -> list[list[str]]:
        """
        Return every recorded argv for one subcommand.

        :param subcommand: e.g. ``"config"``.
        :returns: The matching argv lists, in call order.
        """
        return [call.argv for call in self.calls if _subcommand(call.argv) == subcommand]

    def run(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        """Stand in for ``subprocess.run``, recording and replying."""
        self.calls.append(_Call(argv=list(argv), stdin=kwargs.get("stdin")))
        queue = self.replies.get(_subcommand(argv), [])
        reply = queue.pop(0) if queue else _Reply()
        return subprocess.CompletedProcess(
            args=argv,
            returncode=reply.returncode,
            stdout=reply.stdout,
            stderr=reply.stderr,
        )


def _subcommand(argv: list[str]) -> str:
    """
    Extract the ``databricks sandbox <sub>`` subcommand from an argv.

    :param argv: The full argv list.
    :returns: The subcommand, or ``""`` when the argv is not shaped like
        a sandbox call.
    """
    if "sandbox" not in argv:
        return ""
    index = argv.index("sandbox") + 1
    return argv[index] if index < len(argv) else ""


def _install(monkeypatch: pytest.MonkeyPatch, cli: _FakeCLI | None = None) -> _FakeCLI:
    """
    Swap in the fake CLI and make ``shutil.which`` resolve the binary.

    :param monkeypatch: pytest's patcher.
    :param cli: An existing recorder to reuse, or ``None`` for a fresh one.
    :returns: The recorder now backing ``subprocess.run``.
    """
    fake = cli if cli is not None else _FakeCLI()
    monkeypatch.setattr(dbx.subprocess, "run", fake.run)
    monkeypatch.setattr(dbx.shutil, "which", lambda name: f"/fake/{name}")
    return fake


def _remote_command(argv: list[str]) -> str:
    """
    Return the remote command an ``ssh`` argv carries after ``--``.

    :param argv: A recorded ``ssh`` argv.
    :returns: The single remote-command element.
    """
    return argv[argv.index("--") + 1]


# ── capabilities ────────────────────────────────────────────


def test_capabilities_declare_port_forward_and_resume() -> None:
    """
    The two flags that distinguish this provider from every other
    exec-model launcher must be True: `databricks sandbox ssh` is real
    SSH (so `-L` forwarding works) and sandboxes restart with their disk
    intact (so a dormant host resumes under the same id). Both were
    verified live; if either regresses to False the bootstrap silently
    routes around a capability the platform actually has.
    """
    capabilities = DatabricksSandboxLauncher().capabilities
    assert capabilities.local_port_forward is True
    assert capabilities.resume_stopped is True
    assert capabilities.managed_launch is True
    assert capabilities.programmatic_terminate is True
    assert capabilities.file_copy is True


# ── prepare ─────────────────────────────────────────────────


def test_prepare_raises_install_hint_when_cli_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    No `databricks` on PATH must fail with the install remediation, not
    a bare FileNotFoundError from the first subprocess call.
    """
    monkeypatch.setattr(dbx.shutil, "which", lambda name: None)
    with pytest.raises(click.ClickException) as exc:
        DatabricksSandboxLauncher().prepare()
    assert str(exc.value) == INSTALL_HINT


def test_prepare_raises_auth_hint_when_list_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    An installed-but-unauthenticated CLI must point at `databricks auth
    login`, and must surface the CLI's own reason verbatim so an
    operator can tell "wrong workspace" from "key not registered".
    """
    cli = _install(monkeypatch)
    cli.reply_with("list", _Reply(returncode=1, stderr="cannot resolve credentials"))
    with pytest.raises(click.ClickException) as exc:
        DatabricksSandboxLauncher().prepare()
    assert AUTH_HINT in str(exc.value)
    assert "cannot resolve credentials" in str(exc.value)


def test_prepare_accepts_authenticated_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful `sandbox list` is the whole preflight."""
    cli = _install(monkeypatch)
    cli.reply_with("list", _Reply(stdout="[]"))
    DatabricksSandboxLauncher().prepare()
    assert cli.argvs("list") == [["databricks", "sandbox", "list", "-o", "json"]]


# ── provision ───────────────────────────────────────────────


def test_provision_creates_and_returns_sandbox_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    `create --json` is the id source. The label rides as `--name`, and
    the returned id (not the label) is what every later primitive uses.
    """
    cli = _install(monkeypatch)
    cli.reply_with("create", _Reply(stdout=json.dumps({"sandboxId": "spanking-pitta-1649"})))
    sandbox_id = DatabricksSandboxLauncher().provision("managed-a1b2c3d4")
    assert sandbox_id == "spanking-pitta-1649"
    assert cli.argvs("create") == [
        ["databricks", "sandbox", "create", "--name", "managed-a1b2c3d4", "--json"]
    ]


def test_provision_disables_autostop_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    A managed host sits idle between turns, so provision must follow
    `create` with `config --no-autostop`. Without it the sandbox reaps
    itself on the workspace's default idle timeout and the session dies.
    """
    cli = _install(monkeypatch)
    cli.reply_with("create", _Reply(stdout=json.dumps({"sandboxId": "sb-1"})))
    DatabricksSandboxLauncher().provision("managed-a1b2c3d4")
    assert cli.argvs("config") == [["databricks", "sandbox", "config", "sb-1", "--no-autostop"]]


def test_provision_applies_idle_timeout_when_autostop_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    With `no_autostop=False`, a configured idle timeout is applied
    instead — the two knobs are mutually exclusive on the CLI.
    """
    cli = _install(monkeypatch)
    cli.reply_with("create", _Reply(stdout=json.dumps({"sandboxId": "sb-1"})))
    DatabricksSandboxLauncher(no_autostop=False, idle_timeout="4h").provision("label")
    assert cli.argvs("config") == [
        ["databricks", "sandbox", "config", "sb-1", "--idle-timeout", "4h"]
    ]


def test_provision_skips_config_when_no_policy_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No policy configured → no `config` call at all."""
    cli = _install(monkeypatch)
    cli.reply_with("create", _Reply(stdout=json.dumps({"sandboxId": "sb-1"})))
    DatabricksSandboxLauncher(no_autostop=False).provision("label")
    assert cli.argvs("config") == []


def test_provision_raises_when_create_returns_no_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    A create that succeeds but reports no `sandboxId` must fail loud —
    returning an empty id would strand every later primitive on a
    sandbox that does not exist.
    """
    cli = _install(monkeypatch)
    cli.reply_with("create", _Reply(stdout=json.dumps({"status": "Running"})))
    with pytest.raises(click.ClickException, match="no 'sandboxId'"):
        DatabricksSandboxLauncher().provision("label")


def test_provision_raises_on_unparseable_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-JSON stdout must surface as a clear error, not a ValueError."""
    cli = _install(monkeypatch)
    cli.reply_with("create", _Reply(stdout="Creating sandbox…"))
    with pytest.raises(click.ClickException, match="unparseable JSON"):
        DatabricksSandboxLauncher().provision("label")


def test_profile_is_threaded_through_every_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    A configured profile must reach every CLI invocation. A server
    running under a service-principal profile that leaked a single
    unprofiled call would silently fall back to the DEFAULT profile —
    i.e. a different workspace.
    """
    cli = _install(monkeypatch)
    cli.reply_with("create", _Reply(stdout=json.dumps({"sandboxId": "sb-1"})))
    launcher = DatabricksSandboxLauncher(profile="sandbox-sp")
    launcher.provision("label")
    launcher.run("sb-1", "true")
    launcher.terminate("sb-1")
    for call in cli.calls:
        assert "-p" in call.argv, call.argv
        assert call.argv[call.argv.index("-p") + 1] == "sandbox-sp"


def test_profile_flag_never_crosses_the_ssh_separator(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    `-p` is a CLI flag, not an ssh flag: it must appear BEFORE `--`.
    Past the separator the Databricks CLI hands arguments straight to
    ssh, where `-p` means "port" and would silently redirect the
    connection.
    """
    cli = _install(monkeypatch)
    DatabricksSandboxLauncher(profile="sandbox-sp").run("sb-1", "echo hi")
    argv = cli.argvs("ssh")[0]
    assert argv.index("-p") < argv.index("--")


def test_cli_path_override_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit executable path replaces the bare `databricks` name."""
    cli = _install(monkeypatch)
    DatabricksSandboxLauncher(cli_path="/opt/databricks/bin/databricks").run("sb-1", "true")
    assert cli.argvs("ssh")[0][0] == "/opt/databricks/bin/databricks"


# ── run ─────────────────────────────────────────────────────


def test_run_passes_command_as_one_argument_after_separator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The remote command must ride as a SINGLE argv element after `--`.
    Splitting it on whitespace would let the sandbox's login shell see
    `echo` and `hi` as separate ssh arguments and break every command
    containing shell syntax.
    """
    cli = _install(monkeypatch)
    DatabricksSandboxLauncher().run("sb-1", "echo hi; exit 0")
    argv = cli.argvs("ssh")[0]
    assert argv[:4] == ["databricks", "sandbox", "ssh", "sb-1"]
    assert _remote_command(argv) == "echo hi; exit 0"


def test_run_reports_stdout_and_stderr_separately(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Unlike the SDK-backed providers, this transport keeps the two
    streams apart — the launcher must not merge them into `stdout`.
    """
    cli = _install(monkeypatch)
    cli.reply_with("ssh", _Reply(stdout="out\n", stderr="err\n"))
    result = DatabricksSandboxLauncher().run("sb-1", "true")
    assert result.stdout == "out\n"
    assert result.stderr == "err\n"
    assert result.returncode == 0


def test_run_raises_on_nonzero_when_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    """A checked failure names the sandbox, the exit code, and the command."""
    cli = _install(monkeypatch)
    cli.reply_with("ssh", _Reply(returncode=7, stderr="boom\n"))
    with pytest.raises(click.ClickException) as exc:
        DatabricksSandboxLauncher().run("sb-1", "false")
    assert "sb-1" in str(exc.value)
    assert "exit 7" in str(exc.value)
    assert "false" in str(exc.value)


def test_run_returns_failure_when_unchecked(monkeypatch: pytest.MonkeyPatch) -> None:
    """`check=False` reports the exit code instead of raising."""
    cli = _install(monkeypatch)
    cli.reply_with("ssh", _Reply(returncode=7))
    result = DatabricksSandboxLauncher().run("sb-1", "false", check=False)
    assert result.returncode == 7


def test_run_raises_install_hint_when_cli_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    A missing binary surfacing mid-bootstrap (not just at `prepare`)
    must still carry the install remediation.
    """

    def _missing(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        """Stand in for an absent executable."""
        raise FileNotFoundError(argv[0])

    monkeypatch.setattr(dbx.subprocess, "run", _missing)
    with pytest.raises(click.ClickException) as exc:
        DatabricksSandboxLauncher().run("sb-1", "true")
    assert str(exc.value) == INSTALL_HINT


def test_run_raises_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """A wedged SSH session must time out with a named action, not hang."""

    def _hang(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        """Stand in for a command that never returns."""
        raise subprocess.TimeoutExpired(cmd=argv, timeout=float(kwargs["timeout"]))

    monkeypatch.setattr(dbx.subprocess, "run", _hang)
    with pytest.raises(click.ClickException, match="Timed out"):
        DatabricksSandboxLauncher().run("sb-1", "true")


# ── put ─────────────────────────────────────────────────────


def test_put_streams_the_file_over_stdin(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    File shipping is `cat > <path>` fed from local stdin — verified as
    the working transport, since the CLI passes stdin through. The
    remote path must be shell-quoted: it lands in the sandbox's login
    shell, where an unquoted space or `;` would truncate or inject.
    """
    cli = _install(monkeypatch)
    local = tmp_path / "oa-wheels.tgz"
    local.write_bytes(b"wheel-bytes")
    DatabricksSandboxLauncher().put("sb-1", local, "/tmp/dir with space/oa-wheels.tgz")
    argv = cli.argvs("ssh")[0]
    assert _remote_command(argv) == "cat > '/tmp/dir with space/oa-wheels.tgz'"
    # The local file handle — not its bytes, not a path string — is what
    # the CLI child reads from.
    stdin = cli.calls[0].stdin
    assert stdin is not None
    assert stdin.name == str(local)


def test_put_raises_when_local_file_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An unreadable local file must name the path, not the CLI."""
    _install(monkeypatch)
    with pytest.raises(click.ClickException) as exc:
        DatabricksSandboxLauncher().put("sb-1", tmp_path / "absent.tgz", "/tmp/x")
    assert "absent.tgz" in str(exc.value)


def test_put_raises_when_transfer_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A non-zero `cat` (e.g. a read-only destination) must fail loud."""
    cli = _install(monkeypatch)
    cli.reply_with("ssh", _Reply(returncode=1, stderr="Permission denied"))
    local = tmp_path / "f.tgz"
    local.write_bytes(b"x")
    with pytest.raises(click.ClickException) as exc:
        DatabricksSandboxLauncher().put("sb-1", local, "/root/f.tgz")
    assert "Permission denied" in str(exc.value)


# ── lifecycle ───────────────────────────────────────────────


def test_terminate_passes_auto_approve(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    `delete` prompts interactively without `--auto-approve`, which would
    hang a server-side teardown forever. The flag is not optional.
    """
    cli = _install(monkeypatch)
    DatabricksSandboxLauncher().terminate("sb-1")
    assert cli.argvs("delete") == [["databricks", "sandbox", "delete", "sb-1", "--auto-approve"]]


def test_terminate_is_idempotent_for_a_missing_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Deleting an already-gone sandbox is success: the desired end state
    holds, and managed teardown races another cleanup path routinely.
    """
    cli = _install(monkeypatch)
    cli.reply_with("delete", _Reply(returncode=1, stderr="sandbox sb-1 not found"))
    DatabricksSandboxLauncher().terminate("sb-1")


def test_terminate_raises_on_other_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """A delete that fails for any other reason must NOT be swallowed."""
    cli = _install(monkeypatch)
    cli.reply_with("delete", _Reply(returncode=1, stderr="permission denied"))
    with pytest.raises(click.ClickException, match="permission denied"):
        DatabricksSandboxLauncher().terminate("sb-1")


def test_resume_starts_the_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resume is `start <id>`, which blocks until the microVM is Running."""
    cli = _install(monkeypatch)
    DatabricksSandboxLauncher().resume("sb-1")
    assert cli.argvs("start") == [["databricks", "sandbox", "start", "sb-1"]]


def test_attach_starts_a_stopped_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    A stopped sandbox retains its disk, so attaching revives it rather
    than rejecting it — the Daytona posture, not the Modal one.
    """
    cli = _install(monkeypatch)
    cli.reply_with("status", _Reply(stdout=json.dumps({"status": "Stopped"})))
    DatabricksSandboxLauncher().attach("sb-1")
    assert cli.argvs("start") == [["databricks", "sandbox", "start", "sb-1"]]


def test_attach_leaves_a_running_sandbox_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """An already-running sandbox must not be restarted."""
    cli = _install(monkeypatch)
    cli.reply_with("status", _Reply(stdout=json.dumps({"status": "Running"})))
    DatabricksSandboxLauncher().attach("sb-1")
    assert cli.argvs("start") == []


@pytest.mark.parametrize(
    ("status", "expected"),
    [("Running", True), ("Stopped", False), ("Stopping", False), ("Wedged", None)],
)
def test_is_running_maps_control_plane_status(
    monkeypatch: pytest.MonkeyPatch, status: str, expected: bool | None
) -> None:
    """
    Known states map to True/False; anything unrecognized maps to None
    so callers keep their existing liveness behavior instead of acting
    on a guess.
    """
    cli = _install(monkeypatch)
    cli.reply_with("status", _Reply(stdout=json.dumps({"status": status})))
    assert DatabricksSandboxLauncher().is_running("sb-1") is expected


def test_keep_alive_warns_instead_of_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Keep-alive is soft-fail per the launcher contract: a rejected policy
    leaves a usable sandbox that may reap itself, which is worth a
    warning but not an aborted bootstrap.
    """
    cli = _install(monkeypatch)
    cli.reply_with("config", _Reply(returncode=1, stderr="policy rejected"))
    DatabricksSandboxLauncher().keep_alive("sb-1")


# ── forward_local_port ──────────────────────────────────────


class _FakeSshForward:
    """
    Stand-in for the ``ssh -v -N -L`` child.

    Emits a scripted ``ssh -v`` transcript on stdout — readiness is read
    from that transcript, so the fake reproduces the real signal without
    binding a port or reaching a network.

    :param transcript: Lines the fake ssh prints, in order.
    :param returncode: Exit status once the transcript is exhausted, or
        ``None`` to stay alive like a healthy forward.
    """

    def __init__(self, transcript: list[str], returncode: int | None = None) -> None:
        self.stdout = io.StringIO("".join(transcript))
        self._returncode = returncode
        self.terminated = False

    @property
    def returncode(self) -> int | None:
        """Exit status, read by the launcher's error path."""
        return self._returncode

    def poll(self) -> int | None:
        """Report the child's exit status, if it has one."""
        return self._returncode

    def terminate(self) -> None:
        """Record the teardown signal."""
        self.terminated = True
        self._returncode = 0

    def wait(self, timeout: float | None = None) -> int:
        """Reap the fake child."""
        del timeout
        self._returncode = 0 if self._returncode is None else self._returncode
        return self._returncode

    def kill(self) -> None:
        """Nothing to kill — the fake never blocks on terminate."""


_READY_TRANSCRIPT = [
    "debug1: Authentication succeeded (publickey).\n",
    "debug1: Local forwarding listening on 127.0.0.1 port 8022.\n",
    "debug1: Entering interactive session.\n",
]


def test_forward_pins_both_endpoints_to_ipv4_loopback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Both ends of the `-L` spec must be literal 127.0.0.1, never
    `localhost`.

    Unpinned, ssh binds the local port on BOTH ::1 and 127.0.0.1 while
    the sandbox side resolves `localhost` to whichever family it
    prefers — so a callback server bound only to IPv4 loopback can be
    missed entirely.

    The argv must also carry no `-o` option: the Databricks CLI
    re-quotes what it forwards to ssh, and `-o ExitOnForwardFailure=yes`
    was verified live to produce "Bad configuration option" while the
    single-token spelling silently dropped `-N`.
    """
    _install(monkeypatch)
    seen: list[list[str]] = []

    def _fake_popen(argv: list[str], **kwargs: Any) -> _FakeSshForward:
        """Record the argv and hand back a ready forward."""
        seen.append(argv)
        return _FakeSshForward(_READY_TRANSCRIPT)

    monkeypatch.setattr(dbx.subprocess, "Popen", _fake_popen)
    with DatabricksSandboxLauncher().forward_local_port("sb-1", 8022):
        pass
    argv = seen[0]
    assert "-L" in argv
    assert argv[argv.index("-L") + 1] == "127.0.0.1:8022:127.0.0.1:8022"
    assert "localhost" not in " ".join(argv)
    assert "-N" in argv
    # -v is what produces the readiness line the wait depends on.
    assert "-v" in argv
    # No `-o` options: the Databricks CLI re-quotes forwarded arguments
    # and mangles `-o Key=Value`, which cost a live debugging cycle when
    # `ExitOnForwardFailure` silently broke `-N`.
    assert "-o" not in argv
    assert not any(token.startswith("-o") and len(token) > 2 for token in argv)


def test_forward_waits_for_ssh_to_report_listening(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Readiness comes from ssh's own transcript, and NOT from probing the
    port — both probe styles were observed to fail live.

    A *connect* probe traverses the tunnel and opens a real session
    against whatever listens in the sandbox; the App OAuth callback
    server accepts exactly ONE connection, so the probe consumed the very
    request the caller was waiting for. A *bind* probe races ssh for the
    port and, about one run in three, won — leaving ssh's own bind to
    fail and the tunnel dead while the port still looked taken.

    A transcript that never reports listening must therefore time out
    rather than be declared ready by a probe that touched the port.
    """
    _install(monkeypatch)
    monkeypatch.setattr(dbx, "_FORWARD_BIND_TIMEOUT_S", 0.5)
    monkeypatch.setattr(dbx, "_FORWARD_POLL_INTERVAL_S", 0.05)
    silent = ["debug1: Authentication succeeded (publickey).\n"]
    monkeypatch.setattr(dbx.subprocess, "Popen", lambda argv, **kwargs: _FakeSshForward(silent))
    with pytest.raises(click.ClickException, match="did not report listening"):
        with DatabricksSandboxLauncher().forward_local_port("sb-1", 8022):
            pass


def test_forward_fails_fast_when_ssh_cannot_listen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    ssh reporting "cannot listen to port" must abort immediately rather
    than burn the full readiness timeout.

    This is the stand-in for `ExitOnForwardFailure=yes`, which the
    Databricks CLI cannot pass through intact. Without it, a port
    collision costs the caller the entire bind timeout before it learns
    anything.
    """
    _install(monkeypatch)
    # Deliberately generous: a slow path here would mean the failure was
    # detected by the deadline rather than by the transcript.
    monkeypatch.setattr(dbx, "_FORWARD_BIND_TIMEOUT_S", 30.0)
    monkeypatch.setattr(dbx, "_FORWARD_POLL_INTERVAL_S", 0.05)
    refused = [
        "debug1: Authentication succeeded (publickey).\n",
        "bind [127.0.0.1]:8022: Address already in use\n",
        "channel_setup_fwd_listener_tcpip: cannot listen to port: 8022\n",
    ]
    # Still alive — exactly the case ExitOnForwardFailure would have
    # turned into an exit, and the reason the transcript is watched.
    monkeypatch.setattr(dbx.subprocess, "Popen", lambda argv, **kwargs: _FakeSshForward(refused))
    started = time.monotonic()
    with pytest.raises(click.ClickException, match="could not listen"):
        with DatabricksSandboxLauncher().forward_local_port("sb-1", 8022):
            pass
    assert time.monotonic() - started < 10.0


def test_forward_reports_ssh_own_error_when_child_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    An ssh that dies before listening for a reason OTHER than a bind
    failure (bad key, deleted sandbox, gateway refusal) must be reported
    with ssh's own words and exit code, not as a generic timeout the
    operator has to wait out.

    Distinct from the bind-failure path above: there the child is still
    alive and only the transcript reveals the problem; here the child is
    gone and its exit status is the evidence.
    """
    _install(monkeypatch)
    monkeypatch.setattr(dbx, "_FORWARD_BIND_TIMEOUT_S", 2.0)
    monkeypatch.setattr(dbx, "_FORWARD_POLL_INTERVAL_S", 0.05)
    failed = [
        "debug1: Offering public key: /home/me/.ssh/sandbox_ed25519\n",
        "ssh: Permission denied (publickey).\n",
    ]
    monkeypatch.setattr(
        dbx.subprocess, "Popen", lambda argv, **kwargs: _FakeSshForward(failed, returncode=255)
    )
    with pytest.raises(click.ClickException) as exc:
        with DatabricksSandboxLauncher().forward_local_port("sb-1", 8022):
            pass
    assert "Permission denied" in str(exc.value)
    assert "exit 255" in str(exc.value)


def test_forward_tears_down_the_child_on_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Leaving the block must stop ssh — a leaked forward holds the local
    port and keeps a billable SSH session open against the sandbox.
    """
    _install(monkeypatch)
    child = _FakeSshForward(_READY_TRANSCRIPT)
    monkeypatch.setattr(dbx.subprocess, "Popen", lambda argv, **kwargs: child)
    with DatabricksSandboxLauncher().forward_local_port("sb-1", 8022):
        pass
    assert child.terminated is True


# ── wheel install ───────────────────────────────────────────


def test_wheel_install_command_breaks_system_packages() -> None:
    """
    The sandbox image's interpreter is PEP 668 externally-managed, so a
    plain `pip install` refuses outright. `--force-reinstall` is equally
    load-bearing: the baked omnigent shares a version with the local
    build, so pip would otherwise report success and change nothing.
    """
    command = DatabricksSandboxLauncher().wheel_install_command("/tmp/oa-wheels.tgz")
    assert "--break-system-packages" in command
    assert "--force-reinstall" in command
    assert "--no-deps" in command
    assert "/tmp/oa-wheels.tgz" in command


def test_wheel_install_command_quotes_the_tarball_path() -> None:
    """The tarball path is interpolated into a shell command; quote it."""
    command = DatabricksSandboxLauncher().wheel_install_command("/tmp/a b.tgz")
    assert "'/tmp/a b.tgz'" in command


# ── job-delegated bootstrap ──────────────────────────────────
#
# A Databricks App container cannot reach the sandbox SSH gateway on port
# 2222 (private-link stub, only 443 routes there), so `JobBootstrapConfig`
# delegates the one-time bootstrap SSH session to a classic-compute job,
# which CAN reach it (verified live: on classic compute the gateway name
# resolves to a PUBLIC regional address and 2222 returns a real SSH
# banner; on both a serverless job and the App container it resolves to a
# private address that answers only on 443). Classic egress is selective,
# not allow-all -- `github.com:22` and the PyPI mirror both time out
# there -- so the notebook must not assume general internet access.
#
# `_run_via_job` imports `databricks.sdk` lazily inside the method it
# backs (an optional extra, per the module docstring), so these fakes are
# built and the SDK itself imported only inside test bodies — never at
# module level — so this file stays collectible on lanes that never
# installed the `databricks` extra. Every test that touches the fake
# `WorkspaceClient` is marked `@pytest.mark.databricks`, matching
# `tests/db/test_utils.py`'s convention; tests that only exercise script
# composition or the unmodified direct-SSH path carry no marker.


@dataclass
class _FakeSecrets:
    """Records every ``secrets.put_secret`` / ``delete_secret`` call."""

    put_calls: list[tuple[str, str, str]] = field(default_factory=list)
    delete_calls: list[tuple[str, str]] = field(default_factory=list)

    def put_secret(self, *, scope: str, key: str, string_value: str) -> None:
        self.put_calls.append((scope, key, string_value))

    def delete_secret(self, *, scope: str, key: str) -> None:
        self.delete_calls.append((scope, key))


@dataclass
class _FakeWorkspace:
    """Records ``workspace.upload`` / ``delete`` (the driver notebook)."""

    uploads: list[tuple[str, bytes]] = field(default_factory=list)
    upload_kwargs: list[dict[str, object]] = field(default_factory=list)
    deletes: list[str] = field(default_factory=list)

    def upload(
        self,
        path: str,
        *,
        content: bytes,
        format: object = None,
        language: object = None,
        overwrite: bool = True,
    ) -> None:
        self.uploads.append((path, content))
        self.upload_kwargs.append({"format": format, "language": language})

    def delete(self, path: str) -> None:
        self.deletes.append(path)


@dataclass
class _FakeWait:
    """Stands in for the ``Wait`` object ``jobs.submit`` returns."""

    run: SimpleNamespace

    def result(self) -> SimpleNamespace:
        return self.run


@dataclass
class _FakeJobs:
    """
    Fakes the two Jobs-API calls the job-bootstrap path makes.

    :param result_state: The canned terminal ``RunResultState`` the fake
        run reports.
    :param notebook_output: The value ``dbutils.notebook.exit()`` would
        have carried back, or ``None`` to simulate a run whose output
        must be recovered from ``logs``/``error`` instead.
    """

    result_state: Any
    notebook_output: str | None = None
    logs: str | None = None
    error: str | None = None
    submit_calls: list[dict[str, Any]] = field(default_factory=list)
    run_output_calls: list[int] = field(default_factory=list)

    def submit(self, **kwargs: Any) -> _FakeWait:
        self.submit_calls.append(kwargs)
        run = SimpleNamespace(
            state=SimpleNamespace(result_state=self.result_state, life_cycle_state="TERMINATED"),
            tasks=[SimpleNamespace(run_id=999)],
            run_id=111,
        )
        return _FakeWait(run=run)

    def get_run_output(self, run_id: int) -> SimpleNamespace:
        self.run_output_calls.append(run_id)
        notebook_output = (
            SimpleNamespace(result=self.notebook_output)
            if self.notebook_output is not None
            else None
        )
        return SimpleNamespace(notebook_output=notebook_output, logs=self.logs, error=self.error)


@dataclass
class _FakeConfig:
    """The one ``WorkspaceClient.config`` attribute the launcher reads."""

    host: str = "https://fake.cloud.databricks.com"


@dataclass
class _FakeApiClient:
    """
    Records ``api_client.do`` — the REST control plane's only exit.

    :param responses: Canned reply per ``(method, path)``. A ``list`` value
        is consumed one entry per call (so a status poll can change), and an
        ``Exception`` value is raised, standing in for an API error.
    :param default: Reply for any ``(method, path)`` with no canned entry.
    """

    responses: dict[tuple[str, str], Any] = field(default_factory=dict)
    default: Any = field(default_factory=dict)
    calls: list[tuple[str, str, Any, Any]] = field(default_factory=list)

    def do(
        self,
        method: str,
        path: str,
        *,
        body: Any = None,
        query: Any = None,
    ) -> Any:
        self.calls.append((method, path, body, query))
        reply = self.responses.get((method, path), self.default)
        if isinstance(reply, list):
            reply = reply.pop(0) if len(reply) > 1 else reply[0]
        if isinstance(reply, Exception):
            raise reply
        return reply


@dataclass
class _FakeWorkspaceClient:
    """Stands in for ``databricks.sdk.WorkspaceClient``."""

    jobs: _FakeJobs
    secrets: _FakeSecrets = field(default_factory=_FakeSecrets)
    workspace: _FakeWorkspace = field(default_factory=_FakeWorkspace)
    config: _FakeConfig = field(default_factory=_FakeConfig)
    api_client: _FakeApiClient = field(default_factory=_FakeApiClient)

    def get_workspace_id(self) -> int:
        """The org id the driver notebook needs to route to this workspace."""
        return 1234567890


def _install_job_bootstrap(
    monkeypatch: pytest.MonkeyPatch, jobs: _FakeJobs | None = None
) -> _FakeWorkspaceClient:
    """
    Swap in a fake ``WorkspaceClient`` for ``_run_via_job``'s lazy import.

    :param jobs: A pre-configured `_FakeJobs`, or `None` for a canned
        successful run.
    :returns: The fake client backing the launcher's SDK calls.
    """
    import databricks.sdk
    from databricks.sdk.service.jobs import RunResultState

    fake = _FakeWorkspaceClient(jobs=jobs or _FakeJobs(result_state=RunResultState.SUCCESS))
    monkeypatch.setattr(databricks.sdk, "WorkspaceClient", lambda **_kwargs: fake)
    return fake


def _job_bootstrap_config(**overrides: object) -> JobBootstrapConfig:
    """A representative `JobBootstrapConfig`, with any field overridden."""
    defaults: dict[str, object] = {
        "ssh_key_secret_scope": "omnigent-sandbox-bootstrap",
        "ssh_key_secret_key": "sandbox-gateway-key",
        "workspace_notebook_path": "/Shared/omnigent/sandbox-job-bootstrap",
    }
    defaults.update(overrides)
    return JobBootstrapConfig(**defaults)  # type: ignore[arg-type]


@pytest.mark.databricks
def test_start_host_via_job_returns_the_tagged_workspace_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    On success, the job's captured output carries the workspace path the
    composed bootstrap script `printf`'d as its last line — `start_host`
    must parse that tag back out rather than making a second trip into
    the sandbox to ask for it.
    """
    from databricks.sdk.service.jobs import RunResultState

    fake = _install_job_bootstrap(
        monkeypatch,
        jobs=_FakeJobs(
            result_state=RunResultState.SUCCESS,
            notebook_output=f"{dbx._WORKSPACE_TAG}/home/sandbox-agent/workspace\n",
        ),
    )
    launcher = DatabricksSandboxLauncher(job_bootstrap=_job_bootstrap_config())

    workspace = launcher.start_host(
        "sb-1",
        token="armed-token",
        host_id="host-1",
        host_name="host-name",
        server_url="https://omnigent.example.com",
    )

    assert workspace == "/home/sandbox-agent/workspace"
    assert fake.jobs.submit_calls  # exactly one job was submitted


@pytest.mark.databricks
def test_start_host_via_job_never_puts_the_token_in_job_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The armed host token must never appear in the `jobs.submit` kwargs —
    those land verbatim in job run JSON visible to anyone with job-read
    access. It travels inside the SSH argv, staged as a transient secret
    instead (asserted separately below).
    """
    from databricks.sdk.service.jobs import RunResultState

    fake = _install_job_bootstrap(
        monkeypatch,
        jobs=_FakeJobs(
            result_state=RunResultState.SUCCESS,
            notebook_output=f"{dbx._WORKSPACE_TAG}/home/sandbox-agent/workspace\n",
        ),
    )
    launcher = DatabricksSandboxLauncher(job_bootstrap=_job_bootstrap_config())

    launcher.start_host(
        "sb-1",
        token="super-secret-armed-token",
        host_id="host-1",
        host_name="host-name",
        server_url="https://omnigent.example.com",
    )

    submit_kwargs = fake.jobs.submit_calls[0]
    assert "super-secret-armed-token" not in json.dumps(submit_kwargs, default=str)


@pytest.mark.databricks
def test_job_bootstrap_notebook_never_shells_out_to_the_databricks_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The driver notebook must reach the control plane over REST, never via
    the `databricks` CLI.

    Not a style preference: the CLI shipped in DBR images is a stub that
    refuses non-interactive use — ``databricks --version`` exits 1 with
    "only supported for interactive use from the web terminal ... we
    recommend using the Databricks Python SDK" — so any argv whose head is
    `databricks` fails 100% of the time on a job cluster. This asserts the
    uploaded notebook source is free of it.
    """
    from databricks.sdk.service.jobs import RunResultState

    fake = _install_job_bootstrap(
        monkeypatch,
        jobs=_FakeJobs(
            result_state=RunResultState.SUCCESS,
            notebook_output=f"{dbx._WORKSPACE_TAG}/ws\n",
        ),
    )
    launcher = DatabricksSandboxLauncher(job_bootstrap=_job_bootstrap_config())

    launcher.start_host(
        "sb-1",
        token="armed-token",
        host_id="host-1",
        host_name="host-name",
        server_url="https://omnigent.example.com",
    )

    _path, content = fake.workspace.uploads[0]
    source = content.decode("utf-8")
    assert "databricks sandbox" not in source
    assert '"databricks"' not in source
    # It must instead name the REST surface and invoke ssh itself.
    assert dbx._LAKEBOX_API_ROOT in source
    assert '"ssh",' in source


@pytest.mark.databricks
def test_job_bootstrap_notebook_carries_the_workspace_routing_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The notebook is rendered with the workspace host AND org id.

    On classic compute the notebook context's ``apiUrl()`` returns the
    regional control-plane host rather than the workspace-specific one, so
    the gateway rejects the credential unless the workspace is named
    explicitly via ``X-Databricks-Org-Id``. Both values are only knowable
    launcher-side, so a rendering that drops either is a live-only failure.
    """
    from databricks.sdk.service.jobs import RunResultState

    fake = _install_job_bootstrap(
        monkeypatch,
        jobs=_FakeJobs(
            result_state=RunResultState.SUCCESS,
            notebook_output=f"{dbx._WORKSPACE_TAG}/ws\n",
        ),
    )
    launcher = DatabricksSandboxLauncher(job_bootstrap=_job_bootstrap_config())

    launcher.start_host(
        "sb-1",
        token="armed-token",
        host_id="host-1",
        host_name="host-name",
        server_url="https://omnigent.example.com",
    )

    source = fake.workspace.uploads[0][1].decode("utf-8")
    assert fake.config.host in source
    assert str(fake.get_workspace_id()) in source
    assert "X-Databricks-Org-Id" in source


@pytest.mark.databricks
def test_start_host_via_job_stages_the_sandbox_id_with_the_script(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The transient secret carries the sandbox id alongside the script.

    The notebook resolves the gateway host itself (the launcher's process
    cannot), so it needs to be told WHICH sandbox to resolve — and the
    sandbox id travels in the secret rather than as a job parameter purely
    so there is one payload to stage and tear down, not two channels.
    """
    from databricks.sdk.service.jobs import RunResultState

    fake = _install_job_bootstrap(
        monkeypatch,
        jobs=_FakeJobs(
            result_state=RunResultState.SUCCESS,
            notebook_output=f"{dbx._WORKSPACE_TAG}/ws\n",
        ),
    )
    launcher = DatabricksSandboxLauncher(job_bootstrap=_job_bootstrap_config())

    launcher.start_host(
        "sb-42",
        token="armed-token",
        host_id="host-1",
        host_name="host-name",
        server_url="https://omnigent.example.com",
    )

    payload = json.loads(fake.secrets.put_calls[0][2])
    assert payload["sandbox_id"] == "sb-42"
    assert "armed-token" in payload["remote_command"]


@pytest.mark.databricks
def test_job_bootstrap_notebook_uploads_as_python_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The driver notebook must be imported as PYTHON `SOURCE` explicitly.

    Both arguments are load-bearing, measured against a live workspace:
    omit `format` and the import API treats the body as a DBC archive and
    rejects the upload with "The zip archive contains no items" (the SDK
    docstring claims SOURCE is the default; the API disagrees), and the SDK
    only infers `language` from a path suffix, which the per-run notebook
    path deliberately does not have. Getting either wrong fails every
    managed launch before the job is even submitted.
    """
    from databricks.sdk.service.jobs import RunResultState
    from databricks.sdk.service.workspace import ImportFormat, Language

    fake = _install_job_bootstrap(
        monkeypatch,
        jobs=_FakeJobs(
            result_state=RunResultState.SUCCESS,
            notebook_output=f"{dbx._WORKSPACE_TAG}/ws\n",
        ),
    )
    launcher = DatabricksSandboxLauncher(job_bootstrap=_job_bootstrap_config())

    launcher.start_host(
        "sb-1",
        token="armed-token",
        host_id="host-1",
        host_name="host-name",
        server_url="https://omnigent.example.com",
    )

    assert fake.workspace.upload_kwargs == [
        {"format": ImportFormat.SOURCE, "language": Language.PYTHON}
    ]


@pytest.mark.databricks
def test_start_host_via_job_stages_the_argv_as_a_transient_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    With no `payload_secret_scope` configured, the token-bearing SSH argv
    is written to Secrets under the same scope the operator registered the
    sandbox-gateway SSH key in (no second scope to provision), and deleted
    again once the run reaches a terminal state — bounding the token's
    exposure window to roughly the job's own runtime rather than leaving it
    stranded indefinitely.
    """
    from databricks.sdk.service.jobs import RunResultState

    fake = _install_job_bootstrap(
        monkeypatch,
        jobs=_FakeJobs(
            result_state=RunResultState.SUCCESS,
            notebook_output=f"{dbx._WORKSPACE_TAG}/ws\n",
        ),
    )
    config = _job_bootstrap_config()
    launcher = DatabricksSandboxLauncher(job_bootstrap=config)

    launcher.start_host(
        "sb-1",
        token="armed-token",
        host_id="host-1",
        host_name="host-name",
        server_url="https://omnigent.example.com",
    )

    assert len(fake.secrets.put_calls) == 1
    put_scope, put_key, put_value = fake.secrets.put_calls[0]
    assert put_scope == config.ssh_key_secret_scope
    assert "armed-token" in put_value  # the argv (bootstrap script) carries it
    assert fake.secrets.delete_calls == [(put_scope, put_key)]


@pytest.mark.databricks
def test_start_host_via_job_deletes_the_secret_even_on_job_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A failed run must not strand the transient argv secret — cleanup
    runs in a `finally`, independent of the run's result.
    """
    from databricks.sdk.service.jobs import RunResultState

    fake = _install_job_bootstrap(
        monkeypatch,
        jobs=_FakeJobs(
            result_state=RunResultState.FAILED,
            error="databricks sandbox ssh exited 255",
        ),
    )
    launcher = DatabricksSandboxLauncher(job_bootstrap=_job_bootstrap_config())

    with pytest.raises(click.ClickException) as exc:
        launcher.start_host(
            "sb-1",
            token="armed-token",
            host_id="host-1",
            host_name="host-name",
            server_url="https://omnigent.example.com",
        )

    assert "exited 255" in str(exc.value)
    assert len(fake.secrets.delete_calls) == 1


@pytest.mark.databricks
def test_start_host_via_job_stages_the_argv_in_the_configured_payload_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A configured `payload_secret_scope` takes the transient argv secret —
    both the write and the cleanup delete — while the key scope keeps
    holding only the long-lived gateway key.

    This is the least-privilege deployment shape: writing the payload needs
    ``WRITE`` on whatever scope holds it, so pointing it at the key scope
    means whoever launches can also overwrite the private key. Splitting
    them keeps that grant down to ``READ`` on the key scope. The notebook
    must read the payload from the same scope it was written to, or the job
    fails looking for a secret that is not there.
    """
    from databricks.sdk.service.jobs import RunResultState

    fake = _install_job_bootstrap(
        monkeypatch,
        jobs=_FakeJobs(
            result_state=RunResultState.SUCCESS,
            notebook_output=f"{dbx._WORKSPACE_TAG}/ws\n",
        ),
    )
    config = _job_bootstrap_config(payload_secret_scope="omnigent-sandbox-payload")
    launcher = DatabricksSandboxLauncher(job_bootstrap=config)

    launcher.start_host(
        "sb-1",
        token="armed-token",
        host_id="host-1",
        host_name="host-name",
        server_url="https://omnigent.example.com",
    )

    put_scope, put_key, _value = fake.secrets.put_calls[0]
    assert put_scope == "omnigent-sandbox-payload"
    assert fake.secrets.delete_calls == [("omnigent-sandbox-payload", put_key)]
    # The notebook resolves both secrets by scope name, so the payload read
    # must follow the payload scope while the key read stays put.
    notebook = fake.workspace.uploads[0][1].decode()
    assert f'dbutils.secrets.get(scope="omnigent-sandbox-payload", key="{put_key}")' in notebook
    assert (
        f'dbutils.secrets.get(scope="{config.ssh_key_secret_scope}", '
        f'key="{config.ssh_key_secret_key}")'
    ) in notebook


def _launch(launcher: DatabricksSandboxLauncher, sandbox_id: str, token: str) -> None:
    """Drive one job-delegated `start_host`, ignoring the returned path."""
    launcher.start_host(
        sandbox_id,
        token=token,
        host_id="host-1",
        host_name="host-name",
        server_url="https://omnigent.example.com",
    )


@pytest.mark.databricks
def test_start_host_via_job_uses_a_per_run_notebook_path_and_removes_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Each run uploads its driver notebook to a path derived from the
    configured base plus a fresh suffix, and deletes it afterwards.

    A single fixed path would be a correctness bug, not untidiness: two
    concurrent launches would race between upload and run, and the loser
    would execute the winner's notebook — pointed at the winner's payload
    secret, and therefore at the wrong sandbox.
    """
    from databricks.sdk.service.jobs import RunResultState

    fake = _install_job_bootstrap(
        monkeypatch,
        jobs=_FakeJobs(
            result_state=RunResultState.SUCCESS,
            notebook_output=f"{dbx._WORKSPACE_TAG}/ws\n",
        ),
    )
    config = _job_bootstrap_config()
    launcher = DatabricksSandboxLauncher(job_bootstrap=config)

    _launch(launcher, "sb-1", "armed-token")
    _launch(launcher, "sb-2", "armed-token")

    paths = [path for path, _content in fake.workspace.uploads]
    assert len(paths) == 2
    assert len(set(paths)) == 2, "two launches shared one notebook path"
    for path in paths:
        assert path.startswith(f"{config.workspace_notebook_path}-")
    # Nothing token-bearing may outlive the run: both the notebook and the
    # payload secret are removed, and the job ran the path just uploaded.
    assert fake.workspace.deletes == paths
    assert [key for _scope, key in fake.secrets.delete_calls] == [
        key for _scope, key, _value in fake.secrets.put_calls
    ]
    submitted = [
        task.notebook_task.notebook_path
        for kwargs in fake.jobs.submit_calls
        for task in kwargs["tasks"]
    ]
    assert submitted == paths


@pytest.mark.databricks
def test_job_bootstrap_notebook_scrubs_the_armed_token_from_its_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The armed host token must not survive into the notebook's output.

    `dbutils.secrets.get` is redacted by Databricks automatically, but the
    remote command's OWN stdout/stderr is not, and both
    `dbutils.notebook.exit` and an uncaught exception message land in
    durable job-run JSON readable by anyone with job-read access. The
    caller therefore names the token in the payload's `redact` list, and
    the notebook scrubs every output path with it.
    """
    from databricks.sdk.service.jobs import RunResultState

    fake = _install_job_bootstrap(
        monkeypatch,
        jobs=_FakeJobs(
            result_state=RunResultState.SUCCESS,
            notebook_output=f"{dbx._WORKSPACE_TAG}/ws\n",
        ),
    )
    launcher = DatabricksSandboxLauncher(job_bootstrap=_job_bootstrap_config())

    _launch(launcher, "sb-1", "super-secret-armed-token")

    _scope, _key, put_value = fake.secrets.put_calls[0]
    assert json.loads(put_value)["redact"] == ["super-secret-armed-token"]
    notebook = fake.workspace.uploads[0][1].decode()
    # Both exits — the failure path and the success path — go through the
    # scrubber, so neither can carry the token out of the run.
    assert "raise RuntimeError(" in notebook
    assert "scrub(completed.stderr.strip())" in notebook
    assert "dbutils.notebook.exit(scrub(completed.stdout))" in notebook


@pytest.mark.databricks
def test_start_host_via_job_raises_when_output_has_no_workspace_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A run that reports success but whose output never carries the
    workspace tag (e.g. the notebook template drifted from the script)
    must fail loud rather than return `None`/garbage as the workspace.
    """
    from databricks.sdk.service.jobs import RunResultState

    _install_job_bootstrap(
        monkeypatch,
        jobs=_FakeJobs(result_state=RunResultState.SUCCESS, notebook_output="launched\n"),
    )
    launcher = DatabricksSandboxLauncher(job_bootstrap=_job_bootstrap_config())

    with pytest.raises(click.ClickException, match="workspace"):
        launcher.start_host(
            "sb-1",
            token="armed-token",
            host_id="host-1",
            host_name="host-name",
            server_url="https://omnigent.example.com",
        )


def test_start_host_without_job_bootstrap_ssh_directly_as_before(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    `job_bootstrap=None` (the default) must NOT touch the Databricks SDK
    at all — the direct-SSH path this launcher has always used keeps
    working unmodified from a laptop that CAN reach the sandbox gateway.
    """
    cli = _install(monkeypatch)
    cli.reply_with("ssh", _Reply(stdout="/home/sandbox-agent"))
    launcher = DatabricksSandboxLauncher()

    launcher.start_host(
        "sb-1",
        token="armed-token",
        host_id="host-1",
        host_name="host-name",
        server_url="https://omnigent.example.com",
    )

    assert cli.argvs("ssh")  # base start_host's own probe/launch calls ran


def test_compose_bootstrap_script_runs_the_dev_bootstrap_command_first() -> None:
    """
    `bootstrap_command` (the dev-only self-update hook) must still run on
    the job-delegated path, in the same position it runs on direct SSH —
    before workspace setup, on every host start.
    """
    launcher = DatabricksSandboxLauncher(
        bootstrap_command="git -C ~/omnigent pull",
        job_bootstrap=_job_bootstrap_config(),
    )
    script = launcher._compose_bootstrap_script(
        token="tok",
        host_id="hid",
        host_name="hname",
        server_url="https://omnigent.example.com",
        repo_url=None,
        repo_branch=None,
        repo_name=None,
        host_config=None,
    )
    lines = script.splitlines()
    assert lines.index("git -C ~/omnigent pull") < lines.index('home="$(printf %s "$HOME")"')


def test_compose_bootstrap_script_clones_only_when_repo_url_given() -> None:
    """
    No `repo_url` means no clone step — the sandbox image's own checkout
    (or a resumed sandbox's existing one) is used as-is.
    """
    launcher = DatabricksSandboxLauncher(job_bootstrap=_job_bootstrap_config())
    script = launcher._compose_bootstrap_script(
        token="tok",
        host_id="hid",
        host_name="hname",
        server_url="https://omnigent.example.com",
        repo_url=None,
        repo_branch=None,
        repo_name=None,
        host_config=None,
    )
    assert "git clone" not in script

    cloning_script = launcher._compose_bootstrap_script(
        token="tok",
        host_id="hid",
        host_name="hname",
        server_url="https://omnigent.example.com",
        repo_url="https://github.com/example/repo.git",
        repo_branch="main",
        repo_name="repo",
        host_config=None,
    )
    assert "git clone --branch main --single-branch -- " in cloning_script
    assert "https://github.com/example/repo.git" in cloning_script


def test_compose_bootstrap_script_ends_with_the_workspace_tag() -> None:
    """
    The last line must `printf` the workspace tag — `_run_via_job`'s
    caller parses exactly this to recover `start_host`'s return value.
    """
    launcher = DatabricksSandboxLauncher(job_bootstrap=_job_bootstrap_config())
    script = launcher._compose_bootstrap_script(
        token="tok",
        host_id="hid",
        host_name="hname",
        server_url="https://omnigent.example.com",
        repo_url=None,
        repo_branch=None,
        repo_name=None,
        host_config=None,
    )
    assert script.splitlines()[-1] == f'printf "{dbx._WORKSPACE_TAG}%s\\n" "$workspace"'


# ── REST control plane ──────────────────────────────────────
#
# Configuring `job_bootstrap` also moves create/start/status/config/delete
# off the `databricks` CLI and onto the workspace REST API, because that
# mode exists for a Databricks Apps container — which can neither reach
# the sandbox gateway on 2222 nor run a Go CLI binary it does not have.
# Every test here proves the CLI is untouched by making both `shutil.which`
# and `subprocess.run` hostile.


def _install_rest(
    monkeypatch: pytest.MonkeyPatch,
    *,
    responses: dict[tuple[str, str], Any] | None = None,
    default: Any = None,
) -> _FakeWorkspaceClient:
    """
    Install a fake SDK client and make any CLI use an immediate failure.

    :param responses: Canned `api_client.do` replies keyed by `(method, path)`.
    :param default: Reply for un-keyed calls; defaults to a Running sandbox.
    :returns: The fake client backing the launcher.
    """
    import databricks.sdk
    from databricks.sdk.service.jobs import RunResultState

    fake = _FakeWorkspaceClient(jobs=_FakeJobs(result_state=RunResultState.SUCCESS))
    fake.api_client = _FakeApiClient(
        responses=responses or {},
        default=default if default is not None else _sandbox_record("Running"),
    )
    monkeypatch.setattr(databricks.sdk, "WorkspaceClient", lambda **_kwargs: fake)

    def _no_cli(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("the REST control plane must not shell out to the CLI")

    monkeypatch.setattr(dbx.shutil, "which", lambda _name: None)
    monkeypatch.setattr(dbx.subprocess, "run", _no_cli)

    # A virtual clock, not a no-op sleep: the wait loops bound themselves on
    # `monotonic`, so zeroing only `sleep` turns a timeout test into a tight
    # 600-second spin that polls until the process runs out of memory.
    clock = {"now": 0.0}

    def _sleep(seconds: float) -> None:
        clock["now"] += seconds

    monkeypatch.setattr(dbx.time, "sleep", _sleep)
    monkeypatch.setattr(dbx.time, "monotonic", lambda: clock["now"])
    return fake


def _sandbox_record(status: str, sandbox_id: str = "sb-1") -> dict[str, Any]:
    """One control-plane sandbox record, in the API's camelCase spelling."""
    return {
        "sandboxId": sandbox_id,
        "name": "managed-a1b2c3d4",
        "status": status,
        "gatewayHost": "us-east-1.service-direct.cloud.databricks.com",
        "idleTimeout": "0s",
        "noAutostop": True,
    }


def _rest_launcher(**overrides: Any) -> DatabricksSandboxLauncher:
    """A launcher whose control plane is REST (i.e. `job_bootstrap` set)."""
    return DatabricksSandboxLauncher(job_bootstrap=_job_bootstrap_config(), **overrides)


@pytest.mark.databricks
def test_rest_prepare_does_not_require_the_databricks_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Regression guard for the blocker this path exists to fix: `prepare`
    used to demand the CLI on `PATH`, which no Databricks Apps container
    has, so every server-initiated launch failed with `INSTALL_HINT`
    before the job bootstrap could run at all.
    """
    fake = _install_rest(monkeypatch, default={"sandboxes": []})

    _rest_launcher().prepare()

    assert fake.api_client.calls == [
        ("GET", "/api/2.0/lakebox/sandboxes", None, {"page_size": 100})
    ]


@pytest.mark.databricks
def test_rest_provision_creates_waits_for_running_then_applies_autostop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    `provision` must return an id that is immediately usable, which the
    CLI gave for free by blocking — over REST the launcher has to poll.
    Request bodies are snake_case even though responses are camelCase.
    """
    fake = _install_rest(
        monkeypatch,
        responses={
            ("POST", "/api/2.0/lakebox/sandboxes"): {"sandboxId": "sb-1"},
            ("GET", "/api/2.0/lakebox/sandboxes/sb-1"): [
                _sandbox_record("Starting"),
                _sandbox_record("Running"),
            ],
        },
    )

    assert _rest_launcher().provision("managed-a1b2c3d4") == "sb-1"

    methods_and_paths = [(method, path) for method, path, _body, _query in fake.api_client.calls]
    assert methods_and_paths == [
        ("POST", "/api/2.0/lakebox/sandboxes"),
        ("GET", "/api/2.0/lakebox/sandboxes/sb-1"),
        ("GET", "/api/2.0/lakebox/sandboxes/sb-1"),
        ("PATCH", "/api/2.0/lakebox/sandboxes/sb-1"),
    ]
    assert fake.api_client.calls[0][2] == {"sandbox": {"name": "managed-a1b2c3d4"}}
    assert fake.api_client.calls[-1][2] == {"sandbox_id": "sb-1", "no_autostop": True}


@pytest.mark.databricks
def test_rest_provision_fails_when_the_sandbox_never_reaches_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_rest(
        monkeypatch,
        responses={("POST", "/api/2.0/lakebox/sandboxes"): {"sandboxId": "sb-1"}},
        default=_sandbox_record("Starting"),
    )

    with pytest.raises(click.ClickException, match="did not reach running"):
        _rest_launcher().provision("managed-a1b2c3d4")


@pytest.mark.databricks
def test_rest_resume_starts_the_sandbox_and_waits_for_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    `POST .../start` returns as soon as the request is accepted, where the
    CLI's `start` blocks — so this path owns the wait.
    """
    fake = _install_rest(monkeypatch)

    _rest_launcher().resume("sb-1")

    assert [(method, path) for method, path, _body, _query in fake.api_client.calls] == [
        ("POST", "/api/2.0/lakebox/sandboxes/sb-1/start"),
        ("GET", "/api/2.0/lakebox/sandboxes/sb-1"),
    ]


@pytest.mark.databricks
@pytest.mark.parametrize(
    ("status", "expected"),
    [("Running", True), ("Stopped", False), ("Wedged", None)],
)
def test_rest_is_running_maps_the_control_plane_status(
    monkeypatch: pytest.MonkeyPatch, status: str, expected: bool | None
) -> None:
    _install_rest(monkeypatch, default=_sandbox_record(status))

    assert _rest_launcher().is_running("sb-1") is expected


@pytest.mark.databricks
def test_rest_attach_starts_a_stopped_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install_rest(
        monkeypatch,
        responses={
            ("GET", "/api/2.0/lakebox/sandboxes/sb-1"): [
                _sandbox_record("Stopped"),
                _sandbox_record("Running"),
            ]
        },
    )

    _rest_launcher().attach("sb-1")

    assert ("POST", "/api/2.0/lakebox/sandboxes/sb-1/start", None, None) in fake.api_client.calls


@pytest.mark.databricks
def test_rest_keep_alive_sends_the_idle_timeout_in_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The API takes seconds (`"14400s"`) where config carries the human
    spelling (`"4h"`) — the CLI did that conversion, so this path must.
    """
    fake = _install_rest(monkeypatch)

    _rest_launcher(no_autostop=False, idle_timeout="4h").keep_alive("sb-1")

    assert fake.api_client.calls == [
        (
            "PATCH",
            "/api/2.0/lakebox/sandboxes/sb-1",
            {"sandbox_id": "sb-1", "idle_timeout": "14400s"},
            None,
        )
    ]


@pytest.mark.databricks
def test_rest_keep_alive_soft_fails_on_a_rejected_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Launcher contract: a rejected auto-stop setting warns, it does not
    abort the launch — the sandbox is usable, it may just reap on idle.
    """
    _install_rest(
        monkeypatch,
        responses={("PATCH", "/api/2.0/lakebox/sandboxes/sb-1"): RuntimeError("nope")},
    )

    _rest_launcher().keep_alive("sb-1")  # must not raise


@pytest.mark.databricks
def test_rest_terminate_deletes_the_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install_rest(monkeypatch)

    _rest_launcher().terminate("sb-1")

    assert fake.api_client.calls == [("DELETE", "/api/2.0/lakebox/sandboxes/sb-1", None, None)]


@pytest.mark.databricks
def test_rest_terminate_is_idempotent_for_a_missing_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    An already-gone sandbox is the desired end state. The API says so with
    `{"error_code":"NOT_FOUND","message":"sandbox … not found"}`.
    """
    _install_rest(
        monkeypatch,
        responses={
            ("DELETE", "/api/2.0/lakebox/sandboxes/sb-1"): RuntimeError("sandbox sb-1 not found")
        },
    )

    _rest_launcher().terminate("sb-1")  # must not raise


@pytest.mark.databricks
def test_rest_terminate_still_raises_on_a_real_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_rest(
        monkeypatch,
        responses={
            ("DELETE", "/api/2.0/lakebox/sandboxes/sb-1"): RuntimeError("permission denied")
        },
    )

    with pytest.raises(click.ClickException, match="delete Databricks Sandbox"):
        _rest_launcher().terminate("sb-1")


@pytest.mark.databricks
def test_the_control_plane_stays_on_the_cli_without_job_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The laptop path is unchanged: no `job_bootstrap` means no SDK import
    and no REST call, so this launcher keeps working from a machine that
    CAN reach the sandbox gateway.
    """
    cli = _install(monkeypatch)
    cli.reply_with("create", _Reply(stdout=json.dumps({"sandboxId": "sb-1"})))

    launcher = DatabricksSandboxLauncher()
    assert launcher.provision("managed-a1b2c3d4") == "sb-1"
    assert launcher._client is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [("4h", "14400s"), ("90m", "5400s"), ("30s", "30s"), ("1h30m", "5400s")],
)
def test_as_api_duration_normalizes_go_durations(value: str, expected: str) -> None:
    assert dbx._as_api_duration(value) == expected


@pytest.mark.parametrize("value", ["", "   ", "4 hours", "forever", "4d"])
def test_as_api_duration_rejects_what_it_cannot_read(value: str) -> None:
    """
    Fail loudly rather than send an unreadable spelling through: silently
    defaulting would set an idle timeout the operator did not ask for.
    """
    with pytest.raises(click.ClickException, match="idle timeout"):
        dbx._as_api_duration(value)
