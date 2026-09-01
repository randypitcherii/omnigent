"""Tests for ``_agent_clone_root_name``.

Clone rows are named ``"<name> (fork <id>)"`` / ``"<name> (switch <id>)"``
and stack when a clone is itself cloned. These cases pin the peeling
behaviour against the web client's ``agentRootName`` (``forkHarness.ts``),
whose regex this helper mirrors.
"""

from __future__ import annotations

import pytest

from omnigent.server.routes.sessions import _agent_clone_root_name


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        # No suffix — returned verbatim.
        ("claude-native-ui", "claude-native-ui"),
        # A single fork or switch layer is peeled.
        ("claude-native-ui (fork ag_a1b2)", "claude-native-ui"),
        ("claude-native-ui (switch ag_a1b2)", "claude-native-ui"),
        # Stacked layers peel all the way to the root.
        ("claude-native-ui (fork ag_a) (fork ag_b)", "claude-native-ui"),
        ("claude-native-ui (fork ag_a) (switch ag_b)", "claude-native-ui"),
        (
            "claude-native-ui (switch ag_a) (fork ag_b) (switch ag_c)",
            "claude-native-ui",
        ),
        # Parenthesised names that are not clone suffixes are preserved.
        ("my agent (v2)", "my agent (v2)"),
        ("my agent (fork)", "my agent (fork)"),
        ("my agent (forked ag_a)", "my agent (forked ag_a)"),
        # A suffix only counts at the end of the name.
        ("agent (fork ag_a) trailing", "agent (fork ag_a) trailing"),
        # A root name that itself ends in parens keeps them once the
        # clone suffix is peeled.
        ("my agent (v2) (fork ag_a)", "my agent (v2)"),
        ("", ""),
    ],
)
def test_agent_clone_root_name(name: str, expected: str) -> None:
    """Peeling every clone suffix yields the root agent name."""
    assert _agent_clone_root_name(name) == expected
