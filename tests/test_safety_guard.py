"""Tests for the VibeFlow Safety Guard hook."""

from __future__ import annotations

import importlib.util
import json
import sys
from io import StringIO
from pathlib import Path

import pytest

# Load safety_guard.py from .claude/hooks/ (not a regular Python package)
_HOOK_PATH = (
    Path(__file__).resolve().parent.parent / ".claude" / "hooks" / "safety_guard.py"
)
_spec = importlib.util.spec_from_file_location("safety_guard", str(_HOOK_PATH))
assert _spec is not None, f"Could not find hook at {_HOOK_PATH}"
_safety_guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_safety_guard)  # type: ignore[arg-type]

is_dangerous = _safety_guard.is_dangerous
main = _safety_guard.main


class TestIsDangerous:
    """Unit tests for the pure is_dangerous() function."""

    # ── dangerous commands — should be blocked ──────────────────────

    @pytest.mark.parametrize(
        "command,expected_reason_keyword",
        [
            ("rm -rf vibeflow", "rm -rf"),
            ("rm  -rf  vibeflow", "rm -rf"),          # extra spaces
            ("rm -fr knowledge", "rm -fr"),
            ("rm -fr /tmp/cache", "rm -fr"),
            ("git reset --hard HEAD~1", "git reset --hard"),
            ("git  reset  --hard  HEAD~1", "git reset --hard"),  # extra spaces
            ("git clean -fd", "git clean -fd"),
            ("git clean -df", "git clean -df"),
            ("git clean -fdx", "git clean -fd"),      # -fdx caught by -fd[x]?
            ("git push --force origin main", "git push --force"),
            ("git push origin --force", "git push --force"),
            ("git push -f origin main", "git push -f"),
            ("git push origin -f", "git push -f"),
            ("git checkout -- .", "git checkout -- ."),
            ("git  checkout  --  .", "git checkout -- ."),  # extra spaces
            ("git restore .", "git restore ."),
        ],
    )
    def test_blocks_dangerous_command(self, command: str, expected_reason_keyword: str) -> None:
        dangerous, reason = is_dangerous(command)
        assert dangerous is True, f"Expected '{command}' to be blocked"
        assert reason is not None
        assert expected_reason_keyword in reason

    # ── safe commands — should be allowed ──────────────────────────

    @pytest.mark.parametrize(
        "command",
        [
            "git status",
            "git diff",
            "git log --oneline",
            "pytest -q",
            "python -m vibeflow.main",
            "python -m vibeflow.search_cli",
            "ls",
            "pwd",
        ],
    )
    def test_allows_safe_command(self, command: str) -> None:
        dangerous, reason = is_dangerous(command)
        assert dangerous is False, f"Expected '{command}' to be allowed, got: {reason}"

    # ── edge cases ─────────────────────────────────────────────────

    def test_empty_string(self) -> None:
        dangerous, reason = is_dangerous("")
        assert dangerous is False
        assert reason is None

    def test_none_input(self) -> None:
        dangerous, reason = is_dangerous(None)
        assert dangerous is False
        assert reason is None

    def test_whitespace_only(self) -> None:
        dangerous, reason = is_dangerous("   ")
        assert dangerous is False
        assert reason is None

    def test_non_string_int(self) -> None:
        dangerous, reason = is_dangerous(42)  # type: ignore[arg-type]
        assert dangerous is False
        assert reason is None


class TestMainProtocol:
    """Tests for the main() stdin/stdout protocol."""

    def _run_main(self, input_json: str) -> tuple[str, int]:
        """Run main() with the given stdin, return (stdout, exit_code)."""
        original_stdin = sys.stdin
        original_stdout = sys.stdout
        try:
            sys.stdin = StringIO(input_json)
            sys.stdout = StringIO()
            main()
            output = sys.stdout.getvalue()
        finally:
            sys.stdin = original_stdin
            sys.stdout = original_stdout
        return output

    def test_deny_output_format(self) -> None:
        """Denied command must output valid JSON with all required fields."""
        input_data = json.dumps({
            "tool_name": "Bash",
            "tool_input": {"command": "rm -rf vibeflow"},
        })
        output = self._run_main(input_data)

        parsed = json.loads(output)
        hook_output = parsed["hookSpecificOutput"]
        assert hook_output["hookEventName"] == "PreToolUse"
        assert hook_output["permissionDecision"] == "deny"
        assert "rm -rf" in hook_output["permissionDecisionReason"]

    def test_safe_command_produces_no_output(self) -> None:
        """Safe commands should exit cleanly with no stdout."""
        input_data = json.dumps({
            "tool_name": "Bash",
            "tool_input": {"command": "git status"},
        })
        output = self._run_main(input_data)
        assert output.strip() == ""

    def test_non_bash_tool_ignored(self) -> None:
        """Non-Bash tools should produce no output."""
        input_data = json.dumps({
            "tool_name": "Read",
            "tool_input": {"file_path": "/tmp/test.txt"},
        })
        output = self._run_main(input_data)
        assert output.strip() == ""

    def test_invalid_json_no_exception(self) -> None:
        """Invalid JSON must not cause an unhandled exception."""
        output = self._run_main("not valid json {{{")
        assert output.strip() == ""

    def test_missing_command_no_exception(self) -> None:
        """Missing 'command' field must not cause an unhandled exception."""
        input_data = json.dumps({
            "tool_name": "Bash",
            "tool_input": {},
        })
        output = self._run_main(input_data)
        assert output.strip() == ""

    def test_missing_tool_input_no_exception(self) -> None:
        """Missing 'tool_input' field must not cause an unhandled exception."""
        input_data = json.dumps({"tool_name": "Bash"})
        output = self._run_main(input_data)
        assert output.strip() == ""
