"""VibeFlow Safety Guard — PreToolUse Hook.

Reads tool invocation JSON from stdin and blocks known-dangerous Bash commands.
"""

from __future__ import annotations

import json
import re
import sys

# (regex_pattern, chinese_description) pairs
DANGEROUS_PATTERNS: list[tuple[str, str]] = [
    (r"\brm\s+-rf\b", "rm -rf（强制递归删除）"),
    (r"\brm\s+-fr\b", "rm -fr（强制递归删除）"),
    (r"\bgit\s+reset\s+--hard\b", "git reset --hard（硬重置，不可恢复）"),
    (r"\bgit\s+clean\s+-fd[x]?\b", "git clean -fd/-fdx（强制清理未追踪文件）"),
    (r"\bgit\s+clean\s+-df\b", "git clean -df（强制清理未追踪文件）"),
    (r"\bgit\s+push\b.*\s--force\b", "git push --force（强制推送，覆盖远程历史）"),
    (r"\bgit\s+push\b.*\s-f\b", "git push -f（强制推送，覆盖远程历史）"),
    (r"\bgit\s+checkout\s+--\s+\.", "git checkout -- .（丢弃所有本地修改）"),
    (r"\bgit\s+restore\s+\.", "git restore .（恢复所有文件，丢弃修改）"),
]


def is_dangerous(command: str | None) -> tuple[bool, str | None]:
    """Check whether *command* matches a known-dangerous pattern.

    Returns ``(True, reason)`` when a dangerous pattern is found, or
    ``(False, None)`` when the command is safe (or empty / missing).
    """
    if not isinstance(command, str) or not command:
        return False, None

    stripped = command.strip()
    if not stripped:
        return False, None

    for pattern, description in DANGEROUS_PATTERNS:
        if re.search(pattern, stripped):
            return True, f"拦截危险命令：{description}"

    return False, None


def main() -> None:
    """Read PreToolUse JSON from stdin and deny dangerous Bash commands."""
    try:
        raw = sys.stdin.read()
        input_data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        # Invalid JSON — don't block, let normal flow decide
        return

    if not isinstance(input_data, dict):
        return

    tool_name = input_data.get("tool_name", "")
    if tool_name != "Bash":
        return

    tool_input = input_data.get("tool_input", {})
    if not isinstance(tool_input, dict):
        return

    command = tool_input.get("command", "")
    dangerous, reason = is_dangerous(command)

    if dangerous:
        output = {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        }
        print(json.dumps(output, ensure_ascii=False))
        return

    # Safe command — exit 0, no output; normal permission flow continues.
    return


if __name__ == "__main__":
    main()
