"""Unified CLI: ``python -m agent_ai.cli <command> [args...]``."""
from __future__ import annotations

import sys

from agent_ai.cli.bootstrap import ensure_repo_on_path
from agent_ai.cli.dispatch import list_commands, run_command


def main(argv: list[str] | None = None) -> int:
    ensure_repo_on_path()
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        commands = list_commands()
        print("Agent-AI CLI")
        print()
        print("Usage: python -m agent_ai.cli <command> [args...]")
        print("       python -m agent_ai.cli list")
        print()
        print("Commands:")
        for name in commands:
            print(f"  {name}")
        return 0
    if args[0] == "list":
        for name in list_commands():
            print(name)
        return 0
    command, *rest = args
    return run_command(command, argv=rest)


if __name__ == "__main__":
    raise SystemExit(main())
