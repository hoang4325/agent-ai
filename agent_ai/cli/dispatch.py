"""Dispatch helpers for CLI entrypoints and ``python -m agent_ai.cli``."""
from __future__ import annotations

import importlib
import runpy
import sys
from types import ModuleType
from typing import Any, Callable

from agent_ai.cli.bootstrap import ensure_repo_on_path
from agent_ai.cli.registry_data import COMMANDS


def _resolve_main(module: ModuleType) -> Callable[..., Any] | None:
    main = getattr(module, "main", None)
    if callable(main):
        return main
    # Some modules expose only parse_args + side-effect main blocks.
    return None


def run_module_main(module_path: str, argv: list[str] | None = None) -> int:
    """
    Import ``module_path`` and execute its ``main()`` if present.

    Falls back to ``runpy.run_module`` for scripts that only use
    ``if __name__ == "__main__"`` blocks.
    """
    ensure_repo_on_path()
    if argv is not None:
        sys.argv = [sys.argv[0], *argv]

    module = importlib.import_module(module_path)
    main = _resolve_main(module)
    if main is not None:
        try:
            result = main()
        except SystemExit as exc:
            code = exc.code
            if code is None:
                return 0
            if isinstance(code, int):
                return code
            return 1
        if result is None:
            return 0
        if isinstance(result, int):
            return result
        return 0

    # No main() — execute module as __main__
    try:
        runpy.run_module(module_path, run_name="__main__")
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        return 1
    return 0


def run_command(command: str, argv: list[str] | None = None) -> int:
    """Run a registered CLI command name."""
    ensure_repo_on_path()
    key = command.strip()
    if key not in COMMANDS:
        known = ", ".join(sorted(COMMANDS))
        raise SystemExit(f"Unknown command {command!r}. Known: {known}")
    return run_module_main(COMMANDS[key], argv=argv)


def list_commands() -> list[str]:
    """Return sorted domain command names (hide legacy stage* / run_* aliases)."""
    import re

    preferred: set[str] = set()
    for name in COMMANDS:
        if name.startswith("run_"):
            continue
        # Hide temporary stage-number aliases from the public catalog.
        if re.search(r"stage\d", name) or name.startswith("stage"):
            continue
        preferred.add(name)
    return sorted(preferred)
