"""Shim: ``stage4.stop_scene_alignment`` → ``agent_ai.runtime.stop_scene_alignment``."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("agent_ai.runtime.stop_scene_alignment")
sys.modules[__name__] = _module
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
