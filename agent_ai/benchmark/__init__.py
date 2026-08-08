"""
Benchmark package: corpus, orchestration, and domain subpackages.

Subpackages
-----------
- ``gates``    — contract/artifact/preflight/stop audits & metric packs
- ``shadow``   — MPC/agent shadow evaluation and smoke tools
- ``takeover`` — control-authority / takeover sandboxes and rings
- ``assist``   — stage-8-style assist arbitration adapters

Core modules remain at package root: ``io``, ``contracts``, ``metrics``,
``orchestration``, ``runner_core``, ``provenance``, frozen corpus tools.
"""
from __future__ import annotations

__all__ = ["gates", "shadow", "takeover", "assist"]
