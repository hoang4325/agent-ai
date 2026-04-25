from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_stage4_visualization_bundle(
    *,
    output_root: str | Path,
    tick_records: Iterable[Dict[str, Any]],
) -> None:
    output_root = Path(output_root)
    tick_records = list(tick_records)
    behavior_counts = Counter(
        record["selected_behavior"] for record in tick_records if record.get("selected_behavior")
    )
    _write_json(
        output_root / "behavior_counts.json",
        {"behavior_counts": dict(behavior_counts)},
    )
    _write_json(
        output_root / "latency_series.json",
        {
            "ticks": [
                {
                    "tick_id": int(record["tick_id"]),
                    "perception_latency_ms": record.get("perception_latency_ms"),
                    "world_update_latency_ms": record.get("world_update_latency_ms"),
                    "behavior_latency_ms": record.get("behavior_latency_ms"),
                    "execution_latency_ms": record.get("execution_latency_ms"),
                }
                for record in tick_records
            ]
        },
    )
