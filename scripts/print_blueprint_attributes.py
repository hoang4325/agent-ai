from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import carla

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print CARLA blueprint attributes for a given blueprint id.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--blueprint", required=True, help="Exact blueprint id, e.g. sensor.other.radar")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    client = carla.Client(args.host, args.port)
    client.set_timeout(5.0)
    world = client.get_world()
    blueprint = world.get_blueprint_library().find(args.blueprint)

    print(f"blueprint: {blueprint.id}")
    for attr in blueprint:
        recommended = list(attr.recommended_values)
        print(
            f"- {attr.id}: type={attr.type} modifiable={attr.is_modifiable} "
            f"default={attr.as_str()} recommended={recommended}"
        )


if __name__ == "__main__":
    main()
