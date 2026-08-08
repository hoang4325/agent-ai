from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent_ai.shared.artifact_io import (
    append_jsonl,
    load_json,
    load_json_object,
    load_jsonl,
    touch_jsonl,
    write_json,
    write_jsonl,
)
from agent_ai.shared.numeric import clamp, clamp01


class ArtifactIoTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_write_and_load_json(self) -> None:
        path = self.root / "nested" / "payload.json"
        write_json(path, {"ok": True, "n": 3})
        self.assertEqual(load_json(path), {"ok": True, "n": 3})
        self.assertTrue(path.exists())

    def test_write_json_trailing_newline(self) -> None:
        path = self.root / "with_nl.json"
        write_json(path, {"a": 1}, trailing_newline=True)
        self.assertTrue(path.read_text(encoding="utf-8").endswith("\n"))

    def test_jsonl_roundtrip(self) -> None:
        path = self.root / "events.jsonl"
        append_jsonl(path, {"i": 1})
        append_jsonl(path, {"i": 2})
        self.assertEqual(load_jsonl(path), [{"i": 1}, {"i": 2}])
        write_jsonl(path, [{"i": 9}])
        self.assertEqual(load_jsonl(path), [{"i": 9}])

    def test_load_json_missing_ok(self) -> None:
        missing = self.root / "missing.json"
        self.assertEqual(load_json(missing, default={"x": 1}, missing_ok=True), {"x": 1})

    def test_load_json_object_rejects_list(self) -> None:
        path = self.root / "arr.json"
        path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        with self.assertRaises(ValueError):
            load_json_object(path)

    def test_load_jsonl_objects_only(self) -> None:
        path = self.root / "mixed.jsonl"
        path.write_text('{"a":1}\n[1,2]\n{"b":2}\n', encoding="utf-8")
        self.assertEqual(load_jsonl(path, objects_only=True), [{"a": 1}, {"b": 2}])

    def test_touch_jsonl(self) -> None:
        path = self.root / "empty.jsonl"
        touch_jsonl(path)
        self.assertEqual(path.read_text(encoding="utf-8"), "")

    def test_permissive_json_serialization(self) -> None:
        path = self.root / "perm.json"
        write_json(path, {"p": Path("/tmp/x")}, permissive=True)
        self.assertEqual(load_json(path)["p"], "/tmp/x")

    def test_clamp_helpers(self) -> None:
        self.assertEqual(clamp(5, 0, 3), 3)
        self.assertEqual(clamp(-1, 0, 3), 0)
        self.assertEqual(clamp01(1.5), 1.0)
        self.assertEqual(clamp01(-0.2), 0.0)


if __name__ == "__main__":
    unittest.main()
