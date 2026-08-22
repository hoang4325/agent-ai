from __future__ import annotations

import threading
import time
import unittest

from benchmark.async_agent_worker import LatestOnlyAgentWorker


class LatestOnlyAgentWorkerTests(unittest.TestCase):
    def test_call_runs_off_control_thread(self) -> None:
        caller_thread = threading.get_ident()

        def call(payload):
            return {"value": payload["value"], "thread": threading.get_ident()}

        worker = LatestOnlyAgentWorker(call)
        try:
            worker.submit(
                frame_id=10,
                frame_idx=2,
                sim_timestamp_s=0.2,
                payload={"value": 7},
            )
            deadline = time.monotonic() + 1.0
            result = None
            while result is None and time.monotonic() < deadline:
                result = worker.poll()
                time.sleep(0.005)
            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result.intent_record["value"], 7)
            self.assertNotEqual(result.intent_record["thread"], caller_thread)
            self.assertGreaterEqual(result.latency_ms, 0.0)
        finally:
            worker.close()

    def test_new_submission_replaces_only_pending_request(self) -> None:
        release_first = threading.Event()
        started_first = threading.Event()

        def call(payload):
            if payload["value"] == 1:
                started_first.set()
                release_first.wait(timeout=1.0)
            return payload["value"]

        worker = LatestOnlyAgentWorker(call)
        try:
            worker.submit(frame_id=1, frame_idx=1, sim_timestamp_s=0.1, payload={"value": 1})
            self.assertTrue(started_first.wait(timeout=1.0))
            second = worker.submit(frame_id=2, frame_idx=2, sim_timestamp_s=0.2, payload={"value": 2})
            third = worker.submit(frame_id=3, frame_idx=3, sim_timestamp_s=0.3, payload={"value": 3})
            self.assertEqual(third.replaced_pending_request_id, second.request_id)
            release_first.set()

            deadline = time.monotonic() + 1.0
            while worker.stats()["completed"] < 2 and time.monotonic() < deadline:
                time.sleep(0.005)
            stats = worker.stats()
            self.assertEqual(stats["started"], 2)
            self.assertEqual(stats["completed"], 2)
            self.assertEqual(stats["pending_replaced"], 1)
        finally:
            release_first.set()
            worker.close()


if __name__ == "__main__":
    unittest.main()
