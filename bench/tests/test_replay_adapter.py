from __future__ import annotations

import unittest

from bench.engine.adapters import RetainedOutputReplayAdapter
from bench.engine.protocol import DEFAULT_RERUNS


class RetainedOutputReplayAdapterTests(unittest.TestCase):
    def _adapter(
        self, *, observation_mode: str = "black_box"
    ) -> RetainedOutputReplayAdapter:
        return RetainedOutputReplayAdapter(
            model_id="hosted:provider/model",
            observation_mode=observation_mode,
            effective_reruns=2,
            outputs={("item-1", "prompt-1"): ["first", "second"]},
            receipts={
                ("item-1", "prompt-1"): [
                    {"capturedAt": "2026-09-24T00:00:00Z", "source": "provider"},
                    {"capturedAt": "2026-09-24T00:00:01Z", "source": "cache"},
                ]
            },
        )

    def test_replays_exact_rerun_and_returns_a_receipt_copy(self) -> None:
        adapter = self._adapter()

        output = adapter.generate(
            "ignored retained prompt",
            {"id": "item-1"},
            {"id": "prompt-1"},
            seed=73,
            rerun_index=1,
        )

        self.assertEqual(output, "second")
        receipt = adapter.last_response_receipt()
        self.assertEqual(
            receipt,
            {"capturedAt": "2026-09-24T00:00:01Z", "source": "cache"},
        )
        assert receipt is not None
        receipt["source"] = "mutated"
        self.assertEqual(adapter.last_response_receipt()["source"], "cache")  # type: ignore[index]

    def test_non_black_box_replay_does_not_claim_provider_receipt(self) -> None:
        adapter = self._adapter(observation_mode="mock")

        self.assertEqual(
            adapter.generate(
                "ignored",
                {"id": "item-1"},
                {"id": "prompt-1"},
                seed=0,
                rerun_index=0,
            ),
            "first",
        )
        self.assertIsNone(adapter.last_response_receipt())

    def test_requires_the_frozen_configured_rerun_count(self) -> None:
        adapter = self._adapter()

        self.assertEqual(adapter.reruns(DEFAULT_RERUNS), 2)
        with self.assertRaisesRegex(ValueError, "frozen rerun count"):
            adapter.reruns(DEFAULT_RERUNS - 1)

    def test_missing_key_or_rerun_fails_closed(self) -> None:
        adapter = self._adapter()

        for item_id, rerun_index in (("missing", 0), ("item-1", 2)):
            with self.subTest(item_id=item_id, rerun_index=rerun_index):
                with self.assertRaisesRegex(
                    ValueError,
                    "raw-output receipt is incomplete for hosted:provider/model",
                ):
                    adapter.generate(
                        "ignored",
                        {"id": item_id},
                        {"id": "prompt-1"},
                        seed=0,
                        rerun_index=rerun_index,
                    )


if __name__ == "__main__":
    unittest.main()
