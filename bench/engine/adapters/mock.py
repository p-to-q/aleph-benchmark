from __future__ import annotations

import hashlib
from typing import Any

from .base import ModelAdapter


PROFILES = {
    "mock-frontier": {
        1: 0.992,
        2: 0.965,
        3: 0.925,
    },
    "mock-mid": {
        1: 0.952,
        2: 0.875,
        3: 0.69,
    },
    "mock-small": {
        1: 0.905,
        2: 0.66,
        3: 0.43,
    },
}
MOCK_ADAPTER_ID = "aleph.mock-profile"
MOCK_ADAPTER_VERSION = "1"


class MockAdapter(ModelAdapter):
    """Deterministic adapter used to validate benchmark plumbing.

    The adapter has access to the item because it is simulated evidence. Hosted
    black-box adapters must not use this shortcut.
    """

    def __init__(self, model_id: str):
        if model_id not in PROFILES:
            raise ValueError(f"unknown mock model: {model_id}")
        super().__init__(model_id=model_id, observation_mode="mock", temperature=0.0)
        self.profile = PROFILES[model_id]

    def cache_identity(self) -> dict[str, Any]:
        identity = super().cache_identity()
        identity.update(
            {
                "adapterId": MOCK_ADAPTER_ID,
                "adapterVersion": MOCK_ADAPTER_VERSION,
            }
        )
        return identity

    def generate(
        self,
        prompt: str,
        item: dict[str, Any],
        ladder_prompt: dict[str, Any],
        *,
        seed: int,
        rerun_index: int,
    ) -> str:
        del prompt
        target = item["target"]["text"]
        rung = int(ladder_prompt["rung"])
        if rung == 0:
            return target
        base = self.profile.get(rung, 0.25)
        wobble = self._wobble(item["id"], rung, ladder_prompt["paraphrase"], seed, rerun_index)
        fidelity = max(0.0, min(1.0, base + wobble))
        return degrade(target, fidelity)

    @staticmethod
    def _wobble(item_id: str, rung: int, paraphrase: int, seed: int, rerun_index: int) -> float:
        raw = f"{item_id}:{rung}:{paraphrase}:{seed}:{rerun_index}".encode("utf-8")
        digest = hashlib.sha256(raw).digest()[0]
        return ((digest % 9) - 4) / 1000


def degrade(target: str, fidelity: float) -> str:
    if fidelity >= 0.985:
        return target
    keep = max(0, min(len(target), int(round(len(target) * fidelity))))
    return target[:keep].rstrip()
