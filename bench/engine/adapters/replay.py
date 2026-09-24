from __future__ import annotations

from typing import Any

from ..protocol import DEFAULT_RERUNS
from .base import ModelAdapter


class RetainedOutputReplayAdapter(ModelAdapter):
    """Replay already-verified raw strings through the scorer without I/O.

    This adapter deliberately does not parse or validate an evidence artifact.
    Callers must first verify coverage, ordering, response types, and receipt
    identities, then pass the resulting lookup tables here for deterministic
    rescoring.
    """

    def __init__(
        self,
        *,
        model_id: str,
        observation_mode: str,
        effective_reruns: int,
        outputs: dict[tuple[str, str], list[str]],
        receipts: dict[tuple[str, str], list[dict[str, str]]],
    ) -> None:
        super().__init__(
            model_id=model_id,
            observation_mode=observation_mode,
            temperature=0.0,
        )
        self.effective_reruns = effective_reruns
        self.outputs = outputs
        self.receipts = receipts
        self._last_response_receipt: dict[str, str] | None = None

    def reruns(self, configured: int) -> int:
        if configured != DEFAULT_RERUNS:
            raise ValueError("receipt replay requires the frozen rerun count")
        return self.effective_reruns

    def generate(
        self,
        prompt: str,
        item: dict[str, Any],
        ladder_prompt: dict[str, Any],
        *,
        seed: int,
        rerun_index: int,
    ) -> str:
        del prompt, seed
        key = (item["id"], ladder_prompt["id"])
        try:
            output = self.outputs[key][rerun_index]
            self._last_response_receipt = (
                dict(self.receipts[key][rerun_index])
                if self.observation_mode == "black_box"
                else None
            )
            return output
        except (KeyError, IndexError) as exc:
            raise ValueError(
                f"raw-output receipt is incomplete for {self.model_id}:{key[0]}:{key[1]}"
            ) from exc

    def last_response_receipt(self) -> dict[str, str] | None:
        return (
            dict(self._last_response_receipt)
            if self._last_response_receipt is not None
            else None
        )
