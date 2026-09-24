from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ModelAdapter:
    model_id: str
    observation_mode: str
    temperature: float = 0.0

    def reruns(self, configured: int) -> int:
        return 1 if self.temperature == 0 else max(1, configured)

    def cache_identity(self) -> dict[str, Any]:
        """Return non-secret fields that determine one adapter request."""

        return {
            "adapterId": f"{type(self).__module__}.{type(self).__qualname__}",
            "adapterVersion": "1",
            "model": self.model_id,
            "observationMode": self.observation_mode,
            "temperature": self.temperature,
        }

    def evidence_identity(self) -> dict[str, Any]:
        """Return the non-secret adapter identity retained with evidence."""

        return self.cache_identity()

    def last_response_receipt(self) -> dict[str, str] | None:
        """Return capture metadata for the most recent generated response."""

        return None

    def generate(
        self,
        prompt: str,
        item: dict[str, Any],
        ladder_prompt: dict[str, Any],
        *,
        seed: int,
        rerun_index: int,
    ) -> str:
        raise NotImplementedError
