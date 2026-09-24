from __future__ import annotations

from .protocol import DEFAULT_LEAKAGE_THRESHOLDS
from .scoring_core import (
    LEAKAGE_UNIT,
    LeakageGateResult,
    lcs_length,
    leakage_score,
    longest_common_span,
    ngrams,
    unicode_lexical_units,
)
from .scoring_core import evaluate_leakage as _evaluate_leakage


DEFAULT_THRESHOLDS = dict(DEFAULT_LEAKAGE_THRESHOLDS)


def evaluate_leakage(
    prompt: str,
    target: str,
    thresholds: dict[str, float | int] | None = None,
) -> LeakageGateResult:
    """Evaluate the canonical scorer with config-backed default thresholds."""

    active = dict(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
    return _evaluate_leakage(prompt, target, active)


__all__ = [
    "DEFAULT_THRESHOLDS",
    "LEAKAGE_UNIT",
    "LeakageGateResult",
    "evaluate_leakage",
    "lcs_length",
    "leakage_score",
    "longest_common_span",
    "ngrams",
    "unicode_lexical_units",
]
