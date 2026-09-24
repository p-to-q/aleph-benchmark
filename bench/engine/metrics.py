from __future__ import annotations

import random
import re
from statistics import mean
from typing import Any, Iterable

from .scoring_core import (
    exact_match_fidelity,
    fidelity,
    levenshtein,
    normalize_lexical_text,
    normalize_unicode,
    normalized_edit_similarity,
    unicode_char_ngram_similarity,
)


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

# Source compatibility for callers that imported the v0.1 function name.
# The v0.2 behavior is deliberately binary; callers needing a gradient must
# request ``normalized_edit_similarity`` explicitly.
exact_fidelity = exact_match_fidelity


def token_count(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def distortion(target: str, output: str, metric_class: str) -> float:
    return round(1.0 - fidelity(target, output, metric_class), 6)


def monotone_lower_envelope(points: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return real candidate points that improve the best known distortion."""
    by_length: dict[int, dict[str, Any]] = {}
    for point in points:
        if point.get("disqualified"):
            continue
        length = int(point["tokens"])
        current = by_length.get(length)
        if current is None or point["distortion"] < current["distortion"]:
            by_length[length] = point

    envelope: list[dict[str, Any]] = []
    best_distortion: float | None = None
    for point in sorted(by_length.values(), key=lambda item: (item["tokens"], item["distortion"])):
        if best_distortion is None or point["distortion"] < best_distortion:
            best_distortion = point["distortion"]
            frontier_point = dict(point)
            frontier_point["frontierRank"] = len(envelope) + 1
            envelope.append(frontier_point)
    return envelope


def aurc(frontier: list[dict[str, Any]], normalizer_tokens: int) -> float:
    """Area under the non-leaking rate-distortion staircase. Lower is better.

    ``normalizer_tokens`` is the right-hand budget anchor used to rescale the
    x-axis into ``[0, 1]``. ``frozen_ladder.evaluate_item`` passes the
    explicit-reconstruction prompt length (the rung-0 prompt — which is
    gated out of scoring but still anchors the budget), or the target text
    length if it is larger:

        explicit_tokens = max(
            token_count(item["frozenLadder"][0]["prompt"]),
            token_count(target),
            1,
        )

    The curve starts at the empty-budget baseline ``distortion = 1.0`` and
    drops only when the first eligible prompt becomes available. Beyond the
    right-hand anchor, the curve is held at its last distortion value, so AURC
    degrades smoothly when a model never reaches the target within the budget.
    The choice matches ``docs/protocol-v0.2.md`` §2's
    ``L_max = |explicit reconstruction prompt|``.
    """

    if not frontier:
        return 1.0
    normalizer = max(1, normalizer_tokens)
    area = 0.0
    previous_x = 0.0
    current_distortion = 1.0
    for point in sorted(frontier, key=lambda item: item["tokens"]):
        x = max(previous_x, min(1.0, point["tokens"] / normalizer))
        area += (x - previous_x) * current_distortion
        current_distortion = point["distortion"]
        previous_x = x
    area += max(0.0, 1.0 - previous_x) * current_distortion
    return round(max(0.0, min(1.0, area)), 6)


def ecl_at_tau(frontier: list[dict[str, Any]], tau: float) -> int | None:
    hits = [point["tokens"] for point in frontier if point["fidelity"] >= tau]
    return min(hits) if hits else None


def elicit_at_k(points: list[dict[str, Any]], tau: float, k: int) -> bool:
    """Return whether an eligible prompt within the k-unit budget reaches tau."""

    return any(point["tokens"] <= k and point["fidelity"] >= tau for point in points)


def ci95(values: list[float], *, seed: int, samples: int) -> dict[str, float] | None:
    if not values:
        return None
    if len(values) == 1:
        value = round(values[0], 6)
        return {"low": value, "high": value}
    rng = random.Random(seed)
    boot = []
    for _ in range(samples):
        draw = [values[rng.randrange(len(values))] for _ in values]
        boot.append(mean(draw))
    boot.sort()
    low_index = int(0.025 * (len(boot) - 1))
    high_index = int(0.975 * (len(boot) - 1))
    return {"low": round(boot[low_index], 6), "high": round(boot[high_index], 6)}


def summarize(values: list[float], *, seed: int, samples: int) -> tuple[float | None, dict[str, float] | None]:
    if not values:
        return None, None
    return round(mean(values), 6), ci95(values, seed=seed, samples=samples)


def rank_by_metric(rows: list[dict[str, Any]], key: str) -> list[str]:
    return [row["model"] for row in sorted(rows, key=lambda row: (row[key], row["model"]))]
