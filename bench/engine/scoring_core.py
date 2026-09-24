"""Pure-standard-library Aleph-Bench v0.2 scorer contract.

This module intentionally has no repository-relative imports so the exact file
can be vendored into platform packages. Engine and packaged scorer conformance
tests execute the same implementation.
"""

from __future__ import annotations

import sys
import unicodedata
import math
from collections import Counter
from dataclasses import dataclass


PROTOCOL_VERSION = "0.2.0"
SCORER_ID = "aleph-unicode"
SCORER_VERSION = "0.2.0"
SCHEMA_VERSION = "0.2.0"
RESPONSE_CAPTURE_VERSION = "raw-v1"
NORMALIZATION_PROFILE = "unicode_nfc_newline_v1"
LEXICAL_PROFILE = "unicode_nfc_casefold_whitespace_char3_multiset_jaccard_v1"
LEAKAGE_UNIT = "unicode_dual_channel_v1"
LEAKAGE_FAIL_CLOSED_POLICY = "bidi_controls_and_cjk_compatibility_ideographs_v1"
REQUIRED_PYTHON_VERSION = "3.13"
REQUIRED_UNICODE_DATABASE_VERSION = "15.1.0"
MAX_SCORING_TEXT_CHARACTERS = 16_384
MAX_NORMALIZED_TEXT_CHARACTERS = 32_768
MAX_QUADRATIC_CELLS = 1_000_000
SUPPORTED_METRIC_CLASSES = (
    "exact",
    "normalized_edit_similarity",
    "unicode_char_ngram",
)


def scoring_profile() -> dict[str, str | int]:
    return {
        "scorerId": SCORER_ID,
        "scorerVersion": SCORER_VERSION,
        "schemaVersion": SCHEMA_VERSION,
        "normalizationProfile": NORMALIZATION_PROFILE,
        "lexicalProfile": LEXICAL_PROFILE,
        "leakageUnit": LEAKAGE_UNIT,
        "leakageFailClosedPolicy": LEAKAGE_FAIL_CLOSED_POLICY,
        "responseCaptureVersion": RESPONSE_CAPTURE_VERSION,
        "unicodeDatabaseVersion": REQUIRED_UNICODE_DATABASE_VERSION,
        "pythonVersion": REQUIRED_PYTHON_VERSION,
        "maxScoringTextCharacters": MAX_SCORING_TEXT_CHARACTERS,
        "maxNormalizedTextCharacters": MAX_NORMALIZED_TEXT_CHARACTERS,
        "maxQuadraticCells": MAX_QUADRATIC_CELLS,
    }


def validate_scoring_runtime() -> None:
    """Fail closed when the runtime cannot reproduce the v0.2 profile."""

    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    if python_version != REQUIRED_PYTHON_VERSION:
        raise RuntimeError(
            "Aleph-Bench v0.2 scoring requires Python "
            f"{REQUIRED_PYTHON_VERSION}; found {python_version}"
        )
    if unicodedata.unidata_version != REQUIRED_UNICODE_DATABASE_VERSION:
        raise RuntimeError(
            "Aleph-Bench v0.2 scoring requires Unicode database "
            f"{REQUIRED_UNICODE_DATABASE_VERSION}; found {unicodedata.unidata_version}"
        )


def normalize_unicode(text: str) -> str:
    """Normalize CRLF/CR to LF and then normalize Unicode to NFC."""

    normalized_newlines = text.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", normalized_newlines)


def normalize_lexical_text(text: str) -> str:
    """Case-fold and collapse Unicode whitespace, retaining other code points."""

    folded = unicodedata.normalize("NFC", normalize_unicode(text).casefold())
    return " ".join(folded.split())


def levenshtein(a: str, b: str) -> int:
    _validate_quadratic_pair("Levenshtein", a, b)
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (0 if char_a == char_b else 1)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def exact_match_fidelity(target: str, output: str) -> float:
    """Binary equality of raw decoded code-point sequences; no normalization."""

    _validate_scoring_text("target", target)
    _validate_scoring_text("output", output)
    return 1.0 if target == output else 0.0


def normalized_edit_similarity(target: str, output: str) -> float:
    """NFC/newline-normalized Levenshtein similarity over code points."""

    _validate_scoring_text("target", target)
    _validate_scoring_text("output", output)
    target_norm = normalize_unicode(target)
    output_norm = normalize_unicode(output)
    _validate_normalized_text("target", target_norm)
    _validate_normalized_text("output", output_norm)
    if not target_norm and not output_norm:
        return 1.0
    if not target_norm or not output_norm:
        return 0.0
    _validate_quadratic_pair("normalized edit", target_norm, output_norm)
    distance = levenshtein(target_norm, output_norm)
    scale = max(len(target_norm), len(output_norm))
    return round(max(0.0, 1.0 - distance / scale), 6)


def unicode_char_ngram_similarity(target: str, output: str, n: int = 3) -> float:
    """Multiset Jaccard over normalized Unicode-code-point n-grams."""

    if n < 1:
        raise ValueError("n must be at least 1")
    _validate_scoring_text("target", target)
    _validate_scoring_text("output", output)

    def grams(value: str) -> Counter[str]:
        clean = normalize_lexical_text(value)
        _validate_normalized_text("lexical fidelity text", clean)
        if len(clean) < n:
            return Counter([clean]) if clean else Counter()
        return Counter(clean[index : index + n] for index in range(len(clean) - n + 1))

    target_grams = grams(target)
    output_grams = grams(output)
    if not target_grams and not output_grams:
        return 1.0
    if not target_grams or not output_grams:
        return 0.0
    keys = target_grams.keys() | output_grams.keys()
    intersection = sum(min(target_grams[key], output_grams[key]) for key in keys)
    union = sum(max(target_grams[key], output_grams[key]) for key in keys)
    return round(intersection / union, 6)


def fidelity(target: str, output: str, metric_class: str) -> float:
    validate_scoring_runtime()
    if not isinstance(target, str) or not isinstance(output, str):
        raise TypeError("target and output must be strings")
    _validate_scoring_text("target", target)
    _validate_scoring_text("output", output)
    if metric_class == "exact":
        return exact_match_fidelity(target, output)
    if metric_class == "normalized_edit_similarity":
        return normalized_edit_similarity(target, output)
    if metric_class == "unicode_char_ngram":
        return unicode_char_ngram_similarity(target, output)
    raise ValueError(f"unsupported metric class: {metric_class}")


@dataclass(frozen=True)
class LeakageGateResult:
    disqualified: bool
    unit: str
    failClosedReason: str | None
    lcsRatio: float
    targetTrigramRecall: float
    verbatimSpanUnits: int
    skeletonLcsRatio: float
    skeletonTargetTrigramRecall: float
    skeletonVerbatimSpanUnits: int
    thresholds: dict[str, float | int]

    def as_dict(self) -> dict[str, object]:
        return {
            "disqualified": self.disqualified,
            "unit": self.unit,
            "failClosedReason": self.failClosedReason,
            "lcsRatio": self.lcsRatio,
            "targetTrigramRecall": self.targetTrigramRecall,
            "verbatimSpanUnits": self.verbatimSpanUnits,
            "skeletonLcsRatio": self.skeletonLcsRatio,
            "skeletonTargetTrigramRecall": self.skeletonTargetTrigramRecall,
            "skeletonVerbatimSpanUnits": self.skeletonVerbatimSpanUnits,
            "thresholds": dict(self.thresholds),
        }


def _is_leakage_ignorable(char: str) -> bool:
    codepoint = ord(char)
    return (
        unicodedata.category(char) in {"Cc", "Cf", "Cs"}
        or 0xFE00 <= codepoint <= 0xFE0F
        or 0xE0100 <= codepoint <= 0xE01EF
        or 0x1F3FB <= codepoint <= 0x1F3FF
    )


def fail_closed_codepoint_reason(text: str) -> str | None:
    for char in text:
        codepoint = ord(char)
        if codepoint in {0x061C, 0x200E, 0x200F} or (
            0x202A <= codepoint <= 0x202E
            or 0x2066 <= codepoint <= 0x2069
        ):
            return "bidi_control"
        if 0xF900 <= codepoint <= 0xFAFF or 0x2F800 <= codepoint <= 0x2FA1F:
            return "cjk_compatibility_ideograph"
    return None


def _normalize_leakage_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = unicodedata.normalize("NFKC", normalized)
    normalized = unicodedata.normalize("NFKC", normalized.casefold())
    normalized = "".join(char for char in normalized if not _is_leakage_ignorable(char))
    normalized = unicodedata.normalize("NFKC", normalized)
    _validate_normalized_text("leakage text", normalized)
    return normalized


def unicode_skeleton_codepoints(text: str) -> list[str]:
    """Return the boundary-insensitive compatibility-normalized channel."""

    return [char for char in _normalize_leakage_text(text) if not char.isspace()]


def unicode_lexical_units(text: str) -> list[str]:
    """Return word, CJK code-point, punctuation, and symbol lexical units."""

    units: list[str] = []
    word: list[str] = []

    def flush_word() -> None:
        if word:
            units.append("".join(word))
            word.clear()

    for char in _normalize_leakage_text(text):
        category = unicodedata.category(char)
        if char.isspace():
            flush_word()
            continue
        if category.startswith("M"):
            if word:
                word.append(char)
            elif units:
                units[-1] += char
            else:
                units.append(char)
            continue
        if category[0] in {"L", "N"} or category == "Pc":
            if ord(char) > 127 and unicodedata.east_asian_width(char) in {"W", "F"}:
                flush_word()
                units.append(char)
            else:
                word.append(char)
            continue
        flush_word()
        units.append(char)
    flush_word()
    return units


def lcs_length(a: list[str], b: list[str]) -> int:
    _validate_quadratic_pair("LCS", a, b)
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for unit_a in a:
        current = [0]
        for index, unit_b in enumerate(b, start=1):
            if unit_a == unit_b:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[index - 1]))
        previous = current
    return previous[-1]


def longest_common_span(a: list[str], b: list[str]) -> int:
    _validate_quadratic_pair("longest common span", a, b)
    best = 0
    previous = [0] * (len(b) + 1)
    for unit_a in a:
        current = [0]
        for index, unit_b in enumerate(b, start=1):
            value = previous[index - 1] + 1 if unit_a == unit_b else 0
            current.append(value)
            best = max(best, value)
        previous = current
    return best


def ngrams(values: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(
        tuple(values[index : index + n])
        for index in range(max(0, len(values) - n + 1))
    )


def evaluate_leakage(
    prompt: str,
    target: str,
    thresholds: dict[str, float | int],
) -> LeakageGateResult:
    validate_scoring_runtime()
    if not isinstance(prompt, str) or not isinstance(target, str):
        raise TypeError("prompt and target must be strings")
    _validate_scoring_text("prompt", prompt)
    _validate_scoring_text("target", target)
    active = _validate_leakage_thresholds(thresholds)
    def channel_metrics(
        prompt_units: list[str], target_units: list[str]
    ) -> tuple[float, float, int]:
        if not prompt_units or not target_units:
            return 0.0, 0.0, 0
        _validate_quadratic_pair("leakage overlap", prompt_units, target_units)
        lcs_ratio = lcs_length(prompt_units, target_units) / len(target_units)
        target_trigrams = ngrams(target_units, 3)
        prompt_trigrams = ngrams(prompt_units, 3)
        target_trigram_count = sum(target_trigrams.values())
        target_trigram_recall = (
            sum((target_trigrams & prompt_trigrams).values()) / target_trigram_count
            if target_trigram_count
            else 0.0
        )
        span = longest_common_span(prompt_units, target_units)
        return lcs_ratio, target_trigram_recall, span

    lcs_ratio, target_trigram_recall, span = channel_metrics(
        unicode_lexical_units(prompt), unicode_lexical_units(target)
    )
    skeleton_lcs, skeleton_trigram_recall, skeleton_span = channel_metrics(
        unicode_skeleton_codepoints(prompt), unicode_skeleton_codepoints(target)
    )
    fail_closed_reason = fail_closed_codepoint_reason(prompt)
    disqualified = (
        fail_closed_reason is not None
        or lcs_ratio >= float(active["lcsRatio"])
        or target_trigram_recall >= float(active["targetTrigramRecall"])
        or span >= int(active["verbatimSpanUnits"])
        or skeleton_lcs >= float(active["skeletonLcsRatio"])
        or skeleton_trigram_recall
        >= float(active["skeletonTargetTrigramRecall"])
        or skeleton_span >= int(active["skeletonVerbatimSpanUnits"])
    )
    return LeakageGateResult(
        disqualified=disqualified,
        unit=LEAKAGE_UNIT,
        failClosedReason=fail_closed_reason,
        lcsRatio=round(lcs_ratio, 6),
        targetTrigramRecall=round(target_trigram_recall, 6),
        verbatimSpanUnits=span,
        skeletonLcsRatio=round(skeleton_lcs, 6),
        skeletonTargetTrigramRecall=round(skeleton_trigram_recall, 6),
        skeletonVerbatimSpanUnits=skeleton_span,
        thresholds=active,
    )


def leakage_score(result: LeakageGateResult) -> float:
    """Return a diagnostic proximity score; never an aggregate penalty."""

    if result.failClosedReason is not None:
        return 1.0
    thresholds = _validate_leakage_thresholds(result.thresholds)
    span_score = min(
        1.0,
        result.verbatimSpanUnits / int(thresholds["verbatimSpanUnits"]),
    )
    skeleton_span_score = min(
        1.0,
        result.skeletonVerbatimSpanUnits
        / int(thresholds["skeletonVerbatimSpanUnits"]),
    )
    return round(
        max(
            result.lcsRatio,
            result.targetTrigramRecall,
            span_score,
            result.skeletonLcsRatio,
            result.skeletonTargetTrigramRecall,
            skeleton_span_score,
        ),
        6,
    )


def _validate_leakage_thresholds(
    thresholds: dict[str, float | int],
) -> dict[str, float | int]:
    if not isinstance(thresholds, dict):
        raise TypeError("leakage thresholds must be an object")
    required = {
        "lcsRatio",
        "targetTrigramRecall",
        "verbatimSpanUnits",
        "skeletonLcsRatio",
        "skeletonTargetTrigramRecall",
        "skeletonVerbatimSpanUnits",
    }
    if set(thresholds) != required:
        raise ValueError(f"leakage thresholds must contain exactly {sorted(required)}")

    active: dict[str, float | int] = {}
    for name in (
        "lcsRatio",
        "targetTrigramRecall",
        "skeletonLcsRatio",
        "skeletonTargetTrigramRecall",
    ):
        value = thresholds[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or (isinstance(value, float) and not math.isfinite(value))
            or not 0.0 <= value <= 1.0
        ):
            raise ValueError(f"leakage threshold {name} must be a finite number in [0, 1]")
        active[name] = float(value)
    for name in ("verbatimSpanUnits", "skeletonVerbatimSpanUnits"):
        span = thresholds[name]
        if isinstance(span, bool) or not isinstance(span, int) or span < 1:
            raise ValueError(f"leakage threshold {name} must be an integer >= 1")
        active[name] = span
    return active


def _validate_scoring_text(role: str, value: str) -> None:
    if len(value) > MAX_SCORING_TEXT_CHARACTERS:
        raise ValueError(
            f"{role} exceeds the scorer limit of {MAX_SCORING_TEXT_CHARACTERS} characters"
        )


def _validate_normalized_text(role: str, value: str) -> None:
    if len(value) > MAX_NORMALIZED_TEXT_CHARACTERS:
        raise ValueError(
            f"{role} exceeds the normalized scorer limit of "
            f"{MAX_NORMALIZED_TEXT_CHARACTERS} characters"
        )


def _validate_quadratic_pair(
    role: str,
    left: str | list[str],
    right: str | list[str],
) -> None:
    left_length = len(left)
    right_length = len(right)
    cells = left_length * right_length
    if cells > MAX_QUADRATIC_CELLS:
        raise ValueError(
            f"{role} would require {cells} quadratic cells; "
            f"limit is {MAX_QUADRATIC_CELLS}"
        )
