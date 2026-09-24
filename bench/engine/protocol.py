from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .scoring_core import (
    PROTOCOL_VERSION,
    RESPONSE_CAPTURE_VERSION,
    scoring_profile,
    validate_scoring_runtime,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_CONFIG_PATH = REPO_ROOT / "bench/config/frozen_ladder-v0.2.json"
EXPECTED_PROTOCOL_CONFIG_SHA256 = (
    "c4cc5d97655de91fcf22674d5133a689faee8ac27dd8c60a1a67caccae648ba8"
)


def load_protocol_config(path: Path = PROTOCOL_CONFIG_PATH) -> dict[str, Any]:
    raw_config = Path(path).read_bytes()
    config = json.loads(raw_config.decode("utf-8"))
    if not isinstance(config, dict):
        raise ValueError("frozen-ladder config must be a JSON object")
    required = (
        "protocolVersion",
        "track",
        "split",
        "stratum",
        "datasetId",
        "datasetItemCount",
        "datasetSha256",
        "datasetHashAlgorithm",
        "tau",
        "k",
        "reruns",
        "bootstrapSamples",
        "leakageThresholds",
        "decoding",
    )
    missing = [name for name in required if name not in config]
    unexpected = sorted(set(config) - set(required))
    if missing or unexpected:
        raise ValueError(
            "frozen-ladder config must contain exactly the frozen fields; "
            f"missing={missing}, unexpected={unexpected}"
        )
    tau = config["tau"]
    if (
        isinstance(tau, bool)
        or not isinstance(tau, (int, float))
        or (isinstance(tau, float) and not math.isfinite(tau))
        or not 0.0 <= tau <= 1.0
    ):
        raise ValueError("frozen-ladder tau must be between 0 and 1")
    for name in ("k", "reruns", "bootstrapSamples"):
        if isinstance(config[name], bool) or not isinstance(config[name], int) or config[name] < 1:
            raise ValueError(f"frozen-ladder {name} must be at least 1")
    thresholds = config["leakageThresholds"]
    if not isinstance(thresholds, dict):
        raise ValueError("frozen-ladder leakageThresholds must be an object")
    expected_thresholds = (
        "lcsRatio",
        "targetTrigramRecall",
        "verbatimSpanUnits",
        "skeletonLcsRatio",
        "skeletonTargetTrigramRecall",
        "skeletonVerbatimSpanUnits",
    )
    missing_thresholds = [name for name in expected_thresholds if name not in thresholds]
    unexpected_thresholds = sorted(set(thresholds) - set(expected_thresholds))
    if missing_thresholds or unexpected_thresholds:
        raise ValueError(
            "frozen-ladder leakageThresholds must contain exactly "
            f"{', '.join(expected_thresholds)}; missing={missing_thresholds}, "
            f"unexpected={unexpected_thresholds}"
        )
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
            raise ValueError(f"frozen-ladder {name} must be between 0 and 1")
    for name in ("verbatimSpanUnits", "skeletonVerbatimSpanUnits"):
        span = thresholds[name]
        if isinstance(span, bool) or not isinstance(span, int) or span < 1:
            raise ValueError(f"frozen-ladder {name} must be at least 1")
    for name in (
        "protocolVersion",
        "track",
        "split",
        "stratum",
        "datasetId",
        "datasetSha256",
        "datasetHashAlgorithm",
    ):
        if not isinstance(config[name], str) or not config[name]:
            raise ValueError(f"frozen-ladder {name} must be a non-empty string")
    if (
        not isinstance(config["decoding"], dict)
        or set(config["decoding"]) != {"mock", "hosted"}
        or any(
            not isinstance(value, str) or not value
            for value in config["decoding"].values()
        )
    ):
        raise ValueError(
            "frozen-ladder decoding must contain non-empty mock and hosted strings"
        )
    if (
        isinstance(config["datasetItemCount"], bool)
        or not isinstance(config["datasetItemCount"], int)
        or config["datasetItemCount"] < 1
    ):
        raise ValueError("frozen-ladder datasetItemCount must be a positive integer")
    if len(config["datasetSha256"]) != 64:
        raise ValueError("frozen-ladder datasetSha256 must be a SHA-256 hex digest")
    try:
        int(config["datasetSha256"], 16)
    except ValueError as exc:
        raise ValueError("frozen-ladder datasetSha256 must be a SHA-256 hex digest") from exc
    observed_digest = hashlib.sha256(raw_config).hexdigest()
    if observed_digest != EXPECTED_PROTOCOL_CONFIG_SHA256:
        raise ValueError(
            "Aleph-Bench v0.2 frozen config bytes changed: "
            f"expected {EXPECTED_PROTOCOL_CONFIG_SHA256}, found {observed_digest}"
        )
    return config


PROTOCOL_CONFIG = load_protocol_config()
if str(PROTOCOL_CONFIG["protocolVersion"]) != PROTOCOL_VERSION:
    raise ValueError(
        "frozen-ladder protocolVersion does not match the canonical scorer: "
        f"{PROTOCOL_CONFIG['protocolVersion']} != {PROTOCOL_VERSION}"
    )
DEFAULT_BOOTSTRAP_SAMPLES = int(PROTOCOL_CONFIG["bootstrapSamples"])
DEFAULT_RERUNS = int(PROTOCOL_CONFIG["reruns"])
DEFAULT_TAU = float(PROTOCOL_CONFIG["tau"])
DEFAULT_K = int(PROTOCOL_CONFIG["k"])
DEFAULT_LEAKAGE_THRESHOLDS: dict[str, float | int] = {
    "lcsRatio": float(PROTOCOL_CONFIG["leakageThresholds"]["lcsRatio"]),
    "targetTrigramRecall": float(
        PROTOCOL_CONFIG["leakageThresholds"]["targetTrigramRecall"]
    ),
    "verbatimSpanUnits": int(PROTOCOL_CONFIG["leakageThresholds"]["verbatimSpanUnits"]),
    "skeletonLcsRatio": float(
        PROTOCOL_CONFIG["leakageThresholds"]["skeletonLcsRatio"]
    ),
    "skeletonTargetTrigramRecall": float(
        PROTOCOL_CONFIG["leakageThresholds"]["skeletonTargetTrigramRecall"]
    ),
    "skeletonVerbatimSpanUnits": int(
        PROTOCOL_CONFIG["leakageThresholds"]["skeletonVerbatimSpanUnits"]
    ),
}
FROZEN_TRACK = str(PROTOCOL_CONFIG["track"])
FROZEN_SPLIT = str(PROTOCOL_CONFIG["split"])
FROZEN_STRATUM = str(PROTOCOL_CONFIG["stratum"])
FROZEN_DATASET_ID = str(PROTOCOL_CONFIG["datasetId"])
FROZEN_DATASET_ITEM_COUNT = int(PROTOCOL_CONFIG["datasetItemCount"])
FROZEN_DATASET_SHA256 = str(PROTOCOL_CONFIG["datasetSha256"])
FROZEN_DATASET_HASH_ALGORITHM = str(PROTOCOL_CONFIG["datasetHashAlgorithm"])
FROZEN_DECODING: dict[str, str] = {
    str(name): str(value) for name, value in PROTOCOL_CONFIG["decoding"].items()
}


def artifact_identity_fingerprint(kind: str, payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        {
            "kind": kind,
            "protocolVersion": PROTOCOL_VERSION,
            "datasetSha256": FROZEN_DATASET_SHA256,
            "payload": payload,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def result_artifact_id(
    artifact: dict[str, Any],
) -> str:
    evaluation_scope = artifact.get("evaluationScope")
    evaluated_item_count = artifact.get("evaluatedItemCount")
    if evaluation_scope not in {"canonical", "smoke"}:
        raise ValueError(f"unsupported evaluation scope: {evaluation_scope!r}")
    if (
        isinstance(evaluated_item_count, bool)
        or not isinstance(evaluated_item_count, int)
        or evaluated_item_count < 1
    ):
        raise ValueError("evaluatedItemCount must be a positive integer")
    scope_label = (
        "m0"
        if evaluation_scope == "canonical"
        else f"smoke-{evaluated_item_count}"
    )
    seed = artifact.get("seed")
    fingerprint = artifact_identity_fingerprint("result", artifact)
    return f"aleph-bench-v0.2-{scope_label}-seed-{seed}-artifact-{fingerprint}"


def manifest_artifact_id(artifact: dict[str, Any]) -> str:
    seed = artifact.get("seed")
    fingerprint = artifact_identity_fingerprint("manifest", artifact)
    return f"aleph-bench-v0.2-m0-manifest-seed-{seed}-artifact-{fingerprint}"


def aleph_run_artifact_id(
    *, item_id: str, seed: int, artifact: dict[str, Any]
) -> str:
    fingerprint = artifact_identity_fingerprint(
        "aleph-run",
        {"itemId": item_id, "seed": seed, "artifact": artifact},
    )
    return f"aleph-run-v0.2-{item_id}-seed-{seed}-artifact-{fingerprint}"


def decoding_for_evidence_mode(evidence_mode: str) -> str:
    if evidence_mode == "black_box":
        return FROZEN_DECODING["hosted"]
    if evidence_mode in {"mock", "fixture", "simulated"}:
        return FROZEN_DECODING["mock"]
    raise ValueError(
        f"protocol {PROTOCOL_VERSION} has no decoding contract for {evidence_mode!r}"
    )


def validate_protocol_settings(
    *,
    track: str = FROZEN_TRACK,
    split: str = FROZEN_SPLIT,
    tau: float = DEFAULT_TAU,
    k: int = DEFAULT_K,
    reruns: int = DEFAULT_RERUNS,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    thresholds: dict[str, float | int] | None = None,
) -> None:
    """Reject values that would be mislabeled as the frozen v0.2 protocol."""

    load_protocol_config(PROTOCOL_CONFIG_PATH)
    active_thresholds = dict(
        DEFAULT_LEAKAGE_THRESHOLDS if thresholds is None else thresholds
    )
    observed = {
        "track": track,
        "split": split,
        "tau": tau,
        "k": k,
        "reruns": reruns,
        "bootstrapSamples": bootstrap_samples,
        "leakageThresholds": active_thresholds,
    }
    expected = {
        "track": FROZEN_TRACK,
        "split": FROZEN_SPLIT,
        "tau": DEFAULT_TAU,
        "k": DEFAULT_K,
        "reruns": DEFAULT_RERUNS,
        "bootstrapSamples": DEFAULT_BOOTSTRAP_SAMPLES,
        "leakageThresholds": DEFAULT_LEAKAGE_THRESHOLDS,
    }
    drift = [
        f"{name}={observed[name]!r} (expected {value!r})"
        for name, value in expected.items()
        if not _same_protocol_value(observed[name], value)
    ]
    if drift:
        raise ValueError(
            f"Aleph-Bench protocol {PROTOCOL_VERSION} is frozen; rejected overrides: "
            + ", ".join(drift)
        )


def _same_protocol_value(observed: Any, expected: Any) -> bool:
    if type(observed) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(observed) == set(expected) and all(
            _same_protocol_value(observed[key], expected[key]) for key in expected
        )
    return observed == expected
