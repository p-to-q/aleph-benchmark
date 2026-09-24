from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev, pvariance
from typing import Any

from .adapters import (
    HostedBlackBoxAdapter,
    MockAdapter,
    ModelAdapter,
    normalize_model_id,
)
from .leakage_gate import DEFAULT_THRESHOLDS, evaluate_leakage, leakage_score
from .metrics import (
    aurc,
    ecl_at_tau,
    elicit_at_k,
    fidelity,
    monotone_lower_envelope,
    summarize,
    token_count,
)
from .protocol import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_K,
    DEFAULT_RERUNS,
    DEFAULT_TAU,
    FROZEN_SPLIT,
    FROZEN_STRATUM,
    FROZEN_TRACK,
    FROZEN_DATASET_HASH_ALGORITHM,
    FROZEN_DATASET_ID,
    FROZEN_DATASET_ITEM_COUNT,
    FROZEN_DATASET_SHA256,
    FROZEN_DECODING,
    PROTOCOL_CONFIG_PATH,
    PROTOCOL_VERSION,
    aleph_run_artifact_id,
    load_protocol_config,
    decoding_for_evidence_mode,
    result_artifact_id,
    scoring_profile,
    validate_protocol_settings,
    validate_scoring_runtime,
)
from .response_cache import ResponseCacheAdapter
from .schema_validation import SchemaValidationError, load_schema, validate
from .scoring_core import (
    SUPPORTED_METRIC_CLASSES,
    fail_closed_codepoint_reason,
    unicode_skeleton_codepoints,
)


FIXED_CREATED_AT = "2026-09-17T00:00:00Z"
REPO_ROOT = Path(__file__).resolve().parents[2]


def stable_dataset_path(data_dir: Path) -> str:
    resolved = data_dir.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return data_dir.as_posix()


def validate_item_protocol(item: dict[str, Any]) -> None:
    item_id = item.get("id", "<missing id>")
    if item.get("protocolVersion") != PROTOCOL_VERSION:
        raise ValueError(
            f"Aleph-Bench scorer protocol {PROTOCOL_VERSION} rejected dataset item "
            f"{item_id}={item.get('protocolVersion', '<missing>')}"
        )
    if item.get("metricClass") not in SUPPORTED_METRIC_CLASSES:
        raise ValueError(
            "Aleph-Bench v0.2 rejected unsupported metric class: "
            f"{item_id}={item.get('metricClass', '<missing>')}"
        )
    if item.get("stratum") != FROZEN_STRATUM:
        raise ValueError(
            f"Aleph-Bench protocol {PROTOCOL_VERSION} requires stratum "
            f"{FROZEN_STRATUM}; {item_id} has {item.get('stratum', '<missing>')}"
        )


def dataset_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        name = path.name.encode("utf-8")
        content = path.read_bytes()
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def load_items(
    data_dir: Path,
    *,
    limit: int | None = None,
    require_canonical: bool = False,
) -> list[dict[str, Any]]:
    if limit is not None and limit < 1:
        raise ValueError("item limit must be at least 1")
    paths = sorted(data_dir.glob("*.json"))
    if not paths:
        raise ValueError(f"no BenchItems found in {data_dir}")
    if limit is not None and limit > len(paths):
        raise ValueError(
            f"item limit {limit} exceeds the available dataset size {len(paths)}"
        )
    if require_canonical:
        observed_digest = dataset_sha256(paths)
        if len(paths) != FROZEN_DATASET_ITEM_COUNT or observed_digest != FROZEN_DATASET_SHA256:
            raise ValueError(
                f"dataset does not match frozen {FROZEN_DATASET_ID}: expected "
                f"{FROZEN_DATASET_ITEM_COUNT} items/{FROZEN_DATASET_SHA256}, found "
                f"{len(paths)} items/{observed_digest}"
            )

    items: list[dict[str, Any]] = []
    item_schema = load_schema(REPO_ROOT / "schemas/v0.2/aleph-bench-item.schema.json")
    for path in paths:
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"could not load BenchItem {path}: {exc}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"BenchItem {path} must be a JSON object")
        validate_item_protocol(item)
        try:
            validate(item, item_schema)
        except SchemaValidationError as exc:
            raise ValueError(f"BenchItem {path} failed v0.2 schema validation: {exc}") from exc
        items.append(item)

    item_ids = [item.get("id") for item in items]
    if any(not isinstance(item_id, str) or not item_id for item_id in item_ids):
        raise ValueError("every BenchItem must have a non-empty string id")
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("BenchItem ids must be unique")
    canaries = {item.get("canaryGuid") for item in items}
    if len(canaries) != 1 or not all(isinstance(value, str) and value for value in canaries):
        raise ValueError("all BenchItems must share one non-empty canaryGuid")
    for item in items:
        target = item["target"]["text"]
        forbidden_target_reason = fail_closed_codepoint_reason(target)
        if forbidden_target_reason is not None:
            raise ValueError(
                f"BenchItem {item['id']} target contains forbidden Unicode policy input: "
                f"{forbidden_target_reason}"
            )
        if not unicode_skeleton_codepoints(target):
            raise ValueError(
                f"BenchItem {item['id']} target is empty after leakage normalization"
            )
        for ladder in item["frozenLadder"]:
            gate = evaluate_leakage(ladder["prompt"], target, DEFAULT_THRESHOLDS)
            expected_disqualified = ladder["expectedLeakage"] == "leaky_anchor"
            if gate.disqualified != expected_disqualified:
                raise ValueError(
                    f"BenchItem {item['id']} prompt {ladder['id']} expectedLeakage "
                    f"does not match the frozen leakage gate"
                )
    return items[:limit] if limit is not None else items


def adapter_for_model(model: str) -> ModelAdapter:
    canonical, _, uses_configured_reruns = normalize_model_id(model)
    if not uses_configured_reruns:
        return MockAdapter(canonical)
    return HostedBlackBoxAdapter(canonical.removeprefix("hosted:"))


def bootstrap_seed(seed: int, model: str, metric: str) -> int:
    identity = "\0".join(
        [PROTOCOL_VERSION, FROZEN_DATASET_SHA256, str(seed), model, metric]
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(identity).digest()[:8], "big")


def summarize_model_runs(
    *,
    model: str,
    evidence_mode: str,
    runs: list[dict[str, Any]],
    seed: int,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    """Derive one model summary solely from its item-run receipts.

    Keeping aggregation in one function lets the offline verifier recompute the
    published summary from retained raw outputs without invoking a provider.
    """

    validate_protocol_settings(bootstrap_samples=bootstrap_samples)
    if not runs:
        raise ValueError(f"cannot summarize model {model!r} without item runs")
    if any(run.get("model") != model for run in runs):
        raise ValueError(f"item runs for {model!r} contain a different model id")
    aurc_values = [run["metrics"]["aurc"] for run in runs]
    ecl_values = [
        run["metrics"]["eclAtTau"]
        for run in runs
        if run["metrics"]["eclAtTau"] is not None
    ]
    elicit_values = [
        1.0 if run["metrics"]["elicitAtK"] else 0.0 for run in runs
    ]
    leakage_values = [run["metrics"]["leakageHitRate"] for run in runs]
    aurc_mean, aurc_ci = summarize(
        aurc_values,
        seed=bootstrap_seed(seed, model, "aurc"),
        samples=bootstrap_samples,
    )
    ecl_mean, ecl_ci = summarize(
        [float(value) for value in ecl_values],
        seed=bootstrap_seed(seed, model, "eclAtTau"),
        samples=bootstrap_samples,
    )
    elicit_mean, elicit_ci = summarize(
        elicit_values,
        seed=bootstrap_seed(seed, model, "elicitAtK"),
        samples=bootstrap_samples,
    )
    return {
        "model": model,
        "evidenceMode": evidence_mode,
        "itemCount": len(runs),
        "aurc": aurc_mean,
        "aurcCi95": aurc_ci,
        "eclAtTau": ecl_mean,
        "eclAtTauCi95": ecl_ci,
        "coverageAtTau": round(len(ecl_values) / len(runs), 6),
        "elicitAtK": elicit_mean,
        "elicitAtKCi95": elicit_ci,
        "leakageHitRate": round(mean(leakage_values), 6),
    }


def result_notes(evidence_modes: list[str]) -> list[str]:
    notes: list[str] = []
    if "mock" in evidence_modes:
        notes.append(
            "Mock rows are deterministic pipeline evidence, not a real model leaderboard."
        )
    if "black_box" in evidence_modes:
        notes.append(
            "Hosted rows are black-box output evidence only; they do not claim logits or model internals."
        )
        notes.append(
            "Hosted response receipts retain original capture times and provider/cache source; createdAt is the run-start timestamp."
        )
    notes.extend(
        [
            "Leakage is a gate. Disqualified ladder prompts are excluded from AURC, ECL@tau, and Elicit@k.",
            "Raw decoded adapter strings are retained in measurements.outputs before scoring normalization.",
            "AURC is area under the non-leaking rate-distortion staircase; lower is better.",
            "Elicit@k uses k as a measured prompt-length budget, not a count of frontier points.",
        ]
    )
    return notes


def evaluate_item(
    item: dict[str, Any],
    adapter: ModelAdapter,
    *,
    seed: int,
    tau: float = DEFAULT_TAU,
    k: int = DEFAULT_K,
    reruns: int = DEFAULT_RERUNS,
    thresholds: dict[str, float | int] | None = None,
    created_at: str = FIXED_CREATED_AT,
) -> dict[str, Any]:
    validate_scoring_runtime()
    thresholds = dict(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
    validate_protocol_settings(tau=tau, k=k, reruns=reruns, thresholds=thresholds)
    validate_item_protocol(item)
    target = item["target"]["text"]
    explicit_tokens = max(token_count(item["frozenLadder"][0]["prompt"]), token_count(target), 1)
    configured_reruns = adapter.reruns(reruns)
    candidates: list[dict[str, Any]] = []
    measurements: list[dict[str, Any]] = []
    scored_points: list[dict[str, Any]] = []
    leakage_hits = 0

    for ladder in item["frozenLadder"]:
        prompt = ladder["prompt"]
        gate = evaluate_leakage(prompt, target, thresholds)
        leak = leakage_score(gate)
        outputs: list[str] = []
        response_receipts: list[dict[str, str]] = []
        if gate.disqualified:
            leakage_hits += 1
            output = ""
            fit = 0.0
            stable = 0.0
            fidelity_variance: float | None = None
            fidelity_stddev: float | None = None
            rerun_count = 0
            note = "Disqualified by leakage gate; excluded from compression metrics."
        else:
            for rerun_index in range(configured_reruns):
                outputs.append(
                    adapter.generate(
                        prompt,
                        item,
                        ladder,
                        seed=seed,
                        rerun_index=rerun_index,
                    )
                )
                if adapter.observation_mode == "black_box":
                    receipt = adapter.last_response_receipt()
                    if receipt is None:
                        raise ValueError(
                            f"black-box adapter {adapter.model_id!r} did not emit a response receipt"
                        )
                    response_receipts.append(receipt)
            invalid_output_types = [
                type(value).__name__ for value in outputs if not isinstance(value, str)
            ]
            if invalid_output_types:
                raise TypeError(
                    f"adapter {adapter.model_id!r} returned non-string output(s): "
                    + ", ".join(invalid_output_types)
                )
            fidelities = [fidelity(target, output, item["metricClass"]) for output in outputs]
            output = outputs[0]
            fit = round(mean(fidelities), 6)
            fidelity_variance = round(pvariance(fidelities), 6)
            fidelity_stddev = round(pstdev(fidelities), 6)
            rerun_count = len(fidelities)
            stable = 1.0 if len(fidelities) == 1 else round(max(0.0, 1.0 - fidelity_stddev), 6)
            note = (
                f"{adapter.observation_mode} evidence; scored with "
                f"{item['metricClass']} under protocol {PROTOCOL_VERSION}."
            )

        tokens = token_count(prompt)
        compression = round(max(0.0, min(1.0, 1.0 - tokens / explicit_tokens)), 6)
        measurement = {
            "promptId": ladder["id"],
            "rung": ladder["rung"],
            "paraphrase": ladder["paraphrase"],
            "tokens": tokens,
            "outputs": outputs,
            "responseReceipts": response_receipts,
            "rerunCount": rerun_count,
            "fidelityMean": fit if not gate.disqualified else None,
            "fidelityVariance": fidelity_variance,
            "fidelityStdDev": fidelity_stddev,
            "disqualified": gate.disqualified,
            "leakageGate": gate.as_dict(),
        }
        candidate = {
            "id": ladder["id"],
            "label": ladder["label"],
            "prompt": prompt,
            "output": output,
            "tokens": tokens,
            "fit": fit,
            "stability": stable,
            "compression": compression,
            "leakage": leak,
            "note": note,
        }
        point = {
            **candidate,
            "itemId": item["id"],
            "model": adapter.model_id,
            "rung": ladder["rung"],
            "paraphrase": ladder["paraphrase"],
            "distortion": round(1.0 - fit, 6),
            "fidelity": fit,
            "rerunCount": rerun_count,
            "fidelityVariance": fidelity_variance,
            "fidelityStdDev": fidelity_stddev,
            "disqualified": gate.disqualified,
            "leakageGate": gate.as_dict(),
            "evidenceMode": adapter.observation_mode,
        }
        measurements.append(measurement)
        candidates.append(candidate)
        scored_points.append(point)

    frontier = monotone_lower_envelope(scored_points)
    frontier_ids = {point["id"]: index + 1 for index, point in enumerate(frontier)}
    for candidate in candidates:
        if candidate["id"] in frontier_ids:
            candidate["frontierRank"] = frontier_ids[candidate["id"]]
    for point in frontier:
        point["frontierRank"] = frontier_ids[point["id"]]

    selected = frontier[0]["id"] if frontier else candidates[0]["id"]
    max_candidate_tokens = max(candidate["tokens"] for candidate in candidates)
    non_leaking_points = [point for point in scored_points if not point["disqualified"]]
    item_aurc = aurc(frontier, explicit_tokens)
    item_ecl = ecl_at_tau(frontier, tau)
    item_elicit = elicit_at_k(non_leaking_points, tau, k)
    leakage_hit_rate = round(leakage_hits / len(item["frozenLadder"]), 6)
    aleph_run = {
        "createdAt": created_at,
        "target": item["target"],
        "config": {
            "model": adapter.model_id,
            "decoding": decoding_for_evidence_mode(adapter.observation_mode),
            "metric": item["metricClass"],
            "budget": {
                "candidates": len(item["frozenLadder"]),
                "maxPromptTokens": max_candidate_tokens,
                "repeatedSamples": configured_reruns,
            },
            "mode": "non_leaking",
        },
        "candidates": candidates,
        "selectedCandidateId": selected,
        "observations": {
            "mode": adapter.observation_mode,
            "evalSuite": [
                {
                    "name": "leakage gate",
                    "passed": leakage_hits >= 1,
                    "score": leakage_hit_rate,
                    "note": "Rung 0 is expected to be disqualified as the explicit reconstruction anchor.",
                },
                {
                    "name": f"Elicit@{k}",
                    "passed": item_elicit,
                    "score": 1.0 if item_elicit else 0.0,
                    "note": f"Threshold tau={tau}.",
                },
            ],
        },
    }
    aleph_run["id"] = aleph_run_artifact_id(
        item_id=item["id"], seed=seed, artifact=aleph_run
    )
    return {
        "itemId": item["id"],
        "model": adapter.model_id,
        "alephRun": aleph_run,
        "measurements": measurements,
        "frontier": frontier,
        "metrics": {
            "aurc": item_aurc,
            "eclAtTau": item_ecl,
            "elicitAtK": item_elicit,
            "leakageHitRate": leakage_hit_rate,
        },
    }


def run_benchmark(
    *,
    data_dir: Path,
    models: list[str],
    split: str = "public",
    seed: int = 0,
    tau: float = DEFAULT_TAU,
    k: int = DEFAULT_K,
    reruns: int = DEFAULT_RERUNS,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    limit: int | None = None,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    validate_protocol_settings(
        split=split,
        tau=tau,
        k=k,
        reruns=reruns,
        bootstrap_samples=bootstrap_samples,
    )
    validate_scoring_runtime()
    if not models:
        raise ValueError("at least one model is required")
    canonical_models = sorted(normalize_model_id(model)[0] for model in models)
    if len(set(canonical_models)) != len(canonical_models):
        raise ValueError("models must have unique canonical logical ids")
    items = load_items(data_dir, limit=limit, require_canonical=True)
    canary = items[0]["canaryGuid"]
    item_runs: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    adapters: list[ModelAdapter] = []
    for model in canonical_models:
        adapter = adapter_for_model(model)
        if cache_dir is not None:
            adapter = ResponseCacheAdapter(adapter, cache_dir, seed=seed)
        adapters.append(adapter)
    logical_model_ids = [adapter.model_id for adapter in adapters]
    if len(set(logical_model_ids)) != len(logical_model_ids):
        raise ValueError("models must have unique canonical logical ids")
    evidence_modes = sorted({adapter.observation_mode for adapter in adapters})
    created_at = (
        FIXED_CREATED_AT
        if evidence_modes == ["mock"]
        else datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )

    for adapter in adapters:
        runs = [
            evaluate_item(
                item,
                adapter,
                seed=seed,
                tau=tau,
                k=k,
                reruns=reruns,
                created_at=created_at,
            )
            for item in items
        ]
        item_runs.extend(runs)
        summary = summarize_model_runs(
            model=adapter.model_id,
            evidence_mode=adapter.observation_mode,
            runs=runs,
            seed=seed,
            bootstrap_samples=bootstrap_samples,
        )
        capture_receipts = [
            receipt
            for run in runs
            for measurement in run["measurements"]
            for receipt in measurement["responseReceipts"]
        ]
        response_capture: dict[str, Any] | None = None
        if adapter.observation_mode == "black_box":
            captured_at = [receipt["capturedAt"] for receipt in capture_receipts]
            response_capture = {
                "responseCount": len(capture_receipts),
                "providerResponseCount": sum(
                    receipt["source"] == "provider" for receipt in capture_receipts
                ),
                "cacheHitCount": sum(
                    receipt["source"] == "cache" for receipt in capture_receipts
                ),
                "capturedAtMin": min(captured_at),
                "capturedAtMax": max(captured_at),
            }
        summary["adapterIdentity"] = adapter.evidence_identity()
        summary["responseCapture"] = response_capture
        summaries.append(summary)

    metric_classes = sorted({item["metricClass"] for item in items})
    evaluation_scope = "canonical" if limit is None else "smoke"
    evaluated_item_count = len(items)
    result = {
        "protocolVersion": PROTOCOL_VERSION,
        "evaluationScope": evaluation_scope,
        "requestedItemLimit": limit,
        "evaluatedItemCount": evaluated_item_count,
        "createdAt": created_at,
        "track": FROZEN_TRACK,
        "split": FROZEN_SPLIT,
        "seed": seed,
        "stratum": FROZEN_STRATUM,
        "metricClasses": metric_classes,
        "tau": tau,
        "k": k,
        "canaryGuid": canary,
        "config": {
            "datasetPath": stable_dataset_path(data_dir),
            "datasetId": FROZEN_DATASET_ID,
            "datasetItemCount": FROZEN_DATASET_ITEM_COUNT,
            "datasetSha256": FROZEN_DATASET_SHA256,
            "datasetHashAlgorithm": FROZEN_DATASET_HASH_ALGORITHM,
            "decoding": dict(FROZEN_DECODING),
            "evidenceModes": evidence_modes,
            "leakageThresholds": dict(DEFAULT_THRESHOLDS),
            "scoring": scoring_profile(),
            "bootstrapSamples": bootstrap_samples,
            "reruns": reruns,
        },
        "models": summaries,
        "itemRuns": item_runs,
        "notes": result_notes(evidence_modes),
    }
    result["id"] = result_artifact_id(result)
    return result
