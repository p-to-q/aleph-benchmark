from __future__ import annotations

from pathlib import Path
from typing import Any

from .adapters import normalize_model_id
from .adapters.hosted_black_box import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_DELAY_SECONDS,
    configured_hosted_max_retries,
    configured_hosted_retry_delay_seconds,
)
from .frozen_ladder import DEFAULT_RERUNS, load_items, stable_dataset_path
from .leakage_gate import DEFAULT_THRESHOLDS, evaluate_leakage
from .metrics import token_count
from .preflight import model_readiness
from .protocol import (
    FROZEN_DATASET_HASH_ALGORITHM,
    FROZEN_DATASET_ID,
    FROZEN_DATASET_ITEM_COUNT,
    FROZEN_DATASET_SHA256,
    FROZEN_SPLIT,
    FROZEN_TRACK,
    PROTOCOL_VERSION,
    manifest_artifact_id,
    scoring_profile,
    validate_protocol_settings,
    validate_scoring_runtime,
)


FIXED_CREATED_AT = "2026-09-17T00:00:00Z"


def manifest_notes() -> list[str]:
    return [
        "Manifest lists only non-leaking prompts that would be sent to model adapters.",
        "Gated prompts are retained as IDs and leakage measurements, without prompt text.",
        "Manifest generation does not call any model and is not model evidence.",
        "estimatedGenerations counts logical samples; estimatedMaxHttpAttempts is a retry-policy upper bound, not observed billing.",
        "Hosted adapter identities retain endpoint hash, provider model, deployment receipt, and transport policy without credentials.",
    ]


def build_prompt_receipts(
    items: list[dict[str, Any]],
    *,
    thresholds: dict[str, float | int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build canonical sendable and gated prompt receipts without model calls."""

    active_thresholds = dict(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
    validate_protocol_settings(thresholds=active_thresholds)
    prompts: list[dict[str, Any]] = []
    gated_prompts: list[dict[str, Any]] = []
    for item in items:
        target = item["target"]["text"]
        for ladder in item["frozenLadder"]:
            gate = evaluate_leakage(ladder["prompt"], target, active_thresholds)
            row = {
                "itemId": item["id"],
                "targetLabel": item["target"].get("label"),
                "promptId": ladder["id"],
                "rung": ladder["rung"],
                "paraphrase": ladder["paraphrase"],
                "label": ladder["label"],
                "tokens": token_count(ladder["prompt"]),
                "leakageGate": gate.as_dict(),
            }
            if gate.disqualified:
                gated_prompts.append(row)
            else:
                prompts.append({**row, "prompt": ladder["prompt"]})
    return prompts, gated_prompts


def build_manifest(
    *,
    data_dir: Path,
    models: list[str],
    split: str = "public",
    seed: int = 0,
    reruns: int = DEFAULT_RERUNS,
    thresholds: dict[str, float | int] | None = None,
) -> dict[str, Any]:
    active_thresholds = dict(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
    validate_protocol_settings(
        split=split,
        reruns=reruns,
        thresholds=active_thresholds,
    )
    validate_scoring_runtime()
    if not models:
        raise ValueError("at least one model is required")
    canonical_models = sorted(normalize_model_id(model)[0] for model in models)
    if len(set(canonical_models)) != len(canonical_models):
        raise ValueError("model ids must have unique canonical identities")
    model_rows = [model_readiness(model, reruns=reruns) for model in canonical_models]
    blocked_rows = [row for row in model_rows if row.get("status") != "ready"]
    if blocked_rows:
        planning_errors: list[str] = []
        for row in blocked_rows:
            details: list[str] = []
            if isinstance(row.get("error"), str) and row["error"]:
                details.append(row["error"])
            missing_env = row.get("missingEnv")
            if isinstance(missing_env, list) and missing_env:
                details.append("missing env: " + ", ".join(missing_env))
            if not details:
                details.append(f"status={row.get('status')!r}")
            planning_errors.append(
                f"{row.get('model', '<unknown model>')}: " + "; ".join(details)
            )
        raise ValueError("manifest planning failed: " + "; ".join(planning_errors))
    items = load_items(data_dir, require_canonical=True)
    prompts, gated_prompts = build_prompt_receipts(
        items,
        thresholds=active_thresholds,
    )
    hosted_max_retries = (
        configured_hosted_max_retries()
        if any(row["evidenceMode"] == "black_box" for row in model_rows)
        else DEFAULT_MAX_RETRIES
    )
    hosted_retry_delay_seconds = (
        configured_hosted_retry_delay_seconds()
        if any(row["evidenceMode"] == "black_box" for row in model_rows)
        else DEFAULT_RETRY_DELAY_SECONDS
    )
    hosted_generations = len(prompts) * sum(
        row["effectiveReruns"]
        for row in model_rows
        if row["evidenceMode"] == "black_box"
    )
    manifest = {
        "protocolVersion": PROTOCOL_VERSION,
        "createdAt": FIXED_CREATED_AT,
        "track": FROZEN_TRACK,
        "split": FROZEN_SPLIT,
        "seed": seed,
        "datasetPath": stable_dataset_path(data_dir),
        "datasetId": FROZEN_DATASET_ID,
        "datasetItemCount": FROZEN_DATASET_ITEM_COUNT,
        "datasetSha256": FROZEN_DATASET_SHA256,
        "datasetHashAlgorithm": FROZEN_DATASET_HASH_ALGORITHM,
        "itemCount": len(items),
        "modelCount": len(canonical_models),
        "models": model_rows,
        "configuredReruns": reruns,
        "promptCount": sum(len(item["frozenLadder"]) for item in items),
        "nonLeakingPromptCount": len(prompts),
        "gatedPromptCount": len(gated_prompts),
        "estimatedGenerations": len(prompts)
        * sum(row["effectiveReruns"] for row in model_rows),
        "hostedMaxRetries": hosted_max_retries,
        "hostedRetryDelaySeconds": hosted_retry_delay_seconds,
        "estimatedMaxHttpAttempts": hosted_generations
        * (hosted_max_retries + 1),
        "leakageThresholds": active_thresholds,
        "scoring": scoring_profile(),
        "prompts": prompts,
        "gatedPrompts": gated_prompts,
        "notes": manifest_notes(),
    }
    manifest["id"] = manifest_artifact_id(manifest)
    return manifest
