from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .frozen_ladder import DEFAULT_RERUNS, load_items, stable_dataset_path
from .adapters import HostedBlackBoxAdapter, MockAdapter, normalize_model_id
from .adapters.hosted_black_box import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_DELAY_SECONDS,
    configured_hosted_max_retries,
    configured_hosted_retry_delay_seconds,
    validate_hosted_base_url,
)
from .leakage_gate import DEFAULT_THRESHOLDS, evaluate_leakage
from .schema_validation import SchemaValidationError, load_schema, validate
from .protocol import validate_protocol_settings, validate_scoring_runtime
from .protocol import (
    FROZEN_DATASET_HASH_ALGORITHM,
    FROZEN_DATASET_ID,
    FROZEN_DATASET_ITEM_COUNT,
    FROZEN_DATASET_SHA256,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
HOSTED_REQUIRED_ENV = [
    "ALEPH_CUSTOM_API_BASE_URL",
    "ALEPH_CUSTOM_API_KEY",
    "ALEPH_CUSTOM_API_DEPLOYMENT_ID",
]


def model_readiness(model: str, *, reruns: int = DEFAULT_RERUNS) -> dict[str, Any]:
    try:
        canonical, evidence_mode, uses_configured_reruns = normalize_model_id(model)
    except ValueError as exc:
        return {
            "model": model,
            "evidenceMode": "unknown",
            "status": "blocked",
            "missingEnv": [],
            "effectiveReruns": 0,
            "adapterIdentity": None,
            "error": str(exc),
        }
    if not uses_configured_reruns:
        return {
            "model": canonical,
            "evidenceMode": evidence_mode,
            "status": "ready",
            "missingEnv": [],
            "effectiveReruns": 1,
            "adapterIdentity": MockAdapter(canonical).evidence_identity(),
        }
    missing = [name for name in HOSTED_REQUIRED_ENV if not os.environ.get(name)]
    if not missing:
        try:
            validate_hosted_base_url(os.environ["ALEPH_CUSTOM_API_BASE_URL"])
            adapter_identity = HostedBlackBoxAdapter(
                canonical.removeprefix("hosted:")
            ).evidence_identity()
        except (RuntimeError, ValueError) as exc:
            return {
                "model": canonical,
                "evidenceMode": evidence_mode,
                "status": "blocked",
                "missingEnv": [],
                "effectiveReruns": reruns,
                "adapterIdentity": None,
                "error": str(exc),
            }
    else:
        adapter_identity = None
    return {
        "model": canonical,
        "evidenceMode": evidence_mode,
        "status": "ready" if not missing else "blocked",
        "missingEnv": missing,
        "effectiveReruns": reruns,
        "adapterIdentity": adapter_identity,
    }


def preflight(
    *,
    data_dir: Path,
    models: list[str],
    reruns: int = DEFAULT_RERUNS,
    thresholds: dict[str, float | int] | None = None,
) -> dict[str, Any]:
    active_thresholds = dict(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
    item_schema = load_schema(REPO_ROOT / "schemas/v0.2/aleph-bench-item.schema.json")
    errors: list[str] = []
    has_hosted_model = any(model.startswith("hosted:") for model in models)
    hosted_max_retries = DEFAULT_MAX_RETRIES
    hosted_retry_delay_seconds = DEFAULT_RETRY_DELAY_SECONDS
    retry_policy_valid = True
    try:
        if has_hosted_model:
            hosted_max_retries = configured_hosted_max_retries()
            hosted_retry_delay_seconds = configured_hosted_retry_delay_seconds()
        validate_protocol_settings(reruns=reruns, thresholds=active_thresholds)
        validate_scoring_runtime()
        items = load_items(data_dir, require_canonical=True)
    except (OSError, RuntimeError, ValueError) as exc:
        items = []
        errors.append(str(exc))
        retry_policy_valid = not (
            has_hosted_model and str(exc).startswith("ALEPH_CUSTOM_API_")
        )
    leakage_by_rung = {"0": 0, "1": 0, "2": 0, "3": 0}
    prompt_count = 0
    disqualified_count = 0

    for item in items:
        try:
            validate(item, item_schema)
        except SchemaValidationError as exc:
            errors.append(f"{item.get('id', '<unknown>')}: {exc}")
            continue
        target = item["target"]["text"]
        for ladder in item["frozenLadder"]:
            prompt_count += 1
            gate = evaluate_leakage(ladder["prompt"], target, active_thresholds)
            if gate.disqualified:
                disqualified_count += 1
                leakage_by_rung[str(ladder["rung"])] += 1

    model_rows = [model_readiness(model, reruns=reruns) for model in models]
    canonical_model_ids = [
        row["model"] for row in model_rows if row["evidenceMode"] != "unknown"
    ]
    if len(set(canonical_model_ids)) != len(canonical_model_ids):
        errors.append("model ids must have unique canonical identities")
    non_leaking_prompt_count = prompt_count - disqualified_count
    estimated_generations = non_leaking_prompt_count * sum(
        row["effectiveReruns"] for row in model_rows
    )
    hosted_generations = non_leaking_prompt_count * sum(
        row["effectiveReruns"]
        for row in model_rows
        if row["evidenceMode"] == "black_box"
    )
    estimated_max_http_attempts = (
        hosted_generations * (hosted_max_retries + 1)
        if retry_policy_valid
        else None
    )
    blocked_models = [row for row in model_rows if row["status"] != "ready"]
    status = "ready" if not errors and not blocked_models else "blocked"
    return {
        "status": status,
        "datasetPath": stable_dataset_path(data_dir),
        "datasetId": FROZEN_DATASET_ID,
        "datasetItemCount": FROZEN_DATASET_ITEM_COUNT,
        "datasetSha256": FROZEN_DATASET_SHA256,
        "datasetHashAlgorithm": FROZEN_DATASET_HASH_ALGORITHM,
        "itemCount": len(items),
        "promptCount": prompt_count,
        "disqualifiedPromptCount": disqualified_count,
        "nonLeakingPromptCount": non_leaking_prompt_count,
        "leakageByRung": leakage_by_rung,
        "configuredReruns": reruns,
        "estimatedGenerations": estimated_generations,
        "hostedMaxRetries": hosted_max_retries if retry_policy_valid else None,
        "hostedRetryDelaySeconds": (
            hosted_retry_delay_seconds if retry_policy_valid else None
        ),
        "estimatedMaxHttpAttempts": estimated_max_http_attempts,
        "models": model_rows,
        "errors": errors,
    }


def format_preflight(report: dict[str, Any]) -> str:
    lines = [
        f"status: {report['status']}",
        f"dataset: {report['datasetPath']}",
        f"items: {report['itemCount']}",
        f"prompts: {report['promptCount']} ({report['nonLeakingPromptCount']} non-leaking, {report['disqualifiedPromptCount']} gated)",
        f"leakage by rung: {json.dumps(report['leakageByRung'], sort_keys=True)}",
        f"estimated logical generations: {report['estimatedGenerations']}",
        f"hosted max retries per logical generation: {report['hostedMaxRetries']}",
        f"hosted retry delay seconds: {report['hostedRetryDelaySeconds']}",
        f"estimated max HTTP attempts: {report['estimatedMaxHttpAttempts']}",
    ]
    for row in report["models"]:
        suffix = ""
        if row.get("missingEnv"):
            suffix = f" missing env: {', '.join(row['missingEnv'])}"
        if row.get("error"):
            suffix = f" error: {row['error']}"
        lines.append(
            f"model {row['model']} [{row['evidenceMode']}]: {row['status']} "
            f"(effective reruns: {row['effectiveReruns']}){suffix}"
        )
    for error in report["errors"]:
        lines.append(f"error: {error}")
    return "\n".join(lines)
