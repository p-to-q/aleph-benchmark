from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema_validation import load_schema, validate


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PROTOCOL_VERSION = "0.2.0"
RESULT_SCHEMA_PATH = REPO_ROOT / "schemas/v0.2/aleph-bench-result.schema.json"


def _ci(value: dict[str, float] | None) -> str:
    if value is None:
        return "n/a"
    return f"{value['low']}-{value['high']}"


def _num(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def _require_v0_2_result(value: Any) -> dict[str, Any]:
    """Reject non-v0.2 inputs before selecting a result schema or rendering."""

    if not isinstance(value, dict):
        raise ValueError("Aleph-Bench result must be a JSON object")
    protocol_version = value.get("protocolVersion")
    if protocol_version != RESULT_PROTOCOL_VERSION:
        raise ValueError(
            "unsupported Aleph-Bench result protocolVersion: "
            f"expected {RESULT_PROTOCOL_VERSION!r}, found {protocol_version!r}"
        )
    return value


def _validate_v0_2_result(value: Any) -> dict[str, Any]:
    result = _require_v0_2_result(value)
    validate(result, load_schema(RESULT_SCHEMA_PATH))
    return result


def load_result(result_path: Path) -> dict[str, Any]:
    parsed = json.loads(result_path.read_text(encoding="utf-8"))
    return _validate_v0_2_result(parsed)


def model_summary_table(result: dict[str, Any]) -> str:
    lines = [
        "| Model | Evidence | AURC | 95% CI | ECL@tau | 95% CI | Coverage@tau | Elicit@k | Leakage hits |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in result["models"]:
        lines.append(
            " | ".join(
                [
                    f"| {model['model']}",
                    model["evidenceMode"],
                    _num(model["aurc"]),
                    _ci(model["aurcCi95"]),
                    _num(model["eclAtTau"]),
                    _ci(model["eclAtTauCi95"]),
                    _num(model["coverageAtTau"]),
                    _num(model["elicitAtK"]),
                    f"{_num(model['leakageHitRate'])} |",
                ]
            )
        )
    return "\n".join(lines)


def per_item_table(result: dict[str, Any]) -> str:
    models = [model["model"] for model in result["models"]]
    by_item: dict[str, dict[str, dict[str, Any]]] = {}
    for run in result["itemRuns"]:
        by_item.setdefault(run["itemId"], {})[run["model"]] = run["metrics"]

    header = "| Item | " + " | ".join(models) + " |"
    align = "|---|" + "|".join("---:" for _ in models) + "|"
    lines = [header, align]
    for item_id in sorted(by_item):
        cells = []
        for model in models:
            metrics = by_item[item_id][model]
            cells.append(f"{metrics['aurc']:.3f} / {metrics['eclAtTau']}")
        lines.append(f"| {item_id} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def hosted_provenance_table(result: dict[str, Any]) -> str:
    lines = [
        "| Model | Deployment | Endpoint SHA-256 | Captured | Provider / cache |",
        "|---|---|---|---|---:|",
    ]
    for model in result["models"]:
        if model["evidenceMode"] != "black_box":
            continue
        identity = model["adapterIdentity"]
        capture = model["responseCapture"]
        lines.append(
            " | ".join(
                [
                    f"| {model['model']}",
                    identity["deploymentId"],
                    f"`{identity['endpointSha256']}`",
                    f"{capture['capturedAtMin']} — {capture['capturedAtMax']}",
                    f"{capture['providerResponseCount']} / {capture['cacheHitCount']} |",
                ]
            )
        )
    return "\n".join(lines)


def render_markdown_report(result: dict[str, Any]) -> str:
    result = _validate_v0_2_result(result)
    evidence_modes = sorted({model["evidenceMode"] for model in result["models"]})
    protocol_version = result["protocolVersion"]
    metric_classes = ", ".join(result.get("metricClasses", []))
    dataset_path = result.get("config", {}).get("datasetPath", "<unknown>")
    lines = [
        "# Aleph-Bench Result Report",
        "",
        f"- Result id: `{result['id']}`",
        f"- Protocol: `{protocol_version}`",
        f"- Evaluation scope: `{result['evaluationScope']}`",
        f"- Evaluated items: `{result['evaluatedItemCount']} / {result['config']['datasetItemCount']}`",
        f"- Track: `{result['track']}`",
        f"- Split: `{result['split']}`",
        f"- Stratum: `{result['stratum']}`",
        f"- Dataset: `{dataset_path}`",
        f"- Metric classes: `{metric_classes}`",
        f"- Seed: `{result['seed']}`",
        f"- Tau: `{result['tau']}`",
        f"- K: `{result['k']}`",
        f"- Evidence modes: `{', '.join(evidence_modes)}`",
    ]
    scoring = result.get("config", {}).get("scoring")
    if scoring:
        lines.extend(
            [
                f"- Scorer: `{scoring['scorerId']}@{scoring['scorerVersion']}`",
                f"- Normalization: `{scoring['normalizationProfile']}`",
                f"- Lexical profile: `{scoring['lexicalProfile']}`",
                f"- Runtime: `Python {scoring['pythonVersion']} / Unicode {scoring['unicodeDatabaseVersion']}`",
            ]
        )
    lines.extend(
        [
            "",
            "Lower AURC and ECL are better. This report is generated from the result JSON; it does not add evidence beyond that file.",
        ]
    )
    if result["evaluationScope"] == "smoke":
        lines.extend(
            [
                "",
                "This is a diagnostic smoke run over "
                f"{result['evaluatedItemCount']} / {result['config']['datasetItemCount']} items. "
                "It is not a canonical benchmark result or a leaderboard score.",
            ]
        )
    if set(evidence_modes) <= {"mock", "fixture", "simulated"}:
        lines.extend(
            [
                "",
                "This result contains only mock, fixture, or simulated evidence. It is pipeline evidence, not a model leaderboard.",
            ]
        )
    lines.extend(
        [
            "",
            "## Model Summary",
            "",
            model_summary_table(result),
            "",
            "## Per-Item Receipt",
            "",
            "Cells are `AURC / ECL@tau`.",
            "",
            per_item_table(result),
            "",
        ]
    )
    if "black_box" in evidence_modes:
        lines.extend(
            [
                "## Hosted Evidence Provenance",
                "",
                hosted_provenance_table(result),
                "",
                "Capture times describe retained provider outputs; result `createdAt` is the run-start timestamp.",
                "",
            ]
        )
    if result.get("notes"):
        lines.extend(["## Result Notes", ""])
        lines.extend(f"- {note}" for note in result["notes"])
        lines.append("")
    return "\n".join(lines)


def render_report_file(result_path: Path) -> str:
    return render_markdown_report(load_result(result_path))
