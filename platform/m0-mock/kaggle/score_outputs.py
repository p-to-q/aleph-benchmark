"""Score a Kaggle / local submission against the M0 frozen ladder.

Inputs:
- items JSONL (`data/public_s2_items.jsonl`)
- prompts JSONL (`data/public_s2_prompts.jsonl`)
- submission rows (CSV path or list of dicts with row_id, model_id, item_id, prompt_id, output_text)

Output: a `BenchResult`-compatible dict (subset of the strict schema; intended
for community use, not as a substitute for `./aleph-bench verify`).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

KAGGLE_DIR = Path(__file__).resolve().parent
if str(KAGGLE_DIR) not in sys.path:
    sys.path.insert(0, str(KAGGLE_DIR))

import _scoring  # noqa: E402


DEFAULT_TAU = 0.9
DEFAULT_K = 3
DEFAULT_BOOTSTRAP_SAMPLES = 500
DEFAULT_SEED = 0


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_submission_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _index_outputs(
    submission_rows: Iterable[dict[str, Any]],
    *,
    valid_prompt_keys: set[tuple[str, str]],
    expected_model_id: str,
) -> dict[tuple[str, str], str]:
    outputs: dict[tuple[str, str], str] = {}
    for row in submission_rows:
        item_id = str(row["item_id"]) if "item_id" in row else None
        prompt_id = str(row["prompt_id"]) if "prompt_id" in row else None
        row_id = str(row.get("row_id") or "")
        row_model_id = str(row.get("model_id") or "")
        text = str(row.get("output_text") or "")
        if item_id is None or prompt_id is None:
            raise ValueError(f"submission row missing item_id/prompt_id: {row!r}")
        key = (item_id, prompt_id)
        expected_row_id = f"{item_id}:{prompt_id}"
        if row_id and row_id != expected_row_id:
            raise ValueError(f"submission row_id mismatch: expected {expected_row_id!r}, found {row_id!r}")
        if row_model_id and row_model_id != expected_model_id:
            raise ValueError(
                f"submission model_id mismatch: expected {expected_model_id!r}, found {row_model_id!r}"
            )
        if key not in valid_prompt_keys:
            raise ValueError(f"submission row references unknown or gated prompt: {expected_row_id}")
        if key in outputs:
            raise ValueError(f"duplicate submission row: {expected_row_id}")
        outputs[key] = text
    return outputs


def _index_prompts(prompt_rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(row["item_id"]), str(row["prompt_id"])): row for row in prompt_rows}


def score_submission(
    *,
    items_path: str | Path,
    prompts_path: str | Path,
    submission_rows: Iterable[dict[str, Any]] | str | Path,
    model_id: str,
    tau: float = DEFAULT_TAU,
    k: int = DEFAULT_K,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_SEED,
    leakage_thresholds: dict[str, float | int] | None = None,
) -> dict[str, Any]:
    items_path = Path(items_path)
    prompts_path = Path(prompts_path)
    items = _load_jsonl(items_path)
    prompt_rows = _load_jsonl(prompts_path)
    if isinstance(submission_rows, (str, Path)):
        submission_rows = _load_submission_rows(Path(submission_rows))
    prompt_index = _index_prompts(prompt_rows)
    valid_prompt_keys = {
        key for key, row in prompt_index.items() if not bool(row.get("disqualified"))
    }
    outputs = _index_outputs(
        submission_rows,
        valid_prompt_keys=valid_prompt_keys,
        expected_model_id=model_id,
    )
    thresholds = dict(_scoring.DEFAULT_LEAKAGE_THRESHOLDS if leakage_thresholds is None else leakage_thresholds)

    item_runs: list[dict[str, Any]] = []
    aurc_values: list[float] = []
    ecl_values: list[float] = []
    elicit_values: list[float] = []
    leakage_values: list[float] = []
    metric_classes: set[str] = set()

    for item in items:
        target = item["target"]["text"]
        metric_class = item["metricClass"]
        metric_classes.add(metric_class)
        explicit_prompt = item["frozenLadder"][0]["prompt"]
        explicit_tokens = max(
            _scoring.token_count(explicit_prompt), _scoring.token_count(target), 1
        )
        scored_points: list[dict[str, Any]] = []
        leakage_hits = 0
        for ladder in item["frozenLadder"]:
            prompt_text = ladder["prompt"]
            key = (str(item["id"]), str(ladder["id"]))
            packed_prompt = prompt_index.get(key)
            gate = _scoring.evaluate_leakage(prompt_text, target, thresholds)
            disqualified = bool((packed_prompt or {}).get("disqualified", gate.disqualified))
            if disqualified:
                leakage_hits += 1
                fid = 0.0
            elif key not in outputs:
                fid = 0.0
            else:
                fid = _scoring.fidelity(target, outputs[key], metric_class)
            tokens = _scoring.token_count(prompt_text)
            scored_points.append(
                {
                    "id": ladder["id"],
                    "tokens": tokens,
                    "fidelity": fid,
                    "distortion": round(1.0 - fid, 6),
                    "disqualified": disqualified,
                    "rung": ladder["rung"],
                    "paraphrase": ladder["paraphrase"],
                }
            )
        frontier = _scoring.monotone_lower_envelope(scored_points)
        item_aurc = _scoring.aurc(frontier, explicit_tokens)
        item_ecl = _scoring.ecl_at_tau(frontier, tau)
        non_leaking = [point for point in scored_points if not point["disqualified"]]
        item_elicit = _scoring.elicit_at_k(non_leaking, tau, k)
        leakage_hit_rate = round(leakage_hits / max(1, len(scored_points)), 6)
        aurc_values.append(item_aurc)
        if item_ecl is not None:
            ecl_values.append(float(item_ecl))
        elicit_values.append(1.0 if item_elicit else 0.0)
        leakage_values.append(leakage_hit_rate)
        item_runs.append(
            {
                "itemId": item["id"],
                "model": model_id,
                "frontier": frontier,
                "metrics": {
                    "aurc": item_aurc,
                    "eclAtTau": item_ecl,
                    "elicitAtK": item_elicit,
                    "leakageHitRate": leakage_hit_rate,
                },
            }
        )

    aurc_mean, aurc_ci = _scoring.summarize(aurc_values, seed=seed + 1000, samples=bootstrap_samples)
    ecl_mean, ecl_ci = _scoring.summarize(ecl_values, seed=seed + 2000, samples=bootstrap_samples)
    elicit_mean, elicit_ci = _scoring.summarize(
        elicit_values, seed=seed + 3000, samples=bootstrap_samples
    )

    return {
        "id": f"aleph-bench-m0-{model_id}",
        "track": "F",
        "split": "public",
        "seed": seed,
        "stratum": "S2",
        "metricClasses": sorted(metric_classes),
        "tau": tau,
        "k": k,
        "canaryGuid": items[0]["canaryGuid"] if items else None,
        "config": {
            "datasetPath": str(items_path),
            "decoding": "user-supplied; this scorer consumes pre-generated outputs",
            "evidenceModes": ["black_box"],
            "leakageThresholds": thresholds,
            "bootstrapSamples": bootstrap_samples,
            "reruns": 1,
        },
        "models": [
            {
                "model": model_id,
                "evidenceMode": "black_box",
                "itemCount": len(item_runs),
                "aurc": aurc_mean,
                "aurcCi95": aurc_ci,
                "eclAtTau": ecl_mean,
                "eclAtTauCi95": ecl_ci,
                "coverageAtTau": round(len(ecl_values) / max(1, len(item_runs)), 6),
                "elicitAtK": elicit_mean,
                "elicitAtKCi95": elicit_ci,
                "leakageHitRate": round(mean(leakage_values) if leakage_values else 0.0, 6),
            }
        ],
        "aggregate": {
            "aurc": aurc_mean,
            "aurcCi95": aurc_ci,
            "eclAtTau": ecl_mean,
            "elicitAtK": elicit_mean,
        },
        "itemRuns": item_runs,
        "notes": [
            "Computed by the vendored kaggle/score_outputs.py scorer.",
            "For full schema validation and audit, re-run ./aleph-bench verify in the upstream Aleph repository.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items-path", default=str(KAGGLE_DIR.parent / "data/public_s2_items.jsonl"))
    parser.add_argument("--prompts-path", default=str(KAGGLE_DIR.parent / "data/public_s2_prompts.jsonl"))
    parser.add_argument("--submission-csv", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    result = score_submission(
        items_path=args.items_path,
        prompts_path=args.prompts_path,
        submission_rows=args.submission_csv,
        model_id=args.model_id,
        tau=args.tau,
        k=args.k,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(payload, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
