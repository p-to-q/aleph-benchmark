"""Local smoke test for the Aleph-Bench M0 Kaggle API-test wrapper.

This script deliberately uses a stub LLM. It proves two contracts before any
hosted Kaggle model access exists:

  1. The I/O contract: aleph_bench_m0_task.run_black_box_model produces one
     submission row per non-leaking ladder prompt, with the exact public
     submission shape.
  2. The vendored scorer contract: score_outputs.score_submission consumes
     those rows, returns a schema-shaped BenchResult, and emits well-formed
     metric values (aurc in [0, 1], monotone non-leaking frontier, rung-0
     prompts gated by the leakage gate).

Neither contract produces model evidence or leaderboard rows.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import sys


KAGGLE_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = KAGGLE_DIR.parent
SUBMISSION_PATH = PACKAGE_ROOT / "data/submission_format.csv"
ITEMS_PATH = PACKAGE_ROOT / "data/public_s2_items.jsonl"
PROMPTS_PATH = PACKAGE_ROOT / "data/public_s2_prompts.jsonl"


def cleanup_runtime_cache() -> None:
    shutil.rmtree(KAGGLE_DIR / "__pycache__", ignore_errors=True)


cleanup_runtime_cache()
sys.dont_write_bytecode = True
sys.path.insert(0, str(KAGGLE_DIR))

import aleph_bench_m0_task  # noqa: E402
import score_outputs  # noqa: E402


class StubLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def prompt(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return f"stub-output-{len(self.prompts):03d}"


def load_submission_rows() -> list[dict[str, str]]:
    with SUBMISSION_PATH.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def row_key(row: dict[str, object]) -> tuple[str, str, str]:
    return (str(row["row_id"]), str(row["item_id"]), str(row["prompt_id"]))


def _check_metric_invariants(bench_result: dict, errors: list[str]) -> None:
    aggregate = bench_result.get("aggregate") or {}
    aurc = aggregate.get("aurc")
    if aurc is None or not (0.0 <= float(aurc) <= 1.0):
        errors.append(f"aggregate.aurc out of [0,1]: {aurc!r}")

    item_runs = bench_result.get("itemRuns") or []
    if not item_runs:
        errors.append("score_submission returned no itemRuns")
        return

    for run in item_runs:
        item_id = run.get("itemId")
        frontier = run.get("frontier") or []
        previous_distortion: float | None = None
        previous_tokens: int | None = None
        for point in frontier:
            tokens = int(point["tokens"])
            distortion = float(point["distortion"])
            if previous_tokens is not None and tokens < previous_tokens:
                errors.append(f"{item_id}: frontier tokens not monotone non-decreasing")
                break
            if previous_distortion is not None and distortion > previous_distortion + 1e-9:
                errors.append(f"{item_id}: frontier distortion not monotone non-increasing")
                break
            previous_tokens = tokens
            previous_distortion = distortion

        leakage_hit_rate = run.get("metrics", {}).get("leakageHitRate")
        if leakage_hit_rate is None or float(leakage_hit_rate) < 0.25 - 1e-9:
            errors.append(
                f"{item_id}: leakage hit rate {leakage_hit_rate!r} below 0.25 — rung-0 anchors should be gated"
            )


def main() -> None:
    expected_rows = load_submission_rows()
    prompts = aleph_bench_m0_task.load_sendable_prompts()
    llm = StubLLM()
    observed_rows = aleph_bench_m0_task.run_black_box_model("stub/model", llm)

    errors: list[str] = []
    if len(prompts) != 180:
        errors.append(f"expected 180 non-leaking prompts, found {len(prompts)}")
    if len(expected_rows) != 180:
        errors.append(f"expected 180 submission rows, found {len(expected_rows)}")
    if len(observed_rows) != len(expected_rows):
        errors.append(f"expected {len(expected_rows)} outputs, found {len(observed_rows)}")
    if len(llm.prompts) != len(expected_rows):
        errors.append(f"expected {len(expected_rows)} prompt() calls, found {len(llm.prompts)}")
    if any(row.get("disqualified") for row in prompts):
        errors.append("load_sendable_prompts returned at least one disqualified prompt")

    expected_keys = [row_key(row) for row in expected_rows]
    observed_keys = [row_key(row) for row in observed_rows]
    if observed_keys != expected_keys:
        errors.append("observed output row order or ids do not match submission_format.csv")

    required_fields = {"row_id", "model_id", "item_id", "prompt_id", "output_text"}
    for index, row in enumerate(observed_rows):
        if set(row) != required_fields:
            errors.append(f"row {index} has fields {sorted(row)}, expected {sorted(required_fields)}")
            break
        if row["model_id"] != "stub/model":
            errors.append(f"row {index} has unexpected model_id {row['model_id']!r}")
            break
        if not row["output_text"]:
            errors.append(f"row {index} has empty output_text")
            break

    bench_result = score_outputs.score_submission(
        items_path=str(ITEMS_PATH),
        prompts_path=str(PROMPTS_PATH),
        submission_rows=observed_rows,
        model_id="stub/model",
    )
    _check_metric_invariants(bench_result, errors)
    aggregate_aurc = bench_result.get("aggregate", {}).get("aurc")

    report = {
        "status": "failed" if errors else "ok",
        "packageRoot": str(PACKAGE_ROOT),
        "sendablePrompts": len(prompts),
        "submissionRows": len(expected_rows),
        "promptCalls": len(llm.prompts),
        "scoredItems": len(bench_result.get("itemRuns") or []),
        "aggregateAurc": aggregate_aurc,
        "firstRow": observed_rows[0] if observed_rows else None,
        "errors": errors,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_runtime_cache()
