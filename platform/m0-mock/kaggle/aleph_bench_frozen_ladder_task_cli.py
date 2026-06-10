"""Kaggle CLI task source for AlephBench Frozen Ladder.

Push with:

    kaggle benchmarks tasks push aleph-bench-frozen-ladder \
      -f kaggle/aleph_bench_frozen_ladder_task_cli.py \
      -d jahyee/alephbench-frozen-ladder-benchmark-package

The file intentionally invokes `.run(...)` at the bottom. Kaggle's benchmark
backend rejects source files that define a task but never create a run file.
The bottom `.run(...)` is the real full-ladder benchmark invocation. Kaggle
uses it both for task creation and later model runs, so it must not be a
registration-only placeholder. `max_tokens` is capped to keep proxy quota
reservation bounded.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import kaggle_benchmarks as kbench


DEFAULT_MAX_TOKENS = 128


def find_data_root() -> Path:
    kaggle_input = Path("/kaggle/input")
    candidates: list[Path] = []

    for prompt_file in kaggle_input.rglob("public_s2_prompts.jsonl"):
        root = prompt_file.parent.parent
        if (root / "kaggle/score_outputs.py").exists():
            candidates.append(root)

    if not candidates:
        raise FileNotFoundError(
            "Could not locate package root containing both "
            "'public_s2_prompts.jsonl' and 'kaggle/score_outputs.py'."
        )

    return candidates[0]


DATA_ROOT = find_data_root()
sys.path.insert(0, str(DATA_ROOT / "kaggle"))

from score_outputs import score_submission  # noqa: E402


def load_prompts() -> pd.DataFrame:
    rows = []
    with (DATA_ROOT / "data/public_s2_prompts.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if not row["disqualified"]:
                rows.append(
                    {
                        "item_id": str(row["item_id"]),
                        "prompt_id": str(row["prompt_id"]),
                        "prompt": str(row["prompt"]),
                    }
                )
    return pd.DataFrame(rows)


def prompt_model(llm, prompt: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return str(llm.prompt(prompt))
    return str(llm.prompt(prompt, extra_api_params={"max_tokens": max_tokens}))


@kbench.task(
    name="aleph_bench_frozen_ladder",
    description="Score target recovery from non-leaking frozen-ladder prompts using inverse AURC.",
    version=1,
)
def aleph_bench_frozen_ladder(
    llm,
    smoke_limit: int = 0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> float:
    prompts = load_prompts()
    if smoke_limit > 0:
        prompts = prompts.head(smoke_limit)
    model_id = str(getattr(llm, "name", None) or getattr(llm, "model", None) or "kaggle_model")

    submission_rows = []
    for _, row in prompts.iterrows():
        # Each prompt must be evaluated in a fresh chat so prior rows do not leak context.
        with kbench.chats.new(f"{row['item_id']}:{row['prompt_id']}"):
            output = prompt_model(llm, str(row["prompt"]), max_tokens)
        submission_rows.append(
            {
                "row_id": f'{row["item_id"]}:{row["prompt_id"]}',
                "model_id": model_id,
                "item_id": str(row["item_id"]),
                "prompt_id": str(row["prompt_id"]),
                "output_text": output,
            }
        )

    bench_result = score_submission(
        items_path=DATA_ROOT / "data/public_s2_items.jsonl",
        prompts_path=DATA_ROOT / "data/public_s2_prompts.jsonl",
        submission_rows=submission_rows,
        model_id=model_id,
    )

    aurc = float(bench_result["aggregate"]["aurc"])
    return 1.0 - aurc


# Kaggle task registration expects the source to invoke the task and emit a run
# file. Kaggle later re-executes this same invocation for selected models, so it
# must remain the real full-ladder benchmark call.
aleph_bench_frozen_ladder.run(
    kbench.llm,
    max_tokens=DEFAULT_MAX_TOKENS,
)
