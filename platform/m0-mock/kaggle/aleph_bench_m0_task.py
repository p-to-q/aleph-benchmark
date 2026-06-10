"""Aleph-Bench M0 Kaggle Community Benchmark scaffold / template.

This file is a *template*. It is `import`-safe and uses only the standard
library so the package smoke test can load it; the `@kbench.task` wiring is
documented inline and meant to be uncommented inside an approved Kaggle
Benchmarks notebook where the `kaggle-benchmarks` SDK and model access are
available. Running this file directly (`python3 aleph_bench_m0_task.py`)
prints a friendly notebook-only message and exits 0 rather than crashing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_PATH = PACKAGE_ROOT / "data/public_s2_prompts.jsonl"


# ---------------------------------------------------------------------------
# Kaggle notebook wiring sketch (uncomment INSIDE the approved benchmark
# notebook, where kaggle-benchmarks is installed and model access is available):
# ---------------------------------------------------------------------------
#
# import kaggle_benchmarks as kbench
#
# @kbench.task(name="aleph_bench_m0_prompt", store_task=False)
# def aleph_bench_m0_prompt(llm, item_id: str, prompt_id: str, prompt: str) -> dict[str, str]:
#     with kbench.chats.new(f"{item_id}:{prompt_id}"):
#         output = llm.prompt(prompt)
#     return {"item_id": item_id, "prompt_id": prompt_id, "output_text": str(output)}
#
# results = aleph_bench_m0_prompt.evaluate(llm=[kbench.llm], evaluation_data=prompt_dataframe())
#
# After collecting `results`, score them with score_outputs.score_submission(...)
# (the vendored AURC / ECL@tau / Elicit@k pipeline lives in `_scoring.py`).


def load_sendable_prompts() -> list[dict[str, Any]]:
    """Return the 180 non-leaking ladder prompts from the packaged JSONL."""

    rows: list[dict[str, Any]] = []
    for line in PROMPTS_PATH.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if not row["disqualified"]:
            rows.append(row)
    return rows


def prompt_dataframe() -> list[dict[str, Any]]:
    """Minimal row-oriented evaluation data suitable for `kbench.task.evaluate`.

    Kaggle Benchmarks accepts any dataframe-like iterable; for local notebooks
    without pandas, the list-of-dicts shape works as drop-in input.
    """

    return [
        {
            "item_id": str(row["item_id"]),
            "prompt_id": str(row["prompt_id"]),
            "prompt": str(row["prompt"]),
        }
        for row in load_sendable_prompts()
    ]


def run_black_box_model(model_id: str, llm: Any) -> list[dict[str, str]]:
    """Run one Kaggle-provided model over all non-leaking M0 prompts.

    `llm` is expected to be the model object provided by the Kaggle Community
    Benchmarks notebook environment. It must expose a `prompt(...)` method
    (string in, string out); exact SDK wiring belongs in the submitted Kaggle
    notebook.
    """

    if not hasattr(llm, "prompt") or not callable(getattr(llm, "prompt")):
        raise TypeError("llm must expose a callable `prompt(str) -> str`")

    outputs: list[dict[str, str]] = []
    prompt_fn: Callable[[str], Any] = llm.prompt
    for row in load_sendable_prompts():
        output = prompt_fn(str(row["prompt"]))
        outputs.append(
            {
                "row_id": f"{row['item_id']}:{row['prompt_id']}",
                "model_id": model_id,
                "item_id": str(row["item_id"]),
                "prompt_id": str(row["prompt_id"]),
                "output_text": str(output),
            }
        )
    return outputs


def main() -> int:
    sendable = load_sendable_prompts()
    print(
        f"Aleph-Bench M0 task scaffold. {len(sendable)} non-leaking prompts ready. "
        "This file is a template: wire it into an approved Kaggle Community Benchmarks "
        "notebook (see the commented @kbench.task block at the top), then score outputs "
        "with score_outputs.score_submission(...).",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
