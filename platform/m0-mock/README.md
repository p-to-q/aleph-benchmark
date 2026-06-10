---
pretty_name: AlephBench Frozen Ladder
language:
- en
license: cc0-1.0
size_categories:
- n<1K
task_categories:
- text-generation
tags:
- benchmark
- prompt-compression
- reverse-prompt-search
- elicitation-efficiency
- rate-distortion
- synthetic
- aleph-bench
version: 0.1.0
milestone: m0
configs:
- config_name: public-s2-items
  data_files:
  - split: test
    path: data/public_s2_items.jsonl
- config_name: public-s2-prompts
  data_files:
  - split: test
    path: data/public_s2_prompts.jsonl
---

# AlephBench Frozen Ladder

> [!WARNING]
> The model rows under `evidence/` are **deterministic mock pipeline outputs**, not real model
> evaluations. This release distributes the benchmark **dataset and procedure** (M0 = milestone 0).
> Real cross-vendor model rows are gated on Kaggle Benchmarks Resource Grant model access; do not
> cite numbers from `evidence/mock_*.csv` as a model leaderboard.

AlephBench Frozen Ladder is the current procedure release for a small frozen-ladder benchmark seed set for comparing how much non-leaking prompt coordinate length a model needs to reproduce synthetic target outputs. **Version**: `m0`. **Evidence mode**: `mock` (no real model rows yet).

The construct is **elicitation efficiency**: fix the target output, then search for the shortest non-leaking prompt that regenerates it. Lower AURC means the model reaches the target with fewer prompt tokens.

## Contents

### Dataset (`data/`)

- `public_s2_items.jsonl` (30 rows): S2 compositional BenchItems with target text, frozen ladder, canary GUID, provenance.
- `public_s2_prompts.jsonl` (240 rows): every frozen-ladder prompt, with leakage-gate measurements and a `disqualified` flag.
- `public_s2_items.csv` / `public_s2_prompts.csv`: flat preview tables for the Kaggle Dataset page.
- `submission_format.csv` (180 rows): the public output shape (`row_id,model_id,item_id,prompt_id,output_text`) restricted to non-leaking prompts.

### Mock evidence (`evidence/`)

- `mock_model_summary.csv` / `mock_item_metrics.csv`: deterministic mock-adapter rows. **Pipeline evidence only**, never to be cited as model rankings.
- `m0-first-run.json`: the full mock `BenchResult` (3 mock models, 30 items).
- `m0-report.md`: pre-rendered tables for the mock result.
- `m0-audit.json` / `m0-bundle.json` / `m0-call-manifest.json` / `m0-evidence.md`: acceptance-gate receipt, bundle digest manifest, no-call prompt manifest, and evidence note.

### Metadata, schemas, and platform scaffolding

- `schemas/`: JSON Schemas validating items, prompts, results, audit, bundle, manifest, and the platform package itself.
- `croissant.json`: MLCommons Croissant 1.1 metadata with `RecordSet` for items and prompts.
- `dataset-metadata.json`: Kaggle Dataset metadata with per-CSV field schemas.
- `huggingface/`: Hugging Face upload preparation (dry-run + optional `--execute`).
- `kaggle/`: Kaggle Community Benchmark task scaffold, vendored scorer, output-scoring helper, and I/O smoke test.
- `PLATFORM_LAUNCH_CHECKLIST.md`: staged launch gates separating prepared, blocked, and not-yet-run work.

## How to evaluate

A Kaggle notebook (or any local runner) can score outputs against the frozen ladder **without cloning the Aleph repository** — the scorer is vendored in `kaggle/`:

```python
import csv, json
from pathlib import Path

# 1) Load the non-leaking prompts (180 of 240).
prompts = [json.loads(line) for line in Path("data/public_s2_prompts.jsonl").read_text().splitlines()]
non_leaking = [row for row in prompts if not row["disqualified"]]

# 2) Generate outputs (replace with your model call). Submission shape is documented in
#    data/submission_format.csv and must contain row_id, model_id, item_id, prompt_id, output_text.
submission = [{"row_id": f"{r['item_id']}:{r['prompt_id']}",
               "model_id": "my-model",
               "item_id": r["item_id"],
               "prompt_id": r["prompt_id"],
               "output_text": my_model(r["prompt"])} for r in non_leaking]

# 3) Score with the vendored AURC / ECL@tau / Elicit@k implementation.
import sys; sys.path.insert(0, "kaggle")
from score_outputs import score_submission
result = score_submission(items_path="data/public_s2_items.jsonl",
                          prompts_path="data/public_s2_prompts.jsonl",
                          submission_rows=submission,
                          model_id="my-model")
print(result["aggregate"])
```

Leakage is a gate, not a penalty. Explicit reconstruction prompts (rung 0) are excluded from compression metrics. AURC is the area under the monotone non-leaking rate-distortion staircase; lower is better. ECL@tau and Elicit@k are interpretable duals.

For deeper validation (schema check + bundle digest + acceptance audit) clone the Aleph repository and run:

```bash
./aleph-bench verify --audit bench/results/m0-audit.json --bundle bench/results/m0-bundle.json
./aleph-bench package --check bench/results/platform/m0-mock/package-manifest.json
```

Hosted black-box results require server-side OpenAI-compatible credentials and should produce their own result, manifest, report, and evidence note. Black-box rows report generated text behavior only; they do not report logits, token NLL, or white-box observations. Native Anthropic / Gemini adapters are M1 scope; M0 cross-vendor coverage requires an OpenAI-compatible proxy (OpenRouter, LiteLLM, ...).

## Citation

A versioned citation (with author list, DOI, and venue) will accompany the first hosted-evidence release. For now, please cite as:

```bibtex
@misc{alephbench_m0_2026,
  title  = {AlephBench Frozen Ladder: A Frozen-Ladder Prompt-Compression Benchmark (Procedure Release)},
  author = {AlephBench Maintainers},
  year   = {2026},
  note   = {Procedure release, mock pipeline evidence; cross-vendor model rows pending Kaggle Benchmarks Resource Grant},
  url    = {https://huggingface.co/datasets/p-to-q/aleph-bench-m0}
}
```

The canonical CITATION template lives at `docs/benchmark/launch-kit/CITATION.cff` in the upstream repository; it will be finalized with the first real evidence release.

## Responsible Use

The targets are rule-generated synthetic English strings with a shared canary GUID stored as metadata, not as target text. The package is meant for benchmark procedure review and community reproduction, not for claims about globally shortest prompts or real model ranking until hosted black-box rows exist. Any `aurc / eclAtTau / elicitAtK` number sourced from `evidence/` is mock pipeline evidence with `evidenceMode = mock`, and downstream summaries (blog posts, slides, leaderboards) must preserve that label.
