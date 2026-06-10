# Aleph-Bench M0 Platform Evaluation Protocol

This package is ready to publish as a Hugging Face Dataset or Kaggle Dataset. It is also ready for a community benchmark dry run because it includes the seed data, prompt rows, schemas, mock evidence, a vendored scorer, and verification commands.

## Evidence Boundary

- The included model rows are deterministic mock evidence and live under `evidence/mock_*.csv` (not under `data/`). Hugging Face and Kaggle dataset previews therefore do not render them as a leaderboard.
- They prove the benchmark pipeline, not model quality.
- Hosted black-box rows must use the same S2 items, frozen ladder, leakage thresholds, tau, k, and AURC/ECL/Elicit definitions.
- Do not report logits, token NLL, bits, or white-box claims for hosted black-box rows.

## Submission Shape

Use `data/submission_format.csv` as the public output shape:

```text
row_id,model_id,item_id,prompt_id,output_text
```

`prompt_id` must refer to a non-leaking prompt from `data/public_s2_prompts.jsonl`. Gated explicit reconstruction prompts are present for auditability but should not be submitted as compression candidates.

## Where the scorer lives

This package is **self-contained for scoring**. You do not need to clone the Aleph repository to compute AURC / ECL@tau / Elicit@k on a `submission_format.csv` of outputs:

- `kaggle/_scoring.py` is a vendored copy of the pure-function metric helpers from `bench/engine/metrics.py` and `bench/engine/leakage_gate.py`. It has no third-party dependencies (standard library only).
- `kaggle/score_outputs.py` reads the items JSONL, prompts JSONL, and a submission row sequence (CSV path or in-memory list) and emits a `BenchResult`-compatible JSON.
- `kaggle/api_test_smoke.py` exercises both the I/O contract and the scorer end-to-end against a stub LLM; passing the smoke test means a Kaggle notebook will be able to call `score_submission(...)` against real outputs without further glue.

A full repository-side verification (schema, bundle, audit, manifest) still requires the Aleph repo:

```bash
git clone https://github.com/p-to-q/aleph
cd aleph
./aleph-bench verify --audit bench/results/m0-audit.json --bundle bench/results/m0-bundle.json
./aleph-bench package --check bench/results/platform/m0-mock/package-manifest.json
python3 bench/results/platform/m0-mock/kaggle/api_test_smoke.py
python3 bench/results/platform/m0-mock/huggingface/upload_dataset.py --dry-run
npm run lint
npm run test
```

## Hosted Evidence Upgrade

Before replacing the mock evidence note with real black-box rows:

```bash
./aleph-bench doctor --model hosted:model-a,hosted:model-b,hosted:model-c
./aleph-bench manifest --model hosted:model-a,hosted:model-b,hosted:model-c --out bench/results/m0-hosted-manifest.json
./aleph-bench run --track F --model hosted:model-a,hosted:model-b,hosted:model-c --split public --seed 0 --cache-dir .cache/aleph-bench/m0-hosted --out bench/results/m0-hosted-run.json
./aleph-bench verify --result bench/results/m0-hosted-run.json --manifest bench/results/m0-hosted-manifest.json
./aleph-bench report --result bench/results/m0-hosted-run.json --out bench/results/m0-hosted-report.md
```

`ALEPH_CUSTOM_API_BASE_URL` expects an OpenAI-compatible `/chat/completions` endpoint. For cross-vendor coverage (Anthropic, Gemini), point it at an OpenAI-compatible proxy such as OpenRouter or LiteLLM; native Anthropic / Gemini adapters are M1 scope.
