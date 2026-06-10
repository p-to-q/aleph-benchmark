# M0 Evidence

This note records the first Aleph-Bench M0 run:

```bash
python3 bench/run.py run --track F --model mock-frontier,mock-mid,mock-small --split public --seed 0 --out bench/results/m0-first-run.json
```

The result is deterministic mock evidence. It proves the Track F pipeline, schemas, leakage gate, metrics, bootstrap CIs, and reproducibility path. It is not a real model leaderboard.

Preflight for the checked-in mock run reports 30 items, 240 ladder prompts, 60 gated explicit reconstruction prompts, 180 non-leaking prompts, and 540 estimated mock generations across three models:

```bash
./aleph-bench doctor --model mock-frontier,mock-mid,mock-small
```

The corresponding no-call manifest is [bench/results/m0-call-manifest.json](m0-call-manifest.json). It validates against [schemas/aleph-bench-manifest.schema.json](../schemas/aleph-bench-manifest.schema.json), lists the 180 non-leaking prompts that would be sent, and records the 60 gated explicit reconstruction anchors without repeating their prompt text.

The hosted black-box path is also covered by offline local `/chat/completions` tests. Those tests verify request shape, bounded retry on 429/5xx failures, error reporting, and a one-item schema-valid `black_box` `BenchResult` without producing model evidence.

Hosted runs may use `--cache-dir .cache/aleph-bench/m0-hosted` to preserve completed calls across interruptions. The cache is a local runtime aid, not a checked-in evidence artifact.

Hosted retries are also runtime aids: by default the adapter retries retryable HTTP failures twice. Retry settings do not change scoring semantics.

The checked-in dataset, result, and manifest can be audited together:

```bash
./aleph-bench verify --audit bench/results/m0-audit.json --bundle bench/results/m0-bundle.json
```

The original M0 acceptance gates can be checked directly:

```bash
./aleph-bench audit
```

The checked-in audit receipt is [bench/results/m0-audit.json](m0-audit.json). It is generated from `./aleph-bench audit --out bench/results/m0-audit.json`, validates against [schemas/aleph-bench-audit.schema.json](../schemas/aleph-bench-audit.schema.json), and records the current acceptance-gate status; it does not add model evidence beyond the mock result.

The audit command is mock-only because it reruns seed 0 and seed 1. Hosted results should be validated with `verify` and rendered with `report` unless a future audit mode explicitly allows adapter calls.

The audit also checks that the S2 seed set has exactly 30 items, one shared field-level canary GUID that does not appear in target or prompt text, a 10/10/10 family split across arithmetic cards, letter lattices, and route receipts, and the expected 4x2 ladder coordinates.

The mock M0 evidence bundle index is [bench/results/m0-bundle.json](m0-bundle.json). It records SHA-256 digests for the result, manifest, audit receipt, generated report, and this evidence note. It validates against [schemas/aleph-bench-bundle.schema.json](../schemas/aleph-bench-bundle.schema.json) and can be checked with `./aleph-bench bundle --check bench/results/m0-bundle.json`.

The model and per-item tables are also available as a generated report: [bench/results/m0-report.md](m0-report.md).

Every item/model run also includes a `measurements` receipt for all eight ladder prompts. It records effective rerun count, fidelity mean, variance, standard deviation, and leakage-gate status per prompt. In the checked-in deterministic mock run, non-leaking prompts have one effective rerun and zero variance; the same field is the audit trail for stochastic hosted adapters when temperature is above zero.

## Run

- Track: F, Frozen Ladder.
- Split: public.
- Stratum: S2 compositional.
- Items: 30.
- Metric class: exact, with normalized edit-distance signal for near misses.
- Threshold: `tau = 0.9`.
- Elicit cutoff: `k = 3`.
- Result file: [bench/results/m0-first-run.json](m0-first-run.json).

## Model Summary

Lower AURC and ECL are better. The leakage hit rate is expected to be `0.25` because each item has two explicit reconstruction prompts out of eight ladder prompts.

| Model | Evidence | AURC | 95% CI | ECL@0.9 | 95% CI | Coverage@0.9 | Elicit@3 | Leakage hits |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| mock-frontier | mock | 0.047515 | 0.043485-0.051577 | 5.333333 | 5.166667-5.5 | 1.0 | 1.0 | 0.25 |
| mock-mid | mock | 0.197422 | 0.182687-0.210575 | 38.333333 | 36.466667-40.3 | 1.0 | 0.0 | 0.25 |
| mock-small | mock | 0.404643 | 0.381256-0.430169 | 38.433333 | 36.8-40.4 | 1.0 | 0.0 | 0.25 |

## Stability Gate

The mock M0 acceptance gate checks seed stability in `bench/tests/test_m0.py`: a full 30-item seed 0 run and a full 30-item seed 1 run both rank `mock-frontier`, `mock-mid`, then `mock-small` by AURC, and adjacent AURC 95% CI bands do not overlap. This is mock pipeline stability evidence only; real black-box stability still requires the hosted run.

## Per-Item Receipt

Cells are `AURC / ECL@0.9`. Full per-item frontiers, prompts, outputs, and leakage gate measurements are in the result JSON.

| Item | mock-frontier | mock-mid | mock-small |
|---|---:|---:|---:|
| s2-001 | 0.039 / 6 | 0.168 / 34 | 0.348 / 34 |
| s2-002 | 0.068 / 5 | 0.251 / 46 | 0.505 / 46 |
| s2-003 | 0.041 / 5 | 0.173 / 35 | 0.363 / 35 |
| s2-004 | 0.038 / 6 | 0.171 / 34 | 0.350 / 34 |
| s2-005 | 0.063 / 5 | 0.253 / 46 | 0.503 / 46 |
| s2-006 | 0.040 / 5 | 0.171 / 35 | 0.361 / 35 |
| s2-007 | 0.040 / 6 | 0.167 / 34 | 0.353 / 34 |
| s2-008 | 0.069 / 5 | 0.253 / 46 | 0.502 / 46 |
| s2-009 | 0.038 / 5 | 0.169 / 35 | 0.362 / 35 |
| s2-010 | 0.038 / 6 | 0.166 / 34 | 0.351 / 34 |
| s2-011 | 0.060 / 5 | 0.253 / 46 | 0.502 / 46 |
| s2-012 | 0.041 / 5 | 0.171 / 35 | 0.364 / 35 |
| s2-013 | 0.037 / 6 | 0.166 / 34 | 0.349 / 34 |
| s2-014 | 0.069 / 5 | 0.254 / 46 | 0.509 / 46 |
| s2-015 | 0.041 / 5 | 0.172 / 35 | 0.363 / 35 |
| s2-016 | 0.041 / 6 | 0.171 / 34 | 0.350 / 37 |
| s2-017 | 0.062 / 5 | 0.254 / 46 | 0.501 / 46 |
| s2-018 | 0.040 / 5 | 0.172 / 35 | 0.360 / 35 |
| s2-019 | 0.038 / 6 | 0.166 / 34 | 0.349 / 34 |
| s2-020 | 0.060 / 5 | 0.248 / 46 | 0.497 / 46 |
| s2-021 | 0.040 / 5 | 0.170 / 35 | 0.363 / 35 |
| s2-022 | 0.039 / 6 | 0.168 / 34 | 0.353 / 34 |
| s2-023 | 0.058 / 5 | 0.250 / 46 | 0.500 / 46 |
| s2-024 | 0.040 / 5 | 0.171 / 35 | 0.359 / 35 |
| s2-025 | 0.040 / 6 | 0.167 / 34 | 0.348 / 34 |
| s2-026 | 0.068 / 5 | 0.259 / 46 | 0.505 / 46 |
| s2-027 | 0.039 / 5 | 0.172 / 35 | 0.362 / 35 |
| s2-028 | 0.039 / 6 | 0.172 / 34 | 0.348 / 34 |
| s2-029 | 0.061 / 5 | 0.251 / 46 | 0.499 / 46 |
| s2-030 | 0.039 / 5 | 0.171 / 35 | 0.362 / 35 |

## Open-Question Choices

- Q1 leakage thresholds: `lcsRatio = 0.65`, `trigramOverlap = 0.5`, `verbatimSpanTokens = 16`. This gates both explicit reconstruction prompts for every item and does not gate descriptive, concept, or minimal prompts in the seed set.
- Q2 reruns: deterministic `temperature=0` adapters use one actual generation per prompt. The configured rerun count remains available for stochastic hosted adapters, and per-prompt variance is recorded in each item run's `measurements`.
- Q3 tau: `0.9` is achievable in the mock run for all models through non-leaking prompts, so M0 keeps it.
- Q4 S2 design: targets are arithmetic cards, letter lattices, and route receipts generated from fixed rules and parameters.
- Q5 model choice: the checked-in first run uses three mock profiles designed to separate the pipeline. Real hosted models remain the next evidence step.
- Q6 exact near misses: exact-class fidelity uses normalized edit distance after exact-match failure, so a nearly correct table or receipt is not collapsed to a binary zero.

## Read

The metric separates the three mock profiles sensibly: `mock-frontier` reaches the threshold with minimal cues, while `mock-mid` and `mock-small` usually need descriptive prompts. That is enough to show the M0 machinery can express model separation without weighted sums.

The next evidence step is to run the same seed set with three real black-box models through the hosted adapter, then replace this note's model summary with black-box rows while preserving the same schema and leakage gate.

Before that hosted run, `./aleph-bench doctor --model hosted:model-a,hosted:model-b,hosted:model-c` should report `status: ready`; without `ALEPH_CUSTOM_API_BASE_URL` and `ALEPH_CUSTOM_API_KEY`, it reports `blocked` rather than pretending the benchmark is runnable.

For a real run, create a hosted manifest first:

```bash
./aleph-bench manifest --model hosted:model-a,hosted:model-b,hosted:model-c --out bench/results/m0-hosted-manifest.json
```

Then run with a local response cache:

```bash
./aleph-bench run --track F --model hosted:model-a,hosted:model-b,hosted:model-c --split public --seed 0 --cache-dir .cache/aleph-bench/m0-hosted --out bench/results/m0-hosted-run.json
```

After a hosted result is written, run `./aleph-bench verify --result bench/results/m0-hosted-run.json --manifest bench/results/m0-hosted-manifest.json`.

Then regenerate hosted tables with `./aleph-bench report --result bench/results/m0-hosted-run.json --out bench/results/m0-hosted-report.md`.
