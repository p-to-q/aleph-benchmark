# Hosted M0 Runbook

This runbook turns the next Aleph-Bench step into a reproducible black-box evidence pass. It does not replace the checked-in mock receipt until the hosted result validates.

## Purpose

The checked-in M0 result is deterministic mock evidence. The hosted M0 run should answer a narrower question: can the same Track F procedure produce three schema-valid `black_box` model rows without changing the dataset, leakage gate, metric definitions, or embedded `AlephRun` contract?

## Model Choice

Choose three hosted models that are likely to separate on S2 compositional targets:

- one frontier or largest available model;
- one mid-tier model from the same or a comparable provider route;
- one smaller or cheaper model that is expected to need more descriptive prompts.

Record the exact provider-facing model ids in the manifest and evidence note. Do not rename rows into marketing tiers after the run; the result file should carry the model ids used for calls.

## Preflight

Credentials must stay server-side:

```bash
export ALEPH_CUSTOM_API_BASE_URL=...
export ALEPH_CUSTOM_API_KEY=...
```

> **Adapter scope (M0).** `ALEPH_CUSTOM_API_BASE_URL` expects an OpenAI-compatible
> `/chat/completions` endpoint
> ([`bench/engine/adapters/hosted_black_box.py`](https://github.com/p-to-q/aleph/blob/main/bench/engine/adapters/hosted_black_box.py)).
> For cross-vendor coverage (Anthropic, Gemini, Grok), point it at an OpenAI-compatible proxy
> such as OpenRouter or LiteLLM, and record the proxy + upstream model id in the manifest and
> evidence note. **Native Anthropic / Gemini / Vertex adapters are M1 scope** — M0 deliberately
> ships a single OpenAI-compatible adapter to keep the surface small and the leaderboard
> contract reviewable.

Optional retry controls:

```bash
export ALEPH_CUSTOM_API_MAX_RETRIES=2
export ALEPH_CUSTOM_API_RETRY_DELAY_SECONDS=1
```

Then check readiness and call budget:

```bash
./aleph-bench doctor --model hosted:model-a,hosted:model-b,hosted:model-c
```

Proceed only if `doctor` reports `status: ready`. The expected M0 budget is 180 non-leaking prompts times three models times one effective rerun: 540 hosted generations.

## No-Call Manifest

Create a manifest before spending calls:

```bash
./aleph-bench manifest \
  --model hosted:model-a,hosted:model-b,hosted:model-c \
  --out bench/results/m0-hosted-manifest.json
```

Review the manifest for:

- three intended model ids;
- 180 sendable prompts;
- 60 gated explicit reconstruction anchors;
- no prompt text repeated in `gatedPrompts`;
- `estimatedGenerations` equal to 540.

Commit a hosted manifest only if it uses the real model ids for the intended run. Do not commit manifests with placeholder model names.

## Run

Use an ignored local cache so an interrupted hosted run can resume without committing provider outputs:

```bash
./aleph-bench run \
  --track F \
  --model hosted:model-a,hosted:model-b,hosted:model-c \
  --split public \
  --seed 0 \
  --cache-dir .cache/aleph-bench/m0-hosted \
  --out bench/results/m0-hosted-run.json
```

The cache is runtime state, not evidence. Keep `.cache/` uncommitted.

## Validation

Run the no-call artifact gate and generated report before changing evidence notes:

```bash
./aleph-bench verify \
  --result bench/results/m0-hosted-run.json \
  --manifest bench/results/m0-hosted-manifest.json

./aleph-bench report \
  --result bench/results/m0-hosted-run.json \
  --out bench/results/m0-hosted-report.md
```

Do not run `./aleph-bench audit` against hosted results. The audit command is intentionally mock-only because it reruns seed 0 and seed 1 to prove M0 acceptance gates without spending hosted calls. A future hosted audit mode must require an explicit adapter-call flag or separate precomputed seed results.

Do not pass the checked-in mock `bench/results/m0-audit.json` or `bench/results/m0-bundle.json` to hosted `verify`; those receipts are tied to `bench/results/m0-first-run.json`.

## Evidence Update

After validation passes:

- keep [bench/results/m0-first-run.json](../evidence/m0-first-run.json) as mock pipeline evidence unless intentionally replacing the first-run artifact;
- add or update `bench/results/m0-hosted-run.json` and `bench/results/m0-hosted-report.md`;
- update [docs/benchmark/../evidence/m0-evidence.md](../evidence/m0-evidence.md) with the hosted model summary, failure notes, latency or retry observations, and any empty-output behavior;
- state that hosted rows are `black_box` behavioral evidence only;
- do not add token NLL, logits, bits, or white-box claims.

## Failure Handling

If a provider returns transient failures, keep the cache and rerun the same command after the provider recovers. If a model repeatedly returns empty or malformed content, record the model id and failure mode in the evidence note instead of silently swapping models after seeing scores.

If model ids must change before the first successful full run, regenerate the manifest so the no-call review artifact matches the result.
