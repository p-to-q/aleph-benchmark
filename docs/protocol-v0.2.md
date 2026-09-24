# Aleph Bench protocol v0.2

> **Migration-stage protocol record — not yet the source-authority cutover.**

This document describes the frozen Aleph Bench v0.2 behavioral measurement
that the standalone repository is being prepared to reproduce. It is a
standalone rewrite of the pinned source documentation, not evidence that every
runtime, release, or hosted-execution component has already migrated.

The interim protocol authority remains the pinned
[`p-to-q/aleph`](https://github.com/p-to-q/aleph) source identified in
[`provenance/aleph/source-v0.2.inventory.json`](../provenance/aleph/source-v0.2.inventory.json).
Repository authority changes only after the clean-clone regeneration gate in
[standalone issue #1](https://github.com/p-to-q/aleph-benchmark/issues/1).

## 1. Measurement boundary

v0.2 implements **Track F: Frozen Ladder** over the public S2 split. Each of
the 30 synthetic, compositional items contains a fixed authored ladder with
four rungs and two paraphrases per rung. A model is observed on those prompts;
the protocol does not search prompt space, optimize a prompt, or estimate an
intrinsic shortest description.

Accordingly, the supported interpretation is:

> behavior on one frozen authored prompt ladder under one frozen scoring
> profile.

“Shortest Found” means the shortest eligible coordinate found **inside that
ladder**. It must not be relabeled “shortest prompt,” prompt-search optimality,
or a model-intrinsic compression limit. The explicit-reconstruction prompt is
an anchor and audit reference, not a non-leaking candidate.

The frozen identity is recorded by
[`bench/config/frozen_ladder-v0.2.json`](../bench/config/frozen_ladder-v0.2.json):

| Field | Frozen value |
| --- | --- |
| Protocol | `0.2.0` |
| Track / split / stratum | `F` / `public` / `S2` |
| Dataset id | `aleph-bench-v0.2-public-s2` |
| Item count | 30 |
| Dataset digest | `6f3a03400ec16405414afb94c7c639f2df07f7f0797c3b58ad1c4229e52f2041` |
| Threshold | `tau = 0.95` |
| Budget | `k = 16` current length units |
| Hosted reruns | 5 generations per eligible, non-disqualified prompt |
| Bootstrap | 500 resamples |

The dataset digest uses
`sha256-length-framed-filename-and-content-v1`; it binds both canonical
filenames and file contents. A limit smaller than 30 items is diagnostic smoke
scope and is not a canonical benchmark result.

Each current item has six eligible prompt coordinates: rung 0 is explicitly
disqualified, while rungs 1–3 contribute two paraphrases each. Canonical hosted
black-box evidence therefore contains
`30 items × 6 eligible prompts × 5 reruns = 900` logical response slots per
hosted model. A complete hosted result must retain all 900 slots for each
hosted model; a resumed run may dispatch fewer new provider calls when verified
cached responses fill earlier slots. Deterministic mock adapters use one
generation per eligible coordinate and remain mock pipeline evidence, not
canonical hosted model evidence.

## 2. Rate–distortion coordinates

The current implementation stores prompt length in a field named `tokens`, but
computes it with the ASCII expression `[A-Za-z0-9]+`. It is therefore an
**ASCII word-like-span proxy**, not a provider tokenizer count, model input
token count, Unicode-aware word count, or language-comparable unit. The name is
retained in v0.2 for compatibility; publications must describe the actual unit.

This limitation is material for CJK and other scripts and is tracked, together
with the future provider-token receipt, in
[`p-to-q/aleph#35`](https://github.com/p-to-q/aleph/issues/35). It is not fixed
silently in this migration slice because changing it would change protocol
semantics.

For each item, eligible points are deduplicated by length and reduced to the
strictly improving lower-distortion envelope. Disqualified leakage points do
not enter that envelope. The right-hand normalization anchor is

```text
L_max = max(
  |frozenLadder[0].prompt|,
  |target text|,
  1
)
```

Here `frozenLadder[0]` is the first authored coordinate, `r0-p0`; the second
rung-0 paraphrase is not considered when selecting the anchor. Both lengths use
the current proxy. The staircase begins at distortion `1.0` at zero budget,
changes only when an eligible point becomes available, and holds its last
distortion through the right-hand anchor.

The current metrics are:

- **AURC:** area under that normalized rate–distortion staircase. Lower is
  better. An item with no eligible frontier has AURC `1.0`.
- **ECL@tau:** the shortest eligible proxy length on an item that reaches
  fidelity `tau`.
- **coverageAtTau:** the share of items with an ECL hit.
- **Elicit@k:** the share of items with an eligible point of length at most `k`
  that reaches `tau`.

The current aggregate ECL is the mean only over items that reach `tau`.
Therefore it is success-conditioned and must always be interpreted with
`coverageAtTau`; it is not a safe standalone ranking headline. An apparently
short ECL with low coverage is not evidence of better overall compression.
Coverage-safe aggregation remains an explicit part of issue #35.

Bootstrap confidence intervals are deterministic for the frozen seed and
sample count. Ranking helpers order ascending by the selected metric and use
model id as the deterministic tie-breaker. These mechanics do not make every
metric suitable as a leaderboard primary key.

## 3. Fidelity contract

v0.2 accepts exactly three declared metric classes:

- `exact`: binary equality of raw decoded Python string code-point sequences;
  no Unicode, newline, case, or whitespace normalization.
- `normalized_edit_similarity`: normalize CRLF and lone CR to LF, normalize to
  NFC, then compute `1 - Levenshtein / max(code-point lengths)`.
- `unicode_char_ngram`: NFC, case-fold, NFC again, collapse Unicode whitespace,
  then compute multiset Jaccard over code-point trigrams.

The public S2 items declare `normalized_edit_similarity`. Unknown metric
classes fail closed. Raw model output is captured before scoring
normalization; transport code must not trim it or coerce a non-string response.

The scorer bounds raw text at 16,384 Unicode code points, normalized text at
32,768 code points, and each quadratic edit/LCS pair at 1,000,000 cells. Those
limits are part of the v0.2 scoring profile, not optional performance tuning.
The shared vectors in
[`bench/conformance/scorer-v0.2.json`](../bench/conformance/scorer-v0.2.json)
lock engine and packaged-scorer behavior.

## 4. Leakage gate

Leakage is an exclusion gate rather than a numeric penalty. The
`unicode_dual_channel_v1` gate compares both a lexical channel and a
boundary-insensitive skeleton channel after compatibility normalization and
case-folding. A candidate is disqualified when any frozen threshold is met:

```text
lexical:  lcsRatio                    >= 0.65
       or targetTrigramRecall         >= 0.50
       or verbatimSpanUnits           >= 16

skeleton: skeletonLcsRatio            >= 0.80
       or skeletonTargetTrigramRecall >= 0.80
       or skeletonVerbatimSpanUnits   >= 32
```

Raw bidi controls and CJK Compatibility Ideographs are fail-closed
disqualifiers regardless of the overlap diagnostics. They do not bypass the
scorer's input and quadratic-work bounds, so an oversized comparison can be
rejected before a disqualification receipt is produced. Private-use and
unassigned code points remain visible rather than being erased. The skeleton
is conservative and is not a Unicode UTS #39 confusable implementation;
cross-script homoglyphs and visual substitutions such as `I`/`l` or `rn`/`m`
remain manual-audit limitations.

## 5. Runtime profile

Canonical scoring requires Python 3.13 and Unicode database 15.1.0.
`bench.engine.scoring_core.validate_scoring_runtime()` enforces that profile
before fidelity scoring. Import and migration tooling are intentionally tested
on Python 3.10–3.13, but successful import on an older interpreter is not proof
that it may produce canonical scores. On a non-conforming runtime, fidelity
scoring must fail closed.

The frozen protocol seed coordinates bootstrap sampling and cache identity. It
is not currently sent to an OpenAI-compatible provider. Hosted prompts are
specified for five calls even at temperature zero so provider-side variance
can remain visible.

## 6. Evidence and claim levels

Keep these evidence classes separate:

| Evidence | What it can show | What it cannot show |
| --- | --- | --- |
| Mock fixture | deterministic pipeline and schema behavior | real model quality or ranking |
| Scorer conformance | fidelity/leakage parity on frozen vectors | hosted transport or full benchmark completion |
| Diagnostic/canary | bounded transport and capture behavior | AURC, leaderboard eligibility, or protocol conformance |
| Canonical result | complete closed-set receipt plus independent replay | claims outside its exact model/runtime/protocol identity |

There is currently no formal v0.2 public model score in this repository.
Historical `platform/m0-mock` material remains mock evidence. A successful
Kaggle task transport, a public Hugging Face dataset, or an old leaderboard row
must not be promoted into a v0.2 score without the complete closed-set receipt
and replay gates.

## 7. Current standalone implementation status

This table prevents normative protocol text from being mistaken for current
repository capability:

| Component | Status in this repository |
| --- | --- |
| Frozen config, S2 data, schemas, conformance vectors | present as pinned E2 copies |
| Scoring core, protocol helpers, leakage gate | present as pinned E2 copies |
| Metric aggregation module | present as the provenance-checked E3a port |
| Full runner and installed CLI | not yet available |
| Report, verification, and v0.2 package orchestration | not yet ported |
| Generated Kaggle capture and diagnostic tasks | not yet regenerated here |
| Deterministic release builders and HF/Kaggle publication | not yet available |
| Standalone source-authority cutover | not complete |

The currently valid local checks are intentionally narrower than a benchmark
run:

```bash
python3 -I -B scripts/check-source-v0.2-import.py
python3 -B -m unittest -v tests.test_metrics_v0_2
python3.13 -B -m unittest -v bench.tests.test_scorer_conformance
```

The first command verifies the content-addressed migration slice offline. The
second exercises aggregation behavior across supported tooling interpreters.
The third is the canonical-runtime scorer-conformance proof. None dispatches a
model call or creates a score.

## 8. Change control

Any change to dataset identity, ladder prompts, length unit, `tau`, `k`, rerun
count, bootstrap count, metric semantics, leakage thresholds, runtime profile,
or aggregate convention requires an explicit versioned protocol decision. It
must not be introduced as a portability fix or a migration-only edit.

The authority migration is tracked in
[`p-to-q/aleph-benchmark#1`](https://github.com/p-to-q/aleph-benchmark/issues/1),
and this metrics/document slice in
[`p-to-q/aleph-benchmark#4`](https://github.com/p-to-q/aleph-benchmark/issues/4).
The measurement redesign remains in
[`p-to-q/aleph#35`](https://github.com/p-to-q/aleph/issues/35).
