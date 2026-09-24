# aleph-benchmark

> **Historical generated mirror — not the protocol authority.**

This repository preserves a public snapshot of earlier Aleph Bench packaging. It is useful for
auditing that snapshot, but it is not independently maintained and must not be used to decide the
current protocol, measurement contract, or release process.

## Current authority and public entrypoints

- **Canonical source and protocol authority:** [p-to-q/aleph](https://github.com/p-to-q/aleph).
  Active v0.2 development currently lives on that repository's
  [`benchmark/source-v0.2`](https://github.com/p-to-q/aleph/tree/benchmark/source-v0.2) branch;
  this statement does not imply that v0.2 is already present on its `main` branch.
- **Sole public dataset entry:** [p-to-q/aleph-bench on Hugging Face](https://huggingface.co/datasets/p-to-q/aleph-bench).
- **Long-lived Kaggle benchmark entry:** [jahyee/aleph-bench](https://www.kaggle.com/benchmarks/jahyee/aleph-bench).

Generated platform artifacts should flow from the canonical source and retain their source commit,
protocol/config digest, dataset digest, and generator version. Visibility on this mirror, Hugging
Face, or Kaggle does not by itself make an artifact a current protocol release or admissible model
result. The authority transition is tracked in
[p-to-q/aleph#29](https://github.com/p-to-q/aleph/issues/29).

## What is here

- `platform/m0-mock/`: a historical AlephBench Frozen Ladder M0 platform package snapshot.
- `references/kaggle-bench/`: local Kaggle Benchmarks syntax notes and a minimal example task.

## Evidence boundary: mock data is not a score

The `platform/m0-mock/evidence/` directory contains deterministic mock pipeline artifacts. They document the benchmark package and scoring flow, but they are not real cross-model leaderboard results.

Do not cite the mock evidence as a benchmark score, a real model ranking, or a current protocol
release. Preserve the `mock` / `evidenceMode = mock` label in every downstream use.

## Immediate entrypoints

- Start with [platform/m0-mock/README.md](platform/m0-mock/README.md) for the historical package overview.
- Read [platform/m0-mock/EVALUATION.md](platform/m0-mock/EVALUATION.md) for that snapshot's evaluation contract.
- Read [platform/m0-mock/PLATFORM_LAUNCH_CHECKLIST.md](platform/m0-mock/PLATFORM_LAUNCH_CHECKLIST.md) for its recorded launch status and remaining gates.
