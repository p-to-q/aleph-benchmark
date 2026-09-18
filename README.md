# Aleph-Bench

Aleph-Bench studies how little information a model needs to reconstruct a
target artifact. It separates prompt leakage, output fidelity, call cost, and
evidence provenance instead of collapsing them into an unqualified score.

> [!WARNING]
> This repository is currently a **historical generated preview**, not the
> canonical benchmark source and not a release-quality leaderboard. The only
> checked-in model evidence is deterministic mock evidence. Do not cite it as
> a ranking of real models.

## Current authority

The imported snapshot at `f37fa45` cannot regenerate or independently verify
its checked-in package: it contains generated files but no engine, tests, or
release workflow. Until the migration gates in
[docs/migration-plan.md](docs/migration-plan.md) pass, reviewable benchmark
source remains on the `benchmark/source-v0.2` branch of
[`p-to-q/aleph`](https://github.com/p-to-q/aleph/tree/benchmark/source-v0.2).

The intended end state is for this repository to own the benchmark engine,
datasets, schemas, tests, protocol documentation, and release tooling. Aleph
will then consume versioned release files rather than hidden cross-repository
imports.

## What is here today

- [`platform/m0-mock/`](platform/m0-mock/) is a frozen v0.1/M0 generated
  preview. Its extra CLI file means it does not match the current immutable
  package receipt; treat the directory as historical material, not an active
  release.
- [`references/kaggle-bench/`](references/kaggle-bench/) contains non-normative
  integration notes from the original import. It is not benchmark authority.

Useful historical documents:

- [package overview](platform/m0-mock/README.md)
- [evaluation contract](platform/m0-mock/EVALUATION.md)
- [launch checklist](platform/m0-mock/PLATFORM_LAUNCH_CHECKLIST.md)

## Evidence vocabulary

- **Mock or fixture evidence** checks pipeline behavior; it is not a model
  observation.
- **Black-box evidence** requires retained raw outputs plus run and provider
  identity.
- **White-box evidence** requires real model internals such as logits; API text
  alone cannot support that claim.
- **Shortest Found**, **Explicit Reconstruction**, and **non-leaking mode** are
  different claims and must remain labeled separately.

## Migration and contribution status

Migration is deliberately staged so review does not mix repository governance,
immutable v0.1 provenance, v0.2 protocol source, CI, and release cutover. See
[the migration plan](docs/migration-plan.md) for the exact commits and gates.

Until the source migration lands, please open an issue before proposing code or
data changes. Do not hand-edit generated files under `platform/`; fixes belong
in the current source repository and must be regenerated from a clean checkout.

Before contributing, read [`CONTRIBUTING.md`](CONTRIBUTING.md) and the
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md). Report sensitive problems through
the path in [`SECURITY.md`](SECURITY.md), not a public issue. Software and
original documentation use [Apache-2.0](LICENSE); benchmark data and imported
reference material have the separate scopes documented in
[`DATA_LICENSE.md`](DATA_LICENSE.md), with imported-source attribution in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Cite a future release with
its exact tag and [`CITATION.cff`](CITATION.cff), not this unreleased snapshot
alone.
