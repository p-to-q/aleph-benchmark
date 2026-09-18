# Standalone source migration plan

- Status: proposed
- Source tracking: `p-to-q/aleph` issue
  [`#29`](https://github.com/p-to-q/aleph/issues/29)
- Kaggle reliability tracking: `p-to-q/aleph` issue
  [`#38`](https://github.com/p-to-q/aleph/issues/38)
- Historical standalone commit:
  `f37fa45f7f7dec11e1cdebb7139027a114dbb631`
- Audited Aleph source commit: `537a593` (`benchmark/source-v0.2`)

## Objective

Make `p-to-q/aleph-benchmark` the reproducible source and release authority for
Aleph-Bench without rewriting historical evidence or importing the Aleph
product application. Every phase below has one acceptance gate and remains a
source candidate until the final cutover gate passes.

## Non-goals

- Do not present deterministic mock outputs as real model rankings.
- Do not change the frozen v0.1 receipt or silently reinterpret its metrics.
- Do not relax the v0.2 Python 3.13 / Unicode 15.1 scoring contract.
- Do not copy Aleph's web application, product CI, or unrelated packages.
- Do not publish Kaggle or Hugging Face artifacts before their exact source,
  revision, digests, license, and evidence boundaries are recorded.

## Phase 1: repository governance

Add a complete code license, a separate data-license statement, contribution
and security guidance, citation metadata, a
[migration decision](decisions/0001-standalone-source-authority.md), and an
offline repository-hygiene check.

This phase does not make the historical generated package a supported release
and does not change benchmark authority.

Acceptance gate: a clean checkout proves that required governance files exist,
the citation contains no placeholders, local Markdown links resolve, no
maintainer-local absolute paths remain, and license scopes are explicit.

## Phase 2: immutable v0.1 provenance

Import the v0.1 verifier and only the source files needed to reproduce the
receipt-identical package from the current Aleph source branch. Preserve any
receipt-bound compatibility file at its recorded path. Do not repair the
existing `platform/m0-mock` preview in place: retain it in Git history or move
it as one clearly labeled legacy snapshot.

Acceptance gate: a clean standalone checkout regenerates and verifies the
v0.1 package, result, audit, and bundle with the receipt's exact aggregate
digest.

## Phase 3: v0.2 protocol and scorer source

Import the benchmark-only paths associated with the reviewed Unicode-safe v0.2
protocol, offline Kaggle receipt path, and fail-closed diagnostic tooling.
JSON Schema is authoritative in this repository. Aleph's TypeScript mirror is
a release consumer and must not become a hidden build dependency.

Acceptance gate: two clean v0.2 package generations are byte-identical, the
closed-world package check passes, and the vendored scorer passes every
conformance vector.

## Phase 4: benchmark-only CI and release tooling

Run Python 3.13 / Unicode 15.1 runtime checks, all benchmark unit tests, v0.1
immutable verification, v0.2 deterministic generation, scorer conformance,
Kaggle diagnostic generation checks, and repository hygiene. Expose one
required `release-candidate` summary job.

Acceptance gate: all offline jobs pass from a clean clone with no access to
maintainer-local files or credentials.

## Phase 5: authority cutover

Wait for the raw-capture schema, Unicode compatibility proof, Hugging Face
publication path, and complete platform generator to be reviewed. Then record
the final Aleph source commit and first standalone source commit, create an
immutable version tag, upload release assets, download them again, and verify
their digests. Update the Aleph authority decision and freeze its temporary
benchmark source branch.

Acceptance gate: assets downloaded from the public release are byte-for-byte
identical to a clean standalone generation, and Kaggle/Hugging Face readbacks
match the release manifest.

## Known blockers

- The current standalone snapshot cannot self-verify and contains one orphaned
  generated Kaggle CLI file.
- The v0.2 package is scorer-conformance staging, not yet a complete platform
  package with capture, aggregation, Croissant, report, and release evidence.
- Kaggle's observed runtime is Python 3.11 / Unicode 14, so canonical v0.2
  scoring must remain offline under Python 3.13 / Unicode 15.1.
- The current Hugging Face repository has a private v0.1/mock default branch;
  public release requires a preserved legacy ref and a truthful new landing
  page before visibility changes.
