# Aleph Bench

> **Source migration staging — not yet the protocol authority.**

This repository is receiving the standalone Aleph Bench source in small,
content-addressed slices. The current closure preserves 78 byte-identical files
from pinned Aleph commit
`10abdc1368439d1ab454ee3862a59e14b67a9530`, together with the reviewed E1
inventory. E3a adds a behavior-preserving metrics port and standalone protocol
rewrite. E3b adds the v0.2-only report renderer with a fail-closed protocol
boundary. E3c adds runtime-neutral deterministic scorer-conformance package
assembly and the ported protocol/package regression suite; scorer execution
itself remains frozen to Python 3.13 and Unicode database 15.1.0. E3d adds the
v0.2-only artifact verifier with bounded snapshot reads, final path binding,
and retained-output replay. E3e adds the v0.2-only repository runner, turns the
byte-identical `aleph-bench` launcher into a working repository-root CLI, and
replaces its excluded legacy output dependency with a descriptor-relative
atomic writer.

The sole source-parity gate for this slice is:

```bash
python3 -I -B scripts/check-source-v0.2-import.py
```

That gate verifies the checked-in inventory, copied bytes, Git modes, paths,
closed managed source trees, the top-level import surface, the reviewed
standalone notice, the two E3a transformations, and the E3b report transform.
It also verifies the two E3c output digests, exact modes, package-tree contract,
and mechanically replays the declared UTF-8/code-point splice sequences in both
directions, including the E3d verifier and E3e runner transformations. With
`--source-git`, it proves that every reconstructed port input matches the exact
pinned Git object. Passing the gate does not make the full benchmark test suite,
release packaging, or authority transition complete.
The migration checker is tested with Python 3.10 through 3.13 on POSIX systems
with no-follow file APIs. This tooling range does not relax the separate v0.2
scoring contract, which remains pinned to Python 3.13 and Unicode 15.1.0.

## Current authority and public entrypoints

- **Interim source and protocol authority:** [p-to-q/aleph](https://github.com/p-to-q/aleph),
  pinned by commit rather than by a mutable branch for this import.
- **Sole public dataset entry:** [p-to-q/aleph-bench on Hugging Face](https://huggingface.co/datasets/p-to-q/aleph-bench).
- **Long-lived Kaggle benchmark entry:** [jahyee/aleph-bench](https://www.kaggle.com/benchmarks/jahyee/aleph-bench).

Generated platform artifacts must retain their source commit, protocol/config
digest, dataset digest, and generator version. Visibility here, on Hugging
Face, or on Kaggle does not by itself make an artifact a current protocol
release or an admissible model result. The authority transition is tracked in
[p-to-q/aleph#29](https://github.com/p-to-q/aleph/issues/29).

## What is here

- `bench/`, `schemas/v0.2/`, `LICENSE`, and `aleph-bench`: the E2 byte-copy
  slice; E3a metrics; E3b report rendering; E3c deterministic
  scorer-conformance package assembly, checking, and protocol tests; the E3d
  v0.2 artifact verifier; and the E3e v0.2 repository runner. From the
  repository root, `./aleph-bench --help` exposes `doctor`, `manifest`, `run`,
  `verify`, `report`, `package-v0.2`, and `validate-croissant`. Direct and
  file-backed reports are schema checked; smoke reports identify themselves as
  non-canonical and non-leaderboard output.
  Package build/check runs on Python 3.10–3.13 without evaluating scoring or
  leakage vectors. The packaged runner returns structured
  `runtime_incompatible` evidence with zero executed vectors outside the frozen
  Python 3.13/UCD 15.1 runtime. Package publication uses descriptor-relative,
  atomic no-replace operations. Failed writes remove staging only while its
  parent-relative name still identifies the recorded inode; a renamed or
  replaced entry is left as a private mode-0700 orphan. This guarantee assumes
  the enforced POSIX owner/mode policy: processes sharing the writer's effective
  UID and mutation rights granted through platform ACLs remain trusted.
- [`docs/protocol-v0.2.md`](docs/protocol-v0.2.md): the standalone protocol and
  claim-boundary record. Its prose remains provenance-pinned to an earlier
  migration slice; a later documentation rewrite will describe the landed CLI.
- `provenance/aleph/source-v0.2.inventory.json`: the byte-identical reviewed E1
  extraction inventory.
- `provenance/aleph/e3a.metrics-and-protocol.json`: source/output digests and
  transformation contracts for the E3a port and rewrite.
- `provenance/aleph/e3b.report.json`: the content-addressed, mechanically
  replayable report-port receipt.
- `provenance/aleph/e3c.platform-package-and-protocol-tests.json`: exact
  source/output identities and reversible splice contracts for both E3c ports,
  plus the ten-file package-tree digest
  `be213ca07782f108815bf94264b4d87f48570c29cbe8511f6346e31296bf0cd8`.
- `provenance/aleph/e3d.verifier.json`: exact source/output identities and a
  reversible splice contract for the standalone v0.2 verifier.
- `provenance/aleph/e3e.runner.json`: exact source/output identities and a
  reversible splice contract for the standalone v0.2 runner.
- `platform/m0-mock/`: a historical Aleph Bench Frozen Ladder M0 platform
  package snapshot.
- `references/kaggle-bench/`: local Kaggle Benchmarks syntax notes and a minimal example task.

## Evidence boundary: mock data is not a score

The `platform/m0-mock/evidence/` directory contains deterministic mock pipeline
artifacts. They document an earlier package and scoring flow; they are not real
cross-model leaderboard results.

Do not cite the mock evidence as a benchmark score, a real model ranking, or a current protocol
release. Preserve the `mock` / `evidenceMode = mock` label in every downstream use.

## Current limitations

- The two generated Kaggle tasks and four remaining documentation rewrites are
  deliberately absent.
- `aleph-bench` is a working repository-root launcher, not an installed console
  entry point. No wheel or release package is produced here yet.
- Canonical scoring remains pinned to Python 3.13 and Unicode database 15.1.0.
  On Python 3.10–3.12, `doctor` reports the structural incompatibility and
  `run` / `manifest` fail before writing output; parser, verifier diagnostics,
  package build/check, and migration tooling remain available in their
  documented fail-closed or runtime-neutral modes.
- `validate-croissant` imports `mlcroissant` only when invoked. In the base
  environment it exits with an installation hint; `mlcroissant` is not a
  mandatory dependency for any other command.
- Copied benchmark tests outside the explicitly listed CI set remain provenance
  material until the task-regeneration and documentation closure lands. E3d
  unlocks the copied `test_v0_2_verify` receipt-replay suite. The
  verifier rejects missing or unknown protocol identities, legacy audit/bundle
  inputs, non-regular and oversized files, and paths or dataset entries that
  change after their byte snapshots are read. Its report binds the dataset,
  result, and manifest by SHA-256. A non-canonical Python or Unicode runtime
  produces a failed replay report rather than a score.
- Among the copied benchmark suite, Python 3.13 CI retains the 62 tests in
  `test_kaggle_capture`,
  `test_kaggle_creation_output`, and `test_replay_adapter`, plus the seven
  scorer-conformance tests unlocked by E3a. E3c separately runs 34
  protocol/package tests on Python 3.13 (33 pass plus the expected
  incompatible-runtime-branch skip). Each Python 3.10–3.12 matrix job runs the
  17 package-focused cases (16 pass plus the expected canonical-runtime skip).
  E3e additionally exercises parser, fail-closed runtime, package command, and
  atomic output boundaries across the Python matrix, with a no-network mock
  lifecycle on the canonical runtime.
  Hosted Linux and macOS 15 jobs are the publication gate for the two
  no-replace syscall paths and the verifier's POSIX input boundaries. These are
  bounded migration checks, not the full benchmark suite or a model score.
  Positive archive cases in `test_kaggle_creation_output` mock the diagnostic
  receipt parser, so this smoke does not prove real receipt parsing or E2E
  Kaggle execution.
- Historical mock evidence remains clearly separated from model scores.
