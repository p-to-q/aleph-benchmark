# Aleph Bench

> **Source migration staging — not yet the protocol authority.**

This repository is receiving the standalone Aleph Bench source in small,
content-addressed slices. The current closure preserves 78 byte-identical files
from pinned Aleph commit
`10abdc1368439d1ab454ee3862a59e14b67a9530`, together with the reviewed E1
inventory, one behavior-preserving metrics port, and one standalone protocol
rewrite.

The sole source-parity gate for this slice is:

```bash
python3 -I -B scripts/check-source-v0.2-import.py
```

That gate verifies the checked-in inventory, copied bytes, Git modes, paths,
closed source tree, reviewed standalone notice, and the two E3a transformations.
With `--source-git`, it also proves that the metrics port differs from the pinned
source by exactly one documentation-path replacement. Passing the gate does not
make the CLI, full benchmark test suite, packaging, or authority transition
complete.
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
  slice plus the E3a metrics port. The launcher intentionally remains
  unavailable until its separately reviewed `bench/run.py` port lands.
- [`docs/protocol-v0.2.md`](docs/protocol-v0.2.md): the standalone protocol and
  claim-boundary record; it labels components that have not migrated yet.
- `provenance/aleph/source-v0.2.inventory.json`: the byte-identical reviewed E1
  extraction inventory.
- `provenance/aleph/e3a.metrics-and-protocol.json`: source/output digests and
  transformation contracts for the E3a port and rewrite.
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

- Five files classified as `port`, the two generated Kaggle tasks, and four
  remaining documentation rewrites are deliberately absent.
- `aleph-bench` therefore must not be presented as a working installed CLI.
- The copied benchmark tests are provenance material in this slice; the full
  suite is not runnable until the port closure lands. Four production/support
  modules and seven test modules remain outside the complete import closure.
- On Python 3.13, CI runs the 62 tests in `test_kaggle_capture`,
  `test_kaggle_creation_output`, and `test_replay_adapter`, plus the seven
  scorer-conformance tests unlocked by E3a. This is a bounded migration smoke
  check, not the full suite.
  Positive archive cases in `test_kaggle_creation_output` mock the diagnostic
  receipt parser, so this smoke does not prove real receipt parsing or E2E
  Kaggle execution.
- No wheel or other release package is produced here yet.
- Historical mock evidence remains clearly separated from model scores.
