# Contributing

Aleph-Bench is in a staged source migration. Please open an issue before a
code, protocol, dataset, evidence, or release change so the relevant authority
and acceptance gate are explicit.

## Evidence-first changes

- State whether material is a fixture, mock, black-box observation, or
  white-box observation.
- Preserve raw outputs and source identity before deriving scores.
- Never present mock output as a model result or an unverified hosted run as a
  release.
- Give every new dataset item provenance and an explicit compatible license.
- Put unsettled ideas in a plan or decision document rather than presenting
  them as benchmark facts.

## Engineering expectations

- Make one reviewable change with one acceptance gate.
- Add tests for success, malformed input, partial execution, and relevant
  resource limits.
- Keep generated files derived from source. Do not hand-edit `platform/`.
- Do not add maintainer-local absolute paths, secrets, tokens, or private model
  outputs.
- Record exact validation commands and distinguish local tests from hosted
  platform evidence.

The current snapshot cannot run benchmark tests on its own. Follow
[`docs/migration-plan.md`](docs/migration-plan.md) until the benchmark-only
source and CI arrive.

The governance slice has one standard-library-only local gate:

```bash
python3 -B -m unittest scripts.test_repo_hygiene
python3 -B scripts/check_repo_hygiene.py
```
