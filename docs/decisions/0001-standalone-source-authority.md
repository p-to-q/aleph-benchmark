# 0001: Stage standalone source authority

- Status: Accepted
- Date: 2026-09-18
- Related plan: [`../migration-plan.md`](../migration-plan.md)
- Source tracking: [`p-to-q/aleph` issue #29](https://github.com/p-to-q/aleph/issues/29)
- Audited source snapshot: [`537a593`](https://github.com/p-to-q/aleph/tree/537a593e0a698a8a67989a5a308ecb5382227ed5)

## Context

The first commit in this repository is a generated preview. It has useful
historical package material, but it does not include the source, tests, or
release workflow needed to reproduce and verify that material. Calling the
standalone repository authoritative now would create two conflicting sources
of truth and would turn generated files into de facto source.

The reviewable v0.2 implementation currently lives on the
`benchmark/source-v0.2` branch of `p-to-q/aleph`. Product code must eventually
consume a benchmark release without owning or silently modifying its protocol.

## Decision

1. During migration, `p-to-q/aleph@benchmark/source-v0.2` remains the
   reviewable source authority. This standalone repository labels its current
   files as historical preview material.
2. Migration happens in the independently reviewable phases and acceptance
   gates in the linked plan. Generated package files are never hand-edited to
   simulate source completeness.
3. Authority moves only after a clean standalone checkout can reproduce and
   verify the frozen v0.1 receipt and the reviewed v0.2 release artifacts.
4. At cutover, a decision update records both exact Git commits, the immutable
   standalone tag, and downloaded release-asset digests. Until then, no branch
   name or hosted upload is evidence of a release.
5. After cutover, this repository owns benchmark engine, datasets, schemas,
   tests, protocol documents, platform packaging, CI, and releases. The Aleph
   product repository consumes explicit versioned artifacts and does not keep
   a second mutable scorer implementation.

## Consequences

- Contributors have one declared current authority and one declared target
  authority instead of guessing from repository names.
- The historical preview remains inspectable without being promoted to a
  supported release.
- Kaggle and Hugging Face publication must identify an immutable source commit
  and pass byte-level readback before they can support public claims.
- Cutover requires another reviewed change to update this decision; it cannot
  happen implicitly through an upload or branch merge.
