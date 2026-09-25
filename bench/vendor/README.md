# Reviewed portable Unicode inputs

This directory is a supply-chain lock for the proposed Aleph-Bench portable
profile. It does not implement a scorer and does not establish equivalence with
the immutable v0.2 Python 3.13 / UCD 15.1 scoring authority.

`portable-unicode-inputs-v1.json` records the exact official filenames, URLs,
acquisition times, sizes, SHA-256 digests, runtime tags, and retained licenses
for three Unicode 15.1 data files and the CPython 3.11 and 3.12 manylinux x86-64
`unicodedata2` 15.1.0 wheels. The wheel files are retained as archives only;
this slice neither installs nor imports them.

Run both offline checks from the repository root:

```bash
python3 scripts/check-v0-2-scoring-authority.py
python3 scripts/check-portable-unicode-inputs.py
```

The checkers have no update mode. They reject manifest drift, symlink or
hardlink aliases, artifact byte drift, unsafe or unexpected wheel members,
invalid wheel metadata/tags/RECORD entries, and retained-license mismatch.
Updating any pinned byte requires an explicit reviewed change to the checker
expectations and lock manifest.

See `THIRD_PARTY_NOTICES.md` for the separate Unicode and `unicodedata2`
attributions. Later table generation, portable scoring, differential proof,
and hosted Kaggle execution are outside this directory's claim.
