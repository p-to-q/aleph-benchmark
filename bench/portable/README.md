# Frozen Python string semantics

This directory contains the zero-model-call string-semantics slice for the
provisional Aleph-Bench portable profile. It is intentionally narrower than a
scorer: it freezes and consumes Python 3.13.2 / UCD 15.1 case folding,
whitespace, and no-argument split behavior without inheriting those operations
from the ambient interpreter.

The authority generator accepts `--write` only on CPython 3.13.2 with the
standard-library UCD 15.1.0. Its normal offline check is safe on other Python
runtimes:

```bash
python3 scripts/generate-portable-string-semantics.py --check
```

Generation has two accepted starting states: the exact checked-in five-file
closure, or a disposable repository-shaped staging root whose
`bench/portable/data` directory contains only the reviewed schema. The latter
is the clean-room path for a changed candidate. A runnable repository-shaped
staging closure contains the generator, the two pinned Unicode inputs, the
schema, `bench/__init__.py`, the `bench.portable` consumer package, and
`bench.engine.schema_validation` (including its package initializer). Run the
copied CLI from that staging root on the authority runtime, review the
resulting hashes, then update the consumer lock in the same change. Extra data
entries and aliased or special files fail closed. The manifest is published
after the receipt and tables.

The check verifies the content-addressed tables, strict manifest and schema,
generation receipt, retained Unicode inputs, all 1,114,112 code-point
positions (including 2,048 surrogates), the reviewed framed-stream digests,
and a deterministic split-transition corpus. On the exact authority runtime it
also regenerates and compares the ambient Python behavior; other runtimes only
self-check and consume the frozen bytes.

Portable consumers use `bench.portable.load_string_semantics()`. The provider
implements `casefold`, `is_whitespace`, and `split_whitespace` from the tables.
AST tests forbid ambient `casefold`, `isspace`, and no-argument `split` calls in
this package.

The manifest locks the schema, receipt, and two tables. The receipt binds the
authority interpreter/UCD identity, pinned Unicode inputs, generator source,
table hashes, and split corpus. It deliberately does **not** claim to hash the
consumer source; repository revision and release provenance cover that source,
and broader closure work remains tracked by #77.

Safe file inspection requires POSIX descriptor-relative operations and
`O_NOFOLLOW`, `O_DIRECTORY`, `O_CLOEXEC`, and `O_NONBLOCK`. Linux and macOS are
the supported generator/checker hosts. Windows currently fails closed instead
of silently weakening alias and race protection; the frozen JSON data remain
portable once supplied through a supported, verified release process.

This slice does not implement or modify a scorer, does not establish numeric
equivalence with protocol 0.2, does not run a model, and does not authorize a
Kaggle or Hugging Face result.
