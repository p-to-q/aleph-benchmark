# Repository license scopes

The historical snapshot mixes benchmark data, deterministic mock evidence,
project software, project documentation, and attributed third-party reference
material. The license is therefore determined by path:

| Path | License or terms | Scope note |
|---|---|---|
| `platform/m0-mock/data/**` | **CC0-1.0** | Public synthetic S2 rows, prompts, and submission-format data. |
| `platform/m0-mock/evidence/**` | **CC0-1.0** | Deterministic mock output artifacts only; these are not observations of real models. |
| `platform/m0-mock/kaggle/**`, `platform/m0-mock/huggingface/**`, and `platform/m0-mock/schemas/**` | **Apache-2.0** | Original integration code, helpers, and schemas. |
| Other files under `platform/m0-mock/` | **Apache-2.0** | Original documentation, manifests, checksums, and dataset metadata. |
| Root project documentation, workflows, and scripts | **Apache-2.0** | Original repository material. |
| `references/**` | Source-specific; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) | Non-normative imported or derived material. |

The canonical CC0-1.0 legal code is available from
[Creative Commons](https://creativecommons.org/publicdomain/zero/1.0/legalcode).
The complete Apache-2.0 text is in [`LICENSE`](LICENSE).

License fields in the frozen historical platform metadata—including
`platform/m0-mock/huggingface/README.md`, `dataset-metadata.json`, and
`croissant.json`—describe the dataset or its data records. They do not
relicense the code, schemas, documentation, or metadata files in the matrix
above. The snapshot is retained unchanged for provenance, so this root
statement resolves those legacy package-level ambiguities.

Files under `references/` are excluded from benchmark release assets.
Distributing a reference in this repository does not claim original authorship
or change its source license.
