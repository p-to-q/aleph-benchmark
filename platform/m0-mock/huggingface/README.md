# Hugging Face Upload Preparation

This directory prepares the M0 package for a Hugging Face Dataset upload. It does not upload anything unless `upload_dataset.py --execute` is used with `HF_TOKEN`.

## Local Dry Run

From the Aleph repository root:

```bash
python3 bench/results/platform/m0-mock/huggingface/upload_dataset.py --dry-run
```

The dry run validates:

- root `README.md` has dataset-card front matter;
- `package-manifest.json` declares the Hugging Face Dataset target;
- every manifest artifact exists and matches its sha256/byte count;
- `checksums.sha256` matches the manifest artifact list;
- required data, schema, evidence, Kaggle, and Hugging Face preparation files are present.

## Actual Upload

After confirming repo ownership and token scope:

```bash
HF_TOKEN=... python3 bench/results/platform/m0-mock/huggingface/upload_dataset.py --execute --repo-id p-to-q/aleph-bench
```

The script creates the dataset repo if needed, then uploads the package folder with `huggingface_hub.upload_folder`.

## Benchmark Boundary

The Hugging Face Dataset upload is a platform distribution step, not a real model leaderboard. A Hugging Face benchmark or leaderboard surface should be created only after hosted black-box rows exist, or as an explicitly labeled mock/demo Space that reads this dataset and refuses to rank models as real evidence.
