"""Hugging Face Dataset upload preparation for AlephBench Frozen Ladder.

Default mode is a local dry run. Real upload requires --execute and HF_TOKEN.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_ID = "p-to-q/aleph-bench"
REQUIRED_FILES = [
    "README.md",
    "EVALUATION.md",
    "PLATFORM_LAUNCH_CHECKLIST.md",
    "package-manifest.json",
    "checksums.sha256",
    "croissant.json",
    "dataset-metadata.json",
    "data/public_s2_items.jsonl",
    "data/public_s2_prompts.jsonl",
    "data/public_s2_items.csv",
    "data/public_s2_prompts.csv",
    "data/submission_format.csv",
    "schemas/aleph-bench-result.schema.json",
    "evidence/m0-first-run.json",
    "evidence/m0-evidence.md",
    "evidence/mock_model_summary.csv",
    "evidence/mock_item_metrics.csv",
    "kaggle/aleph_bench_frozen_ladder_task_cli.py",
    "kaggle/aleph_bench_m0_task.py",
    "kaggle/api_test_smoke.py",
    "kaggle/_scoring.py",
    "kaggle/score_outputs.py",
    "huggingface/README.md",
    "huggingface/upload_dataset.py",
]
FORBIDDEN_FILES = [
    "data/mock_model_summary.csv",
    "data/mock_item_metrics.csv",
]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def expected_checksums(manifest: dict[str, Any]) -> str:
    lines = [f"{artifact['sha256']}  {artifact['path']}" for artifact in manifest["artifacts"]]
    return "\n".join(lines) + "\n"


def validate_package(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    manifest_path = root / "package-manifest.json"
    if not manifest_path.exists():
        errors.append("missing package-manifest.json")
        manifest: dict[str, Any] = {"artifacts": [], "targetPlatforms": []}
    else:
        manifest = load_json(manifest_path)

    for relative in REQUIRED_FILES:
        if not (root / relative).exists():
            errors.append(f"missing required file: {relative}")
    for relative in FORBIDDEN_FILES:
        if (root / relative).exists():
            errors.append(
                f"forbidden file present: {relative} (mock evidence must live under evidence/, "
                "not data/, so HF / Kaggle previews don't render it as a leaderboard)"
            )

    readme_path = root / "README.md"
    if readme_path.exists():
        readme = readme_path.read_text(encoding="utf-8")
        if not readme.startswith("---\n"):
            errors.append("README.md is missing dataset-card YAML front matter")
        if "configs:" not in readme:
            errors.append("README.md does not declare dataset configs")
        if "data/public_s2_items.jsonl" not in readme:
            errors.append("README.md does not declare the public-s2 item file")
        if "data/public_s2_prompts.jsonl" not in readme:
            errors.append("README.md does not declare the public-s2 prompt file")
        if "deterministic mock pipeline outputs" not in readme:
            errors.append(
                "README.md is missing the mock-evidence boundary banner ('deterministic mock pipeline outputs')"
            )

    if "huggingface_dataset" not in manifest.get("targetPlatforms", []):
        errors.append("package manifest does not target huggingface_dataset")
    if manifest.get("evidenceMode") != "mock":
        errors.append("package manifest evidenceMode should remain mock until hosted rows exist")

    for artifact in manifest.get("artifacts", []):
        path = root / artifact["path"]
        if not path.exists():
            errors.append(f"manifest artifact missing: {artifact['path']}")
            continue
        data = path.read_bytes()
        if sha256(data) != artifact["sha256"]:
            errors.append(f"sha256 mismatch: {artifact['path']}")
        if len(data) != artifact["bytes"]:
            errors.append(f"byte count mismatch: {artifact['path']}")

    checksums_path = root / "checksums.sha256"
    if checksums_path.exists() and checksums_path.read_text(encoding="utf-8") != expected_checksums(manifest):
        errors.append("checksums.sha256 does not match manifest artifact list")

    return {
        "status": "failed" if errors else "ok",
        "packageRoot": str(root),
        "repoType": "dataset",
        "repoId": manifest.get("huggingFaceDatasetId", DEFAULT_REPO_ID),
        "artifactCount": len(manifest.get("artifacts", [])),
        "requiredFileCount": len(REQUIRED_FILES),
        "errors": errors,
    }


def upload(root: Path, repo_id: str, private: bool, commit_message: str) -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required for --execute")
    try:
        from huggingface_hub import HfApi, upload_folder
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub before --execute") from exc

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    upload_folder(
        folder_path=str(root),
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        commit_message=commit_message,
    )
    return f"https://huggingface.co/datasets/{repo_id}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Validate upload readiness without uploading.")
    parser.add_argument("--execute", action="store_true", help="Create/update the Hugging Face Dataset repo.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--commit-message", default="Add AlephBench Frozen Ladder platform package")
    args = parser.parse_args()

    if args.dry_run and args.execute:
        raise SystemExit("Use either --dry-run or --execute, not both")

    report = validate_package(PACKAGE_ROOT)
    report["requestedRepoId"] = args.repo_id
    report["mode"] = "execute" if args.execute else "dry-run"

    if report["errors"]:
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(1)

    if args.execute:
        report["url"] = upload(PACKAGE_ROOT, args.repo_id, args.private, args.commit_message)
    else:
        report["url"] = None

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
