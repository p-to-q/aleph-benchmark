#!/usr/bin/env python3
"""Run deterministic, offline hygiene checks for the standalone repository.

The migration starts with a generated historical snapshot.  This checker keeps
the public surface honest while source, tests, and release tooling move here in
later reviewable slices.  It deliberately uses only the Python standard
library so a clean checkout can run it without network access.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
IGNORED_DIRECTORIES = {".git", ".pytest_cache", "__pycache__"}
REQUIRED_FILES = {
    ".gitattributes",
    ".github/workflows/repo-hygiene.yml",
    "CITATION.cff",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "DATA_LICENSE.md",
    "LICENSE",
    "README.md",
    "SECURITY.md",
    "THIRD_PARTY_NOTICES.md",
    "docs/decisions/0001-standalone-source-authority.md",
    "docs/migration-plan.md",
    "scripts/check_repo_hygiene.py",
    "scripts/test_repo_hygiene.py",
}
TEXT_FILE_SUFFIXES = {
    ".cff",
    ".csv",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".sha256",
    ".txt",
    ".yaml",
    ".yml",
}
TEXT_FILE_NAMES = {".gitattributes", ".gitignore", "LICENSE"}
MAX_TEXT_FILE_BYTES = 5 * 1024 * 1024
APACHE_2_LICENSE_SHA256 = (
    "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
)
CITATION_CFF_SHA256 = (
    "5977b75822f6a3d49b3cb3905a13ffbbc0f2d2f90acac39e3f1c97000d73cec7"
)

# The pattern intentionally handles repository documentation rather than the
# full Markdown grammar.  Angle-bracket targets and optional titles are enough
# for the checked-in documents; external URLs are ignored after parsing.
MARKDOWN_LINK = re.compile(
    r"!?\[[^\]]*\]\(\s*(?P<target><[^>]+>|[^\s)]+)(?:\s+['\"][^)]*['\"])?\s*\)"
)
MARKDOWN_REFERENCE_USE = re.compile(
    r"!?\[(?P<text>[^\]]+)\]\[(?P<label>[^\]]*)\]"
)
MARKDOWN_REFERENCE_DEFINITION = re.compile(
    r"^[ \t]{0,3}\[(?P<label>[^\]]+)\]:[ \t]*(?P<target><[^>]+>|\S+)",
    re.MULTILINE,
)
MAINTAINER_LOCAL_PATHS = (
    re.compile("/" + r"Users/[A-Za-z0-9._-]+/"),
    re.compile("/" + r"home/[A-Za-z0-9._-]+/"),
    re.compile(r"\b[A-Za-z]:[\\/]"),
    re.compile("file" + "://", re.IGNORECASE),
)
PLACEHOLDER = re.compile(
    r"(?:\bTODO\b|\bTBD\b|example\.com|your[-_ ]?(?:name|org|doi))",
    re.IGNORECASE,
)


def relative_path(path: Path) -> str:
    """Return repository paths in the same POSIX form on every host."""

    return path.relative_to(REPOSITORY_ROOT).as_posix()


def repository_files() -> tuple[list[Path], list[str]]:
    """Return bounded regular files and any unsafe filesystem findings."""

    files: list[Path] = []
    errors: list[str] = []
    for directory, directory_names, file_names in os.walk(
        REPOSITORY_ROOT, followlinks=False
    ):
        base = Path(directory)
        kept_directories: list[str] = []
        for name in sorted(directory_names):
            candidate = base / name
            if name in IGNORED_DIRECTORIES:
                continue
            if candidate.is_symlink():
                errors.append(
                    f"symbolic-link directory is not allowed: "
                    f"{relative_path(candidate)}"
                )
                continue
            kept_directories.append(name)
        directory_names[:] = kept_directories

        for name in sorted(file_names):
            candidate = base / name
            relative = relative_path(candidate)
            if candidate.is_symlink():
                errors.append(f"symbolic-link file is not allowed: {relative}")
                continue
            if not candidate.is_file():
                errors.append(f"non-regular repository entry is not allowed: {relative}")
                continue
            files.append(candidate)
    return files, errors


def read_text(path: Path) -> tuple[str | None, str | None]:
    """Read one expected text file with an explicit resource bound."""

    size = path.stat().st_size
    if size > MAX_TEXT_FILE_BYTES:
        return None, f"text file exceeds {MAX_TEXT_FILE_BYTES} bytes: {relative_path(path)}"
    try:
        return path.read_text(encoding="utf-8"), None
    except UnicodeDecodeError:
        return None, f"text file is not UTF-8: {relative_path(path)}"


def is_expected_text(path: Path) -> bool:
    return path.name in TEXT_FILE_NAMES or path.suffix.lower() in TEXT_FILE_SUFFIXES


def _normalize_reference_label(value: str) -> str:
    return " ".join(value.split()).casefold()


def _check_markdown_target(path: Path, raw_target: str) -> list[str]:
    errors: list[str] = []
    target = raw_target[1:-1] if raw_target.startswith("<") else raw_target
    target = unquote(target)
    if target.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", target):
        return [f"absolute local Markdown link in {relative_path(path)}: {target}"]
    try:
        parsed = urlsplit(target)
    except ValueError:
        return [f"malformed Markdown link in {relative_path(path)}: {target}"]
    if parsed.scheme or target.startswith("//") or target.startswith("#"):
        return []

    relative_target = parsed.path
    if not relative_target:
        return []
    resolved = (path.parent / relative_target).resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError:
        return [f"Markdown link escapes repository in {relative_path(path)}: {target}"]
    if not resolved.exists():
        errors.append(f"broken local Markdown link in {relative_path(path)}: {target}")
    return errors


def check_local_markdown_links(path: Path, text: str) -> list[str]:
    """Validate inline and full/collapsed reference-style Markdown links."""

    errors: list[str] = []
    for match in MARKDOWN_LINK.finditer(text):
        errors.extend(_check_markdown_target(path, match.group("target")))

    definitions: dict[str, str] = {}
    for match in MARKDOWN_REFERENCE_DEFINITION.finditer(text):
        label = _normalize_reference_label(match.group("label"))
        if label in definitions:
            errors.append(
                f"duplicate Markdown reference label in {relative_path(path)}: {label}"
            )
            continue
        target = match.group("target")
        definitions[label] = target
        errors.extend(_check_markdown_target(path, target))
    for match in MARKDOWN_REFERENCE_USE.finditer(text):
        label = match.group("label") or match.group("text")
        normalized = _normalize_reference_label(label)
        if normalized not in definitions:
            errors.append(
                f"undefined Markdown reference label in {relative_path(path)}: {label}"
            )
    return errors


def require_literal(text: str, literal: str, path: str) -> str | None:
    if literal not in text:
        return f"{path} is missing required value: {literal}"
    return None


def check_governance_contents(text_by_path: dict[str, str]) -> list[str]:
    errors: list[str] = []

    citation = text_by_path.get("CITATION.cff", "")
    for literal in (
        "cff-version: 1.2.0",
        'title: "Aleph-Bench"',
        "type: software",
        'repository-code: "https://github.com/p-to-q/aleph-benchmark"',
        "license: Apache-2.0",
    ):
        error = require_literal(citation, literal, "CITATION.cff")
        if error:
            errors.append(error)
    if PLACEHOLDER.search(citation):
        errors.append("CITATION.cff contains a placeholder value")
    citation_path = REPOSITORY_ROOT / "CITATION.cff"
    if citation_path.is_file():
        observed_citation_sha256 = hashlib.sha256(citation_path.read_bytes()).hexdigest()
        if observed_citation_sha256 != CITATION_CFF_SHA256:
            errors.append(
                "CITATION.cff differs from the reviewed CFF 1.2 record; "
                "update the record and its pinned digest together"
            )

    data_license = text_by_path.get("DATA_LICENSE.md", "")
    for literal in (
        "CC0-1.0",
        "https://creativecommons.org/publicdomain/zero/1.0/legalcode",
        "Apache-2.0",
        "platform/m0-mock/data/**",
        "platform/m0-mock/evidence/**",
        "platform/m0-mock/kaggle/**",
        "platform/m0-mock/huggingface/**",
        "platform/m0-mock/schemas/**",
        "Root project documentation, workflows, and scripts",
        "`references/**`",
        "THIRD_PARTY_NOTICES.md",
    ):
        error = require_literal(data_license, literal, "DATA_LICENSE.md")
        if error:
            errors.append(error)

    third_party_notices = text_by_path.get("THIRD_PARTY_NOTICES.md", "")
    for literal in (
        "references/kaggle-bench/example_task.py",
        "references/kaggle-bench/kaggle_benchmarks_reference.md",
        "0d759286b5dcb83878e27840fcb42b559062abf3",
        "ecf1a830319276871f90b574dd7d46de042f2584",
        "Copyright 2025 Kaggle Inc.",
        "Apache-2.0",
    ):
        error = require_literal(
            third_party_notices, literal, "THIRD_PARTY_NOTICES.md"
        )
        if error:
            errors.append(error)

    kaggle_example = text_by_path.get(
        "references/kaggle-bench/example_task.py", ""
    )
    for literal in (
        "0d759286b5dcb83878e27840fcb42b559062abf3",
        "Copyright 2025 Kaggle Inc.",
        "SPDX-License-Identifier: Apache-2.0",
    ):
        error = require_literal(
            kaggle_example, literal, "references/kaggle-bench/example_task.py"
        )
        if error:
            errors.append(error)

    kaggle_reference = text_by_path.get(
        "references/kaggle-bench/kaggle_benchmarks_reference.md", ""
    )
    for literal in (
        "ecf1a830319276871f90b574dd7d46de042f2584",
        "quick_start.md",
        "cookbook.md",
        "Copyright 2025 Kaggle Inc.",
        "SPDX-License-Identifier: Apache-2.0",
    ):
        error = require_literal(
            kaggle_reference,
            literal,
            "references/kaggle-bench/kaggle_benchmarks_reference.md",
        )
        if error:
            errors.append(error)

    license_text = text_by_path.get("LICENSE", "")
    for literal in (
        "Apache License",
        "Version 2.0, January 2004",
        "http://www.apache.org/licenses/",
        "END OF TERMS AND CONDITIONS",
    ):
        error = require_literal(license_text, literal, "LICENSE")
        if error:
            errors.append(error)
    if len(license_text.encode("utf-8")) < 10_000:
        errors.append("LICENSE is shorter than the complete Apache-2.0 text")
    license_path = REPOSITORY_ROOT / "LICENSE"
    if license_path.is_file():
        observed_license_sha256 = hashlib.sha256(license_path.read_bytes()).hexdigest()
        if observed_license_sha256 != APACHE_2_LICENSE_SHA256:
            errors.append(
                "LICENSE does not match the canonical Apache-2.0 text "
                f"({observed_license_sha256})"
            )

    readme = text_by_path.get("README.md", "")
    for literal in (
        "historical generated preview",
        "deterministic mock evidence",
        "benchmark/source-v0.2",
    ):
        error = require_literal(readme, literal, "README.md")
        if error:
            errors.append(error)

    workflow = text_by_path.get(".github/workflows/repo-hygiene.yml", "")
    action_references = re.findall(
        r"^[ \t]*(?:-[ \t]*)?uses:[ \t]*[^@\s]+@([^\s#]+)",
        workflow,
        re.MULTILINE,
    )
    if not action_references:
        errors.append("repository hygiene workflow has no action references")
    for reference in action_references:
        if re.fullmatch(r"[0-9a-f]{40}", reference) is None:
            errors.append(
                "repository hygiene workflow action is not pinned to a full commit SHA"
            )
    return errors


def main() -> int:
    files, errors = repository_files()
    relative_files = {relative_path(path) for path in files}
    for required in sorted(REQUIRED_FILES - relative_files):
        errors.append(f"required governance file is missing: {required}")

    text_by_path: dict[str, str] = {}
    for path in files:
        if not is_expected_text(path):
            continue
        relative = relative_path(path)
        text, read_error = read_text(path)
        if read_error:
            errors.append(read_error)
            continue
        assert text is not None
        text_by_path[relative] = text
        for pattern in MAINTAINER_LOCAL_PATHS:
            if pattern.search(text):
                errors.append(f"maintainer-local path found in {relative}")
                break
        if path.suffix.lower() == ".md":
            errors.extend(check_local_markdown_links(path, text))

    errors.extend(check_governance_contents(text_by_path))
    if errors:
        for error in sorted(set(errors)):
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"repository hygiene passed ({len(files)} regular files checked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
