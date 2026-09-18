from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import check_repo_hygiene as hygiene


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class RepositoryHygieneTests(unittest.TestCase):
    def test_maintainer_local_path_patterns_cover_common_hosts(self) -> None:
        examples = (
            "/" + "Users/alice/project/file.md",
            "/" + "home/alice/project/file.md",
            "C:" + "\\Users\\alice\\project\\file.md",
            "C:" + "/" + "Users/alice/project/file.md",
            "file" + ":///private/tmp/report.md",
        )
        for example in examples:
            with self.subTest(example=example):
                self.assertTrue(
                    any(pattern.search(example) for pattern in hygiene.MAINTAINER_LOCAL_PATHS)
                )

    def test_inline_and_reference_links_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            document = root / "README.md"
            document.write_text("placeholder\n", encoding="utf-8")
            with patch.object(hygiene, "REPOSITORY_ROOT", root):
                errors = hygiene.check_local_markdown_links(
                    document,
                    "[inline](missing.md)\n[reference][missing]\n",
                )
        self.assertTrue(any("broken local Markdown link" in error for error in errors))
        self.assertTrue(any("undefined Markdown reference" in error for error in errors))

    def test_reference_definition_target_is_validated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            document = root / "README.md"
            document.write_text("placeholder\n", encoding="utf-8")
            with patch.object(hygiene, "REPOSITORY_ROOT", root):
                errors = hygiene.check_local_markdown_links(
                    document,
                    "[reference][target]\n\n[target]: missing.md\n",
                )
        self.assertEqual(len(errors), 1)
        self.assertIn("broken local Markdown link", errors[0])

    def test_windows_absolute_markdown_link_is_not_mistaken_for_url_scheme(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            document = root / "README.md"
            document.write_text("placeholder\n", encoding="utf-8")
            with patch.object(hygiene, "REPOSITORY_ROOT", root):
                errors = hygiene.check_local_markdown_links(
                    document,
                    "[local](" + "C:" + "\\Users\\alice\\file.md)\n",
                )
        self.assertEqual(len(errors), 1)
        self.assertIn("absolute local Markdown link", errors[0])

    def test_citation_record_is_pinned_after_independent_validation(self) -> None:
        citation = (REPOSITORY_ROOT / "CITATION.cff").read_text(encoding="utf-8")
        license_text = (REPOSITORY_ROOT / "LICENSE").read_text(encoding="utf-8")
        data_license = (REPOSITORY_ROOT / "DATA_LICENSE.md").read_text(
            encoding="utf-8"
        )
        readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
        workflow = (
            REPOSITORY_ROOT / ".github/workflows/repo-hygiene.yml"
        ).read_text(encoding="utf-8")
        values = {
            "CITATION.cff": citation,
            "DATA_LICENSE.md": data_license,
            "LICENSE": license_text,
            "README.md": readme,
            "THIRD_PARTY_NOTICES.md": (
                REPOSITORY_ROOT / "THIRD_PARTY_NOTICES.md"
            ).read_text(encoding="utf-8"),
            "references/kaggle-bench/example_task.py": (
                REPOSITORY_ROOT / "references/kaggle-bench/example_task.py"
            ).read_text(encoding="utf-8"),
            "references/kaggle-bench/kaggle_benchmarks_reference.md": (
                REPOSITORY_ROOT
                / "references/kaggle-bench/kaggle_benchmarks_reference.md"
            ).read_text(encoding="utf-8"),
            ".github/workflows/repo-hygiene.yml": workflow,
        }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "CITATION.cff").write_text(citation, encoding="utf-8")
            (root / "LICENSE").write_text(license_text, encoding="utf-8")
            with patch.object(hygiene, "REPOSITORY_ROOT", root):
                self.assertEqual(hygiene.check_governance_contents(values), [])

                (root / "CITATION.cff").write_text(
                    citation + "broken: [\n", encoding="utf-8"
                )
                errors = hygiene.check_governance_contents(values)
        self.assertTrue(any("reviewed CFF 1.2 record" in error for error in errors))

        incomplete_notice = dict(values)
        incomplete_notice["THIRD_PARTY_NOTICES.md"] = incomplete_notice[
            "THIRD_PARTY_NOTICES.md"
        ].replace("ecf1a830319276871f90b574dd7d46de042f2584", "")
        errors = hygiene.check_governance_contents(incomplete_notice)
        self.assertTrue(
            any(
                "THIRD_PARTY_NOTICES.md is missing required value" in error
                and "ecf1a830319276871f90b574dd7d46de042f2584" in error
                for error in errors
            )
        )

    def test_action_references_must_be_immutable(self) -> None:
        citation = (REPOSITORY_ROOT / "CITATION.cff").read_text(encoding="utf-8")
        license_text = (REPOSITORY_ROOT / "LICENSE").read_text(encoding="utf-8")
        values = {
            "CITATION.cff": citation,
            "DATA_LICENSE.md": (REPOSITORY_ROOT / "DATA_LICENSE.md").read_text(
                encoding="utf-8"
            ),
            "LICENSE": license_text,
            "README.md": (REPOSITORY_ROOT / "README.md").read_text(
                encoding="utf-8"
            ),
            "THIRD_PARTY_NOTICES.md": (
                REPOSITORY_ROOT / "THIRD_PARTY_NOTICES.md"
            ).read_text(encoding="utf-8"),
            "references/kaggle-bench/example_task.py": (
                REPOSITORY_ROOT / "references/kaggle-bench/example_task.py"
            ).read_text(encoding="utf-8"),
            "references/kaggle-bench/kaggle_benchmarks_reference.md": (
                REPOSITORY_ROOT
                / "references/kaggle-bench/kaggle_benchmarks_reference.md"
            ).read_text(encoding="utf-8"),
            ".github/workflows/repo-hygiene.yml": (
                "steps:\n  - uses: actions/checkout@v4\n"
            ),
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "CITATION.cff").write_text(citation, encoding="utf-8")
            (root / "LICENSE").write_text(license_text, encoding="utf-8")
            with patch.object(hygiene, "REPOSITORY_ROOT", root):
                errors = hygiene.check_governance_contents(values)
        self.assertTrue(any("full commit SHA" in error for error in errors))

    def test_relative_contract_paths_use_posix_separators(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            nested = root / "docs" / "decision.md"
            nested.parent.mkdir()
            nested.write_text("ok\n", encoding="utf-8")
            with patch.object(hygiene, "REPOSITORY_ROOT", root):
                self.assertEqual(hygiene.relative_path(nested), "docs/decision.md")


if __name__ == "__main__":
    unittest.main()
