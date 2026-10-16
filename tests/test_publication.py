"""Verify publication checks against staged bytes and deleted history content."""
from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.check_publication import audit, inspect_content


class PublicationTests(unittest.TestCase):
    def test_public_payload_exemption_requires_exact_hash_and_still_checks_secrets(self):
        name = "data/original/text_sources/example.txt"
        content = ("/" + "home" + "/example/task.py").encode()
        checksums = {name: hashlib.sha256(content).hexdigest()}
        self.assertEqual(inspect_content(name, content, checksums, []), [])
        self.assertIn("machine_absolute_path", inspect_content(name, content + b"!", checksums, []))
        self.assertIn("private_marker", inspect_content(name, content, checksums, ["example"]))
        credential = ("ghp_" + "a" * 36).encode()
        checksums[name] = hashlib.sha256(credential).hexdigest()
        self.assertIn("credential_pattern", inspect_content(name, credential, checksums, []))

    def test_private_artifacts_are_rejected(self):
        for name in ["results/record.json", "prompt-snapshots/manifest.json", ".env", "client-result.json"]:
            with self.subTest(name=name):
                self.assertIn("private_artifact_name", inspect_content(name, b"{}", {}, []))
        marked = json.dumps({"private_artifact": True}).encode()
        self.assertIn("marked_private_artifact", inspect_content("notes.json", marked, {}, []))

    def test_staged_check_reads_index_and_history_finds_deleted_private_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "reference").mkdir()
            (root / "reference/data-checksums.json").write_text("{}")

            def git(*args):
                subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)

            git("init", "-q")
            git("config", "user.name", "Test")
            git("config", "user.email", "test@users.noreply.github.com")
            (root / "notes.txt").write_text("public text")
            git("add", ".")
            (root / "notes.txt").write_text("restricted-marker")
            self.assertTrue(audit(root, staged=True, history=False, private_markers=["restricted-marker"])["passed"])
            self.assertFalse(audit(root, staged=False, history=False, private_markers=["restricted-marker"])["passed"])
            git("add", ".")
            git("commit", "-qm", "fixture with private marker")
            git("rm", "notes.txt")
            git("commit", "-qm", "remove fixture")
            report = audit(root, staged=True, history=True, private_markers=["restricted-marker"])
            self.assertFalse(report["passed"])
            self.assertTrue(any(item["scope"] == "history" and "private_marker" in item["rules"] for item in report["findings"]))


if __name__ == "__main__":
    unittest.main()
