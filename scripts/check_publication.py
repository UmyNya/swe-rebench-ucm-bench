#!/usr/bin/env python3
"""Check publication files and reachable Git history without printing secrets.

Only byte-identical public dataset payloads are exempt from the absolute-path
rule: their original task text contains public example and working paths.
Private markers and credential patterns are still checked in those files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRECTORIES = {".git", "__pycache__", ".pytest_cache", "results", "runs", "local", "private", "prompt-snapshots"}
PRIVATE_NAMES = {"client-result.json", "client-config.json", "request-records.jsonl", "owned-processes.json", "launch.json"}
MACHINE_PATH = re.compile(rb"/(?:home|root|data|tmp|mnt|workspace)/[A-Za-z0-9_.~+/-]+|[A-Za-z]:\\Users\\")
CREDENTIAL = re.compile(rb"(?:gh[pousr]_[A-Za-z0-9_]{30,}|github_pat_[A-Za-z0-9_]{40,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)")
URL_CREDENTIAL = re.compile(rb"https?://[^\s/\"']+:[^\s/\"']+@")


def git(*args: str, root: Path = ROOT) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL)


def is_public_payload(name: str) -> bool:
    return (
        name.startswith("data/original/text_sources/")
        or name.endswith(".parquet")
        or name.endswith("/tools.json")
        or name.endswith("/tokenizer.json")
    )


def inspect_content(name: str, content: bytes, checksums: dict, private_markers: list[str]) -> list[str]:
    """Return rule names, never matching values or surrounding private text."""
    problems = []
    path = Path(name)
    if (
        any(part in EXCLUDED_DIRECTORIES - {".git"} for part in path.parts)
        or path.name in PRIVATE_NAMES
        or path.name.startswith(".env")
        or path.name.endswith((".local.json", ".u32le.gz", ".log"))
        or "prompt-snapshot" in path.name
    ):
        problems.append("private_artifact_name")
    for marker in private_markers:
        if marker and (marker.encode() in content or marker in name):
            problems.append("private_marker")
            break
    trusted_payload = is_public_payload(name) and checksums.get(name) == hashlib.sha256(content).hexdigest()
    if MACHINE_PATH.search(content) and not trusted_payload:
        problems.append("machine_absolute_path")
    if CREDENTIAL.search(content) or URL_CREDENTIAL.search(content):
        problems.append("credential_pattern")
    if re.search(rb'"private_artifact"\s*:\s*true', content):
        problems.append("marked_private_artifact")
    return problems


def audit(root: Path, *, staged: bool, history: bool, private_markers: list[str]) -> dict:
    checksums = json.loads((root / "reference/data-checksums.json").read_text())
    findings = []
    checked = 0
    if staged:
        names = [name for name in git("ls-files", "-z", root=root).decode().split("\0") if name]
        files = [(name, git("show", ":" + name, root=root)) for name in names]
    else:
        files = []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            name = str(path.relative_to(root))
            if any(part in EXCLUDED_DIRECTORIES or part.startswith(".venv") for part in path.relative_to(root).parts):
                continue
            files.append((name, path.read_bytes()))
    for name, content in files:
        checked += 1
        rules = inspect_content(name, content, checksums, private_markers)
        if rules:
            findings.append({"scope": "index" if staged else "tree", "file": name, "rules": rules})

    history_objects = 0
    if history:
        objects = git("rev-list", "--objects", "--all", root=root).decode().splitlines()
        for item in objects:
            object_id, _, name = item.partition(" ")
            kind = git("cat-file", "-t", object_id, root=root).strip()
            if kind not in {b"blob", b"commit"}:
                continue
            content = git("cat-file", "-p", object_id, root=root)
            history_objects += 1
            rules = inspect_content(name or "commit-message", content, checksums, private_markers)
            if kind == b"commit":
                identities = re.findall(rb"^(?:author|committer) [^\n]*<([^>]+)>", content, re.MULTILINE)
                if any(not email.endswith(b"@users.noreply.github.com") for email in identities):
                    rules.append("non_noreply_commit_email")
            if rules:
                findings.append({"scope": "history", "object": object_id, "file": name, "rules": rules})
    return {"passed": not findings, "files_checked": checked, "history_objects_checked": history_objects, "findings": findings}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true", help="Inspect Git index bytes instead of worktree files")
    parser.add_argument("--history", action="store_true", help="Also inspect every reachable commit and blob")
    parser.add_argument("--private-markers-file", type=Path, help="Local JSON array of private identifiers; values are never printed")
    args = parser.parse_args()
    markers = json.loads(args.private_markers_file.read_text()) if args.private_markers_file else []
    report = audit(ROOT, staged=args.staged, history=args.history, private_markers=markers)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
