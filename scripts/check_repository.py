"""Check the public-repository requirements before submission."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDERS = ("[MEMBER", "[REPLACE", "[ADD MEMBER")
FORBIDDEN_TEXT = (
    "/" + "Users/",
    "/" + "home/" + "700053286/",
    "UAEU" + "HPCLB",
    "UAEU" + "HPC",
)
TEXT_SUFFIXES = {".md", ".py", ".sh", ".json", ".yml", ".yaml", ".txt", ".csv"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-placeholders", action="store_true",
                        help="Allow member and URL placeholders during private preparation")
    args = parser.parse_args()
    failures: list[str] = []

    required = [
        "README.md", "DATASET.md", "TEAM_COMMIT_PLAN.md", ".gitignore",
        "requirements.txt", "environment.yml", "config/protocol.json",
        "src/study.py", "src/report.py", "tests/test_experiment.py",
        "scripts/run_all.sh",
    ]
    for name in required:
        if not (ROOT / name).is_file():
            failures.append(f"missing required file: {name}")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for expected in [
        "https://github.com/zalandoresearch/fashion-mnist",
        "MIT", "22 September 2026", "Reproduce from a fresh clone",
        "Contributions", "AI and library disclosure",
    ]:
        if expected not in readme:
            failures.append(f"README is missing: {expected}")

    contribution_section = readme.split("## Contributions", 1)[-1].split("\n## ", 1)[0]
    contribution_rows = [
        line for line in contribution_section.splitlines()
        if line.startswith("| **")
    ]
    if len(contribution_rows) != 3:
        failures.append("README must contain exactly three named contribution rows")

    if not args.allow_placeholders:
        for placeholder in PLACEHOLDERS:
            if placeholder in readme:
                failures.append(f"replace README placeholder beginning {placeholder!r}")

    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        rel = path.relative_to(ROOT).as_posix()
        if path.suffix == ".ipynb":
            failures.append(f"notebook found; repository uses scripts instead: {rel}")
        if path.stat().st_size > 10 * 1024 * 1024:
            failures.append(f"file larger than 10 MiB: {rel}")
        if path.suffix.lower() in TEXT_SUFFIXES or path.name == ".gitignore":
            text = path.read_text(encoding="utf-8", errors="replace")
            for forbidden in FORBIDDEN_TEXT:
                if forbidden in text:
                    failures.append(f"private or hard-coded path token {forbidden!r} in {rel}")

    study = (ROOT / "src/study.py").read_text(encoding="utf-8")
    if not re.search(r"ROOT\s*=\s*Path\(__file__\)\.resolve\(\)\.parents\[1\]", study):
        failures.append("src/study.py does not anchor paths to the repository root")
    if "choices=[\"prepare\", \"train\", \"evaluate\", \"report\", \"all\"]" not in study:
        failures.append("src/study.py does not expose the complete staged pipeline")

    result = {
        "status": "failed" if failures else "passed",
        "repository_root": str(ROOT),
        "placeholders_allowed": args.allow_placeholders,
        "failures": failures,
    }
    print(json.dumps(result, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
