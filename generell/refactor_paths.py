#!/usr/bin/env python3
"""
Final working version - uses the proven broad detection logic
"""

import argparse
import re
from pathlib import Path
from datetime import datetime

OLD_PREFIX = r'C:\Users\stefa\Documents\UNI\Master\Masterarbeit\Daten'


def get_project_root_block() -> str:
    return """
from pathlib import Path

# ===============================================
# PATHS UPDATED TO RELATIVE (Daten root)
# ===============================================
SCRIPT = Path(__file__).resolve()

PROJECT_ROOT = SCRIPT
while PROJECT_ROOT.name != "Daten":
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_ROOT = PROJECT_ROOT
""".strip()


def make_relative_path(path_str: str) -> str:
    normalized = path_str.replace("\\", "/").lower()
    prefix_norm = OLD_PREFIX.replace("\\", "/").lower()
    rel = normalized.replace(prefix_norm, "").lstrip("/")

    if not rel:
        return "DATA_ROOT"

    parts = [p for p in rel.split("/") if p]
    return "DATA_ROOT / " + " / ".join(f'"{p}"' for p in parts)


def process_file(file_path: Path, apply: bool):
    suffix = file_path.suffix.lower()
    is_python = suffix == ".py"
    is_config = suffix in {".yaml", ".yml", ".json"}

    if not (is_python or is_config):
        return False, 0

    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
    except:
        return False, 0

    # === Use the proven broad detection ===
    pattern = r'["\']([^"\']*Daten[^"\']*)["\']'
    matches = re.findall(pattern, content, flags=re.IGNORECASE)

    if not matches:
        return False, 0

    count = len(matches)

    if not apply:
        return True, count

    # === Replacement ===
    def replacer(match):
        return make_relative_path(match.group(1))

    new_content = re.sub(pattern, replacer, content, flags=re.IGNORECASE)

    if is_python:
        # Insert clean PROJECT_ROOT block
        if "DATA_ROOT = PROJECT_ROOT" not in new_content:
            lines = new_content.splitlines(keepends=True)
            last_import = -1
            for i, line in enumerate(lines):
                if line.strip().startswith(("import ", "from ")):
                    last_import = i

            block = "\n" + get_project_root_block() + "\n\n"
            if last_import >= 0:
                new_content = "".join(lines[:last_import+1]) + block + "".join(lines[last_import+1:])
            else:
                new_content = block + new_content

    # Backup + write
    backup = file_path.with_suffix(file_path.suffix + ".bak")
    backup.write_text(content, encoding="utf-8")
    file_path.write_text(new_content, encoding="utf-8")

    return True, count


def backup_all_python_files(code_dir: Path):
    """Create .bak for every .py file (even if it has no absolute paths)."""
    print("Creating .bak backups for all Python files...")
    for py_file in code_dir.rglob("*.py"):
        backup_path = py_file.with_suffix(py_file.suffix + ".bak")
        if not backup_path.exists():
            backup_path.write_text(py_file.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
    print("Backups created.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    root = args.root.resolve()
    code_dir = root / "CODE"

    print(f"Scanning inside: {code_dir}")
    print(f"Mode: {'APPLY' if args.apply else 'DRY-RUN'}")
    print("-" * 60)

    if args.apply:
        backup_all_python_files(code_dir)

    total_files = total_paths = 0

    for f in code_dir.rglob("*"):
        if f.is_file():
            changed, count = process_file(f, apply=args.apply)
            if changed:
                print(f"[{'changed' if args.apply else 'would change'}] {f.relative_to(root)} ({count} paths)")
                total_files += 1
                total_paths += count

    print("-" * 60)
    print(f"Files with absolute paths: {total_files}")
    print(f"Total paths found: {total_paths}")

    if args.apply:
        print("\nDone. All .py files have .bak backups.")
        print("Test your pipelines now.")
    else:
        print("\nDry-run finished. Use --apply when ready.")


if __name__ == "__main__":
    main()