"""Copy report/results assets into site/ for Quarto render (not committed)."""
from __future__ import annotations

import shutil
from pathlib import Path

SITE = Path(__file__).resolve().parents[1]
ROOT = SITE.parent
ASSETS = SITE / "assets" / "ext"


def copy_tree(src: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)


def copy_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)

    copy_tree(ROOT / "results" / "figures", ASSETS / "figures")
    copy_file(ROOT / "reports" / "report.pdf", ASSETS / "report.pdf")
    copy_file(ROOT / "reports" / "report.md", ASSETS / "report.md")
    copy_file(ROOT / "results" / "RESULTS_INDEX.md", ASSETS / "RESULTS_INDEX.md")

    print(f"Synced resources into {ASSETS}")


if __name__ == "__main__":
    main()
