#!/usr/bin/env python
"""Put the input tables of final_figures.py in data/.

    python scripts/link_inputs.py [SOURCE] [--copy]

SOURCE = a directory with the layout of analysis_village/cc-inclusive/data (default: ../cc-inclusive/data next to this folder).
For every file of final_figures.INPUTS a relative link data/<name> -> SOURCE/<name> is made (--copy: a copy instead, so the folder
no longer depends on SOURCE). Existing entries in data/ are left alone; a missing source file is an error.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True                   # no scripts/__pycache__ from importing the figure script for its input list
from final_figures import DATA, FINAL, INPUTS  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="link (or copy) the inputs of final_figures.py into data/")
    p.add_argument("source", nargs="?", default=str(FINAL.parent / "cc-inclusive" / "data"), help="directory holding the tables (layout of cc-inclusive/data)")
    p.add_argument("--copy", action="store_true", help="copy the files instead of linking them")
    a = p.parse_args()
    source = Path(a.source).resolve()
    missing = [str(f.relative_to(DATA)) for f in INPUTS if not (source / f.relative_to(DATA)).is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} input files are not in {source}:\n  " + "\n  ".join(missing))
    made = kept = 0
    for f in INPUTS:
        src = source / f.relative_to(DATA)
        if f.exists() or f.is_symlink():
            kept += 1
            continue
        f.parent.mkdir(parents=True, exist_ok=True)
        if a.copy:
            shutil.copy2(src, f)
        else:
            f.symlink_to(os.path.relpath(src, f.parent))
        made += 1
    print(f"{'copied' if a.copy else 'linked'} {made} files into {DATA} from {source}; {kept} already there")


if __name__ == "__main__":
    main()
