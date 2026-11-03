"""Join the rows of ``bench/slurm/s10_shooting_timing.sbatch`` into one table.

usage: python bench/slurm/s10_shooting_summary.py DIR

One line per row JSON (``bench/s10_fused_shooting.py --task timing --out``): the
kind, N, segments x steps, cards, the median seconds (and min, IQR), per step for
the segment gradients, the compiles of the cold call, and CONTENDED when another
process was on the card.
"""

from __future__ import annotations

import glob
import json
import os
import sys


def summarize(directory: str) -> list:
    rows = []
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        with open(path) as fh:
            rec = json.load(fh)
        cfg = rec.get("config", {})
        for key, row in (rec.get("timing") or {}).items():
            if not isinstance(row, dict) or "median_s" not in row:
                continue
            rows.append(
                {
                    "tag": os.path.basename(path)[:-5],
                    "program": f"{cfg.get('only')}:{key}",
                    "n": cfg.get("n"),
                    "segments": cfg.get("segments"),
                    "steps": cfg.get("steps"),
                    "cards": cfg.get("devices") or 1,
                    "optimizer": cfg.get("optimizer"),
                    "median_s": row["median_s"],
                    "min_s": row["min_s"],
                    "iqr_s": row["iqr_s"],
                    "per_step_ms": row.get("per_step_ms"),
                    "trials": row.get("line_search_trials_per_iteration"),
                    "contended": rec.get("contended"),
                }
            )
    for r in rows:
        extra = ""
        if r["per_step_ms"] is not None:
            extra += f"  {r['per_step_ms']:.1f} ms/step"
        if r["trials"]:
            extra += f"  {r['trials']:.1f} line-search trials"
        print(
            f"{r['tag']:<22} {r['program']:<22} N {r['n']:>11,} {r['segments']}x{r['steps']:<3} "
            f"cards {r['cards']}  {r['median_s']:9.3f} s (min {r['min_s']:.3f}, IQR {r['iqr_s']:.3f})"
            + extra
            + ("  CONTENDED" if r["contended"] else "")
        )
    return rows


if __name__ == "__main__":
    summarize(sys.argv[1])
