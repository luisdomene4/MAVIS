"""
E3 (scene) GE2 subset definition — DETERMINISTIC source of truth.

GE2 cannot afford the full 1029-video scene pass (≈11 days of RPD), so E3 runs
GE2 only over a small matched subset, while open-source models (Qwen-2B/8B,
WAVE) run over all 1029 videos with ≥2 scenes. This script fixes that subset
ONCE and writes it to a committed JSON so that BOTH the GE2 embedding job and
the analysis notebook read exactly the same raw_ids.

Subset = all 50 temporal_edit=1 videos (with ≥2 scenes) + 50 temporal_edit=0
control videos matched 1:1 by number of scenes. Matching is reproducible via a
fixed seed.

Usage (local, no cluster, no model):
    conda run -n yolo python scripts/preprocessing/build_e3_subset.py

Output:
    experiments/GroundLie360/e3_ge2_subset.json   (committed)
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INDEX = REPO_ROOT / "experiments/GroundLie360/groundlie_index_filtered.json"
DEFAULT_OUT = REPO_ROOT / "experiments/GroundLie360/e3_ge2_subset.json"


def parse_args():
    p = argparse.ArgumentParser(description="Build the deterministic GE2 subset for E3")
    p.add_argument("--index-json", default=str(DEFAULT_INDEX))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def n_scenes(entry: dict) -> int:
    return len(entry.get("scene_frames", []))


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    with open(args.index_json, encoding="utf-8") as f:
        index = json.load(f)

    eligible = [e for e in index if n_scenes(e) >= 2]
    te1 = sorted((e for e in eligible if e["temporal_edit"] == 1), key=lambda e: e["raw_id"])
    te0 = [e for e in eligible if e["temporal_edit"] == 0]

    # Pool of control videos grouped by exact scene count, shuffled deterministically.
    pool = defaultdict(list)
    for e in sorted(te0, key=lambda e: e["raw_id"]):
        pool[n_scenes(e)].append(e)
    for k in pool:
        rng.shuffle(pool[k])

    controls = []
    for e in te1:
        k = n_scenes(e)
        if pool[k]:
            controls.append(pool[k].pop())
        else:
            # Fallback: nearest scene count with remaining controls (kept for safety;
            # not exercised on the current index where every bucket has enough).
            avail = [kk for kk in pool if pool[kk]]
            nearest = min(avail, key=lambda kk: (abs(kk - k), kk))
            controls.append(pool[nearest].pop())

    def row(e, label):
        return {"raw_id": e["raw_id"], "temporal_edit": label, "n_scenes": n_scenes(e)}

    videos = [row(e, 1) for e in te1] + [row(e, 0) for e in controls]
    videos.sort(key=lambda r: (1 - r["temporal_edit"], r["raw_id"]))

    out = {
        "description": "Deterministic GE2 subset for E3 (scene similarity). "
                       "Source of truth shared by GE2 embedding job and analysis.",
        "seed": args.seed,
        "matched_on": "n_scenes",
        "eligibility": "n_scenes >= 2",
        "n_temporal_edit": len(te1),
        "n_control": len(controls),
        "total": len(videos),
        "raw_ids": [r["raw_id"] for r in videos],
        "videos": videos,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    # Report
    print(f"Eligible (>=2 scenes): {len(eligible)}")
    print(f"  temporal_edit=1: {len(te1)}")
    print(f"  temporal_edit=0: {len(te0)}")
    print(f"Subset: {len(videos)} videos ({len(te1)} te=1 + {len(controls)} control)")
    matched = sum(1 for a, b in zip(te1, controls) if n_scenes(a) == n_scenes(b))
    print(f"Exact scene-count matches: {matched}/{len(te1)}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
