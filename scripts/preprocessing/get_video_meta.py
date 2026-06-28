"""
Get duration and FPS for all GroundLie360 videos via ffprobe.
Saves experiments/GroundLie360/video_meta.json.

Usage (from repo root on cluster):
    python scripts/preprocessing/get_video_meta.py

No GPU or special dependencies needed — only standard library + ffprobe.
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

csv.field_size_limit(10**7)

REPO_ROOT   = Path(__file__).resolve().parent.parent.parent
DATA_CSV    = REPO_ROOT / "data" / "GroundLie360" / "dataset" / "data.csv"
VIDEOS_DIR  = REPO_ROOT / "data" / "GroundLie360" / "vid_groundlie"
OUTPUT_JSON = REPO_ROOT / "experiments" / "GroundLie360" / "video_meta.json"


def ffprobe_meta(video_path: Path) -> dict:
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_streams", "-show_format",
                str(video_path),
            ],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            return {"duration_seconds": None, "fps": None, "error": "ffprobe non-zero"}

        d = json.loads(result.stdout)
        duration = float(d["format"]["duration"])

        fps_str = next(
            (s["r_frame_rate"] for s in d.get("streams", []) if s.get("codec_type") == "video"),
            "25/1",
        )
        num, den = map(int, fps_str.split("/"))
        fps = round(num / den, 3) if den else 25.0

        return {"duration_seconds": round(duration, 3), "fps": fps}

    except subprocess.TimeoutExpired:
        return {"duration_seconds": None, "fps": None, "error": "timeout"}
    except Exception as exc:
        return {"duration_seconds": None, "fps": None, "error": str(exc)}


def main():
    if not DATA_CSV.exists():
        print(f"ERROR: data.csv not found at {DATA_CSV}", file=sys.stderr)
        sys.exit(1)

    with open(DATA_CSV, encoding="utf-8") as f:
        video_ids = [r["video_id"] for r in csv.DictReader(f)]

    print(f"Processing {len(video_ids)} videos from {VIDEOS_DIR}")

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    meta = {}
    missing = 0
    for i, vid_id in enumerate(video_ids, 1):
        video_path = VIDEOS_DIR / f"{vid_id}.mp4"
        if not video_path.exists():
            meta[vid_id] = {"duration_seconds": None, "fps": None, "error": "file_not_found"}
            missing += 1
        else:
            meta[vid_id] = ffprobe_meta(video_path)

        if i % 100 == 0 or i == len(video_ids):
            filled = sum(1 for m in meta.values() if m.get("duration_seconds") is not None)
            print(f"  {i}/{len(video_ids)}  filled={filled}  missing={missing}")
            sys.stdout.flush()

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    filled = sum(1 for m in meta.values() if m.get("duration_seconds") is not None)
    print(f"\nDone: {filled}/{len(meta)} videos with duration/fps")
    print(f"Saved: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
