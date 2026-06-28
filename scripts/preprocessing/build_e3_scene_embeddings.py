"""
E3 (Phase 2c) — per-scene VIDEO embeddings.

For every video with >=2 scenes, embed each scene as its own short clip so the
notebook can compute mean cos(scene_i, scene_{i+1}) and correlate it with
temporal_edit (hypothesis: edited videos have lower inter-scene similarity).

Coverage:
  - Open-source (qwen2b / qwen8b / wave): ALL 1029 videos with >=2 scenes.
  - GE2: ONLY the raw_ids in e3_ge2_subset.json (the matched 100-video subset),
    to stay within the daily RPD budget. That file is the committed source of
    truth shared with the analysis notebook.

Each scene [start_s, end_s] (from the dataset's scene_frames / fps) is cut to a
temporary clip with ffmpeg and embedded with the SAME video pathway the model
used for its global video embedding. Writes ONLY to `segment_embeddings`
(segment_type 'scene', modality 'video'). Never touches the `embeddings` table.
Resumable: existing (raw_id, scene_idx) rows are skipped before any ffmpeg/embed.

Usage (from repo root on cluster):
    # Qwen-2B (env tfg2)
    python scripts/preprocessing/build_e3_scene_embeddings.py --model qwen2b \
        --data-dir data/GroundLie360 \
        --model-dir src/models/Qwen3-VL-Embedding-2B \
        --qwen-repo src/models/qwen3vl_embedding_repo --quantize
    # WAVE-7B (env tfg-wave)
    python scripts/preprocessing/build_e3_scene_embeddings.py --model wave \
        --data-dir data/GroundLie360 \
        --model-dir src/models/WAVE-7B --wave-repo src/models/wave_repo \
        --beats-path src/models/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt --quantize
    # GE2 (env tfg) — subset only
    python scripts/preprocessing/build_e3_scene_embeddings.py --model ge2 \
        --data-dir data/GroundLie360
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
for p in (REPO_ROOT / "src", REPO_ROOT / "src" / "experiments"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from utils.db_schema import init_db, save_segment_embedding, load_segment_embeddings

DEFAULT_INDEX = REPO_ROOT / "experiments/GroundLie360/groundlie_index_filtered.json"
DEFAULT_SUBSET = REPO_ROOT / "experiments/GroundLie360/e3_ge2_subset.json"

MODEL_DBS = {
    "qwen2b": "experiments/GroundLie360/open_source/results/qwen3vl_2b/qwen3vl_cache.db",
    "qwen8b": "experiments/GroundLie360/open_source/results/qwen3vl_8b/qwen3vl_cache.db",
    "wave":   "experiments/GroundLie360/open_source/results/WAVE7B/wave_cache.db",
    "ge2":    "experiments/GroundLie360/google_embeddings2/results/groundlie_ge2.db",
}

SEGMENT_TYPE = "scene"
MODALITY = "video"


def parse_args():
    p = argparse.ArgumentParser(description="Build E3 per-scene video embeddings")
    p.add_argument("--model", required=True, choices=list(MODEL_DBS))
    p.add_argument("--data-dir", required=True, help="Dir containing vid_groundlie/")
    p.add_argument("--index-json", default=str(DEFAULT_INDEX))
    p.add_argument("--subset-json", default=str(DEFAULT_SUBSET),
                   help="GE2 subset source of truth (used only when --model ge2)")
    p.add_argument("--output-db", default=None,
                   help="Target DB (default: the selected model's own cache)")
    p.add_argument("--limit", type=int, default=None, help="First N videos (smoke test)")
    # Qwen
    p.add_argument("--model-dir", default=None)
    p.add_argument("--qwen-repo", default=None)
    p.add_argument("--max-frames", type=int, default=16)
    p.add_argument("--fps", type=float, default=1.0)
    # WAVE
    p.add_argument("--wave-repo", default=None)
    p.add_argument("--beats-path", default=None)
    p.add_argument("--max-frame-side", type=int, default=256)
    p.add_argument("--quantize", action="store_true")
    p.add_argument("--no-instruction", action="store_true")
    # GE2
    p.add_argument("--api-key", default=None)
    p.add_argument("--rpm-target", type=float, default=3)
    p.add_argument("--rpd-limit", type=int, default=950)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Scene geometry + clip extraction
# ---------------------------------------------------------------------------

def scenes_of(entry: dict) -> list:
    """[(scene_idx, start_s, end_s), ...] from scene_frames + fps."""
    fps = entry.get("fps") or 25.0
    out = []
    for i, (sf, ef) in enumerate(entry.get("scene_frames", [])):
        out.append((i, round(sf / fps, 4), round(ef / fps, 4)))
    return out


def cut_clip(src: str, start_s: float, end_s: float, dst: str) -> bool:
    """Cut [start_s, end_s] of src to dst (video-only re-encode). Returns success."""
    dur = max(end_s - start_s, 0.04)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start_s:.3f}", "-i", src, "-t", f"{dur:.3f}",
        "-map", "0:v:0", "-an", "-c:v", "libx264", "-preset", "ultrafast",
        dst,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=90)
        return Path(dst).is_file() and Path(dst).stat().st_size > 0
    except Exception as exc:
        print(f"    [ffmpeg] cut failed {start_s:.2f}-{end_s:.2f}: {exc}", flush=True)
        return False


# ---------------------------------------------------------------------------
# Per-model video embedders — each reuses its global-video pathway
# ---------------------------------------------------------------------------

def make_embedder(args):
    model = args.model

    if model in ("qwen2b", "qwen8b"):
        import run_groundlie360_qwen3vl as q
        if not args.model_dir or not args.qwen_repo:
            sys.exit("qwen models require --model-dir and --qwen-repo")
        instruction = None if args.no_instruction else q.DEFAULT_INSTRUCTION
        m = q.load_model(args.model_dir, args.qwen_repo,
                         max_frames=args.max_frames, fps=args.fps, quantize=args.quantize)
        return lambda clip: q.get_video_embedding(clip, m, instruction), None

    if model == "wave":
        import run_groundlie360_wave as w
        if not args.model_dir or not args.wave_repo or not args.beats_path:
            sys.exit("wave requires --model-dir, --wave-repo and --beats-path")
        m, proc = w.load_wave_model(args.model_dir, args.wave_repo, args.beats_path, args.quantize)
        return (lambda clip: w.get_video_embedding(clip, m, proc, args.max_frames, args.max_frame_side)), None

    # GE2
    import run_groundlie360_ge2 as g
    client = g.setup_client(args.api_key)
    rl = g.RateLimiter(rpm_target=args.rpm_target, rpd_limit=args.rpd_limit)

    def embed_ge2(clip):
        rl.wait_and_record()
        return g.embed_video_single(client, clip, g.MODEL_NAME)

    return embed_ge2, g.RPDLimitReached


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    out_db = args.output_db or str(REPO_ROOT / MODEL_DBS[args.model])
    data_dir = Path(args.data_dir)

    with open(args.index_json, encoding="utf-8") as f:
        index = json.load(f)

    # Eligible videos: >=2 scenes; GE2 further restricted to the subset.
    eligible = [e for e in index if len(e.get("scene_frames", [])) >= 2]
    if args.model == "ge2":
        with open(args.subset_json, encoding="utf-8") as f:
            subset_ids = set(json.load(f)["raw_ids"])
        eligible = [e for e in eligible if e["raw_id"] in subset_ids]
        print(f"GE2 subset: {len(eligible)} videos (from {args.subset_json})")
    if args.limit:
        eligible = eligible[: args.limit]

    total_scenes = sum(len(e["scene_frames"]) for e in eligible)
    print(f"=== E3 scene embeddings | model={args.model} ===")
    print(f"Videos: {len(eligible)}  |  total scenes: {total_scenes}")

    db = init_db(out_db)
    print(f"Output DB: {out_db}")

    embed, rpd_exc = make_embedder(args)

    tmp_dir = Path(tempfile.mkdtemp(prefix="e3_scenes_"))
    clip_path = str(tmp_dir / "scene.mp4")

    saved, skipped, errors = 0, 0, 0
    try:
        for vi, entry in enumerate(eligible):
            rid = entry["raw_id"]
            vpath = str((data_dir / entry["video_path"]).resolve())
            if not Path(vpath).is_file():
                print(f"  [MISSING] {vpath}")
                continue

            existing = {s["segment_idx"] for s in load_segment_embeddings(db, rid, SEGMENT_TYPE, MODALITY)}
            for scene_idx, start_s, end_s in scenes_of(entry):
                if scene_idx in existing:
                    continue
                if not cut_clip(vpath, start_s, end_s, clip_path):
                    errors += 1
                    continue
                try:
                    vec = embed(clip_path)
                    save_segment_embedding(db, rid, SEGMENT_TYPE, scene_idx,
                                           start_s, end_s, MODALITY, vec)
                    saved += 1
                except Exception as exc:
                    if rpd_exc is not None and isinstance(exc, rpd_exc):
                        raise
                    errors += 1
                    print(f"  [{rid}] scene {scene_idx} ERROR: {exc}", flush=True)

            if (vi + 1) % 25 == 0 or (vi + 1) == len(eligible):
                print(f"  [{vi+1}/{len(eligible)}] saved={saved} errors={errors}", flush=True)

    except Exception as exc:
        if rpd_exc is not None and isinstance(exc, rpd_exc):
            print(f"\n[STOP] {exc}\nResubmit tomorrow — resumes automatically.")
        else:
            raise
    finally:
        try:
            Path(clip_path).unlink(missing_ok=True)
            tmp_dir.rmdir()
        except OSError:
            pass

    print(f"\nDone. saved={saved} skipped={skipped} errors={errors}")


if __name__ == "__main__":
    main()
