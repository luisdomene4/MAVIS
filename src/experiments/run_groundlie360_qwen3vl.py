"""
MAVIS — Qwen3-VL-Embedding pipeline for GroundLie360 dataset.

Computes global embeddings for every video in groundlie_index_filtered.json.
Uses the same model-loading and embedding functions as run_qwen3vl.py but:
  - Reads from groundlie_index_filtered.json (pre-filtered ≤120s)
  - Embeds the single title (modality: text_title) instead of real/fake title pair
  - Saves scene_metadata from index (no TransNetV2 needed)
  - Saves groundlie_labels + groundlie_bboxes from index
  - Saves Whisper word timestamps to transcript_words
  - Does NOT run E1/E2 experiments (those live in analysis.ipynb)

Usage (from repo root on cluster):
    python src/experiments/run_groundlie360_qwen3vl.py \
        --data-dir   data/GroundLie360 \
        --model-dir  src/models/Qwen3-VL-Embedding-8B \
        --qwen-repo  src/models/qwen3vl_embedding_repo \
        --output-dir experiments/GroundLie360/open_source/results/qwen3vl_8b \
        [--index-json experiments/GroundLie360/groundlie_index_filtered.json] \
        [--bbox-json  experiments/GroundLie360/bbox_index.json] \
        [--limit 5]

Outputs:
    <output-dir>/qwen3vl_cache.db    -- resumable SQLite cache (8 tables)
"""

import argparse
import gc
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from faster_whisper import WhisperModel

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from utils.db_schema import (
    init_db, cache_progress,
    save_embedding, load_embedding,
    save_transcript_text, load_transcript_text,
    save_transcript_words, load_transcript_words,
    save_video_metadata, load_video_metadata,
    save_scene_metadata, load_scene_metadata,
    save_groundlie_labels, save_groundlie_bboxes,
)

DEFAULT_INSTRUCTION = "Represent the input for multimodal fact-checking verification."
DEFAULT_INDEX = "experiments/GroundLie360/groundlie_index_filtered.json"
DEFAULT_BBOX  = "experiments/GroundLie360/bbox_index.json"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Qwen3-VL embedding pipeline for GroundLie360")
    p.add_argument("--data-dir",    required=True, help="Dir containing vid_groundlie/ videos")
    p.add_argument("--model-dir",   required=True, help="Qwen3-VL-Embedding checkpoint dir (2B or 8B)")
    p.add_argument("--qwen-repo",   required=True, help="Dir with QwenLM/Qwen3-VL-Embedding source")
    p.add_argument("--output-dir",  required=True, help="Output dir for qwen3vl_cache.db")
    p.add_argument("--index-json",  default=DEFAULT_INDEX, help="groundlie_index_filtered.json path")
    p.add_argument("--bbox-json",   default=DEFAULT_BBOX,  help="bbox_index.json path")
    p.add_argument("--max-frames",  type=int,   default=16)
    p.add_argument("--fps",         type=float, default=1.0)
    p.add_argument("--quantize",    action="store_true")
    p.add_argument("--no-instruction", action="store_true")
    p.add_argument("--limit",       type=int,   default=None,
                   help="Process only the first N index entries (smoke test)")
    p.add_argument("--skip-ids",    default="",
                   help="Comma-separated raw_ids to skip in Phase B (debug)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading (identical to run_qwen3vl.py)
# ---------------------------------------------------------------------------

def load_model(model_dir, qwen_repo, max_frames=16, fps=1.0, quantize=False):
    import importlib.util

    qwen_repo_abs = str(Path(qwen_repo).resolve())
    embedder_file = Path(qwen_repo_abs) / "src" / "models" / "qwen3_vl_embedding.py"
    if not embedder_file.exists():
        raise FileNotFoundError(f"Qwen3VLEmbedder not found at {embedder_file}")

    if qwen_repo_abs not in sys.path:
        sys.path.insert(0, qwen_repo_abs)
    spec = importlib.util.spec_from_file_location("qwen3_vl_embedding_mod", str(embedder_file))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    Qwen3VLEmbedder = mod.Qwen3VLEmbedder

    print(f"Loading {Path(model_dir).name} from {model_dir}")
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {torch.cuda.get_device_name(0)} ({vram:.1f} GB VRAM)")

    kwargs = {"torch_dtype": torch.float16}
    if quantize:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )
        kwargs.pop("torch_dtype", None)
    print(f"  max_frames={max_frames}  fps={fps}  quantize={quantize}")

    model = Qwen3VLEmbedder(model_name_or_path=str(model_dir), max_frames=max_frames, fps=fps, **kwargs)
    print("Qwen3VLEmbedder ready")
    return model


def load_whisper():
    try:
        wm = WhisperModel("small", device="cuda", compute_type="float16")
        print("Whisper: GPU")
    except Exception:
        wm = WhisperModel("small", device="cpu", compute_type="int8")
        print("Whisper: CPU")
    return wm


# ---------------------------------------------------------------------------
# Embedding helpers (identical to run_qwen3vl.py)
# ---------------------------------------------------------------------------

def get_text_embedding(text, model, instruction):
    inp = [{"text": text, "instruction": instruction}] if instruction else [{"text": text}]
    emb = model.process(inp, normalize=False)
    return emb[0].float().cpu().numpy()


class _VideoTimeout(Exception):
    pass

def get_video_embedding(video_path, model, instruction, timeout_s=120):
    inp = [{"video": video_path, **({"instruction": instruction} if instruction else {})}]

    def _handler(signum, frame):
        raise _VideoTimeout(f"model.process timed out after {timeout_s}s")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout_s)
    try:
        print(f"    [decord] reading frames...", flush=True)
        emb = model.process(inp, normalize=False)
        print(f"    [inference] done", flush=True)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    return emb[0].float().cpu().numpy()


def get_video_duration(video_path):
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", video_path],
            capture_output=True, text=True, timeout=30,
        )
        streams = json.loads(probe.stdout).get("streams", [])
        duration, has_audio = None, False
        for s in streams:
            if s.get("codec_type") == "video" and duration is None:
                duration = float(s.get("duration", 0)) or None
            if s.get("codec_type") == "audio":
                has_audio = True
        return duration, has_audio
    except Exception:
        return None, False


# ---------------------------------------------------------------------------
# GroundLie360 helpers
# ---------------------------------------------------------------------------

def scene_metadata_from_entry(entry: dict) -> list:
    """Convert index scene_frames [[sf, ef], ...] + fps → scene_metadata rows."""
    fps = entry.get("fps") or 25.0
    scenes = []
    for i, (sf, ef) in enumerate(entry.get("scene_frames", [])):
        scenes.append({
            "scene_idx": i,
            "start_s": round(sf / fps, 4),
            "end_s":   round(ef / fps, 4),
            "detector": "groundlie_dataset",
            "confidence": 1.0,
        })
    return scenes


def whisper_words(video_path: str, whisper_model) -> tuple[str, list]:
    """Transcribe and return (full_text, word_list). Returns ('', []) on failure."""
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", video_path],
            capture_output=True, text=True,
        )
        streams = json.loads(probe.stdout).get("streams", [])
        if not any(s.get("codec_type") == "audio" for s in streams):
            return "", []

        segments, _ = whisper_model.transcribe(
            video_path, language="en", word_timestamps=True
        )
        words = []
        texts = []
        for seg in segments:
            texts.append(seg.text.strip())
            if seg.words:
                for w in seg.words:
                    words.append({"word": w.word, "start_s": w.start, "end_s": w.end, "probability": w.probability})
        return " ".join(texts).strip(), words
    except Exception as exc:
        print(f"  Whisper error: {exc}")
        return "", []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    instruction = None if args.no_instruction else DEFAULT_INSTRUCTION
    print(f"Model: {Path(args.model_dir).name}")
    print(f"Instruction: {'DISABLED' if instruction is None else repr(instruction)}")

    # Load index
    with open(args.index_json, encoding="utf-8") as f:
        index = json.load(f)
    with open(args.bbox_json, encoding="utf-8") as f:
        bbox_index = json.load(f)

    if args.limit:
        index = index[: args.limit]
    print(f"Index: {len(index)} entries")

    # Resolve absolute video paths
    for entry in index:
        entry["_abs_video"] = str((data_dir / entry["video_path"]).resolve())

    missing = [e for e in index if not os.path.isfile(e["_abs_video"])]
    print(f"Videos: {len(index) - len(missing)} found, {len(missing)} MISSING")
    for e in missing[:10]:
        print(f"  MISSING: {e['_abs_video']}")
    if len(index) == len(missing):
        print("ERROR: No videos found. Check --data-dir.")
        sys.exit(1)

    # Init DB
    db = init_db(str(output_dir / "qwen3vl_cache.db"))
    print(f"Cache: {cache_progress(db)}")

    # Load models
    model = load_model(
        args.model_dir, args.qwen_repo,
        max_frames=args.max_frames, fps=args.fps, quantize=args.quantize,
    )
    whisper_model = load_whisper()

    # Phase 0: index metadata (labels, scenes, bboxes) — model-independent, save once
    print("\n=== Phase 0: Index metadata ===")
    for entry in index:
        rid = entry["raw_id"]

        # groundlie_labels
        conn_row = db.execute("SELECT 1 FROM groundlie_labels WHERE raw_id=?", (rid,)).fetchone()
        if conn_row is None:
            save_groundlie_labels(db, rid, entry)

        # video_metadata (duration from index, already ffprobed)
        if load_video_metadata(db, rid) is None:
            duration = entry.get("duration_seconds")
            # has_audio from ffprobe (needed for transcript skip logic later)
            _, has_audio = get_video_duration(entry["_abs_video"]) if duration is None else (None, True)
            save_video_metadata(db, rid, duration, has_audio, dataset_origin="groundlie360")

        # scene_metadata
        if not load_scene_metadata(db, rid):
            scenes = scene_metadata_from_entry(entry)
            if scenes:
                save_scene_metadata(db, rid, scenes)

        # groundlie_bboxes
        if entry.get("has_bbox") and rid in bbox_index:
            existing = db.execute("SELECT 1 FROM groundlie_bboxes WHERE raw_id=?", (rid,)).fetchone()
            if existing is None:
                save_groundlie_bboxes(db, rid, bbox_index[rid])

    print("Phase 0 done.")

    # Phase A: title text embeddings
    print("\n=== Phase A: Title embeddings (text_title) ===")
    pending_text = [e for e in index if load_embedding(db, e["raw_id"], "text_title") is None]
    print(f"Pending: {len(pending_text)}")
    for i, entry in enumerate(pending_text):
        try:
            emb = get_text_embedding(entry["title"], model, instruction)
            save_embedding(db, entry["raw_id"], "text_title", emb)
        except Exception as exc:
            print(f"  Error title {entry['raw_id']}: {exc}")
        if (i + 1) % 100 == 0 or (i + 1) == len(pending_text):
            print(f"  [{i+1}/{len(pending_text)}] titles done")

    # Phase A': transcripts via Whisper (text + word timestamps + transcript embedding)
    print("\n=== Phase A': Transcript (Whisper) ===")
    pending_tr = [
        e for e in index
        if load_embedding(db, e["raw_id"], "transcript") is None
        and load_transcript_text(db, e["raw_id"]) is None
    ]
    print(f"Pending: {len(pending_tr)}")
    tr_errors = []
    for i, entry in enumerate(pending_tr):
        rid   = entry["raw_id"]
        vpath = entry["_abs_video"]
        # Dataset curators: if no transcript annotated, treat as no audio — skip Whisper
        if not entry.get("has_transcript", True):
            save_transcript_text(db, rid, "", model_name="NO_AUDIO")
            tr_errors.append((rid, "no_transcript_in_dataset"))
            if (i + 1) % 50 == 0 or (i + 1) == len(pending_tr):
                print(f"  [{i+1}/{len(pending_tr)}] transcripts done (errors: {len(tr_errors)})")
            continue
        # Fast audio check — avoids hanging Whisper on silent/corrupt videos
        _, has_audio = get_video_duration(vpath)
        if not has_audio:
            save_transcript_text(db, rid, "", model_name="NO_AUDIO")
            tr_errors.append((rid, "no audio stream"))
            if (i + 1) % 50 == 0 or (i + 1) == len(pending_tr):
                print(f"  [{i+1}/{len(pending_tr)}] transcripts done (errors: {len(tr_errors)})")
            continue
        try:
            text, words = whisper_words(vpath, whisper_model)
            if not text:
                tr_errors.append((rid, "empty transcript"))
                continue
            save_transcript_text(db, rid, text, model_name="whisper-small", language="en")
            if words:
                save_transcript_words(db, rid, words)
            emb = get_text_embedding(text, model, instruction)
            save_embedding(db, rid, "transcript", emb)
        except Exception as exc:
            tr_errors.append((rid, str(exc)))
            print(f"  Error transcript {rid}: {exc}")
        if (i + 1) % 50 == 0 or (i + 1) == len(pending_tr):
            print(f"  [{i+1}/{len(pending_tr)}] transcripts done (errors: {len(tr_errors)})")

    # Free Whisper from GPU before video inference to maximise VRAM
    del whisper_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"VRAM freed. Available: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB")

    # Phase B: video embeddings
    skip_ids = {s.strip() for s in args.skip_ids.split(",") if s.strip()}
    if skip_ids:
        print(f"Skip list: {skip_ids}")
    print(f"\n=== Phase B: Video embeddings (max_frames={args.max_frames}, fps={args.fps}) ===")
    pending_v = [e for e in index if load_embedding(db, e["raw_id"], "video") is None]
    print(f"Pending: {len(pending_v)}")
    v_errors = []
    for i, entry in enumerate(pending_v):
        rid   = entry["raw_id"]
        vpath = entry["_abs_video"]

        if rid in skip_ids:
            print(f"  [{i+1}/{len(pending_v)}] SKIP {rid} (--skip-ids)", flush=True)
            continue

        # Defence-in-depth: skip if somehow over 120s (index should have filtered already)
        duration = entry.get("duration_seconds")
        if duration is not None and duration > 120:
            print(f"  [SKIP] {rid} is {duration:.1f}s > 120s (index filter error)")
            continue

        t0 = time.time()
        print(f"  [{i+1}/{len(pending_v)}] START {rid} ({entry.get('duration_seconds', '?'):.1f}s)", flush=True)
        try:
            emb = get_video_embedding(vpath, model, instruction)
            save_embedding(db, rid, "video", emb)
            print(f"  [{i+1}/{len(pending_v)}] OK    {rid}  ({time.time()-t0:.1f}s)", flush=True)
        except _VideoTimeout as exc:
            v_errors.append((rid, str(exc)))
            print(f"  [{i+1}/{len(pending_v)}] TIMEOUT {rid}  ({time.time()-t0:.1f}s)", flush=True)
        except Exception as exc:
            v_errors.append((rid, str(exc)))
            print(f"  [{i+1}/{len(pending_v)}] ERROR {rid}: {exc}", flush=True)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if (i + 1) % 20 == 0 or (i + 1) == len(pending_v):
            print(f"  [{i+1}/{len(pending_v)}] videos done (errors: {len(v_errors)})")

    print(f"\nCache final: {cache_progress(db)}")
    print("Done.")


if __name__ == "__main__":
    main()
