"""
MAVIS — Qwen3-VL-Embedding pipeline (open-source baseline)
Replicates E1 + E2 experiments from mavis_embeddings.ipynb using a local Qwen3-VL model.

Compatible with both Qwen3-VL-Embedding-2B and 8B checkpoints (same Qwen3VLEmbedder API,
only weights and embedding dimension differ: 2B→2048-dim, 8B→4096-dim).

Uses the official Qwen3VLEmbedder API (QwenLM/Qwen3-VL-Embedding repo) which
supports native video input — no manual frame extraction needed.

Usage:
    python src/experiments/run_qwen3vl.py \
        --data-dir   data/FakeVV_testset \
        --model-dir  src/models/Qwen3-VL-Embedding-8B \
        --qwen-repo  src/models/qwen3vl_embedding_repo \
        --output-dir experiments/open_source/results/qwen3vl_8b \
        [--max-frames 16] [--fps 1.0] [--quantize] [--no-instruction]

Token budget:
    Each video frame ≈ 360 tokens. max_length = 8192.
    max-frames=16 → ~5760 video tokens (safe margin).
    max-frames=22 → ~7920 (theoretical max, risky).

Outputs:
    <output-dir>/qwen3vl_cache.db          -- SQLite embedding + transcript cache (resumable)
    <output-dir>/OS_E1_baseline.csv        -- E1 results (with over_120s flag)
    <output-dir>/OS_E2_transcript.csv      -- E2 results (with over_120s flag)
    <output-dir>/figures/                  -- PNG plots

Note: Embeddings stored RAW (non-normalized) in DB.
      sklearn cosine_similarity normalizes internally at comparison time.
      GE2 comparison: report metrics for ALL videos AND ≤120s subset.
      Audio NOT processed from video — both GE2 and Qwen3VL are vision-only for video.
"""

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from faster_whisper import WhisperModel
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use("Agg")  # headless — no display required on cluster
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_INSTRUCTION = "Represent the input for multimodal fact-checking verification."


def parse_args():
    p = argparse.ArgumentParser(description="Qwen3-VL embedding pipeline for FakeVV")
    p.add_argument("--data-dir",      required=True, help="Dir containing test_index.json and videos")
    p.add_argument("--model-dir",     required=True, help="Dir with Qwen3-VL-Embedding checkpoints (2B or 8B)")
    p.add_argument("--qwen-repo",     required=True, help="Dir with cloned QwenLM/Qwen3-VL-Embedding source code")
    p.add_argument("--output-dir",    required=True, help="Dir for cache, CSVs, figures")
    p.add_argument("--quantize",      action="store_true", help="Use INT4 quantization")
    p.add_argument("--max-frames",    type=int,   default=16,
                   help="Max video frames fed to model (default 16; 16×360≈5760 tokens < 8192 limit)")
    p.add_argument("--fps",           type=float, default=1.0,
                   help="Video sampling fps (default 1.0)")
    p.add_argument("--no-instruction", action="store_true",
                   help="Disable task instruction prefix (ablation study)")
    p.add_argument("--instruction",   default=DEFAULT_INSTRUCTION,
                   help="Custom task instruction (ignored if --no-instruction)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# SQLite cache
# ---------------------------------------------------------------------------

def init_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            raw_id     TEXT,
            modality   TEXT,
            vector     BLOB,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (raw_id, modality)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS transcripts (
            raw_id     TEXT PRIMARY KEY,
            text       TEXT,
            model      TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS video_metadata (
            raw_id           TEXT PRIMARY KEY,
            duration_seconds REAL,
            has_audio        INTEGER,
            n_frames_used    INTEGER,
            created_at       TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


def save_embedding(conn, raw_id, modality, emb):
    conn.execute(
        "INSERT OR IGNORE INTO embeddings (raw_id, modality, vector) VALUES (?, ?, ?)",
        (raw_id, modality, emb.astype(np.float32).tobytes()),
    )
    conn.commit()


def load_embedding(conn, raw_id, modality):
    row = conn.execute(
        "SELECT vector FROM embeddings WHERE raw_id=? AND modality=?",
        (raw_id, modality),
    ).fetchone()
    if row is None:
        return None
    return np.frombuffer(row[0], dtype=np.float32)


def save_transcript_text(conn, raw_id, text, model_name="whisper-small"):
    conn.execute(
        "INSERT OR IGNORE INTO transcripts (raw_id, text, model) VALUES (?, ?, ?)",
        (raw_id, text, model_name),
    )
    conn.commit()


def load_transcript_text(conn, raw_id):
    row = conn.execute("SELECT text FROM transcripts WHERE raw_id=?", (raw_id,)).fetchone()
    return row[0] if row else None


def save_video_metadata(conn, raw_id, duration_seconds, has_audio, n_frames_used=None):
    conn.execute(
        """INSERT OR REPLACE INTO video_metadata
           (raw_id, duration_seconds, has_audio, n_frames_used) VALUES (?, ?, ?, ?)""",
        (raw_id, duration_seconds, int(has_audio), n_frames_used),
    )
    conn.commit()


def load_video_metadata(conn, raw_id):
    row = conn.execute(
        "SELECT duration_seconds, has_audio, n_frames_used FROM video_metadata WHERE raw_id=?",
        (raw_id,),
    ).fetchone()
    if row is None:
        return None
    return {"duration_seconds": row[0], "has_audio": bool(row[1]), "n_frames_used": row[2]}


def cache_progress(conn):
    rows = conn.execute(
        "SELECT modality, COUNT(*) FROM embeddings GROUP BY modality"
    ).fetchall()
    return {mod: cnt for mod, cnt in rows}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(model_dir, qwen_repo, max_frames=16, fps=1.0, quantize=False):
    """Load Qwen3VLEmbedder from the official QwenLM/Qwen3-VL-Embedding repo.

    Compatible with 2B (hidden=2048) and 8B (hidden=4096) checkpoints.
    max_frames=16 default: 16×~360 tokens ≈ 5760 < 8192 token limit.
    64-frame default would crash with token mismatch error.
    """
    import importlib.util

    qwen_repo_abs = str(Path(qwen_repo).resolve())
    embedder_file = Path(qwen_repo_abs) / "src" / "models" / "qwen3_vl_embedding.py"
    if not embedder_file.exists():
        raise FileNotFoundError(f"Qwen3VLEmbedder not found at {embedder_file}")

    if qwen_repo_abs not in sys.path:
        sys.path.insert(0, qwen_repo_abs)
    spec = importlib.util.spec_from_file_location("qwen3_vl_embedding_mod", str(embedder_file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    Qwen3VLEmbedder = mod.Qwen3VLEmbedder

    model_name = Path(model_dir).name
    print(f"Loading {model_name} from {model_dir}")
    print(f"  CUDA: {torch.cuda.is_available()}")
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
        print("  dtype: INT4 (quantized)")
    else:
        print("  dtype: FP16")
    print(f"  max_frames={max_frames}  fps={fps}  (budget: ~{max_frames * 360} video tokens / 8192 max)")

    model = Qwen3VLEmbedder(
        model_name_or_path=str(model_dir),
        max_frames=max_frames,
        fps=fps,
        **kwargs,
    )
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
# Embedding functions
# ---------------------------------------------------------------------------
# NOTE: Embeddings are stored RAW (not normalized) in SQLite.
#       Cosine similarity (sklearn) handles normalization internally at comparison time.
# ---------------------------------------------------------------------------

def get_text_embedding(text, model, instruction):
    """Embed a text string. instruction=None disables the task prefix."""
    inp = [{"text": text, "instruction": instruction}] if instruction else [{"text": text}]
    emb = model.process(inp, normalize=False)
    return emb[0].float().cpu().numpy()


def get_video_embedding(video_path, model, instruction):
    """Embed a video file. Frame sampling controlled by max_frames/fps set at model load time."""
    inp = [{
        "video": video_path,
        **({"instruction": instruction} if instruction else {}),
    }]
    emb = model.process(inp, normalize=False)
    return emb[0].float().cpu().numpy()


def get_video_duration(video_path):
    """Return (duration_seconds, has_audio) via ffprobe. Returns (None, False) on error."""
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", video_path],
            capture_output=True, text=True, timeout=30,
        )
        streams = json.loads(probe.stdout).get("streams", [])
        duration = None
        has_audio = False
        for s in streams:
            if s.get("codec_type") == "video" and duration is None:
                duration = float(s.get("duration", 0)) or None
            if s.get("codec_type") == "audio":
                has_audio = True
        return duration, has_audio
    except Exception:
        print("ERROR extrayenodo datos")
        return None, False


def resolve_video_path(data_dir: Path, video_path_in_index: str) -> Path:
    """Resolve video path to absolute. Falls back from test_video/X to flat X."""
    candidate = (data_dir / video_path_in_index).resolve()
    if candidate.exists():
        return candidate
    # Flat fallback: data_dir/filename (no subdirectory)
    flat = (data_dir / Path(video_path_in_index).name).resolve()
    if flat.exists():
        return flat
    # Return original resolved path (will fail at embedding time with clear error)
    return candidate


def get_transcript(video_path, whisper_model):
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", video_path],
        capture_output=True, text=True,
    )
    streams = json.loads(probe.stdout).get("streams", [])
    if not any(s.get("codec_type") == "audio" for s in streams):
        raise ValueError(f"No audio: {os.path.basename(video_path)}")
    segments, _ = whisper_model.transcribe(video_path, language="en")
    return " ".join(seg.text.strip() for seg in segments).strip()


def get_transcript_embedding(video_path, whisper_model, model, instruction, db=None, raw_id=None):
    """Transcribe via Whisper, optionally save text to DB, return embedding."""
    transcript = get_transcript(video_path, whisper_model)
    if not transcript:
        raise ValueError(f"Empty transcript: {os.path.basename(video_path)}")
    if db is not None and raw_id is not None:
        save_transcript_text(db, raw_id, transcript)
    return get_text_embedding(transcript, model, instruction)


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def run_e1(test_index, db, output_dir):
    results = []
    skipped = 0
    for item in test_index:
        rid = item["raw_id"]
        ev = load_embedding(db, rid, "video")
        er = load_embedding(db, rid, "text_real")
        ef = load_embedding(db, rid, "text_fake")
        if ev is None or er is None or ef is None:
            skipped += 1
            continue
        sim_real = float(cosine_similarity(ev.reshape(1, -1), er.reshape(1, -1))[0][0])
        sim_fake = float(cosine_similarity(ev.reshape(1, -1), ef.reshape(1, -1))[0][0])
        meta = load_video_metadata(db, rid)
        results.append({
            "raw_id": rid, "category": item["category"],
            "sim_real": sim_real, "sim_fake": sim_fake,
            "margin": sim_real - sim_fake, "correct": sim_real > sim_fake,
            "duration_seconds": meta["duration_seconds"] if meta else None,
            "over_120s": (meta["duration_seconds"] > 120) if (meta and meta["duration_seconds"]) else None,
        })

    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "OS_E1_baseline.csv")
    df.to_csv(csv_path, index=False)
    print(f"E1 results: {len(df)} videos, {skipped} skipped")
    print(f"  ALL    — Accuracy: {df['correct'].mean():.4f} ({df['correct'].sum()}/{len(df)}), margin mean: {df['margin'].mean():.4f}")
    df_le120 = df[df["over_120s"] == False]  # noqa: E712
    if len(df_le120) > 0:
        print(f"  ≤120s  — Accuracy: {df_le120['correct'].mean():.4f} ({df_le120['correct'].sum()}/{len(df_le120)})  [GE2-comparable subset]")
    return df


def run_e2(test_index, db, output_dir):
    results = []
    skipped = 0
    for item in test_index:
        rid = item["raw_id"]
        ev  = load_embedding(db, rid, "video")
        er  = load_embedding(db, rid, "text_real")
        ef  = load_embedding(db, rid, "text_fake")
        et  = load_embedding(db, rid, "transcript")
        if ev is None or er is None or ef is None:
            skipped += 1
            continue
        svr = float(cosine_similarity(ev.reshape(1, -1), er.reshape(1, -1))[0][0])
        svf = float(cosine_similarity(ev.reshape(1, -1), ef.reshape(1, -1))[0][0])
        meta = load_video_metadata(db, rid)
        row = {
            "raw_id": rid, "category": item["category"],
            "sim_vid_real": svr, "sim_vid_fake": svf,
            "e1_correct": svr > svf, "e1_margin": svr - svf,
            "has_transcript": et is not None,
            "duration_seconds": meta["duration_seconds"] if meta else None,
            "over_120s": (meta["duration_seconds"] > 120) if (meta and meta["duration_seconds"]) else None,
        }
        if et is not None:
            str_ = float(cosine_similarity(et.reshape(1, -1), er.reshape(1, -1))[0][0])
            stf  = float(cosine_similarity(et.reshape(1, -1), ef.reshape(1, -1))[0][0])
            cr   = (svr + str_) / 2
            cf   = (svf + stf) / 2
            row.update({
                "sim_trans_real": str_, "sim_trans_fake": stf,
                "e2_correct": str_ > stf, "e2_margin": str_ - stf,
                "e2c_correct": cr > cf, "e2c_margin": cr - cf,
            })
        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "OS_E2_transcript.csv"), index=False)
    df_t = df[df["has_transcript"]]
    print(f"E2: {len(df)} total, {len(df_t)} with transcript, {skipped} skipped")
    if len(df_t) > 0:
        print(f"  ALL   — E1: {df_t['e1_correct'].mean():.4f}  E2a: {df_t['e2_correct'].mean():.4f}  E2c: {df_t['e2c_correct'].mean():.4f}")
        df_t_le120 = df_t[df_t["over_120s"] == False]  # noqa: E712
        if len(df_t_le120) > 0:
            print(f"  ≤120s — E1: {df_t_le120['e1_correct'].mean():.4f}  E2a: {df_t_le120['e2_correct'].mean():.4f}  E2c: {df_t_le120['e2c_correct'].mean():.4f}  [GE2-comparable]")
    return df, df_t


def save_figures(df_e1, df_e2, df_e2_trans, output_dir, model_name="Qwen3-VL"):
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    if len(df_e1) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(f"E1: {model_name} — sim(Video, Title)", fontsize=13)

        axes[0].hist(df_e1["margin"], bins=40, color="steelblue", edgecolor="white")
        axes[0].axvline(0, color="red", linestyle="--")
        axes[0].set_title("Margin distribution")
        axes[0].set_xlabel("margin")

        cat_acc = df_e1.groupby("category")["correct"].mean().sort_values()
        cat_acc.plot(kind="barh", ax=axes[1], color="steelblue")
        axes[1].axvline(df_e1["correct"].mean(), color="red", linestyle="--")
        axes[1].set_title("Accuracy by category")

        colors = ["green" if c else "red" for c in df_e1["correct"]]
        axes[2].scatter(df_e1["sim_fake"], df_e1["sim_real"], c=colors, alpha=0.4, s=10)
        axes[2].set_title("sim_real vs sim_fake")

        plt.tight_layout()
        fig_name = f"OS_E1_{model_name.replace(' ', '_').replace('/', '_')}.png"
        plt.savefig(os.path.join(fig_dir, fig_name), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Figure saved: {fig_name}")

    if len(df_e2_trans) > 0:
        fig, ax = plt.subplots(figsize=(7, 4))
        methods = ["E1: Video", "E2a: Transcript", "E2c: Combined"]
        accs = [df_e2_trans["e1_correct"].mean(), df_e2_trans["e2_correct"].mean(), df_e2_trans["e2c_correct"].mean()]
        bars = ax.bar(methods, accs, color=["steelblue", "darkorange", "seagreen"])
        ax.set_ylim(0, 1)
        ax.set_title(f"{model_name} — Accuracy by method")
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.01, f"{acc:.3f}", ha="center", fontsize=10)
        plt.tight_layout()
        fig_name2 = f"OS_E2_methods_{model_name.replace(' ', '_').replace('/', '_')}.png"
        plt.savefig(os.path.join(fig_dir, fig_name2), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Figure saved: {fig_name2}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    data_dir   = Path(args.data_dir)
    model_dir  = Path(args.model_dir)
    qwen_repo  = Path(args.qwen_repo)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = model_dir.name  # e.g. "Qwen3-VL-Embedding-8B"

    # Resolve instruction
    instruction = None if args.no_instruction else args.instruction
    print(f"Model: {model_name}")
    print(f"Instruction: {'DISABLED (ablation)' if instruction is None else repr(instruction)}")
    print(f"max_frames={args.max_frames}  fps={args.fps}")

    # Load test index
    test_index_path = data_dir / "test_index.json"
    with open(test_index_path, encoding="utf-8") as f:
        test_index = json.load(f)

    # Resolve video paths (flat fallback: test_video/X → X if not found)
    for item in test_index:
        item["video_path"] = str(resolve_video_path(data_dir, item["video_path"]))

    # --- Validate that video files actually exist ---
    missing_videos = []
    for item in test_index:
        if not os.path.isfile(item["video_path"]):
            missing_videos.append(item["video_path"])
    found_count = len(test_index) - len(missing_videos)
    print(f"Test set: {len(test_index)} videos ({found_count} found, {len(missing_videos)} MISSING)")
    if missing_videos:
        for mv in missing_videos[:10]:  # show first 10
            print(f"  MISSING: {mv}")
        if len(missing_videos) > 10:
            print(f"  ... and {len(missing_videos) - 10} more")
    if found_count == 0:
        print("ERROR: No video files found. Check --data-dir path.")
        sys.exit(1)

    # Init cache
    db = init_db(str(output_dir / "qwen3vl_cache.db"))
    print(f"Cache progress: {cache_progress(db)}")

    # Load models
    model = load_model(
        str(model_dir), str(qwen_repo),
        max_frames=args.max_frames, fps=args.fps, quantize=args.quantize,
    )
    whisper_model = load_whisper()

    # Phase A: text embeddings
    print("\n=== Phase A: Text embeddings ===")
    pending = [(it["raw_id"], "text_real", it["title"]) for it in test_index
               if load_embedding(db, it["raw_id"], "text_real") is None]
    pending += [(it["raw_id"], "text_fake", it["fake_title"]) for it in test_index
                if load_embedding(db, it["raw_id"], "text_fake") is None]
    print(f"Pending text: {len(pending)}")
    for i, (rid, mod, text) in enumerate(pending):
        try:
            emb = get_text_embedding(text, model, instruction)
            save_embedding(db, rid, mod, emb)
        except Exception as e:
            print(f"Error text {rid}/{mod}: {e}")
        if (i + 1) % 100 == 0 or (i + 1) == len(pending):
            print(f"  [{i+1}/{len(pending)}] texts done")

    # Phase A: transcripts
    print("\nTranscripts...")
    pending_t = [it for it in test_index if load_embedding(db, it["raw_id"], "transcript") is None]
    print(f"Pending transcripts: {len(pending_t)}")
    t_errors = []
    for i, item in enumerate(pending_t):
        try:
            emb = get_transcript_embedding(
                item["video_path"], whisper_model, model, instruction,
                db=db, raw_id=item["raw_id"],
            )
            save_embedding(db, item["raw_id"], "transcript", emb)
        except Exception as e:
            t_errors.append((item["raw_id"], str(e)))
            print(f"  Error transcript {item['raw_id'][:8]}: {e}")
        if (i + 1) % 50 == 0 or (i + 1) == len(pending_t):
            print(f"  [{i+1}/{len(pending_t)}] transcripts done (errors: {len(t_errors)})")

    # Phase B: video embeddings + metadata
    print(f"\n=== Phase B: Video embeddings (max_frames={args.max_frames}, fps={args.fps}) ===")
    pending_v = [it for it in test_index if load_embedding(db, it["raw_id"], "video") is None]
    print(f"Pending videos: {len(pending_v)}")
    v_errors = []
    for i, item in enumerate(pending_v):
        rid = item["raw_id"]
        vpath = item["video_path"]
        try:
            # Gather metadata before embedding
            meta = load_video_metadata(db, rid)
            if meta is None:
                duration, has_audio = get_video_duration(vpath)
                save_video_metadata(db, rid, duration, has_audio)
                meta = {"duration_seconds": duration, "has_audio": has_audio}
                if duration is not None:
                    over = duration > 120
                    print(f"  [{rid[:8]}] {duration:.1f}s  audio={has_audio}  {'OVER 120s' if over else 'ok'}")

            duration = meta.get("duration_seconds")
            if duration is not None and duration > 120:
                print(f"  [SKIP] Video {rid[:8]} is {duration:.1f}s (> 120s)")
                continue

            emb = get_video_embedding(vpath, model, instruction)
            save_embedding(db, rid, "video", emb)
        except Exception as e:
            v_errors.append((rid, str(e)))
            print(f"  Error video {rid}: {e}")
        finally:
            # Prevent RAM / VRAM memory leaks from video decoding libraries
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if (i + 1) % 20 == 0 or (i + 1) == len(pending_v):
            print(f"  [{i+1}/{len(pending_v)}] videos done (errors: {len(v_errors)})")

    print(f"\nCache final: {cache_progress(db)}")

    # Experiments
    print("\n=== Experiments ===")
    df_e1 = run_e1(test_index, db, str(output_dir))
    df_e2, df_e2_trans = run_e2(test_index, db, str(output_dir))
    save_figures(df_e1, df_e2, df_e2_trans, str(output_dir), model_name=model_name)

    print("\nDone.")


if __name__ == "__main__":
    main()
