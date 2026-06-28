"""
MAVIS — WAVE-7B embedding pipeline (open-source baseline)
Replicates E1 + E2 experiments using WAVE-7B (Tsinghua, ICLR 2026).

Usage:
    python src/experiments/run_wave.py \
        --data-dir data/FakeVV_testset \
        --model-dir src/models/WAVE-7B \
        --wave-repo src/models/wave_repo \
        --beats-path src/models/BEATs/BEATs_iter3_plus.pt \
        --output-dir experiments/open_source/results/WAVE7B

Prerequisites:
    src/models/WAVE-7B/          -- WAVE-7B checkpoints (~18.8 GB, HF: tsinghua-ee/WAVE-7B)
    src/models/BEATs/            -- BEATs_iter3_plus.pt (~300 MB)
                                    Download: https://1drv.ms/u/s!AqeByhGUtINrgcpj8ujXH1YUtxooEg?e=E9Ncea
    src/models/wave_repo/        -- git clone https://github.com/TCL606/WAVE.git

Outputs:
    <output-dir>/wave_cache.db         -- SQLite embedding + metadata cache (resumable)
    <output-dir>/OS_E1_wave.csv        -- E1 results (video vs text, ALL + ≤120s)
    <output-dir>/OS_E2_wave.csv        -- E2 results (audio + audiovisual)
    <output-dir>/figures/              -- PNG plots

Note: Embeddings stored RAW (non-normalized) in DB.
      Duration tracking matches run_qwen3vl.py schema for cross-model comparison.
      WAVE processes ALL video lengths (no token budget constraint unlike Qwen3VL).
      ≤120s subset reported for direct GE2 comparability.
"""

import argparse
import gc
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import av
import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use("Agg")  # headless — no display on cluster
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="WAVE-7B embedding pipeline for FakeVV")
    p.add_argument("--data-dir",   required=True, help="Dir containing test_index.json and test_video/")
    p.add_argument("--model-dir",  required=True, help="Dir with WAVE-7B checkpoints")
    p.add_argument("--wave-repo",  required=True, help="Dir with cloned TCL606/WAVE source code")
    p.add_argument("--beats-path", required=True, help="Path to BEATs_iter3_plus.pt")
    p.add_argument("--output-dir", required=True, help="Dir for cache, CSVs, figures")
    p.add_argument("--max-frames", type=int, default=32, help="Max video frames (default 32)")
    p.add_argument(
        "--max-frame-side",
        type=int,
        default=256,
        help="Downscale decoded frames so max(H,W)<=this (px). Prevents CUDA OOM on high-res videos. Set 0 to disable.",
    )
    p.add_argument("--quantize",   action="store_true",
                   help="Force INT4 NF4 quantization (needed when GPU VRAM is limited)")
    p.add_argument("--no-prefix",  action="store_true",
                   help="Disable text query prefix for ablation study")
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


def cache_progress(conn):
    rows = conn.execute(
        "SELECT modality, COUNT(*) FROM embeddings GROUP BY modality"
    ).fetchall()
    return {mod: cnt for mod, cnt in rows}


# ---------------------------------------------------------------------------
# Video metadata helpers (mirrors run_qwen3vl.py for cross-model comparisons)
# ---------------------------------------------------------------------------

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
        return None, False


def resolve_video_path(data_dir: Path, video_path_in_index: str) -> Path:
    """Resolve video path to absolute. Falls back from test_video/X to flat X."""
    candidate = (data_dir / video_path_in_index).resolve()
    if candidate.exists():
        return candidate
    flat = (data_dir / Path(video_path_in_index).name).resolve()
    if flat.exists():
        return flat
    return candidate


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_wave_model(model_dir, wave_repo, beats_path, quantize=False):
    # Add WAVE source code to Python path
    if str(wave_repo) not in sys.path:
        sys.path.insert(0, str(wave_repo))

    from transformers import BitsAndBytesConfig
    from qwenvl.data.processing_qwen2_5_omni import Qwen2_5OmniProcessor
    from qwenvl.model.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniThinkerConfig
    from qwenvl.model.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration

    print(f"Loading WAVE-7B from {model_dir}")
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU: {torch.cuda.get_device_name(0)} ({vram:.1f} GB VRAM)")
    else:
        print("  No GPU — CPU mode (very slow)")

    model_config = Qwen2_5OmniThinkerConfig.from_pretrained(str(model_dir))
    model_config.train_classify = True
    model_config.classify_type = "all_layer"
    model_config.sim_temperature = 0.02
    model_config.audio_config.beats_path = str(beats_path)
    model_config.audio_config.beats_only = False

    bnb_config = None
    if quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["beats"],  # BEATs uses weight_norm → incompatible with deepcopy
        )
        print("  Mode: INT4 NF4 (quantized, BEATs kept in FP16)")
    else:
        print("  Mode: FP16")

    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        str(model_dir),
        config=model_config,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=bnb_config,
    )

    # Load BEATs weights
    beats_ckpt = torch.load(str(beats_path), map_location="cpu")
    model.beats.load_state_dict(beats_ckpt["model"])
    print("  BEATs loaded")

    model.eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(str(model_dir))
    print("WAVE-7B ready")
    return model, processor


# ---------------------------------------------------------------------------
# Embedding functions
# ---------------------------------------------------------------------------

TEXT_QUERY_PREFIX = "task: fact checking | query: "
# Prompts EXACTOS del entrenamiento de WAVE (scripts/ret_*.json). El embedding sale del
# hidden state del ULTIMO token, entrenado con estos prompts; cambiarlos lo saca de distribucion.
VIDEO_PROMPT      = "Please describe the video."
AUDIO_PROMPT      = "Please describe the audio."
AV_PROMPT         = "Please describe the video."


def free_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _to_numpy(v):
    """Convert tensor or array to raw float32 numpy (no normalization)."""
    if isinstance(v, torch.Tensor):
        v = v.float().cpu().numpy()
    return np.array(v, dtype=np.float32)


def _normalize(v):
    """Normalize a vector to unit length (use only at comparison time)."""
    v = _to_numpy(v)
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-9 else v


def _run_wave(model, inputs_dict):
    with torch.no_grad():
        inputs_dict["pred_embeds"] = True
        outputs = model(**inputs_dict)
    # Store raw (non-normalized) embeddings; normalization applied at comparison time
    return _to_numpy(outputs.mllm_embeds.squeeze(0))


def _wave_chat_text(processor, prompt_text, media=None):
    """Construye el texto de entrada igual que el entrenamiento de WAVE (data_qwen.py:279-316):
    aplica el chat template y conserva el turno de usuario terminando en <|im_end|>, de modo que
    mllm_embeds (hidden state del ultimo token) se lea en la posicion que espera la cabeza
    contrastiva. Sin esto la lectura cae en un token de contenido fuera de distribucion ->
    embeddings anisotropos (~0.90 de coseno entre items distintos)."""
    content = []
    if media == "video":
        content.append({"type": "video"})
    elif media == "audio":
        content.append({"type": "audio"})
    if prompt_text:
        content.append({"type": "text", "text": prompt_text})
    conv = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(conv, add_generation_prompt=False, tokenize=False)
    if isinstance(text, list):
        text = text[0]
    return text.split("<|im_start|>user\n")[-1].strip()


def get_text_embedding(text, model, processor, prefix=TEXT_QUERY_PREFIX):
    full = f"{prefix}{text}" if prefix else text
    chat = _wave_chat_text(processor, full, media=None)
    inputs = processor(
        text=[chat], return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    inputs["types"] = "text"
    emb = _run_wave(model, inputs)
    free_vram()
    return emb


def _extract_audio(video_path, sample_rate=16000):
    container = av.open(video_path)
    audio_streams = [s for s in container.streams if s.type == "audio"]
    if not audio_streams:
        container.close()
        return None, None
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=sample_rate)
    frames = []
    for frame in container.decode(audio=0):
        for rf in resampler.resample(frame):
            frames.append(rf.to_ndarray()[0])
    container.close()
    if not frames:
        return None, None
    return np.concatenate(frames).astype(np.float32), sample_rate


def _extract_video_frames(video_path, max_frames=32):
    try:
        import decord
        decord.bridge.set_bridge("numpy")
        vr = decord.VideoReader(video_path)
        total_frames = len(vr)
        if total_frames > max_frames:
            indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        else:
            indices = np.arange(total_frames)
        return list(vr.get_batch(indices).asnumpy())
    except ImportError:
        import av
        
        def _get_total_frames(path):
            with av.open(path) as cont:
                stream = cont.streams.video[0]
                if stream.frames and stream.frames > 0:
                    return stream.frames
                return sum(1 for _ in cont.decode(stream))
                
        total_frames = _get_total_frames(video_path)
        if total_frames == 0:
            raise ValueError(f"No frames in {video_path}")
            
        if total_frames > max_frames:
            indices = set(np.linspace(0, total_frames - 1, max_frames, dtype=int))
        else:
            indices = set(np.arange(total_frames))
            
        frames = []
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            for i, frame in enumerate(container.decode(stream)):
                if i in indices:
                    frames.append(frame.to_ndarray(format='rgb24'))
                if len(frames) >= len(indices):
                    break
        return frames


def _downscale_frame(frame: np.ndarray, max_side: int) -> np.ndarray:
    """Downscale a single RGB frame to a maximum side length.

    This is a pragmatic guardrail: WAVE/Qwen2.5-Omni can otherwise attempt
    attention over extremely high-resolution frames, causing massive VRAM usage.
    """
    if max_side is None or max_side <= 0:
        return frame
    if not isinstance(frame, np.ndarray) or frame.ndim < 2:
        return frame

    h, w = frame.shape[:2]
    if max(h, w) <= max_side:
        return frame

    scale = max_side / float(max(h, w))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    try:
        from PIL import Image

        img = Image.fromarray(frame)
        resample = getattr(Image, "Resampling", Image).BICUBIC
        img = img.resize((new_w, new_h), resample=resample)
        out = np.asarray(img)
        return out.astype(np.uint8, copy=False)
    except Exception:
        import torch.nn.functional as F

        t = torch.from_numpy(frame)
        if t.ndim == 2:
            t = t.unsqueeze(-1)
        if t.shape[-1] == 3:
            t = t.permute(2, 0, 1)
        t = t.unsqueeze(0).float()  # 1xCxHxW on CPU
        t = F.interpolate(t, size=(new_h, new_w), mode="bilinear", align_corners=False)
        t = t.clamp(0, 255).byte().squeeze(0)
        if t.shape[0] == 3:
            t = t.permute(1, 2, 0)
        return t.numpy()


def get_video_embedding(video_path, model, processor, max_frames, max_frame_side):
    frames = _extract_video_frames(video_path, max_frames)
    if max_frame_side and max_frame_side > 0:
        frames = [_downscale_frame(f, max_frame_side) for f in frames]
    inputs = processor(
        text=[_wave_chat_text(processor, VIDEO_PROMPT, media="video")],
        videos=[frames],
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    inputs["types"] = "video"
    inputs["use_audio"] = False
    emb = _run_wave(model, inputs)
    free_vram()
    return emb


_AUDIO_WINDOW_S = 30  # seconds per chunk for audio/AV embedding


def _chunk_avg_normalize(embs):
    """L2-normalize each embedding, average, then re-normalize."""
    normed = [e / (np.linalg.norm(e) + 1e-9) for e in embs]
    avg = np.mean(normed, axis=0)
    norm = np.linalg.norm(avg)
    return avg / norm if norm > 1e-9 else avg


def get_audio_embedding(video_path, model, processor):
    audio, sr = _extract_audio(video_path)
    if audio is None:
        raise ValueError(f"No audio: {os.path.basename(video_path)}")
    win = _AUDIO_WINDOW_S * sr
    total = len(audio)
    starts = list(range(0, total, win))
    embs = []
    for s in starts:
        chunk = audio[s: s + win]
        inputs = processor(
            text=[_wave_chat_text(processor, AUDIO_PROMPT, media="audio")],
            audio=[chunk],
            return_tensors="pt",
        )
        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        inputs["types"] = "audio"
        inputs["use_audio"] = True
        inputs["input_raw_wav"] = [torch.tensor(chunk).to(model.device)]
        embs.append(_run_wave(model, inputs))
        free_vram()
    return _chunk_avg_normalize(embs)


def _extract_video_frames_window(video_path, t_start, t_end, max_frames):
    """Extract up to max_frames from [t_start, t_end) seconds."""
    try:
        import decord
        decord.bridge.set_bridge("numpy")
        vr = decord.VideoReader(video_path)
        fps = vr.get_avg_fps()
        total = len(vr)
        f_start = min(int(t_start * fps), total - 1)
        f_end   = min(int(t_end   * fps), total)
        if f_end <= f_start:
            f_end = min(f_start + 1, total)
        n = f_end - f_start
        if n > max_frames:
            indices = np.linspace(f_start, f_end - 1, max_frames, dtype=int)
        else:
            indices = np.arange(f_start, f_end)
        return list(vr.get_batch(indices).asnumpy())
    except ImportError:
        # fallback: decode all, slice by index estimate
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate or 25)
            total = stream.frames or 0
        f_start = min(int(t_start * fps), max(total - 1, 0))
        f_end   = min(int(t_end   * fps), total) if total else int(t_end * fps)
        if f_end <= f_start:
            f_end = f_start + 1
        n = max(f_end - f_start, 1)
        if n > max_frames:
            pick = set(np.linspace(f_start, f_end - 1, max_frames, dtype=int))
        else:
            pick = set(range(f_start, f_end))
        frames = []
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            for i, frame in enumerate(container.decode(stream)):
                if i in pick:
                    frames.append(frame.to_ndarray(format="rgb24"))
                if i >= f_end or len(frames) >= len(pick):
                    break
        return frames


def get_audiovisual_embedding(video_path, model, processor, max_frames, max_frame_side):
    audio, sr = _extract_audio(video_path)
    if audio is None:
        raise ValueError(f"No audio for AV: {os.path.basename(video_path)}")
    win = _AUDIO_WINDOW_S * sr
    total = len(audio)
    starts = list(range(0, total, win))
    embs = []
    for s in starts:
        t_start = s / sr
        t_end   = min((s + win) / sr, total / sr)
        audio_chunk = audio[s: s + win]
        frames_win = _extract_video_frames_window(video_path, t_start, t_end, max_frames)
        if not frames_win:
            continue
        if max_frame_side and max_frame_side > 0:
            frames_win = [_downscale_frame(f, max_frame_side) for f in frames_win]
        inputs = processor(
            text=[_wave_chat_text(processor, AV_PROMPT, media="video")],
            videos=[frames_win],
            audio=[audio_chunk],
            return_tensors="pt",
        )
        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        inputs["types"] = "video"
        inputs["use_audio"] = True
        inputs["input_raw_wav"] = [torch.tensor(audio_chunk).to(model.device)]
        embs.append(_run_wave(model, inputs))
        free_vram()
    return _chunk_avg_normalize(embs)


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def sim(a, b):
    """Cosine similarity with normalization applied at comparison time."""
    an = _normalize(a)
    bn = _normalize(b)
    return float(cosine_similarity(an.reshape(1, -1), bn.reshape(1, -1))[0][0])


def run_experiments(test_index, db, output_dir):
    results = []
    skipped = 0

    for item in test_index:
        rid = item["raw_id"]
        ev   = load_embedding(db, rid, "video")
        er   = load_embedding(db, rid, "text_real")
        ef   = load_embedding(db, rid, "text_fake")
        ea   = load_embedding(db, rid, "audio")
        eav  = load_embedding(db, rid, "audiovisual")

        if ev is None or er is None or ef is None:
            skipped += 1
            continue

        meta = load_video_metadata(db, rid)
        duration = meta["duration_seconds"] if meta else None

        row = {
            "raw_id": rid, "category": item["category"],
            "duration_seconds": duration,
            "over_120s": (duration > 120) if duration is not None else None,
            "sim_vid_real": sim(ev, er), "sim_vid_fake": sim(ev, ef),
            "e1_correct": sim(ev, er) > sim(ev, ef),
            "e1_margin":  sim(ev, er) - sim(ev, ef),
            "has_audio": ea is not None, "has_av": eav is not None,
        }

        if ea is not None:
            row["sim_audio_real"] = sim(ea, er)
            row["sim_audio_fake"] = sim(ea, ef)
            row["e2_correct"] = row["sim_audio_real"] > row["sim_audio_fake"]
            row["e2_margin"]  = row["sim_audio_real"] - row["sim_audio_fake"]
            cr = (row["sim_vid_real"] + row["sim_audio_real"]) / 2
            cf = (row["sim_vid_fake"] + row["sim_audio_fake"]) / 2
            row["e2c_correct"] = cr > cf
            row["e2c_margin"]  = cr - cf

        if eav is not None:
            row["sim_av_real"] = sim(eav, er)
            row["sim_av_fake"] = sim(eav, ef)
            row["e3_correct"]  = row["sim_av_real"] > row["sim_av_fake"]
            row["e3_margin"]   = row["sim_av_real"] - row["sim_av_fake"]

        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "OS_E1_wave.csv"), index=False)
    df.to_csv(os.path.join(output_dir, "OS_E2_wave.csv"), index=False)

    print(f"Results: {len(df)} videos, {skipped} skipped")
    print(f"  E1 (video):  {df['e1_correct'].mean():.4f}  ({df['e1_correct'].sum()}/{len(df)})")
    df_a = df[df["has_audio"]]
    if len(df_a) > 0:
        print(f"  E2 (audio):  {df_a['e2_correct'].mean():.4f}")
        print(f"  E2c (V+A):   {df_a['e2c_correct'].mean():.4f}")
    df_av = df[df["has_av"]]
    if len(df_av) > 0:
        print(f"  E3 (AV):     {df_av['e3_correct'].mean():.4f}")

    return df


def save_figures(df, output_dir):
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    if len(df) == 0:
        return

    methods, accs = ["E1: Video"], [df["e1_correct"].mean()]
    df_a = df[df["has_audio"]]
    if len(df_a) > 0:
        methods += ["E2: Audio", "E2c: V+A"]
        accs += [df_a["e2_correct"].mean(), df_a["e2c_correct"].mean()]
    df_av = df[df["has_av"]]
    if len(df_av) > 0:
        methods.append("E3: AV")
        accs.append(df_av["e3_correct"].mean())

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(methods, accs, color=["steelblue", "darkorange", "seagreen", "mediumpurple"][:len(methods)])
    ax.set_ylim(0, 1)
    ax.set_title("WAVE-7B — Accuracy by method")
    ax.set_ylabel("Accuracy")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.01, f"{acc:.3f}", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "OS_wave_methods.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Figure saved: OS_wave_methods.png")

    # Category breakdown E1
    cat_acc = df.groupby("category")["e1_correct"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    cat_acc.plot(kind="barh", ax=ax, color="steelblue")
    ax.axvline(df["e1_correct"].mean(), color="red", linestyle="--")
    ax.set_title("WAVE-7B E1 — Accuracy by category")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "OS_wave_E1_by_category.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Figure saved: OS_wave_E1_by_category.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    data_dir   = Path(args.data_dir)
    model_dir  = Path(args.model_dir)
    wave_repo  = Path(args.wave_repo)
    beats_path = Path(args.beats_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve text prefix
    text_prefix = None if args.no_prefix else TEXT_QUERY_PREFIX
    print(f"Text prefix: {'DISABLED (ablation)' if text_prefix is None else repr(text_prefix)}")

    # Verify prerequisites
    for p, name in [(model_dir, "model-dir"), (wave_repo, "wave-repo"), (beats_path, "beats-path")]:
        if not Path(p).exists():
            raise FileNotFoundError(f"Missing {name}: {p}")

    # Load test index
    with open(data_dir / "test_index.json", encoding="utf-8") as f:
        test_index = json.load(f)

    # Resolve video paths (flat fallback: test_video/X → X if not found)
    for item in test_index:
        item["video_path"] = str(resolve_video_path(data_dir, item["video_path"]))

    # Validate video files exist
    missing_videos = [it for it in test_index if not os.path.isfile(it["video_path"])]
    found_count = len(test_index) - len(missing_videos)
    print(f"Test set: {len(test_index)} videos ({found_count} found, {len(missing_videos)} MISSING)")
    for mv in missing_videos[:10]:
        print(f"  MISSING: {mv['video_path']}")
    if len(missing_videos) > 10:
        print(f"  ... and {len(missing_videos) - 10} more")
    if found_count == 0:
        print("ERROR: No video files found. Check --data-dir path.")
        sys.exit(1)

    # Init cache
    db = init_db(str(output_dir / "wave_cache.db"))
    print(f"Cache progress: {cache_progress(db)}")

    # Phase 0: collect video metadata and filter to ≤120s (mirrors run_qwen3vl.py)
    print("\n=== Phase 0: Video metadata + ≤120s filter ===")
    for item in test_index:
        rid = item["raw_id"]
        if load_video_metadata(db, rid) is None:
            duration, has_audio = get_video_duration(item["video_path"])
            save_video_metadata(db, rid, duration, has_audio)

    test_index_all = test_index
    test_index = [
        it for it in test_index_all
        if (lambda m: m is None or m["duration_seconds"] is None or m["duration_seconds"] <= 120)(
            load_video_metadata(db, it["raw_id"])
        )
    ]
    skipped_long = len(test_index_all) - len(test_index)
    print(f"  Total: {len(test_index_all)}  |  ≤120s: {len(test_index)}  |  skipped (>120s): {skipped_long}")

    # Load model
    model, processor = load_wave_model(model_dir, wave_repo, beats_path, quantize=args.quantize)

    # Phase A: text embeddings
    print("\n=== Phase A: Text embeddings ===")
    pending = [(it["raw_id"], "text_real", it["title"]) for it in test_index
               if load_embedding(db, it["raw_id"], "text_real") is None]
    pending += [(it["raw_id"], "text_fake", it["fake_title"]) for it in test_index
                if load_embedding(db, it["raw_id"], "text_fake") is None]
    print(f"Pending: {len(pending)}")
    for i, (rid, mod, text) in enumerate(pending):
        try:
            emb = get_text_embedding(text, model, processor, prefix=text_prefix)
            save_embedding(db, rid, mod, emb)
        except Exception as e:
            print(f"Error {rid}/{mod}: {e}")
        if (i + 1) % 100 == 0 or (i + 1) == len(pending):
            print(f"  [{i+1}/{len(pending)}] texts done")

    # Phase B: video embeddings
    print(
        f"\n=== Phase B: Video embeddings (max {args.max_frames} frames, max_side {args.max_frame_side}px) ==="
    )
    pending_v = [it for it in test_index if load_embedding(db, it["raw_id"], "video") is None]
    print(f"Pending: {len(pending_v)}")
    v_errors = []
    for i, item in enumerate(pending_v):
        rid   = item["raw_id"]
        vpath = item["video_path"]
        try:
            emb = get_video_embedding(vpath, model, processor, args.max_frames, args.max_frame_side)
            save_embedding(db, rid, "video", emb)
        except Exception as e:
            v_errors.append((rid, str(e)))
            print(f"  Error {rid}: {e}")
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if (i + 1) % 20 == 0 or (i + 1) == len(pending_v):
            print(f"  [{i+1}/{len(pending_v)}] videos done (errors: {len(v_errors)})")

    # Phase C: audio embeddings
    print("\n=== Phase C: Audio embeddings ===")
    pending_a = [it for it in test_index if load_embedding(db, it["raw_id"], "audio") is None]
    print(f"Pending: {len(pending_a)}")
    a_errors = []
    for i, item in enumerate(pending_a):
        try:
            emb = get_audio_embedding(item["video_path"], model, processor)
            save_embedding(db, item["raw_id"], "audio", emb)
        except Exception as e:
            a_errors.append((item["raw_id"], str(e)))
        if (i + 1) % 20 == 0 or (i + 1) == len(pending_a):
            print(f"  [{i+1}/{len(pending_a)}] audio done (errors: {len(a_errors)})")

    # Phase D: audiovisual embeddings
    print("\n=== Phase D: Audiovisual embeddings ===")
    pending_av = [it for it in test_index if load_embedding(db, it["raw_id"], "audiovisual") is None]
    print(f"Pending: {len(pending_av)}")
    av_errors = []
    for i, item in enumerate(pending_av):
        try:
            emb = get_audiovisual_embedding(
                item["video_path"], model, processor, args.max_frames, args.max_frame_side
            )
            save_embedding(db, item["raw_id"], "audiovisual", emb)
        except Exception as e:
            av_errors.append((item["raw_id"], str(e)))
        if (i + 1) % 20 == 0 or (i + 1) == len(pending_av):
            print(f"  [{i+1}/{len(pending_av)}] AV done (errors: {len(av_errors)})")

    print(f"\nCache final: {cache_progress(db)}")

    # Experiments
    print("\n=== Experiments ===")
    df = run_experiments(test_index, db, str(output_dir))
    save_figures(df, str(output_dir))

    print("\nDone.")


if __name__ == "__main__":
    main()
