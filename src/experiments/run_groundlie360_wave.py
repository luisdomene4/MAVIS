"""
MAVIS — WAVE-7B embedding pipeline for GroundLie360 dataset.

Computes global embeddings for every video in groundlie_index_filtered.json.
Uses the same model-loading and embedding functions as run_wave.py but:
  - Reads from groundlie_index_filtered.json (pre-filtered ≤120s)
  - Embeds the single title (modality: text_title) instead of real/fake title pair
  - Computes audio + audiovisual embeddings (needed for E6)
  - Saves scene_metadata from index (no TransNetV2 needed)
  - Saves groundlie_labels + groundlie_bboxes from index
  - Does NOT run experiments (those live in analysis.ipynb)

Usage (from repo root on cluster):
    python src/experiments/run_groundlie360_wave.py \
        --data-dir   data/GroundLie360 \
        --model-dir  src/models/WAVE-7B \
        --wave-repo  src/models/wave_repo \
        --beats-path src/models/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt \
        --output-dir experiments/GroundLie360/open_source/results/WAVE7B \
        [--index-json experiments/GroundLie360/groundlie_index_filtered.json] \
        [--bbox-json  experiments/GroundLie360/bbox_index.json] \
        [--limit 5]

Outputs:
    <output-dir>/wave_cache.db    -- resumable SQLite cache (8 tables)
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import av
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from utils.db_schema import (
    init_db, cache_progress,
    save_embedding, load_embedding,
    save_video_metadata, load_video_metadata,
    save_scene_metadata, load_scene_metadata,
    save_groundlie_labels, save_groundlie_bboxes,
)

DEFAULT_INDEX = "experiments/GroundLie360/groundlie_index_filtered.json"
DEFAULT_BBOX  = "experiments/GroundLie360/bbox_index.json"

TEXT_QUERY_PREFIX = "task: fact checking | query: "
# Prompts EXACTOS del entrenamiento de WAVE (scripts/ret_*.json). El embedding sale del
# hidden state del ULTIMO token, entrenado con estos prompts; cambiarlos lo saca de distribucion.
VIDEO_PROMPT      = "Please describe the video."
AUDIO_PROMPT      = "Please describe the audio."
AV_PROMPT         = "Please describe the video."


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="WAVE-7B embedding pipeline for GroundLie360")
    p.add_argument("--data-dir",       required=True)
    p.add_argument("--model-dir",      required=True)
    p.add_argument("--wave-repo",      required=True)
    p.add_argument("--beats-path",     required=True)
    p.add_argument("--output-dir",     required=True)
    p.add_argument("--index-json",     default=DEFAULT_INDEX)
    p.add_argument("--bbox-json",      default=DEFAULT_BBOX)
    p.add_argument("--max-frames",     type=int, default=32)
    p.add_argument("--max-frame-side", type=int, default=256)
    p.add_argument("--quantize",       action="store_true")
    p.add_argument("--no-prefix",      action="store_true")
    p.add_argument("--limit",          type=int, default=None)
    p.add_argument("--qwen-db",        default=None,
                   help="Path to a Qwen3VL qwen3vl_cache.db to read transcript text from. "
                        "Required for the 'transcript' modality (E6). WAVE runs after Qwen.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading (identical to run_wave.py)
# ---------------------------------------------------------------------------

def load_wave_model(model_dir, wave_repo, beats_path, quantize=False):
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

    model_config = Qwen2_5OmniThinkerConfig.from_pretrained(str(model_dir))
    model_config.train_classify = True
    model_config.classify_type = "all_layer"
    model_config.sim_temperature = 0.02
    model_config.audio_config.beats_path = str(beats_path)
    model_config.audio_config.beats_only = False

    bnb_config = None
    if quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["beats"],
        )
    print(f"  Mode: {'INT4 NF4' if quantize else 'FP16'}")

    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        str(model_dir),
        config=model_config,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=bnb_config,
    )
    beats_ckpt = torch.load(str(beats_path), map_location="cpu")
    model.beats.load_state_dict(beats_ckpt["model"])
    print("  BEATs loaded")
    model.eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(str(model_dir))
    print("WAVE-7B ready")
    return model, processor


# ---------------------------------------------------------------------------
# Embedding helpers (identical to run_wave.py)
# ---------------------------------------------------------------------------

def free_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _to_numpy(v):
    if isinstance(v, torch.Tensor):
        v = v.float().cpu().numpy()
    return np.array(v, dtype=np.float32)


def _run_wave(model, inputs_dict):
    with torch.no_grad():
        inputs_dict["pred_embeds"] = True
        outputs = model(**inputs_dict)
    return _to_numpy(outputs.mllm_embeds.squeeze(0))


def _wave_chat_text(processor, prompt_text, media=None):
    """Construye el texto igual que el entrenamiento de WAVE (data_qwen.py:279-316): chat template
    conservando el turno de usuario terminando en <|im_end|>, para que mllm_embeds (hidden state del
    ultimo token) se lea en la posicion que espera la cabeza contrastiva. Sin esto -> anisotropia ~0.90."""
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
    inputs = processor(text=[chat], return_tensors="pt", padding=True, truncation=True, max_length=512)
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
        total = len(vr)
        indices = np.linspace(0, total - 1, min(max_frames, total), dtype=int)
        return list(vr.get_batch(indices).asnumpy())
    except ImportError:
        frames = []
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            total = stream.frames or sum(1 for _ in container.decode(stream))
        indices = set(np.linspace(0, total - 1, min(max_frames, total), dtype=int))
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            for i, frame in enumerate(container.decode(stream)):
                if i in indices:
                    frames.append(frame.to_ndarray(format="rgb24"))
                if len(frames) >= len(indices):
                    break
        return frames


def _downscale_frame(frame: np.ndarray, max_side: int) -> np.ndarray:
    if not max_side or max_side <= 0:
        return frame
    h, w = frame.shape[:2]
    if max(h, w) <= max_side:
        return frame
    scale = max_side / float(max(h, w))
    new_h, new_w = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    try:
        from PIL import Image
        img = Image.fromarray(frame).resize((new_w, new_h), Image.BICUBIC)
        return np.asarray(img).astype(np.uint8, copy=False)
    except Exception:
        import torch.nn.functional as F
        t = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float()
        t = F.interpolate(t, size=(new_h, new_w), mode="bilinear", align_corners=False)
        return t.clamp(0, 255).byte().squeeze(0).permute(1, 2, 0).numpy()


def get_video_embedding(video_path, model, processor, max_frames, max_frame_side):
    frames = _extract_video_frames(video_path, max_frames)
    if max_frame_side and max_frame_side > 0:
        frames = [_downscale_frame(f, max_frame_side) for f in frames]
    inputs = processor(text=[_wave_chat_text(processor, VIDEO_PROMPT, media="video")], videos=[frames], return_tensors="pt")
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
        inputs = processor(text=[_wave_chat_text(processor, AUDIO_PROMPT, media="audio")], audio=[chunk], return_tensors="pt")
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
        indices = np.linspace(f_start, f_end - 1, min(n, max_frames), dtype=int)
        return list(vr.get_batch(indices).asnumpy())
    except ImportError:
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate or 25)
            total = stream.frames or 0
        f_start = min(int(t_start * fps), max(total - 1, 0))
        f_end   = min(int(t_end   * fps), total) if total else int(t_end * fps)
        if f_end <= f_start:
            f_end = f_start + 1
        n = max(f_end - f_start, 1)
        pick = set(np.linspace(f_start, f_end - 1, min(n, max_frames), dtype=int))
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
        inputs = processor(text=[_wave_chat_text(processor, AV_PROMPT, media="video")], videos=[frames_win], audio=[audio_chunk], return_tensors="pt")
        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        inputs["types"] = "video"
        inputs["use_audio"] = True
        inputs["input_raw_wav"] = [torch.tensor(audio_chunk).to(model.device)]
        embs.append(_run_wave(model, inputs))
        free_vram()
    return _chunk_avg_normalize(embs)


# ---------------------------------------------------------------------------
# GroundLie360 helpers
# ---------------------------------------------------------------------------

def scene_metadata_from_entry(entry: dict) -> list:
    fps = entry.get("fps") or 25.0
    return [
        {"scene_idx": i, "start_s": round(sf / fps, 4), "end_s": round(ef / fps, 4),
         "detector": "groundlie_dataset", "confidence": 1.0}
        for i, (sf, ef) in enumerate(entry.get("scene_frames", []))
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    text_prefix = None if args.no_prefix else TEXT_QUERY_PREFIX
    print(f"Text prefix: {'DISABLED' if text_prefix is None else repr(text_prefix)}")

    for p, name in [(args.model_dir, "model-dir"), (args.wave_repo, "wave-repo"), (args.beats_path, "beats-path")]:
        if not Path(p).exists():
            raise FileNotFoundError(f"Missing {name}: {p}")

    with open(args.index_json, encoding="utf-8") as f:
        index = json.load(f)
    with open(args.bbox_json, encoding="utf-8") as f:
        bbox_index = json.load(f)

    if args.limit:
        index = index[: args.limit]
    print(f"Index: {len(index)} entries")

    for entry in index:
        entry["_abs_video"] = str((data_dir / entry["video_path"]).resolve())

    missing = [e for e in index if not os.path.isfile(e["_abs_video"])]
    print(f"Videos: {len(index) - len(missing)} found, {len(missing)} MISSING")
    for e in missing[:10]:
        print(f"  MISSING: {e['_abs_video']}")
    if len(index) == len(missing):
        print("ERROR: No videos found. Check --data-dir.")
        sys.exit(1)

    db = init_db(str(output_dir / "wave_cache.db"))
    print(f"Cache: {cache_progress(db)}")

    # Phase 0: index metadata (labels, scenes, bboxes) — model-independent
    print("\n=== Phase 0: Index metadata ===")
    for entry in index:
        rid = entry["raw_id"]

        if db.execute("SELECT 1 FROM groundlie_labels WHERE raw_id=?", (rid,)).fetchone() is None:
            save_groundlie_labels(db, rid, entry)

        if load_video_metadata(db, rid) is None:
            save_video_metadata(db, rid, entry.get("duration_seconds"), True,
                                dataset_origin="groundlie360")

        if not load_scene_metadata(db, rid):
            scenes = scene_metadata_from_entry(entry)
            if scenes:
                save_scene_metadata(db, rid, scenes)

        if entry.get("has_bbox") and rid in bbox_index:
            if db.execute("SELECT 1 FROM groundlie_bboxes WHERE raw_id=?", (rid,)).fetchone() is None:
                save_groundlie_bboxes(db, rid, bbox_index[rid])

    print("Phase 0 done.")

    # Load model
    model, processor = load_wave_model(args.model_dir, args.wave_repo, args.beats_path, args.quantize)

    # Phase 0: video metadata for ≤120s gate (mirrors run_wave.py)
    print("\n=== Phase 0b: Video metadata + ≤120s filter ===")
    test_index_all = index
    index = [
        it for it in test_index_all
        if (it.get("duration_seconds") is None or it["duration_seconds"] <= 120)
    ]
    skipped_long = len(test_index_all) - len(index)
    print(f"  Total: {len(test_index_all)}  |  ≤120s: {len(index)}  |  skipped (>120s): {skipped_long}")

    # Phase A: title text embeddings
    print("\n=== Phase A: Title embeddings (text_title) ===")
    pending_text = [e for e in index if load_embedding(db, e["raw_id"], "text_title") is None]
    print(f"Pending: {len(pending_text)}")
    for i, entry in enumerate(pending_text):
        try:
            emb = get_text_embedding(entry["title"], model, processor, prefix=text_prefix)
            save_embedding(db, entry["raw_id"], "text_title", emb)
        except Exception as exc:
            print(f"  Error title {entry['raw_id']}: {exc}")
        if (i + 1) % 100 == 0 or (i + 1) == len(pending_text):
            print(f"  [{i+1}/{len(pending_text)}] titles done")

    # Phase A': transcript embeddings (text from Qwen DB, embedded with WAVE)
    if args.qwen_db:
        print("\n=== Phase A': Transcript embeddings (from Qwen DB) ===")
        import sqlite3 as _sqlite3
        qconn = _sqlite3.connect(args.qwen_db)
        pending_tr = [e for e in index if load_embedding(db, e["raw_id"], "transcript") is None]
        print(f"Pending: {len(pending_tr)}")
        tr_errors = []
        for i, entry in enumerate(pending_tr):
            rid = entry["raw_id"]
            row = qconn.execute("SELECT text FROM transcripts WHERE raw_id=?", (rid,)).fetchone()
            if row is None or not row[0]:
                continue
            try:
                emb = get_text_embedding(row[0], model, processor, prefix=text_prefix)
                save_embedding(db, rid, "transcript", emb)
            except Exception as exc:
                tr_errors.append((rid, str(exc)))
                print(f"  Error transcript {rid}: {exc}")
            if (i + 1) % 100 == 0 or (i + 1) == len(pending_tr):
                print(f"  [{i+1}/{len(pending_tr)}] transcripts done (errors: {len(tr_errors)})")
        qconn.close()
    else:
        print("\n[SKIP] Phase A': --qwen-db not provided, transcript modality will be absent.")
        print("       Run Qwen3VL first, then re-run with --qwen-db to populate 'transcript'.")

    # Phase B: video embeddings
    print(f"\n=== Phase B: Video embeddings (max_frames={args.max_frames}) ===")
    pending_v = [e for e in index if load_embedding(db, e["raw_id"], "video") is None]
    print(f"Pending: {len(pending_v)}")
    v_errors = []
    for i, entry in enumerate(pending_v):
        rid   = entry["raw_id"]
        vpath = entry["_abs_video"]
        try:
            emb = get_video_embedding(vpath, model, processor, args.max_frames, args.max_frame_side)
            save_embedding(db, rid, "video", emb)
        except Exception as exc:
            v_errors.append((rid, str(exc)))
            print(f"  Error video {rid}: {exc}")
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if (i + 1) % 20 == 0 or (i + 1) == len(pending_v):
            print(f"  [{i+1}/{len(pending_v)}] videos done (errors: {len(v_errors)})")

    # Phase C: audio embeddings
    print("\n=== Phase C: Audio embeddings ===")
    pending_a = [e for e in index if load_embedding(db, e["raw_id"], "audio") is None]
    print(f"Pending: {len(pending_a)}")
    a_errors = []
    for i, entry in enumerate(pending_a):
        rid   = entry["raw_id"]
        vpath = entry["_abs_video"]
        try:
            emb = get_audio_embedding(vpath, model, processor)
            save_embedding(db, rid, "audio", emb)
        except Exception as exc:
            a_errors.append((rid, str(exc)))
        if (i + 1) % 20 == 0 or (i + 1) == len(pending_a):
            print(f"  [{i+1}/{len(pending_a)}] audio done (errors: {len(a_errors)})")

    # Phase D: audiovisual embeddings
    print("\n=== Phase D: Audiovisual embeddings ===")
    pending_av = [e for e in index if load_embedding(db, e["raw_id"], "audiovisual") is None]
    print(f"Pending: {len(pending_av)}")
    av_errors = []
    for i, entry in enumerate(pending_av):
        rid   = entry["raw_id"]
        vpath = entry["_abs_video"]
        try:
            emb = get_audiovisual_embedding(vpath, model, processor, args.max_frames, args.max_frame_side)
            save_embedding(db, rid, "audiovisual", emb)
        except Exception as exc:
            av_errors.append((rid, str(exc)))
        if (i + 1) % 20 == 0 or (i + 1) == len(pending_av):
            print(f"  [{i+1}/{len(pending_av)}] AV done (errors: {len(av_errors)})")

    print(f"\nCache final: {cache_progress(db)}")
    print("Done.")


if __name__ == "__main__":
    main()
