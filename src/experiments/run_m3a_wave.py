"""
MAVIS — WAVE-7B embedding pipeline for the M3A dataset (Xu et al. 2024).

Clone of run_groundlie360_wave.py adapted to M3A:
  - Reads m3a_index_<N>.json
  - Phase 0 saves m3a_meta + m3a_nem (no scenes/bboxes)
  - Phase A embeds the BART Summary -> `text_summary`
  - Phase A-NEM embeds NEM fake texts -> `text_nem_{subtype}` (+ optional `text_mtg`)
  - Phase A' transcript: read transcript text from the Qwen DB and embed with WAVE
  - Phase B video embeddings
  - NO raw audio / audiovisual phases — the WAVE audio encoder is unreliable; the audio
    channel is represented via the Whisper transcript (text), comparable across all models.

Usage (after running run_m3a_qwen3vl.py to populate transcripts):
    python src/experiments/run_m3a_wave.py \
        --data-dir   data/M3A \
        --model-dir  src/models/WAVE-7B \
        --wave-repo  src/models/wave_repo \
        --beats-path src/models/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt \
        --output-dir experiments/M3A/open_source/results/WAVE7B \
        --index-json experiments/M3A/m3a_index_2000.json \
        --qwen-db    experiments/M3A/open_source/results/qwen3vl_2b/qwen3vl_cache.db \
        [--limit 5] [--quantize]
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from utils.db_schema import (
    init_db, cache_progress,
    save_embedding, load_embedding,
    save_video_metadata, load_video_metadata,
    save_m3a_meta, save_m3a_nem,
)

DEFAULT_INDEX = "experiments/M3A/m3a_index_2000.json"
NEM_SUBTYPES = ["person", "location", "organization", "complete"]

TEXT_QUERY_PREFIX = "task: fact checking | query: "
# Prompt EXACTO del entrenamiento de WAVE (scripts/ret_*.json). El embedding sale del hidden
# state del ULTIMO token, entrenado con este prompt; cambiarlo lo saca de distribucion.
VIDEO_PROMPT      = "Please describe the video."


def parse_args():
    p = argparse.ArgumentParser(description="WAVE-7B embedding pipeline for M3A")
    p.add_argument("--data-dir",       required=True)
    p.add_argument("--model-dir",      required=True)
    p.add_argument("--wave-repo",      required=True)
    p.add_argument("--beats-path",     required=True)
    p.add_argument("--output-dir",     required=True)
    p.add_argument("--index-json",     default=DEFAULT_INDEX)
    p.add_argument("--max-frames",     type=int, default=32)
    p.add_argument("--max-frame-side", type=int, default=256)
    p.add_argument("--quantize",       action="store_true")
    p.add_argument("--no-prefix",      action="store_true")
    p.add_argument("--with-mtg",       action="store_true")
    p.add_argument("--limit",          type=int, default=None)
    p.add_argument("--qwen-db",        default=None,
                   help="Path to a run_m3a_qwen3vl qwen3vl_cache.db to read transcript text.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading + embedding helpers (identical to run_wave.py / groundlie)
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


def _extract_video_frames(video_path, max_frames=32):
    try:
        import decord
        decord.bridge.set_bridge("numpy")
        vr = decord.VideoReader(video_path)
        total = len(vr)
        indices = np.linspace(0, total - 1, min(max_frames, total), dtype=int)
        return list(vr.get_batch(indices).asnumpy())
    except ImportError:
        import av
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


def _downscale_frame(frame, max_side):
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
    if args.limit:
        index = index[: args.limit]
    print(f"Index: {len(index)} entries")

    for entry in index:
        entry["_abs_video"] = str((data_dir / entry["video_path"]).resolve())

    missing = [e for e in index if not os.path.isfile(e["_abs_video"])]
    print(f"Videos: {len(index) - len(missing)} found, {len(missing)} MISSING")
    if len(index) == len(missing):
        print("ERROR: No videos found. Check --data-dir / materialize step.")
        sys.exit(1)

    db = init_db(str(output_dir / "wave_cache.db"))
    print(f"Cache: {cache_progress(db)}")

    # Phase 0: index metadata (m3a_meta + m3a_nem)
    print("\n=== Phase 0: Index metadata (m3a_meta, m3a_nem) ===")
    for entry in index:
        rid = entry["raw_id"]
        if db.execute("SELECT 1 FROM m3a_meta WHERE raw_id=?", (rid,)).fetchone() is None:
            save_m3a_meta(db, rid, entry)
            save_m3a_nem(db, rid, entry.get("nem_texts", {}))
        if load_video_metadata(db, rid) is None:
            save_video_metadata(db, rid, entry.get("duration_seconds"),
                                entry.get("has_audio", False), dataset_origin="m3a")
    print("Phase 0 done.")

    # Load model
    model, processor = load_wave_model(args.model_dir, args.wave_repo, args.beats_path, args.quantize)

    # Phase A: Summary text embeddings (text_summary)
    print("\n=== Phase A: Summary embeddings (text_summary) ===")
    pending_text = [e for e in index if load_embedding(db, e["raw_id"], "text_summary") is None]
    print(f"Pending: {len(pending_text)}")
    for i, entry in enumerate(pending_text):
        try:
            emb = get_text_embedding(entry["summary"], model, processor, prefix=text_prefix)
            save_embedding(db, entry["raw_id"], "text_summary", emb)
        except Exception as exc:
            print(f"  Error summary {entry['raw_id']}: {exc}")
        if (i + 1) % 100 == 0 or (i + 1) == len(pending_text):
            print(f"  [{i+1}/{len(pending_text)}] summaries done")

    # Phase A-NEM: NEM fake-text embeddings
    print("\n=== Phase A-NEM: NEM fake-text embeddings ===")
    subtypes = list(NEM_SUBTYPES) + (["mtg"] if args.with_mtg else [])
    for sub in subtypes:
        mod_name = f"text_{'mtg' if sub == 'mtg' else 'nem_' + sub}"
        if sub == "mtg":
            pend = [e for e in index if e.get("mtg_text")
                    and load_embedding(db, e["raw_id"], mod_name) is None]
        else:
            pend = [e for e in index if e.get("nem_texts", {}).get(sub)
                    and load_embedding(db, e["raw_id"], mod_name) is None]
        print(f"  {mod_name}: pending {len(pend)}")
        for entry in pend:
            txt = entry["mtg_text"] if sub == "mtg" else entry["nem_texts"][sub]
            try:
                emb = get_text_embedding(txt, model, processor, prefix=text_prefix)
                save_embedding(db, entry["raw_id"], mod_name, emb)
            except Exception as exc:
                print(f"    Error {mod_name} {entry['raw_id']}: {exc}")

    # Phase A': transcript embeddings (text from Qwen DB, embedded with WAVE)
    if args.qwen_db:
        print("\n=== Phase A': Transcript embeddings (from Qwen DB) ===")
        import sqlite3 as _sqlite3
        qconn = _sqlite3.connect(args.qwen_db)
        pending_tr = [e for e in index if load_embedding(db, e["raw_id"], "transcript") is None]
        print(f"Pending: {len(pending_tr)}")
        for i, entry in enumerate(pending_tr):
            rid = entry["raw_id"]
            row = qconn.execute("SELECT text FROM transcripts WHERE raw_id=?", (rid,)).fetchone()
            if row is None or not row[0]:
                continue
            try:
                emb = get_text_embedding(row[0], model, processor, prefix=text_prefix)
                save_embedding(db, rid, "transcript", emb)
            except Exception as exc:
                print(f"  Error transcript {rid}: {exc}")
            if (i + 1) % 100 == 0 or (i + 1) == len(pending_tr):
                print(f"  [{i+1}/{len(pending_tr)}] transcripts done")
        qconn.close()
    else:
        print("\n[SKIP] Phase A': --qwen-db not provided; transcript modality absent.")

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

    print(f"\nCache final: {cache_progress(db)}")
    print("Done.")


if __name__ == "__main__":
    main()
