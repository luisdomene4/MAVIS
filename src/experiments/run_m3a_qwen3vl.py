"""
MAVIS — Qwen3-VL-Embedding pipeline for the M3A dataset (Xu et al. 2024).

Clone of run_groundlie360_qwen3vl.py adapted to M3A:
  - Reads m3a_index_<N>.json (built by build_m3a_index.py + materialize_m3a_subset.py)
  - Phase 0 saves m3a_meta + m3a_nem (no scenes/bboxes — M3A has none)
  - Phase A embeds the BART Summary  -> modality `text_summary`
  - Phase A-NEM embeds each present NEM fake text -> `text_nem_{person,location,organization,complete}`
  - Phase A' transcript (Whisper) and Phase B video are identical to GroundLie360
  - MM / MTG re-pairing happens later in analysis.ipynb (mappings live in the index JSON)

Usage (from repo root on cluster):
    python src/experiments/run_m3a_qwen3vl.py \
        --data-dir   data/M3A \
        --model-dir  src/models/Qwen3-VL-Embedding-8B \
        --qwen-repo  src/models/qwen3vl_embedding_repo \
        --output-dir experiments/M3A/open_source/results/qwen3vl_8b \
        --index-json experiments/M3A/m3a_index_2000.json \
        [--limit 5] [--quantize]
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
    save_transcript_words,
    save_video_metadata, load_video_metadata,
    save_m3a_meta, save_m3a_nem,
)

DEFAULT_INSTRUCTION = "Represent the input for multimodal fact-checking verification."
DEFAULT_INDEX = "experiments/M3A/m3a_index_2000.json"
NEM_SUBTYPES = ["person", "location", "organization", "complete"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Qwen3-VL embedding pipeline for M3A")
    p.add_argument("--data-dir",    required=True, help="Dir containing videos/ (data/M3A)")
    p.add_argument("--model-dir",   required=True, help="Qwen3-VL-Embedding checkpoint dir (2B or 8B)")
    p.add_argument("--qwen-repo",   required=True, help="Dir with QwenLM/Qwen3-VL-Embedding source")
    p.add_argument("--output-dir",  required=True, help="Output dir for qwen3vl_cache.db")
    p.add_argument("--index-json",  default=DEFAULT_INDEX, help="m3a_index_<N>.json path")
    p.add_argument("--max-frames",  type=int,   default=16)
    p.add_argument("--fps",         type=float, default=1.0)
    p.add_argument("--quantize",    action="store_true")
    p.add_argument("--no-instruction", action="store_true")
    p.add_argument("--with-mtg",    action="store_true", help="Also embed MTG text -> text_mtg")
    p.add_argument("--limit",       type=int,   default=None,
                   help="Process only the first N index entries (smoke test)")
    p.add_argument("--skip-ids",    default="",
                   help="Comma-separated raw_ids to skip in Phase B (debug)")
    p.add_argument("--qwen-db",     default=None,
                   help="If set, reuse transcript text from this DB instead of running Whisper "
                        "(no Whisper load → más VRAM para el modelo grande, p.ej. 8B).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading (identical to run_qwen3vl.py / run_groundlie360_qwen3vl.py)
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


def whisper_words(video_path: str, whisper_model):
    """Transcribe (auto-detected language) and return (full_text, word_list, lang).
    Returns ('', [], None) on failure. M3A is multilingual (60 global outlets) so the
    language is auto-detected per video rather than forced to English."""
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", video_path],
            capture_output=True, text=True,
        )
        streams = json.loads(probe.stdout).get("streams", [])
        if not any(s.get("codec_type") == "audio" for s in streams):
            return "", [], None

        segments, info = whisper_model.transcribe(
            video_path, language=None, word_timestamps=True
        )
        words, texts = [], []
        for seg in segments:
            texts.append(seg.text.strip())
            if seg.words:
                for w in seg.words:
                    words.append({"word": w.word, "start_s": w.start, "end_s": w.end, "probability": w.probability})
        return " ".join(texts).strip(), words, getattr(info, "language", None)
    except Exception as exc:
        print(f"  Whisper error: {exc}")
        return "", [], None


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
        print("ERROR: No videos found. Check --data-dir / materialize step.")
        sys.exit(1)

    # Init DB
    db = init_db(str(output_dir / "qwen3vl_cache.db"))
    print(f"Cache: {cache_progress(db)}")

    # Load models
    model = load_model(
        args.model_dir, args.qwen_repo,
        max_frames=args.max_frames, fps=args.fps, quantize=args.quantize,
    )
    whisper_model = None if args.qwen_db else load_whisper()

    # Phase 0: index metadata (m3a_meta + m3a_nem) — model-independent, save once
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

    # Phase A: Summary text embeddings (text_summary)
    print("\n=== Phase A: Summary embeddings (text_summary) ===")
    pending_text = [e for e in index if load_embedding(db, e["raw_id"], "text_summary") is None]
    print(f"Pending: {len(pending_text)}")
    for i, entry in enumerate(pending_text):
        try:
            emb = get_text_embedding(entry["summary"], model, instruction)
            save_embedding(db, entry["raw_id"], "text_summary", emb)
        except Exception as exc:
            print(f"  Error summary {entry['raw_id']}: {exc}")
        if (i + 1) % 100 == 0 or (i + 1) == len(pending_text):
            print(f"  [{i+1}/{len(pending_text)}] summaries done")

    # Phase A-NEM: NEM fake-text embeddings (text_nem_<subtype>)
    print("\n=== Phase A-NEM: NEM fake-text embeddings ===")
    subtypes = list(NEM_SUBTYPES)
    if args.with_mtg:
        subtypes = subtypes + ["mtg"]
    for sub in subtypes:
        mod_name = f"text_{'mtg' if sub == 'mtg' else 'nem_' + sub}"
        if sub == "mtg":
            pend = [e for e in index if e.get("mtg_text")
                    and load_embedding(db, e["raw_id"], mod_name) is None]
        else:
            pend = [e for e in index if e.get("nem_texts", {}).get(sub)
                    and load_embedding(db, e["raw_id"], mod_name) is None]
        print(f"  {mod_name}: pending {len(pend)}")
        for i, entry in enumerate(pend):
            txt = entry["mtg_text"] if sub == "mtg" else entry["nem_texts"][sub]
            try:
                emb = get_text_embedding(txt, model, instruction)
                save_embedding(db, entry["raw_id"], mod_name, emb)
            except Exception as exc:
                print(f"    Error {mod_name} {entry['raw_id']}: {exc}")

    # Phase A': transcript embeddings.
    #   --qwen-db given  -> reuse transcript text from that DB (NO Whisper load).
    #   otherwise        -> run Whisper (auto-detect language) + word timestamps.
    pending_tr = [e for e in index if load_embedding(db, e["raw_id"], "transcript") is None]
    tr_errors = []
    if args.qwen_db:
        print("\n=== Phase A': Transcript embeddings (reuse from --qwen-db, no Whisper) ===")
        print(f"Pending: {len(pending_tr)}")
        import sqlite3 as _sqlite3
        qconn = _sqlite3.connect(args.qwen_db)
        for i, entry in enumerate(pending_tr):
            rid = entry["raw_id"]
            row = qconn.execute("SELECT text FROM transcripts WHERE raw_id=?", (rid,)).fetchone()
            if row is None or not row[0]:
                continue  # NO_AUDIO / empty in source DB
            try:
                emb = get_text_embedding(row[0], model, instruction)
                save_embedding(db, rid, "transcript", emb)
            except Exception as exc:
                tr_errors.append((rid, str(exc)))
            if (i + 1) % 100 == 0 or (i + 1) == len(pending_tr):
                print(f"  [{i+1}/{len(pending_tr)}] transcripts done (errors: {len(tr_errors)})")
        qconn.close()
    else:
        print("\n=== Phase A': Transcript (Whisper, auto-detect) ===")
        pending_tr = [e for e in pending_tr if load_transcript_text(db, e["raw_id"]) is None]
        print(f"Pending: {len(pending_tr)}")
        for i, entry in enumerate(pending_tr):
            rid   = entry["raw_id"]
            vpath = entry["_abs_video"]
            if not entry.get("has_transcript", True):
                save_transcript_text(db, rid, "", model_name="NO_AUDIO")
                tr_errors.append((rid, "no_audio_in_index"))
                continue
            _, has_audio = get_video_duration(vpath)
            if not has_audio:
                save_transcript_text(db, rid, "", model_name="NO_AUDIO")
                tr_errors.append((rid, "no audio stream"))
                continue
            try:
                text, words, lang = whisper_words(vpath, whisper_model)
                if not text:
                    tr_errors.append((rid, "empty transcript"))
                    continue
                save_transcript_text(db, rid, text, model_name="whisper-small", language=lang)
                if words:
                    save_transcript_words(db, rid, words)
                emb = get_text_embedding(text, model, instruction)
                save_embedding(db, rid, "transcript", emb)
            except Exception as exc:
                tr_errors.append((rid, str(exc)))
                print(f"  Error transcript {rid}: {exc}")
            if (i + 1) % 50 == 0 or (i + 1) == len(pending_tr):
                print(f"  [{i+1}/{len(pending_tr)}] transcripts done (errors: {len(tr_errors)})")

    # Free Whisper from GPU before video inference (only if it was loaded)
    if whisper_model is not None:
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
