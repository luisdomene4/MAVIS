"""
MAVIS — Google Embeddings 2 pipeline for the M3A dataset (cluster execution).

Clone of run_groundlie360_ge2.py adapted to M3A. Same conservative rate limiter
(20 RPM target, 800 RPD/session, 3s gap, crash-safe, resumable). Phases:
  0  — m3a_meta + m3a_nem from index (no API)
  A  — text_summary batch embeddings (50 texts/call)
  N  — NEM fake-text batch embeddings -> text_nem_{subtype} (+ optional text_mtg)
  B  — transcript batch embeddings (text read from --qwen-db)
  C  — video embeddings one-by-one (uploads MP4 bytes)

Usage (from repo root on cluster):
    python src/experiments/run_m3a_ge2.py \
        --data-dir   data/M3A \
        --output-dir experiments/M3A/google_embeddings2/results \
        --index-json experiments/M3A/m3a_index_2000.json \
        --qwen-db    experiments/M3A/open_source/results/qwen3vl_2b/qwen3vl_cache.db \
        [--phases 0,A,N,B,C] [--rpm-target 20] [--rpd-limit 800] [--with-mtg]

Requirements: pip install google-generativeai python-dotenv ; GEMINI_API_KEY in env/.env
"""

import argparse
import collections
import json
import os
import sqlite3
import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from utils.db_schema import (
    init_db, cache_progress,
    save_embedding, load_embedding,
    save_video_metadata, load_video_metadata,
    save_m3a_meta, save_m3a_nem,
)

DEFAULT_INDEX  = str(REPO_ROOT / "experiments/M3A/m3a_index_2000.json")
DEFAULT_OUTPUT = str(REPO_ROOT / "experiments/M3A/google_embeddings2/results")
MODEL_NAME     = "gemini-embedding-2-preview"
OUTPUT_DIM     = 3072
TEXT_PREFIX    = "task: fact checking | query: "
DOC_PREFIX     = "task: fact checking | document: "
NEM_SUBTYPES   = ["person", "location", "organization", "complete"]


def parse_args():
    p = argparse.ArgumentParser(description="GE2 embedding pipeline for M3A (cluster)")
    p.add_argument("--data-dir",    default=None, help="data/M3A (required for Phase C)")
    p.add_argument("--output-dir",  default=DEFAULT_OUTPUT)
    p.add_argument("--index-json",  default=DEFAULT_INDEX)
    p.add_argument("--qwen-db",     default=None, help="qwen3vl_cache.db for transcript text (Phase B)")
    p.add_argument("--phases",      default="0,A,N,B,C")
    p.add_argument("--rpm-target",  type=float, default=20)
    p.add_argument("--rpd-limit",   type=int,   default=800)
    p.add_argument("--batch-size",  type=int,   default=50)
    p.add_argument("--with-mtg",    action="store_true")
    p.add_argument("--model",       default=MODEL_NAME)
    p.add_argument("--api-key",     default=None)
    p.add_argument("--limit",       type=int,   default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Rate limiter (identical to groundlie)
# ---------------------------------------------------------------------------

class RPDLimitReached(Exception):
    pass


class RateLimiter:
    def __init__(self, rpm_target: int = 20, rpd_limit: int = 800):
        self.rpm_target    = rpm_target
        self.rpd_limit     = rpd_limit
        self.min_interval  = 60.0 / rpm_target
        self._window       = collections.deque()
        self._session_rpd  = 0
        self._last_call_ts = 0.0
        self._lock         = threading.Lock()

    def _purge(self):
        now = time.time()
        while self._window and now - self._window[0] > 60:
            self._window.popleft()

    def wait_and_record(self):
        with self._lock:
            if self._session_rpd >= self.rpd_limit:
                raise RPDLimitReached(
                    f"Session RPD limit reached: {self._session_rpd}/{self.rpd_limit}. "
                    "Resubmit the job tomorrow to continue."
                )
            elapsed = time.time() - self._last_call_ts
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self._purge()
            if len(self._window) >= self.rpm_target:
                wait_s = 60.0 - (time.time() - self._window[0]) + 1.0
                if wait_s > 0:
                    print(f"  [rate] RPM ceiling hit, waiting {wait_s:.0f}s ...", flush=True)
                    time.sleep(wait_s)
                    self._purge()
            now = time.time()
            self._window.append(now)
            self._last_call_ts = now
            self._session_rpd += 1

    def status(self) -> str:
        self._purge()
        return (f"RPM {len(self._window)}/{self.rpm_target} | "
                f"session RPD {self._session_rpd}/{self.rpd_limit}")


# ---------------------------------------------------------------------------
# GE2 API helpers (identical to groundlie)
# ---------------------------------------------------------------------------

def setup_client(api_key):
    if api_key is None:
        try:
            from dotenv import load_dotenv
            load_dotenv(REPO_ROOT / ".env")
        except ImportError:
            pass
        api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found (--api-key / env / .env).")
    from google import genai
    return genai.Client(api_key=api_key)


def _norm(arr):
    n = np.linalg.norm(arr)
    return arr / n if n > 1e-9 else arr


def embed_texts_batch(client, texts_with_ids, rl, model, batch_size=50):
    from google.genai import types
    saved = 0
    total_batches = (len(texts_with_ids) + batch_size - 1) // batch_size
    for b_idx, start in enumerate(range(0, len(texts_with_ids), batch_size)):
        batch = texts_with_ids[start : start + batch_size]
        contents = [types.Content(parts=[types.Part(text=t)]) for _, _, _, t in batch]
        for attempt in range(6):
            try:
                rl.wait_and_record()
                response = client.models.embed_content(
                    model=model, contents=contents,
                    config=types.EmbedContentConfig(output_dimensionality=OUTPUT_DIM),
                )
                break
            except RPDLimitReached:
                raise
            except Exception as exc:
                msg = str(exc).lower()
                if "429" in msg or "quota" in msg or "resource_exhausted" in msg:
                    wait_s = 60 * (2 ** attempt)
                    print(f"  [429] Backoff {wait_s}s (attempt {attempt+1}/6) ...", flush=True)
                    time.sleep(wait_s)
                else:
                    print(f"  [error] {exc}")
                    raise
        else:
            raise RuntimeError("Max retries exceeded on batch embed.")
        for (db_conn, raw_id, modality, _), emb_obj in zip(batch, response.embeddings):
            save_embedding(db_conn, raw_id, modality, _norm(np.array(emb_obj.values, dtype=np.float32)))
            saved += 1
        pct = 100 * (b_idx + 1) / total_batches
        print(f"  batch [{b_idx+1}/{total_batches}] ({pct:.0f}%) | saved {saved} | {rl.status()}", flush=True)
    return saved


def embed_video_single(client, video_path, model):
    from google.genai import types
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    for attempt in range(6):
        try:
            response = client.models.embed_content(
                model=model,
                contents=[types.Part.from_bytes(data=video_bytes, mime_type="video/mp4")],
                config=types.EmbedContentConfig(output_dimensionality=OUTPUT_DIM),
            )
            return _norm(np.array(response.embeddings[0].values, dtype=np.float32))
        except RPDLimitReached:
            raise
        except Exception as exc:
            msg = str(exc).lower()
            if "429" in msg or "quota" in msg or "resource_exhausted" in msg:
                wait_s = 60 * (2 ** attempt)
                print(f"  [429] Backoff {wait_s}s (attempt {attempt+1}/6) ...", flush=True)
                time.sleep(wait_s)
            else:
                raise
    raise RuntimeError("Max retries exceeded on video embed.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    phases = set(args.phases.upper().split(","))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== GE2 M3A pipeline ===")
    print(f"Model: {args.model} | Phases: {sorted(phases)} | "
          f"RPM {args.rpm_target} (gap {60/args.rpm_target:.1f}s) | RPD {args.rpd_limit} | batch {args.batch_size}")

    with open(args.index_json, encoding="utf-8") as f:
        index = json.load(f)
    if args.limit:
        index = index[: args.limit]
    print(f"Index: {len(index)} entries\n")

    db = init_db(str(output_dir / "m3a_ge2.db"))
    print(f"Cache: {cache_progress(db)}\n")

    # Phase 0: index metadata — no API
    if "0" in phases:
        print("=== Phase 0: Index metadata (no API) ===")
        for entry in index:
            rid = entry["raw_id"]
            if db.execute("SELECT 1 FROM m3a_meta WHERE raw_id=?", (rid,)).fetchone() is None:
                save_m3a_meta(db, rid, entry)
                save_m3a_nem(db, rid, entry.get("nem_texts", {}))
            if load_video_metadata(db, rid) is None:
                save_video_metadata(db, rid, entry.get("duration_seconds"),
                                    entry.get("has_audio", False), dataset_origin="m3a")
        print("Phase 0 done.\n")

    if phases & {"A", "N", "B", "C"}:
        client = setup_client(args.api_key)
        rl = RateLimiter(rpm_target=args.rpm_target, rpd_limit=args.rpd_limit)

    try:
        # Phase A: summary embeddings
        if "A" in phases:
            print("=== Phase A: Summary embeddings (text_summary) ===")
            pending = [
                (db, e["raw_id"], "text_summary", f"{TEXT_PREFIX}{e['summary']}")
                for e in index if e.get("summary") and load_embedding(db, e["raw_id"], "text_summary") is None
            ]
            print(f"Pending: {len(pending)}")
            if pending:
                print(f"Phase A done: {embed_texts_batch(client, pending, rl, args.model, args.batch_size)} saved.\n")

        # Phase N: NEM (+ MTG) fake-text embeddings
        if "N" in phases:
            print("=== Phase N: NEM fake-text embeddings ===")
            subtypes = list(NEM_SUBTYPES) + (["mtg"] if args.with_mtg else [])
            for sub in subtypes:
                mod_name = f"text_{'mtg' if sub == 'mtg' else 'nem_' + sub}"
                pending = []
                for e in index:
                    if load_embedding(db, e["raw_id"], mod_name) is not None:
                        continue
                    txt = e.get("mtg_text") if sub == "mtg" else e.get("nem_texts", {}).get(sub)
                    if txt:
                        pending.append((db, e["raw_id"], mod_name, f"{TEXT_PREFIX}{txt}"))
                print(f"  {mod_name}: pending {len(pending)}")
                if pending:
                    embed_texts_batch(client, pending, rl, args.model, args.batch_size)
            print("Phase N done.\n")

        # Phase B: transcript embeddings (from Qwen DB)
        if "B" in phases:
            print("=== Phase B: Transcript embeddings (from Qwen DB) ===")
            if not args.qwen_db or not Path(args.qwen_db).exists():
                print("[SKIP] --qwen-db not provided or not found.\n")
            else:
                qconn = sqlite3.connect(args.qwen_db)
                pending = []
                for e in index:
                    rid = e["raw_id"]
                    if load_embedding(db, rid, "transcript") is not None:
                        continue
                    row = qconn.execute("SELECT text FROM transcripts WHERE raw_id=?", (rid,)).fetchone()
                    if row and row[0]:
                        pending.append((db, rid, "transcript", f"{DOC_PREFIX}{row[0]}"))
                qconn.close()
                print(f"Pending: {len(pending)}")
                if pending:
                    print(f"Phase B done: {embed_texts_batch(client, pending, rl, args.model, args.batch_size)} saved.\n")

        # Phase C: video embeddings (one-by-one)
        if "C" in phases:
            print("=== Phase C: Video embeddings (one-by-one) ===")
            if not args.data_dir:
                print("[SKIP] --data-dir not provided.\n")
            else:
                data_dir = Path(args.data_dir)
                pending_v = [e for e in index if load_embedding(db, e["raw_id"], "video") is None]
                print(f"Pending: {len(pending_v)}")
                v_errors = 0
                for i, entry in enumerate(pending_v):
                    rid   = entry["raw_id"]
                    vpath = str((data_dir / entry["video_path"]).resolve())
                    dur   = entry.get("duration_seconds", 0) or 0
                    if not os.path.isfile(vpath):
                        print(f"  [MISSING] {vpath}")
                        continue
                    if dur > 120:
                        print(f"  [SKIP] {rid} duration {dur:.1f}s > 120s")
                        continue
                    try:
                        rl.wait_and_record()
                        emb = embed_video_single(client, vpath, args.model)
                        save_embedding(db, rid, "video", emb)
                    except RPDLimitReached:
                        raise
                    except Exception as exc:
                        v_errors += 1
                        print(f"  [error] {rid}: {exc}", flush=True)
                        continue
                    if (i + 1) % 10 == 0 or (i + 1) == len(pending_v):
                        print(f"  [{i+1}/{len(pending_v)}] videos done (errors: {v_errors}) | {rl.status()}", flush=True)
                print(f"Phase C done. Errors: {v_errors}\n")

    except RPDLimitReached as e:
        print(f"\n[STOP] {e}")
        print("Resubmit the job tomorrow — the script will resume automatically.")

    print(f"\nFinal cache: {cache_progress(db)}")
    print("Done.")


if __name__ == "__main__":
    main()
