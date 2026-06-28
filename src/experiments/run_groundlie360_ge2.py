"""
MAVIS — Google Embeddings 2 pipeline for GroundLie360 (cluster execution).

Designed for unattended cluster runs with a very conservative rate limiter:
  - Target 20 RPM (well under the 100 RPM free-tier limit)
  - Stop after 800 RPD per session (well under the 1000 RPD free-tier limit)
  - 3-second enforced gap between every API call (batch or individual)
  - Commits to SQLite after every successful API call — zero data loss on crash
  - Resumes automatically from where it stopped (per-item existence check)
  - Exits cleanly when session RPD limit is reached; resubmit next day to continue

Phases (all resumable):
  0  — groundlie_labels + scene_metadata + groundlie_bboxes from index (no API)
  A  — text_title batch embeddings (50 texts per API call)
  B  — transcript batch embeddings (text from --qwen-db, 50 per API call)
  C  — video embeddings one-by-one (uploads MP4 bytes to GE2 API)

Usage (from repo root on cluster):
    python src/experiments/run_groundlie360_ge2.py \
        --data-dir   data/GroundLie360 \
        --output-dir experiments/GroundLie360/google_embeddings2/results \
        [--qwen-db   experiments/GroundLie360/open_source/results/qwen3vl_2b/qwen3vl_cache.db] \
        [--phases    0,A,B,C] \
        [--rpm-target 20] [--rpd-limit 800]

Outputs:
    <output-dir>/groundlie_ge2.db   -- resumable SQLite cache (8 tables)

Requirements:
    pip install google-generativeai python-dotenv
    GEMINI_API_KEY env var or .env file in repo root.
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
    save_scene_metadata, load_scene_metadata,
    save_groundlie_labels, save_groundlie_bboxes,
)

DEFAULT_INDEX  = str(REPO_ROOT / "experiments/GroundLie360/groundlie_index_filtered.json")
DEFAULT_BBOX   = str(REPO_ROOT / "experiments/GroundLie360/bbox_index.json")
DEFAULT_OUTPUT = str(REPO_ROOT / "experiments/GroundLie360/google_embeddings2/results")
MODEL_NAME     = "gemini-embedding-2-preview"
OUTPUT_DIM     = 3072
TEXT_PREFIX    = "task: fact checking | query: "
DOC_PREFIX     = "task: fact checking | document: "


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="GE2 embedding pipeline for GroundLie360 (cluster)")
    p.add_argument("--data-dir",    default=None,
                   help="Dir containing vid_groundlie/ (required for Phase C)")
    p.add_argument("--output-dir",  default=DEFAULT_OUTPUT)
    p.add_argument("--index-json",  default=DEFAULT_INDEX)
    p.add_argument("--bbox-json",   default=DEFAULT_BBOX)
    p.add_argument("--qwen-db",     default=None,
                   help="Path to qwen3vl_cache.db to read transcript text (Phase B)")
    p.add_argument("--phases",      default="0,A,B,C",
                   help="Comma-separated phases to run: 0,A,B,C (default all)")
    p.add_argument("--rpm-target",  type=float, default=20,
                   help="Target RPM — enforces 60/RPM seconds between requests (default 20)")
    p.add_argument("--rpd-limit",   type=int,   default=800,
                   help="Max API calls per session; script exits cleanly when reached (default 800)")
    p.add_argument("--batch-size",  type=int,   default=50,
                   help="Texts per batch API call (default 50, max 100)")
    p.add_argument("--model",       default=MODEL_NAME)
    p.add_argument("--api-key",     default=None,
                   help="Gemini API key (default: GEMINI_API_KEY env var or .env file)")
    p.add_argument("--limit",       type=int,   default=None,
                   help="Process only first N index entries (smoke test)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Rate limiter — conservative, cluster-safe
# ---------------------------------------------------------------------------

class RPDLimitReached(Exception):
    pass


class RateLimiter:
    """
    Enforces a minimum gap of 60/rpm_target seconds between every API call.
    Also tracks a per-session RPD counter and raises RPDLimitReached when
    the session budget is exhausted — the script then exits cleanly and the
    user resubmits the next day.

    Why 60/rpm_target and not just a sliding window?
    The sliding window allows bursts (e.g., 20 calls in 1 second then idle).
    The enforced minimum gap prevents any burst, which is safer for unattended
    cluster runs where we can't intervene on a 429.
    """

    def __init__(self, rpm_target: int = 20, rpd_limit: int = 800):
        self.rpm_target    = rpm_target
        self.rpd_limit     = rpd_limit
        self.min_interval  = 60.0 / rpm_target   # seconds between calls
        self._window       = collections.deque()  # timestamps in last 60s
        self._session_rpd  = 0
        self._last_call_ts = 0.0
        self._lock         = threading.Lock()

    def _purge(self):
        now = time.time()
        while self._window and now - self._window[0] > 60:
            self._window.popleft()

    def wait_and_record(self):
        """Block until safe to make an API call, then record it."""
        with self._lock:
            if self._session_rpd >= self.rpd_limit:
                raise RPDLimitReached(
                    f"Session RPD limit reached: {self._session_rpd}/{self.rpd_limit}. "
                    "Resubmit the job tomorrow to continue."
                )

            # Enforce minimum gap since last call
            elapsed = time.time() - self._last_call_ts
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)

            # Sliding-window RPM check (secondary guard)
            self._purge()
            if len(self._window) >= self.rpm_target:
                wait_s = 60.0 - (time.time() - self._window[0]) + 1.0
                if wait_s > 0:
                    print(f"  [rate] RPM ceiling hit ({len(self._window)}/{self.rpm_target}), "
                          f"waiting {wait_s:.0f}s ...", flush=True)
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
# GE2 API helpers
# ---------------------------------------------------------------------------

def setup_client(api_key: str | None):
    # Load from .env if not passed explicitly
    if api_key is None:
        try:
            from dotenv import load_dotenv
            load_dotenv(REPO_ROOT / ".env")
        except ImportError:
            pass
        api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not found. Set it via --api-key, the GEMINI_API_KEY "
            "environment variable, or a .env file in the repo root."
        )

    try:
        from google import genai
    except ImportError:
        raise ImportError(
            "google-generativeai not installed. Run:\n"
            "  pip install google-generativeai"
        )

    client = genai.Client(api_key=api_key)
    return client


def _norm(arr: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(arr)
    return arr / n if n > 1e-9 else arr


def embed_texts_batch(client, texts_with_ids, rl: RateLimiter, model: str,
                      batch_size: int = 50) -> int:
    """
    Batch-embed a list of (raw_id, modality, prefixed_text) tuples.
    Each batch = 1 API call = 1 session RPD.
    Saves to DB immediately after each batch (crash-safe).
    Returns total embeddings saved.
    """
    from google.genai import types

    # Only process items not yet in DB (passed in already filtered, but double-check)
    saved = 0
    total_batches = (len(texts_with_ids) + batch_size - 1) // batch_size

    for b_idx, start in enumerate(range(0, len(texts_with_ids), batch_size)):
        batch = texts_with_ids[start : start + batch_size]
        contents = [
            types.Content(parts=[types.Part(text=full_text)])
            for _, _, _, full_text in batch
        ]

        for attempt in range(6):
            try:
                rl.wait_and_record()
                response = client.models.embed_content(
                    model=model,
                    contents=contents,
                    config=types.EmbedContentConfig(output_dimensionality=OUTPUT_DIM),
                )
                break
            except RPDLimitReached:
                raise  # propagate immediately
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
            emb = _norm(np.array(emb_obj.values, dtype=np.float32))
            save_embedding(db_conn, raw_id, modality, emb)
            saved += 1

        pct = 100 * (b_idx + 1) / total_batches
        print(f"  batch [{b_idx+1}/{total_batches}] ({pct:.0f}%) | saved {saved} | {rl.status()}",
              flush=True)

    return saved


def embed_video_single(client, video_path: str, model: str) -> np.ndarray:
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
# GroundLie360 helpers (same as other pipeline scripts)
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
    phases = set(args.phases.upper().split(","))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== GE2 GroundLie360 pipeline ===")
    print(f"Model:      {args.model}")
    print(f"Phases:     {sorted(phases)}")
    print(f"RPM target: {args.rpm_target}  (min gap: {60/args.rpm_target:.1f}s between calls)")
    print(f"RPD limit:  {args.rpd_limit} per session")
    print(f"Batch size: {args.batch_size}")

    # Load index
    with open(args.index_json, encoding="utf-8") as f:
        index = json.load(f)
    with open(args.bbox_json, encoding="utf-8") as f:
        bbox_index = json.load(f)

    if args.limit:
        index = index[: args.limit]
    print(f"Index: {len(index)} entries\n")

    # Init DB
    db = init_db(str(output_dir / "groundlie_ge2.db"))
    print(f"Cache: {cache_progress(db)}\n")

    # -----------------------------------------------------------------------
    # Phase 0: index metadata — no API calls
    # -----------------------------------------------------------------------
    if "0" in phases:
        print("=== Phase 0: Index metadata (no API) ===")
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
        print("Phase 0 done.\n")

    # Setup API client (needed for phases A, B, C)
    if phases & {"A", "B", "C"}:
        client = setup_client(args.api_key)
        rl = RateLimiter(rpm_target=args.rpm_target, rpd_limit=args.rpd_limit)

    try:
        # -------------------------------------------------------------------
        # Phase A: title text embeddings
        # -------------------------------------------------------------------
        if "A" in phases:
            print("=== Phase A: Title embeddings (text_title) ===")
            pending = [
                (db, e["raw_id"], "text_title", f"{TEXT_PREFIX}{e['title']}")
                for e in index
                if load_embedding(db, e["raw_id"], "text_title") is None
            ]
            print(f"Pending: {len(pending)}")
            if pending:
                saved = embed_texts_batch(client, pending, rl, args.model, args.batch_size)
                print(f"Phase A done: {saved} saved.\n")
            else:
                print("Phase A: nothing pending.\n")

        # -------------------------------------------------------------------
        # Phase B: transcript embeddings (from Qwen DB)
        # -------------------------------------------------------------------
        if "B" in phases:
            print("=== Phase B: Transcript embeddings (from Qwen DB) ===")
            if not args.qwen_db or not Path(args.qwen_db).exists():
                print("[SKIP] --qwen-db not provided or not found. Run Phase A of Qwen first.\n")
            else:
                qconn = sqlite3.connect(args.qwen_db)
                pending = []
                for e in index:
                    rid = e["raw_id"]
                    if load_embedding(db, rid, "transcript") is not None:
                        continue
                    row = qconn.execute(
                        "SELECT text FROM transcripts WHERE raw_id=?", (rid,)
                    ).fetchone()
                    if row and row[0]:
                        pending.append((db, rid, "transcript", f"{DOC_PREFIX}{row[0]}"))
                qconn.close()
                print(f"Pending: {len(pending)}")
                if pending:
                    saved = embed_texts_batch(client, pending, rl, args.model, args.batch_size)
                    print(f"Phase B done: {saved} saved.\n")
                else:
                    print("Phase B: nothing pending.\n")

        # -------------------------------------------------------------------
        # Phase C: video embeddings (one-by-one)
        # -------------------------------------------------------------------
        if "C" in phases:
            print("=== Phase C: Video embeddings (one-by-one) ===")
            if not args.data_dir:
                print("[SKIP] --data-dir not provided. Required for video upload.\n")
            else:
                data_dir = Path(args.data_dir)
                pending_v = [
                    e for e in index
                    if load_embedding(db, e["raw_id"], "video") is None
                ]
                print(f"Pending: {len(pending_v)}")
                v_errors = 0
                for i, entry in enumerate(pending_v):
                    rid    = entry["raw_id"]
                    vpath  = str((data_dir / entry["video_path"]).resolve())
                    dur    = entry.get("duration_seconds", 0) or 0

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
                        print(f"  [{i+1}/{len(pending_v)}] videos done "
                              f"(errors: {v_errors}) | {rl.status()}", flush=True)

                print(f"Phase C done. Errors: {v_errors}\n")

    except RPDLimitReached as e:
        print(f"\n[STOP] {e}")
        print("Resubmit the job tomorrow — the script will resume automatically.")

    print(f"\nFinal cache: {cache_progress(db)}")
    print("Done.")


if __name__ == "__main__":
    main()
