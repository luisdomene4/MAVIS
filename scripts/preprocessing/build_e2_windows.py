"""
E2 (Phase 2b) — false_speech TEXTUAL windows.

E2 is a *text-only* experiment. For every video with false_speech=1 AND
false_title=0 (N=20 — the only ones whose title is a trustworthy anchor), we
split the Whisper transcript by the dataset-annotated false_speech span(s):

    T_fake  = words inside  any false_speech span
    T_clean = words outside every false_speech span

and embed both with each model's *text* encoder, exactly as that model embedded
its global transcript. The paired analysis signal cos(T_clean, title) −
cos(T_fake, title) is computed later in the notebook.

Word-level timing comes from `transcript_words`, which only the Qwen pipelines
populate (they run Whisper). So this script reads the canonical word timings
from a Qwen DB (--qwen-db, the qwen3vl_2b cache by default) and embeds the same
two strings with whichever --model is selected — mirroring how WAVE/GE2 already
reuse Qwen's transcript text for a fair comparison.

Writes ONLY to `segment_embeddings` (segment_type 'window_fake' / 'window_clean',
modality 'transcript'). Never touches the `embeddings` table. Resumable via
INSERT OR IGNORE + per-row existence check.

Usage (from repo root on cluster):
    # Qwen-2B (env tfg2)
    python scripts/preprocessing/build_e2_windows.py --model qwen2b \
        --model-dir src/models/Qwen3-VL-Embedding-2B \
        --qwen-repo src/models/qwen3vl_embedding_repo --quantize
    # WAVE-7B (env tfg-wave)
    python scripts/preprocessing/build_e2_windows.py --model wave \
        --model-dir src/models/WAVE-7B --wave-repo src/models/wave_repo \
        --beats-path src/models/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt --quantize
    # GE2 (env tfg)
    python scripts/preprocessing/build_e2_windows.py --model ge2
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
for p in (REPO_ROOT / "src", REPO_ROOT / "src" / "experiments"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from utils.db_schema import (
    init_db, load_transcript_words, save_segment_embedding, load_segment_embeddings,
)

DEFAULT_INDEX = REPO_ROOT / "experiments/GroundLie360/groundlie_index_filtered.json"
DEFAULT_QWEN_DB = REPO_ROOT / "experiments/GroundLie360/open_source/results/qwen3vl_2b/qwen3vl_cache.db"

MODEL_DBS = {
    "qwen2b": "experiments/GroundLie360/open_source/results/qwen3vl_2b/qwen3vl_cache.db",
    "qwen8b": "experiments/GroundLie360/open_source/results/qwen3vl_8b/qwen3vl_cache.db",
    "wave":   "experiments/GroundLie360/open_source/results/WAVE7B/wave_cache.db",
    "ge2":    "experiments/GroundLie360/google_embeddings2/results/groundlie_ge2.db",
}

MODALITY = "transcript"


def parse_args():
    p = argparse.ArgumentParser(description="Build E2 false_speech text windows")
    p.add_argument("--model", required=True, choices=list(MODEL_DBS))
    p.add_argument("--index-json", default=str(DEFAULT_INDEX))
    p.add_argument("--qwen-db", default=str(DEFAULT_QWEN_DB),
                   help="DB to read transcript_words from (canonical = qwen3vl_2b)")
    p.add_argument("--output-db", default=None,
                   help="Target DB (default: the selected model's own cache)")
    # Qwen
    p.add_argument("--model-dir", default=None)
    p.add_argument("--qwen-repo", default=None)
    # WAVE
    p.add_argument("--wave-repo", default=None)
    p.add_argument("--beats-path", default=None)
    p.add_argument("--quantize", action="store_true")
    p.add_argument("--no-instruction", action="store_true")
    # GE2
    p.add_argument("--api-key", default=None)
    p.add_argument("--rpm-target", type=float, default=3)
    p.add_argument("--rpd-limit", type=int, default=950)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Transcript span splitting
# ---------------------------------------------------------------------------

def split_by_spans(words: list, spans: list):
    """Return (fake_text, clean_text). A word is 'fake' if its midpoint falls
    inside any false_speech span; words without timestamps default to clean."""
    fake, clean = [], []
    for w in words:
        s, e = w.get("start_s"), w.get("end_s")
        token = w.get("word", "")
        if s is None or e is None:
            clean.append(token)
            continue
        mid = 0.5 * (s + e)
        inside = any(sp["start_s"] <= mid <= sp["end_s"] for sp in spans)
        (fake if inside else clean).append(token)
    return "".join(fake).strip(), "".join(clean).strip()


# ---------------------------------------------------------------------------
# Per-model text embedders — each reuses its global-transcript convention
# ---------------------------------------------------------------------------

def make_embedder(args):
    """Return embed(text) -> np.float32 vector, using the same prefix/instruction
    each model used for its global transcript embeddings."""
    model = args.model

    if model in ("qwen2b", "qwen8b"):
        import run_groundlie360_qwen3vl as q
        if not args.model_dir or not args.qwen_repo:
            sys.exit("qwen models require --model-dir and --qwen-repo")
        instruction = None if args.no_instruction else q.DEFAULT_INSTRUCTION
        m = q.load_model(args.model_dir, args.qwen_repo, quantize=args.quantize)
        return lambda text: q.get_text_embedding(text, m, instruction)

    if model == "wave":
        import run_groundlie360_wave as w
        if not args.model_dir or not args.wave_repo or not args.beats_path:
            sys.exit("wave requires --model-dir, --wave-repo and --beats-path")
        m, proc = w.load_wave_model(args.model_dir, args.wave_repo, args.beats_path, args.quantize)
        return lambda text: w.get_text_embedding(text, m, proc, prefix=w.TEXT_QUERY_PREFIX)

    # GE2
    import run_groundlie360_ge2 as g
    client = g.setup_client(args.api_key)
    rl = g.RateLimiter(rpm_target=args.rpm_target, rpd_limit=args.rpd_limit)
    from google.genai import types

    def embed_ge2(text):
        rl.wait_and_record()
        resp = client.models.embed_content(
            model=g.MODEL_NAME,
            contents=[types.Content(parts=[types.Part(text=f"{g.DOC_PREFIX}{text}")])],
            config=types.EmbedContentConfig(output_dimensionality=g.OUTPUT_DIM),
        )
        return g._norm(np.array(resp.embeddings[0].values, dtype=np.float32))

    return embed_ge2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    out_db = args.output_db or str(REPO_ROOT / MODEL_DBS[args.model])

    with open(args.index_json, encoding="utf-8") as f:
        index = json.load(f)
    targets = [e for e in index if e.get("false_speech") == 1 and e.get("false_title") == 0]
    print(f"=== E2 windows | model={args.model} ===")
    print(f"Target videos (false_speech=1 AND false_title=0): {len(targets)}")

    qconn = sqlite3.connect(args.qwen_db)
    db = init_db(out_db)
    print(f"Output DB: {out_db}")

    embed = make_embedder(args)

    done, skipped, errors = 0, 0, 0
    for i, entry in enumerate(targets):
        rid = entry["raw_id"]
        spans = entry.get("false_speech_spans") or []
        words = load_transcript_words(qconn, rid)
        if not words:
            print(f"  [{rid}] no transcript_words in qwen-db — skip")
            skipped += 1
            continue
        if not spans:
            print(f"  [{rid}] no false_speech_spans — skip")
            skipped += 1
            continue

        fake_text, clean_text = split_by_spans(words, spans)
        span_start = min(sp["start_s"] for sp in spans)
        span_end = max(sp["end_s"] for sp in spans)

        for seg_type, text, s0, s1 in (
            ("window_fake",  fake_text,  span_start, span_end),
            ("window_clean", clean_text, None,       None),
        ):
            if not text:
                print(f"  [{rid}] {seg_type} empty text — skip")
                skipped += 1
                continue
            # resumable: skip if already present
            if load_segment_embeddings(db, rid, seg_type, MODALITY):
                continue
            try:
                vec = embed(text)
                save_segment_embedding(
                    db, rid, seg_type, 0, s0, s1, MODALITY, vec,
                    extra_json={"text": text, "spans": spans},
                )
                done += 1
            except Exception as exc:
                errors += 1
                print(f"  [{rid}] {seg_type} ERROR: {exc}", flush=True)

        if (i + 1) % 5 == 0 or (i + 1) == len(targets):
            print(f"  [{i+1}/{len(targets)}] saved={done} skipped={skipped} errors={errors}", flush=True)

    qconn.close()
    print(f"\nDone. saved={done} skipped={skipped} errors={errors}")


if __name__ == "__main__":
    main()