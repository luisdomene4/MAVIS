"""
Resumen de las 4 DBs de GroundLie360 + análisis preliminar E1:
¿cos(video, title) es mayor para vídeos REALES que para FAKES?

Uso:
    python scripts/inspect_groundlie_dbs.py
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

DBS = {
    "Qwen3VL-2B": ROOT / "experiments/GroundLie360/open_source/results/qwen3vl_2b/qwen3vl_cache.db",
    "Qwen3VL-8B": ROOT / "experiments/GroundLie360/open_source/results/qwen3vl_8b/qwen3vl_cache.db",
    "WAVE-7B":    ROOT / "experiments/GroundLie360/open_source/results/WAVE7B/wave_cache.db",
    "GE2":        ROOT / "experiments/GroundLie360/google_embeddings2/results/groundlie_ge2.db",
}

EXPECTED_TOTAL = 1722  # ≤120s filter


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def load_emb(conn, raw_id, modality):
    row = conn.execute(
        "SELECT vector FROM embeddings WHERE raw_id=? AND modality=?",
        (raw_id, modality),
    ).fetchone()
    return np.frombuffer(row[0], dtype=np.float32) if row else None


def summarize(name, db_path):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  {db_path.relative_to(ROOT)}")
    print(f"{'='*60}")
    if not db_path.exists():
        print("  [MISSING]")
        return None

    conn = sqlite3.connect(db_path)

    # tablas
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()]
    print(f"  Tablas: {', '.join(tables)}")

    # embeddings por modalidad
    print("  Embeddings:")
    for mod, n in conn.execute(
        "SELECT modality, COUNT(*) FROM embeddings GROUP BY modality ORDER BY modality").fetchall():
        pct = 100 * n / EXPECTED_TOTAL
        print(f"    {mod:<14} {n:>5}/{EXPECTED_TOTAL}  ({pct:5.1f}%)")

    # transcripts
    n_tr = conn.execute("SELECT COUNT(*) FROM transcripts").fetchone()[0]
    n_real = conn.execute(
        "SELECT COUNT(*) FROM transcripts WHERE model != 'NO_AUDIO' AND text != ''"
    ).fetchone()[0]
    n_noaudio = conn.execute(
        "SELECT COUNT(*) FROM transcripts WHERE model = 'NO_AUDIO'"
    ).fetchone()[0]
    print(f"  Transcripts: {n_tr} filas — {n_real} con texto, {n_noaudio} marcadas NO_AUDIO")

    # labels
    n_lbl = conn.execute("SELECT COUNT(*) FROM groundlie_labels").fetchone()[0]
    n_real_v = conn.execute(
        "SELECT COUNT(*) FROM groundlie_labels WHERE binary_label = 0").fetchone()[0]
    n_fake_v = conn.execute(
        "SELECT COUNT(*) FROM groundlie_labels WHERE binary_label = 1").fetchone()[0]
    print(f"  Labels: {n_lbl} ({n_real_v} real, {n_fake_v} fake)")

    # scene & bbox
    n_sc = conn.execute("SELECT COUNT(DISTINCT raw_id) FROM scene_metadata").fetchone()[0]
    n_bb = conn.execute("SELECT COUNT(DISTINCT raw_id) FROM groundlie_bboxes").fetchone()[0]
    print(f"  Scenes (vídeos distintos): {n_sc}")
    print(f"  Bboxes  (vídeos distintos): {n_bb}")

    conn.close()
    return db_path


def e1_analysis(name, db_path):
    """E1: cos(video, text_title) — agrupar por etiqueta fina, no solo binary."""
    print(f"\n--- E1: cos(video, text_title) — {name} ---")
    if not db_path or not db_path.exists():
        print("  [SKIP]")
        return

    conn = sqlite3.connect(db_path)
    rows = conn.execute("""
        SELECT l.raw_id, l.binary_label, l.false_title, l.false_speech, l.temporal_edit, l.cgi
        FROM groundlie_labels l
        WHERE EXISTS (SELECT 1 FROM embeddings e WHERE e.raw_id=l.raw_id AND e.modality='video')
          AND EXISTS (SELECT 1 FROM embeddings e WHERE e.raw_id=l.raw_id AND e.modality='text_title')
    """).fetchall()

    if not rows:
        print("  [no hay videos con video+title embeddings]")
        conn.close()
        return

    records = []
    for rid, lbl, ft, fs, te, cg in rows:
        v = load_emb(conn, rid, "video")
        t = load_emb(conn, rid, "text_title")
        records.append({
            "rid": rid, "label": lbl, "cos": cosine(v, t),
            "false_title": ft, "false_speech": fs, "temporal_edit": te, "cgi": cg,
        })
    conn.close()

    # Agrupaciones
    def stats(group):
        sims = np.array([r["cos"] for r in group])
        if len(sims) == 0:
            return None
        return f"N={len(sims):4d}  mean={sims.mean():.4f}  med={np.median(sims):.4f}  std={sims.std():.4f}"

    real     = [r for r in records if r["label"] == 0]
    fake     = [r for r in records if r["label"] == 1]
    ft_yes   = [r for r in records if r["false_title"] == 1]
    ft_no    = [r for r in records if r["false_title"] == 0]
    fake_ftyes = [r for r in fake if r["false_title"] == 1]  # fake con título engañoso
    fake_ftno  = [r for r in fake if r["false_title"] == 0]  # fake en otro aspecto

    print(f"  REAL              : {stats(real)}")
    print(f"  FAKE (total)      : {stats(fake)}")
    print(f"   |- false_title=1 : {stats(fake_ftyes)}   <- titulo enganoso")
    print(f"   |- false_title=0 : {stats(fake_ftno)}    <- titulo OK, mentira en otro lado")
    print(f"  false_title=1 (todos): {stats(ft_yes)}")
    print(f"  false_title=0 (todos): {stats(ft_no)}")
    if real and fake_ftyes:
        diff = np.mean([r["cos"] for r in real]) - np.mean([r["cos"] for r in fake_ftyes])
        print(f"  Diff (REAL - FAKE_falseTitle): {diff:+.4f}  --> {'OK real > fake_ft (esperado)' if diff > 0 else 'WARN'}")

    # Top/bot 10
    records.sort(key=lambda x: x["cos"], reverse=True)
    real_in_top10 = sum(1 for r in records[:10] if r["label"] == 0)
    real_in_bot10 = sum(1 for r in records[-10:] if r["label"] == 0)
    print(f"  Top 10 mas alineados:   {real_in_top10}/10 son REAL  (esperado: >5)")
    print(f"  Bot 10 menos alineados: {real_in_bot10}/10 son REAL  (esperado: <5)")


def main():
    print(f"\nGroundLie360 — Resumen de DBs ({EXPECTED_TOTAL} vídeos esperados)")
    print(f"Ruta base: {ROOT}")

    valid = {}
    for name, path in DBS.items():
        result = summarize(name, path)
        if result:
            valid[name] = result

    print(f"\n\n{'#'*60}")
    print("#  Análisis preliminar E1: cos(video, text_title)")
    print(f"{'#'*60}")
    print("Hipotesis: para videos REALES la alineacion video-titulo debe ser mayor.")

    for name, path in valid.items():
        e1_analysis(name, path)

    print(f"\n{'='*60}\nFin del análisis.\n")


if __name__ == "__main__":
    main()
