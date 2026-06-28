"""
MAVIS — Comparative Analysis: Google Embeddings 2 vs Qwen3-VL 2B vs Qwen3-VL 8B

Loads SQLite databases, finds common videos (≤120s, complete embeddings),
runs E1/E2/E4a experiments on all three models, and generates comparison
tables + figures suitable for the TFG report.

Usage:
    python src/evaluation/compare_ge2_vs_qwen3vl.py \
        --ge2-db    experiments/FakeVV_testset/google_embeddings2/results/test_set_fakevvGE2.db \
        --qwen2b-db experiments/FakeVV_testset/open_source/results/qwen3vl_2b/qwen3vl_cache.db \
        --qwen8b-db experiments/FakeVV_testset/open_source/results/qwen3vl_8b/qwen3vl_cache.db \
        --index     experiments/FakeVV_testset/google_embeddings2/test_index.json \
        --output    experiments/FakeVV_testset/results \
        --max-duration 120

Requires: numpy, pandas, sklearn, matplotlib  (conda env: tfg)
"""

import argparse
import json
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Global palette — consistent colors and display names across ALL figures
# ---------------------------------------------------------------------------

MODEL_KEYS = ["ge2", "qw2b", "qw8b"]

MODEL_DISPLAY = {
    "ge2":  "GE2",
    "qw2b": "Qwen3-VL 2B",
    "qw8b": "Qwen3-VL 8B",
}

MODEL_COLOR = {
    "ge2":  "#4472C4",  # blue
    "qw2b": "#70AD47",  # green
    "qw8b": "#ED7D31",  # orange
}

# Shared rc-param overrides applied via plt.rc_context in every figure
_RC = {
    "font.family":        "sans-serif",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.18,
    "grid.linestyle":     "--",
    "axes.titlesize":     13,
    "axes.labelsize":     11,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    8,
    "figure.dpi":         150,
}

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def load_embedding(conn, raw_id, modality):
    if conn is None:
        return None
    row = conn.execute(
        "SELECT vector FROM embeddings WHERE raw_id=? AND modality=?",
        (raw_id, modality),
    ).fetchone()
    if row is None:
        return None
    return np.frombuffer(row[0], dtype=np.float32)


def get_modality_counts(conn):
    if conn is None:
        return {}
    rows = conn.execute(
        "SELECT modality, COUNT(*) FROM embeddings GROUP BY modality"
    ).fetchall()
    return {mod: cnt for mod, cnt in rows}


def get_video_duration_from_qwen_db(conn, raw_id):
    if conn is None:
        return None
    try:
        row = conn.execute(
            "SELECT duration_seconds FROM video_metadata WHERE raw_id=?",
            (raw_id,),
        ).fetchone()
        return row[0] if row else None
    except Exception:
        return None


def cos_sim(a, b):
    return float(cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0][0])

# ---------------------------------------------------------------------------
# Find common valid videos
# ---------------------------------------------------------------------------

def find_common_videos(test_index, ge2_conn, qw2b_conn, qw8b_conn, max_duration=120.0):
    """Return list of items present in ALL provided DBs with duration ≤ max_duration."""
    common = []
    skipped_missing = 0
    skipped_duration = 0

    for item in test_index:
        rid = item["raw_id"]

        ge2_v = load_embedding(ge2_conn, rid, "video")
        ge2_r = load_embedding(ge2_conn, rid, "text_real")
        ge2_f = load_embedding(ge2_conn, rid, "text_fake")
        if ge2_v is None or ge2_r is None or ge2_f is None:
            skipped_missing += 1
            continue

        if qw2b_conn:
            qw2b_v = load_embedding(qw2b_conn, rid, "video")
            qw2b_r = load_embedding(qw2b_conn, rid, "text_real")
            qw2b_f = load_embedding(qw2b_conn, rid, "text_fake")
            if qw2b_v is None or qw2b_r is None or qw2b_f is None:
                skipped_missing += 1
                continue
        else:
            qw2b_v, qw2b_r, qw2b_f = None, None, None

        if qw8b_conn:
            qw8b_v = load_embedding(qw8b_conn, rid, "video")
            qw8b_r = load_embedding(qw8b_conn, rid, "text_real")
            qw8b_f = load_embedding(qw8b_conn, rid, "text_fake")
            if qw8b_v is None or qw8b_r is None or qw8b_f is None:
                skipped_missing += 1
                continue
        else:
            qw8b_v, qw8b_r, qw8b_f = None, None, None

        dur = get_video_duration_from_qwen_db(qw8b_conn or qw2b_conn, rid)
        if dur is not None and dur > max_duration:
            skipped_duration += 1
            continue

        ge2_t  = load_embedding(ge2_conn,  rid, "transcript")
        qw2b_t = load_embedding(qw2b_conn, rid, "transcript") if qw2b_conn else None
        qw8b_t = load_embedding(qw8b_conn, rid, "transcript") if qw8b_conn else None

        common.append({
            **item,
            "ge2":  {"video": ge2_v,  "text_real": ge2_r,  "text_fake": ge2_f,  "transcript": ge2_t},
            "qw2b": {"video": qw2b_v, "text_real": qw2b_r, "text_fake": qw2b_f, "transcript": qw2b_t},
            "qw8b": {"video": qw8b_v, "text_real": qw8b_r, "text_fake": qw8b_f, "transcript": qw8b_t},
            "duration":           dur,
            "has_transcript_ge2":  ge2_t  is not None,
            "has_transcript_qw2b": qw2b_t is not None,
            "has_transcript_qw8b": qw8b_t is not None,
        })

    print(f"Common videos: {len(common)}  (skipped: {skipped_missing} missing, {skipped_duration} over {max_duration}s)")
    return common

# ---------------------------------------------------------------------------
# E1: Video ↔ Title
# ---------------------------------------------------------------------------

def run_e1_comparison(common_videos):
    rows = []
    for item in common_videos:
        rid, cat = item["raw_id"], item["category"]
        r = {"raw_id": rid, "category": cat}
        for pfx in MODEL_KEYS:
            d = item[pfx]
            if d["video"] is None:
                continue
            r[f"{pfx}_sim_real"] = cos_sim(d["video"], d["text_real"])
            r[f"{pfx}_sim_fake"] = cos_sim(d["video"], d["text_fake"])
            r[f"{pfx}_margin"]   = r[f"{pfx}_sim_real"] - r[f"{pfx}_sim_fake"]
            r[f"{pfx}_correct"]  = r[f"{pfx}_sim_real"] > r[f"{pfx}_sim_fake"]
        rows.append(r)
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# E2: Transcript ↔ Title
# ---------------------------------------------------------------------------

def run_e2_comparison(common_videos):
    rows = []
    for item in common_videos:
        valid = item["has_transcript_ge2"]
        if item["qw2b"]["video"] is not None:
            valid = valid and item["has_transcript_qw2b"]
        if item["qw8b"]["video"] is not None:
            valid = valid and item["has_transcript_qw8b"]
        if not valid:
            continue

        rid, cat = item["raw_id"], item["category"]
        r = {"raw_id": rid, "category": cat}
        for pfx in MODEL_KEYS:
            d = item[pfx]
            if d["video"] is None:
                continue
            svr  = cos_sim(d["video"],      d["text_real"])
            svf  = cos_sim(d["video"],      d["text_fake"])
            str_ = cos_sim(d["transcript"], d["text_real"])
            stf  = cos_sim(d["transcript"], d["text_fake"])
            cr = (svr + str_) / 2
            cf = (svf + stf)  / 2
            r[f"{pfx}_e1_correct"]  = svr  > svf
            r[f"{pfx}_e2a_correct"] = str_ > stf
            r[f"{pfx}_e2c_correct"] = cr   > cf
            r[f"{pfx}_e1_margin"]   = svr  - svf
            r[f"{pfx}_e2a_margin"]  = str_ - stf
        rows.append(r)
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# E4a: Real vs Fake title similarity by category
# ---------------------------------------------------------------------------

def run_e4a_comparison(common_videos):
    rows = []
    for item in common_videos:
        rid, cat = item["raw_id"], item["category"]
        r = {"raw_id": rid, "category": cat}
        for pfx in MODEL_KEYS:
            d = item[pfx]
            if d["video"] is None:
                continue
            r[f"{pfx}_sim_titles"] = cos_sim(d["text_real"], d["text_fake"])
            r[f"{pfx}_e1_correct"] = cos_sim(d["video"], d["text_real"]) > cos_sim(d["video"], d["text_fake"])
        rows.append(r)
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Tables for report
# ---------------------------------------------------------------------------

def _table_header(active_keys, first_col, first_col_w=15):
    h = f"{first_col:<{first_col_w}} {'N':>5}"
    for k in active_keys:
        h += f"  {MODEL_DISPLAY[k]:>14}"
    return h


def print_e1_table(df_e1, output_dir, active_keys):
    categories = sorted(df_e1["category"].unique())
    print("\n" + "=" * 80)
    print("TABLE 1: E1 Accuracy — Video ↔ Title (≤120 s common subset)")
    print("=" * 80)
    print(_table_header(active_keys, "Category"))
    print("-" * 80)

    table_rows = []
    for cat in categories:
        sub = df_e1[df_e1["category"] == cat]
        r = {"Category": cat, "N": len(sub)}
        row_str = f"{cat:<15} {len(sub):>5}"
        for k in active_keys:
            acc = sub[f"{k}_correct"].mean()
            r[MODEL_DISPLAY[k]] = acc
            row_str += f"  {acc:>14.4f}"
        print(row_str)
        table_rows.append(r)

    r = {"Category": "OVERALL", "N": len(df_e1)}
    row_str = f"{'OVERALL':<15} {len(df_e1):>5}"
    for k in active_keys:
        acc = df_e1[f"{k}_correct"].mean()
        r[MODEL_DISPLAY[k]] = acc
        row_str += f"  {acc:>14.4f}"
    print("-" * 80)
    print(row_str)
    table_rows.append(r)

    df_table = pd.DataFrame(table_rows)
    df_table.to_csv(os.path.join(output_dir, "table_E1_comparison.csv"), index=False)
    return df_table


def print_e2_table(df_e2, output_dir, active_keys):
    if df_e2 is None or len(df_e2) == 0:
        print("\n[E2] No common videos with transcripts.")
        return None

    print("\n" + "=" * 80)
    print("TABLE 2: E2 Accuracy — Multi-method (common subset with transcripts)")
    print("=" * 80)
    print(_table_header(active_keys, "Method", first_col_w=30))
    print("-" * 80)

    methods = [
        ("E1: Video ↔ Title",       "e1"),
        ("E2a: Transcript ↔ Title", "e2a"),
        ("E2c: Combined",           "e2c"),
    ]
    table_rows = []
    for label, key in methods:
        r = {"Method": label, "N": len(df_e2)}
        row_str = f"{label:<30} {len(df_e2):>5}"
        for k in active_keys:
            acc = df_e2[f"{k}_{key}_correct"].mean()
            r[MODEL_DISPLAY[k]] = acc
            row_str += f"  {acc:>14.4f}"
        print(row_str)
        table_rows.append(r)

    df_table = pd.DataFrame(table_rows)
    df_table.to_csv(os.path.join(output_dir, "table_E2_comparison.csv"), index=False)
    return df_table


def print_e4a_table(df_e4a, output_dir, active_keys):
    categories = sorted(df_e4a["category"].unique())
    print("\n" + "=" * 80)
    print("TABLE 3: E4a — Mean Title Similarity (Real vs Fake) by Category")
    print("=" * 80)
    print(_table_header(active_keys, "Category"))
    print("-" * 80)

    table_rows = []
    for cat in categories:
        sub = df_e4a[df_e4a["category"] == cat]
        r = {"Category": cat, "N": len(sub)}
        row_str = f"{cat:<15} {len(sub):>5}"
        for k in active_keys:
            if f"{k}_sim_titles" not in sub.columns:
                continue
            sim = sub[f"{k}_sim_titles"].mean()
            r[MODEL_DISPLAY[k]] = sim
            row_str += f"  {sim:>14.4f}"
        print(row_str)
        table_rows.append(r)

    r = {"Category": "OVERALL", "N": len(df_e4a)}
    row_str = f"{'OVERALL':<15} {len(df_e4a):>5}"
    for k in active_keys:
        if f"{k}_sim_titles" not in df_e4a.columns:
            continue
        sim = df_e4a[f"{k}_sim_titles"].mean()
        r[MODEL_DISPLAY[k]] = sim
        row_str += f"  {sim:>14.4f}"
    print("-" * 80)
    print(row_str)
    table_rows.append(r)

    df_table = pd.DataFrame(table_rows)
    df_table.to_csv(os.path.join(output_dir, "table_E4a_comparison.csv"), index=False)
    return df_table

# ---------------------------------------------------------------------------
# Shared plotting helper
# ---------------------------------------------------------------------------

def _grouped_bar(ax, categories, data_per_key, active_keys,
                 ylabel, title, ylim=(0, 1.12), value_fmt=".1%"):
    """Grouped bar chart with consistent palette, annotations, and 50% baseline."""
    n = len(active_keys)
    x = np.arange(len(categories))
    w = 0.72 / n
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * w

    ax.set_axisbelow(True)  # grid drawn behind bars

    for key, offset in zip(active_keys, offsets):
        vals = data_per_key[key]
        bars = ax.bar(x + offset, vals, w,
                      label=MODEL_DISPLAY[key],
                      color=MODEL_COLOR[key],
                      edgecolor="white", linewidth=0.6)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.012,
                format(v, value_fmt),
                ha="center", va="bottom", fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.set_title(title, pad=10)
    # Visible red baseline at chance level — zorder=5 draws above bars
    ax.axhline(0.5, color="#E74C3C", lw=1.4, linestyle="--", alpha=0.85,
               zorder=5, label="Random baseline (50%)")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.85)

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def save_comparison_figures(df_e1, df_e2, df_e4a, output_dir, active_keys):
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # ── Fig 1: E1 accuracy by category ─────────────────────────────────────
    if len(df_e1) > 0:
        cats = sorted(df_e1["category"].unique())
        data = {k: [df_e1[df_e1["category"] == c][f"{k}_correct"].mean() for c in cats]
                for k in active_keys}
        with plt.rc_context(_RC):
            fig, ax = plt.subplots(figsize=(10, 5))
            _grouped_bar(ax, cats, data, active_keys,
                         ylabel="Accuracy",
                         title="E1 — Video ↔ Title Accuracy by Category (≤120 s)")
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, "comparison_E1_by_category.png"), dpi=150)
            plt.close()

    # ── Fig 2: E1 overall accuracy ──────────────────────────────────────────
    if len(df_e1) > 0:
        data = {k: [df_e1[f"{k}_correct"].mean()] for k in active_keys}
        with plt.rc_context(_RC):
            fig, ax = plt.subplots(figsize=(5, 4))
            _grouped_bar(ax, ["All categories"], data, active_keys,
                         ylabel="Accuracy",
                         title="E1 — Video ↔ Title Overall Accuracy")
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, "comparison_E1_overall.png"), dpi=150)
            plt.close()

    # ── Fig 3: E2 method comparison (overall) ──────────────────────────────
    if df_e2 is not None and len(df_e2) > 0:
        method_labels = [
            ("E1\n(Video ↔ Title)",        "e1"),
            ("E2a\n(Transcript ↔ Title)",  "e2a"),
            ("E2c\n(Combined)",            "e2c"),
        ]
        cats = [lbl for lbl, _ in method_labels]
        data = {k: [df_e2[f"{k}_{key}_correct"].mean() for _, key in method_labels]
                for k in active_keys}
        with plt.rc_context(_RC):
            fig, ax = plt.subplots(figsize=(9, 5))
            _grouped_bar(ax, cats, data, active_keys,
                         ylabel="Accuracy",
                         title="E1 / E2 — Method Comparison (Subset with Transcripts)")
            ax.set_xlabel("Method")
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, "comparison_E2_methods.png"), dpi=150)
            plt.close()

    # ── Fig 4: E2a accuracy by category ────────────────────────────────────
    if df_e2 is not None and len(df_e2) > 0:
        cats = sorted(df_e2["category"].unique())
        data = {k: [df_e2[df_e2["category"] == c][f"{k}_e2a_correct"].mean() for c in cats]
                for k in active_keys}
        with plt.rc_context(_RC):
            fig, ax = plt.subplots(figsize=(10, 5))
            _grouped_bar(ax, cats, data, active_keys,
                         ylabel="Accuracy",
                         title="E2a — Transcript ↔ Title Accuracy by Category")
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, "comparison_E2a_by_category.png"), dpi=150)
            plt.close()

    # ── Fig 5: E4a mean title similarity by category (bar) ─────────────────
    if df_e4a is not None and len(df_e4a) > 0:
        cats = sorted(df_e4a["category"].unique())
        data = {k: [df_e4a[df_e4a["category"] == c][f"{k}_sim_titles"].mean() for c in cats]
                for k in active_keys if f"{k}_sim_titles" in df_e4a.columns}
        if data:
            with plt.rc_context(_RC):
                fig, ax = plt.subplots(figsize=(10, 5))
                _grouped_bar(ax, cats, data, list(data.keys()),
                             ylabel="Cosine Similarity",
                             title="E4a — Mean Title Similarity (Real vs Fake) by Category",
                             ylim=(0, 1.12),
                             value_fmt=".3f")
                # Remove the accuracy baseline (not meaningful for similarity)
                for line in ax.get_lines():
                    line.set_visible(False)
                ax.legend(fontsize=8, loc="lower right", framealpha=0.85)
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
                plt.tight_layout()
                plt.savefig(os.path.join(fig_dir, "comparison_E4a_title_similarity.png"), dpi=150)
                plt.close()

    # ── Fig 6: E4a title similarity distribution (box plots) ───────────────
    if df_e4a is not None and len(df_e4a) > 0:
        valid_keys = [k for k in active_keys if f"{k}_sim_titles" in df_e4a.columns]
        if valid_keys:
            cats_sorted = sorted(df_e4a["category"].unique())
            with plt.rc_context(_RC):
                fig, axes = plt.subplots(1, len(valid_keys),
                                         figsize=(4.5 * len(valid_keys), 5),
                                         sharey=True)
                if len(valid_keys) == 1:
                    axes = [axes]
                for ax, k in zip(axes, valid_keys):
                    bp_data = [df_e4a[df_e4a["category"] == c][f"{k}_sim_titles"].values
                               for c in cats_sorted]
                    bp = ax.boxplot(bp_data, patch_artist=True,
                                    medianprops=dict(color="black", lw=1.5),
                                    whiskerprops=dict(lw=0.8),
                                    capprops=dict(lw=0.8),
                                    flierprops=dict(marker="o", markersize=3, alpha=0.5))
                    for patch in bp["boxes"]:
                        patch.set_facecolor(MODEL_COLOR[k])
                        patch.set_alpha(0.65)
                    ax.set_xticklabels(cats_sorted, rotation=20, ha="right")
                    ax.set_title(MODEL_DISPLAY[k])
                    ax.set_xlabel("Category")
                axes[0].set_ylabel("Cosine Similarity (Real vs Fake Title)")
                fig.suptitle("E4a — Title Similarity Distribution by Category", y=1.01)
                plt.tight_layout()
                plt.savefig(os.path.join(fig_dir, "comparison_E4a_boxplot.png"),
                            dpi=150, bbox_inches="tight")
                plt.close()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    matplotlib.use("Agg")

    p = argparse.ArgumentParser(
        description="Compare GE2 vs Qwen3-VL (2B and/or 8B) — E1, E2, E4a"
    )
    p.add_argument("--ge2-db",       required=True,  help="Path to GE2 SQLite DB")
    p.add_argument("--qwen2b-db",    default=None,   help="Path to Qwen3-VL 2B SQLite DB (optional)")
    p.add_argument("--qwen8b-db",    default=None,   help="Path to Qwen3-VL 8B SQLite DB (optional)")
    p.add_argument("--index",        required=True,  help="Path to test_index.json")
    p.add_argument("--output",       required=True,  help="Output directory for tables and figures")
    p.add_argument("--max-duration", type=float, default=120.0,
                   help="Max video duration in seconds (default: 120)")
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)

    with open(args.index, encoding="utf-8") as f:
        test_index = json.load(f)

    ge2_conn  = sqlite3.connect(args.ge2_db)
    qw2b_conn = sqlite3.connect(args.qwen2b_db) if args.qwen2b_db  else None
    qw8b_conn = sqlite3.connect(args.qwen8b_db) if args.qwen8b_db  else None

    print(f"\nGE2   modalities: {get_modality_counts(ge2_conn)}")
    if qw2b_conn: print(f"QW-2B modalities: {get_modality_counts(qw2b_conn)}")
    if qw8b_conn: print(f"QW-8B modalities: {get_modality_counts(qw8b_conn)}")

    active_keys = ["ge2"]
    if qw2b_conn: active_keys.append("qw2b")
    if qw8b_conn: active_keys.append("qw8b")

    print(f"\nFinding common videos (≤{args.max_duration}s)...")
    common = find_common_videos(test_index, ge2_conn, qw2b_conn, qw8b_conn, args.max_duration)
    if not common:
        print("ERROR: No common videos found.")
        sys.exit(1)

    print("\n--- Running E1 ---")
    df_e1 = run_e1_comparison(common)
    print_e1_table(df_e1, args.output, active_keys)

    print("\n--- Running E2 ---")
    df_e2 = run_e2_comparison(common)
    print_e2_table(df_e2, args.output, active_keys)

    print("\n--- Running E4a ---")
    df_e4a = run_e4a_comparison(common)
    print_e4a_table(df_e4a, args.output, active_keys)

    # Save raw data
    df_e1.to_csv(os.path.join(args.output, "raw_E1_comparison.csv"), index=False)
    df_e4a.to_csv(os.path.join(args.output, "raw_E4a_comparison.csv"), index=False)
    if df_e2 is not None and len(df_e2) > 0:
        df_e2.to_csv(os.path.join(args.output, "raw_E2_comparison.csv"), index=False)

    save_comparison_figures(df_e1, df_e2, df_e4a, args.output, active_keys)

    if ge2_conn:  ge2_conn.close()
    if qw2b_conn: qw2b_conn.close()
    if qw8b_conn: qw8b_conn.close()
    print("\nDone. Results saved to:", args.output)


if __name__ == "__main__":
    main()
