"""
mavis_analysis.py — Reusable analysis helpers for MAVIS TFG multimodal embedding experiments.

Consolidates DB access, math, statistics, and plotting helpers shared across
GroundLie360, FakeVV and M3A analysis notebooks.

Usage (from any notebook / script):
    import sys; sys.path.insert(0, str(REPO_ROOT))
    from experiments.analysis.mavis_analysis import *
"""

import json
import sqlite3
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Repository root & DB registry
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

DB_PATHS = {
    ('FakeVV', 'qw2b'): 'experiments/FakeVV_testset/open_source/results/qwen3vl_2b/qwen3vl_cache.db',
    ('FakeVV', 'qw8b'): 'experiments/FakeVV_testset/open_source/results/qwen3vl_8b/qwen3vl_cache.db',
    ('FakeVV', 'wave'): 'experiments/open_source/results/WAVE7B/wave_cache.db',
    ('FakeVV', 'ge2'):  'experiments/FakeVV_testset/google_embeddings2/results/test_set_fakevvGE2.db',
    ('GroundLie360', 'qw2b'): 'experiments/GroundLie360/open_source/results/qwen3vl_2b/qwen3vl_cache.db',
    ('GroundLie360', 'qw8b'): 'experiments/GroundLie360/open_source/results/qwen3vl_8b/qwen3vl_cache.db',
    ('GroundLie360', 'wave'): 'experiments/GroundLie360/open_source/results/WAVE7B/wave_cache.db',
    ('GroundLie360', 'ge2'):  'experiments/GroundLie360/google_embeddings2/results/groundlie_ge2.db',
    ('M3A', 'qw2b'): 'experiments/M3A/open_source/results/qwen3vl_2b/qwen3vl_cache.db',
    ('M3A', 'qw8b'): 'experiments/M3A/open_source/results/qwen3vl_8b/qwen3vl_cache.db',
    ('M3A', 'wave'): 'experiments/M3A/open_source/results/WAVE7B/wave_cache.db',
    ('M3A', 'ge2'):  'experiments/M3A/google_embeddings2/results/m3a_ge2.db',
}
# Resolve all paths against REPO_ROOT
DB_PATHS = {k: REPO_ROOT / v for k, v in DB_PATHS.items()}

# Separate GE2 segments DB for GroundLie360 (parallelised embedding run)
GE2_GL360_SEGMENTS_DB = REPO_ROOT / 'experiments/GroundLie360/google_embeddings2/results/groundlie_ge2_segments.db'

MODELS_BY_DATASET = {
    'FakeVV':      ['qw2b', 'qw8b', 'wave', 'ge2'],
    'GroundLie360': ['qw2b', 'qw8b', 'wave', 'ge2'],
    'M3A':         ['qw2b', 'qw8b', 'wave', 'ge2'],
}


def db_path(dataset: str, model: str) -> Path:
    """Return resolved Path for a (dataset, model) pair. Raises KeyError if unknown."""
    return DB_PATHS[(dataset, model)]


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

MODEL_DISPLAY = {
    'ge2':  'GE2',
    'qw2b': 'Qwen3-VL 2B',
    'qw8b': 'Qwen3-VL 8B',
    'wave': 'WAVE-7B',
}

MODEL_COLOR = {
    'ge2':  '#4472C4',
    'qw2b': '#70AD47',
    'qw8b': '#ED7D31',
    'wave': '#7030A0',
}

_RC = {
    'font.family':       'sans-serif',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.18,
    'grid.linestyle':    '--',
    'axes.titlesize':    13,
    'axes.labelsize':    11,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'legend.fontsize':   9,
    'figure.dpi':        150,
}

# ---------------------------------------------------------------------------
# DB access helpers
# ---------------------------------------------------------------------------

def connect(path):
    return sqlite3.connect(str(path))


def load_globals(conn, modality):
    """Return {raw_id: np.float32 vector} for a modality from the embeddings table."""
    rows = conn.execute(
        'SELECT raw_id, vector FROM embeddings WHERE modality=?', (modality,)
    ).fetchall()
    return {r[0]: np.frombuffer(r[1], dtype=np.float32) for r in rows}


def load_segments(conn, segment_type, modality):
    """Return {raw_id: list of {segment_idx, start_s, end_s, vector, extra}} ordered by raw_id, segment_idx."""
    rows = conn.execute(
        'SELECT raw_id, segment_idx, start_s, end_s, vector, extra_json '
        'FROM segment_embeddings '
        'WHERE segment_type=? AND modality=? ORDER BY raw_id, segment_idx',
        (segment_type, modality),
    ).fetchall()
    out = {}
    for rid, idx, s0, s1, blob, extra in rows:
        out.setdefault(rid, []).append({
            'segment_idx': idx,
            'start_s': s0,
            'end_s': s1,
            'vector': np.frombuffer(blob, dtype=np.float32),
            'extra': json.loads(extra) if extra else None,
        })
    return out


def load_labels(conn):
    """Return groundlie_labels table indexed by raw_id (GroundLie360 only)."""
    return pd.read_sql_query('SELECT * FROM groundlie_labels', conn).set_index('raw_id')


def groundlie_event_ids(data_csv: Path = None) -> dict:
    """
    Deterministic Snopes-event grouping for GroundLie360 (anti-leakage in CV).

    GroundLie360 has no explicit fact-check-article id, so we approximate the
    event partition with a threshold-free union-find over two deterministic links:
      (1) identical normalised title (lowercase, strip punctuation/whitespace);
      (2) a shared auxiliary video id (original/debunking/evidence_videoid) —
          two target videos referencing the same source are the same event.

    Returns {raw_id (video_id): event_id}. Singletons map to their own id.
    With the shipped data.csv this yields ~1789 events for 2044 videos
    (396 videos fall in 141 multi-video events); the paper reports 1466 events,
    so this proxy is conservative (never merges unrelated videos, may leave some
    same-event videos separate). Use as `groups=` in GroupKFold.
    """
    import csv as _csv
    import re as _re
    from collections import defaultdict as _dd
    _csv.field_size_limit(10 ** 7)
    if data_csv is None:
        data_csv = REPO_ROOT / 'data' / 'data.csv'

    def _norm(t):
        return _re.sub(r'\s+', ' ', _re.sub(r'[^a-z0-9 ]', '', (t or '').lower())).strip()

    def _ids(s):
        s = (s or '').strip()
        if not s or s.lower() in ('nan', '[]'):
            return []
        return _re.findall(r'[A-Za-z0-9_\-]+', s)

    rows = list(_csv.DictReader(open(data_csv, encoding='utf-8')))
    parent = {r['video_id']: r['video_id'] for r in rows}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    by_title = _dd(list)
    for r in rows:
        nt = _norm(r.get('title', ''))
        if nt:
            by_title[nt].append(r['video_id'])
    for g in by_title.values():
        for v in g[1:]:
            union(g[0], v)

    for col in ('original_videoid', 'debunking_videoid', 'evidence_videoid'):
        by_aux = _dd(list)
        for r in rows:
            for x in _ids(r.get(col, '')):
                by_aux[x].append(r['video_id'])
        for g in by_aux.values():
            for v in g[1:]:
                union(g[0], v)

    return {r['video_id']: find(r['video_id']) for r in rows}


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def l2(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


def perimeter3(a, b, c):
    """Triangle perimeter in cosine-distance space for 3 embedding vectors."""
    return (1 - cos(a, b)) + (1 - cos(a, c)) + (1 - cos(b, c))


def tri_area(da, db, dc):
    """Triangle area via Heron's formula given three side lengths."""
    s = (da + db + dc) / 2
    return float(np.sqrt(max(s * (s - da) * (s - db) * (s - dc), 0.0)))


def auc_fake(score, is_fake):
    """Wilcoxon-Mann-Whitney AUC: P(score_fake > score_real). High score => more fake."""
    s = np.asarray(score, float)
    y = np.asarray(is_fake, int)
    ok = ~np.isnan(s)
    s, y = s[ok], y[ok]
    if len(np.unique(y)) < 2:
        return np.nan
    order = s.argsort()
    ranks = np.empty_like(order, float)
    ranks[order] = np.arange(1, len(s) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def bootstrap_ci(values, stat=np.mean, n=10000, seed=0, alpha=0.05):
    """Bootstrap confidence interval for stat(values). Returns (low, high)."""
    rng = np.random.default_rng(seed)
    vals = np.asarray(values, float)
    boots = [stat(rng.choice(vals, size=len(vals), replace=True)) for _ in range(n)]
    lo = np.percentile(boots, 100 * alpha / 2)
    hi = np.percentile(boots, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def summarize_paired(real, fake, name):
    """
    Paired real-vs-fake summary.

    Returns dict with keys:
        exp, n, mean_real, mean_fake, mean_diff, acc, t, p, cohen_d, ci_low, ci_high
    acc = mean(real > fake) (paper-style accuracy).
    t, p from scipy.stats.ttest_rel.
    cohen_d = mean(diff) / std(diff, ddof=1).
    ci_low/ci_high = 95% bootstrap CI (10 000 resamples, seed=0) of mean(diff).
    """
    real = np.asarray(real, float)
    fake = np.asarray(fake, float)
    diff = real - fake
    acc = float(np.mean(real > fake))
    if len(diff) > 1:
        t, p = stats.ttest_rel(real, fake)
    else:
        t, p = np.nan, np.nan
    std_d = float(np.std(diff, ddof=1))
    cohen_d = float(diff.mean() / std_d) if std_d > 0 else np.nan
    ci_low, ci_high = bootstrap_ci(diff, stat=np.mean, n=10000, seed=0)
    return {
        'exp':       name,
        'n':         len(diff),
        'mean_real': round(float(real.mean()), 4),
        'mean_fake': round(float(fake.mean()), 4),
        'mean_diff': round(float(diff.mean()), 4),
        'acc':       round(acc, 4),
        't':         round(float(t), 4),
        'p':         f'{p:.1e}',
        'cohen_d':   round(cohen_d, 4),
        'ci_low':    round(ci_low, 4),
        'ci_high':   round(ci_high, 4),
    }


def mannwhitney(a, b):
    """Mann-Whitney U test (two-sided). Returns (U, p)."""
    U, p = stats.mannwhitneyu(a, b, alternative='two-sided')
    return float(U), float(p)


def cohens_d_indep(a, b):
    """Independent-groups Cohen's d using pooled standard deviation."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2) / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else np.nan


# ---------------------------------------------------------------------------
# Availability introspection
# ---------------------------------------------------------------------------

def available(dataset: str, model: str) -> dict:
    """
    Probe a (dataset, model) DB for what is available.

    Returns {'exists': False} if the DB file is missing.
    Otherwise returns:
        {'exists': True, 'tables': [...], 'modalities': {mod: count},
         'segments': {seg_type: count}}
    """
    try:
        path = db_path(dataset, model)
    except KeyError:
        return {'exists': False}
    if not path.exists():
        return {'exists': False}
    conn = sqlite3.connect(str(path))
    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        modalities = {}
        if 'embeddings' in tables:
            rows = conn.execute(
                'SELECT modality, COUNT(*) FROM embeddings GROUP BY modality'
            ).fetchall()
            modalities = {mod: cnt for mod, cnt in rows}
        segments = {}
        try:
            if 'segment_embeddings' in tables:
                rows = conn.execute(
                    'SELECT segment_type, COUNT(*) FROM segment_embeddings GROUP BY segment_type'
                ).fetchall()
                segments = {st: cnt for st, cnt in rows}
        except Exception:
            pass
        # Algunos modelos (GE2 GroundLie) guardan los segmentos en una DB hermana *_segments.db
        if not segments:
            seg_path = path.with_name(path.stem + '_segments.db')
            if seg_path.exists():
                sconn = sqlite3.connect(str(seg_path))
                try:
                    stabs = [r[0] for r in sconn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()]
                    if 'segment_embeddings' in stabs:
                        rows = sconn.execute(
                            'SELECT segment_type, COUNT(*) FROM segment_embeddings GROUP BY segment_type'
                        ).fetchall()
                        segments = {st: cnt for st, cnt in rows}
                finally:
                    sconn.close()
        return {'exists': True, 'tables': tables, 'modalities': modalities, 'segments': segments}
    finally:
        conn.close()


def availability_table() -> pd.DataFrame:
    """
    Build a DataFrame showing what modalities and segments are available for every
    (dataset, model) pair. Counts are shown as integers; '-' where absent.

    Columns: dataset, model, video, text_title, text_summary, transcript, audio,
             scene_segs, window_segs
    """
    MOD_COLS = ['video', 'text_title', 'text_summary', 'transcript', 'audio']
    SEG_COLS = ['scene', 'window_fake']

    rows = []
    for (dataset, model) in DB_PATHS:
        info = available(dataset, model)
        row = {'dataset': dataset, 'model': model}
        if not info['exists']:
            for c in MOD_COLS:
                row[c] = '-'
            for c in SEG_COLS:
                row[c + '_segs'] = '-'
        else:
            mods = info.get('modalities', {})
            segs = info.get('segments', {})
            # FakeVV guarda el titulo como text_real/text_fake (no text_title)
            if dataset == 'FakeVV' and 'text_title' not in mods and 'text_real' in mods:
                mods = {**mods, 'text_title': mods['text_real']}
            for c in MOD_COLS:
                row[c] = mods.get(c, '-')
            for c in SEG_COLS:
                row[c + '_segs'] = segs.get(c, '-')
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(['dataset', 'model']).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

FIG_DIR = REPO_ROOT / 'experiments/analysis/figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig, fname):
    path = FIG_DIR / fname
    fig.savefig(str(path), dpi=150, bbox_inches='tight')
    return fig


def grouped_bar(data, ylabel, title, fname, baseline=None, pct=False):
    """
    Grouped bar chart.

    data: dict model->value  OR  dict group->dict model->value
    Colors from MODEL_COLOR, x-labels from MODEL_DISPLAY.
    baseline: optional float for a horizontal reference line.
    pct: use PercentFormatter on y-axis.
    Returns the figure.
    """
    # Normalise input to group -> {model: value}
    first = next(iter(data.values()))
    if isinstance(first, dict):
        groups = list(data.keys())
        model_vals = data  # group -> {model: value}
        all_models = list({m for g in model_vals.values() for m in g})
    else:
        groups = ['']
        model_vals = {'': data}
        all_models = list(data.keys())

    # Preserve insertion order where possible
    all_models = [m for m in MODEL_COLOR if m in all_models]

    x = np.arange(len(groups))
    n = len(all_models)
    w = 0.72 / n
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * w

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(max(5, 2.5 * len(groups)), 4))
        ax.set_axisbelow(True)
        for model, offset in zip(all_models, offsets):
            vals = [model_vals[g].get(model, np.nan) for g in groups]
            bars = ax.bar(
                x + offset, vals, w,
                label=MODEL_DISPLAY.get(model, model),
                color=MODEL_COLOR.get(model, '#888888'),
                edgecolor='white', linewidth=0.6,
            )
            for bar, v in zip(bars, vals):
                if np.isnan(v):
                    continue
                fmt = f'{v:.1%}' if pct else f'{v:.3f}'
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                        fmt, ha='center', va='bottom', fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=20, ha='right')
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=10)
        if baseline is not None:
            ax.axhline(baseline, color='#E74C3C', lw=1.4, linestyle='--', alpha=0.85,
                       zorder=5, label=f'Baseline ({baseline})')
        if pct:
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.legend(fontsize=8, loc='lower right', framealpha=0.85)
        plt.tight_layout()
        return _save(fig, fname)


# ---------------------------------------------------------------------------
# UMAP modality-gap visualizations
# ---------------------------------------------------------------------------

# Canonical modality type (for color grouping)
_MODALITY_TYPE = {
    'text_title':    'text',
    'text_summary':  'text',
    'transcript':    'text',
    'video':         'video',
    'audio':         'audio',
    'audiovisual':   'audiovisual',
}

MODALITY_TYPE_COLOR = {
    'text':        '#2196F3',   # blue
    'video':       '#E53935',   # red
    'audio':       '#43A047',   # green
    'audiovisual': '#FB8C00',   # orange
}

MODALITY_TYPE_LABEL = {
    'text':        'Text (title / summary / transcript)',
    'video':       'Video',
    'audio':       'Audio',
    'audiovisual': 'Audiovisual',
}

DATASET_COLOR = {
    'FakeVV':      '#E53935',
    'GroundLie360': '#1E88E5',
    'M3A':         '#43A047',
}

# Which modalities to load per (dataset, model) for the UMAP plots
_UMAP_MODALITIES = {
    ('FakeVV',       'qw2b'): ['text_title', 'transcript', 'video'],
    ('FakeVV',       'qw8b'): ['text_title', 'transcript', 'video'],
    ('FakeVV',       'ge2'):  ['text_title', 'transcript', 'video'],
    ('GroundLie360', 'qw2b'): ['text_title', 'transcript', 'video'],
    ('GroundLie360', 'qw8b'): ['text_title', 'transcript', 'video'],
    ('GroundLie360', 'ge2'):  ['text_title', 'transcript', 'video'],
    ('GroundLie360', 'wave'): ['text_title', 'transcript', 'video', 'audio', 'audiovisual'],
    ('M3A',          'qw2b'): ['text_summary', 'transcript', 'video'],
    ('M3A',          'qw8b'): ['text_summary', 'transcript', 'video'],
    ('M3A',          'ge2'):  ['text_summary', 'transcript', 'video'],
    ('M3A',          'wave'): ['text_summary', 'transcript', 'video'],
}


def _load_umap_matrix(dataset: str, model: str, max_per_modality: int = 2000):
    """
    Load embeddings for (dataset, model) and return (matrix, modality_types, raw_ids).

    Vectors are L2-normalised. Up to max_per_modality rows per modality (random sample,
    seed=42) to keep UMAP tractable. Returns None if DB doesn't exist.
    """
    path = db_path(dataset, model)
    if not path.exists():
        return None
    modalities = _UMAP_MODALITIES.get((dataset, model), [])
    conn = connect(path)
    rng = np.random.default_rng(42)
    vectors, types = [], []
    for mod in modalities:
        data = load_globals(conn, mod)
        if not data:
            continue
        ids = list(data.keys())
        if len(ids) > max_per_modality:
            ids = rng.choice(ids, size=max_per_modality, replace=False).tolist()
        for rid in ids:
            v = data[rid]
            n = np.linalg.norm(v)
            vectors.append(v / n if n > 1e-9 else v)
            types.append(_MODALITY_TYPE.get(mod, mod))
    conn.close()
    if not vectors:
        return None
    return np.array(vectors, dtype=np.float32), types


def plot_modality_gap_umap(dataset: str, n_neighbors: int = 30, min_dist: float = 0.1,
                           max_per_modality: int = 2000, fname: str = None):
    """
    Plot 1: For a given dataset, 4 subplots (one per model), colors = modality type.

    Returns the figure.
    """

    # He quitado a WAVE hasta que acabe.
    import umap as umap_lib

    models_ordered = [m for m in ['ge2', 'qw2b', 'qw8b', 'wave'] if m in MODELS_BY_DATASET[dataset]]

    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, len(models_ordered),
                                 figsize=(5 * len(models_ordered), 5))
        if len(models_ordered) == 1:
            axes = [axes]

        for ax, model in zip(axes, models_ordered):
            result = _load_umap_matrix(dataset, model, max_per_modality)
            if result is None:
                ax.set_title(f'{MODEL_DISPLAY[model]}\n(no data)')
                ax.axis('off')
                continue
            matrix, types = result
            reducer = umap_lib.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                                    random_state=42, verbose=False)
            proj = reducer.fit_transform(matrix)

            # Plot by modality type (legend dedup)
            seen = set()
            for mtype in ['text', 'video', 'audio', 'audiovisual']:
                mask = np.array([t == mtype for t in types])
                if not mask.any():
                    continue
                label = MODALITY_TYPE_LABEL[mtype] if mtype not in seen else '_nolegend_'
                seen.add(mtype)
                ax.scatter(proj[mask, 0], proj[mask, 1],
                           c=MODALITY_TYPE_COLOR[mtype],
                           s=4, alpha=0.4, linewidths=0,
                           label=label)

            ax.set_title(MODEL_DISPLAY[model], fontsize=11)
            ax.set_xlabel('UMAP-1', fontsize=8)
            ax.set_ylabel('UMAP-2', fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=7, markerscale=3, loc='best', framealpha=0.8)

        fig.suptitle(f'Modality gap — {dataset}', fontsize=13, y=1.02)
        plt.tight_layout()
        out_fname = fname or f'umap_modality_gap_{dataset.lower()}.png'
        return _save(fig, out_fname)


def plot_dataset_gap_umap(n_neighbors: int = 30, min_dist: float = 0.1,
                          max_per_modality: int = 1000, fname: str = None):
    """
    Plot 2: 4 subplots (one per model), colors = dataset. All modalities pooled.

    max_per_modality is per (dataset × modality) to keep balance.
    Returns the figure.
    """
    import umap as umap_lib

    all_models = ['ge2', 'qw2b', 'qw8b', 'wave']
    all_datasets = ['FakeVV', 'GroundLie360', 'M3A']

    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, len(all_models),
                                 figsize=(5 * len(all_models), 5))

        for ax, model in zip(axes, all_models):
            vectors_all, dataset_labels = [], []
            for ds in all_datasets:
                if model not in MODELS_BY_DATASET.get(ds, []):
                    continue
                result = _load_umap_matrix(ds, model, max_per_modality)
                if result is None:
                    continue
                matrix, _ = result
                vectors_all.append(matrix)
                dataset_labels.extend([ds] * len(matrix))

            if not vectors_all:
                ax.set_title(f'{MODEL_DISPLAY[model]}\n(no data)')
                ax.axis('off')
                continue

            matrix = np.vstack(vectors_all)
            reducer = umap_lib.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                                    random_state=42, verbose=False)
            proj = reducer.fit_transform(matrix)

            seen = set()
            for ds in all_datasets:
                mask = np.array([d == ds for d in dataset_labels])
                if not mask.any():
                    continue
                label = ds if ds not in seen else '_nolegend_'
                seen.add(ds)
                ax.scatter(proj[mask, 0], proj[mask, 1],
                           c=DATASET_COLOR[ds],
                           s=4, alpha=0.4, linewidths=0,
                           label=label)

            ax.set_title(MODEL_DISPLAY[model], fontsize=11)
            ax.set_xlabel('UMAP-1', fontsize=8)
            ax.set_ylabel('UMAP-2', fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=8, markerscale=3, loc='best', framealpha=0.8)

        fig.suptitle('Dataset separation in embedding space', fontsize=13, y=1.02)
        plt.tight_layout()
        out_fname = fname or 'umap_dataset_gap.png'
        return _save(fig, out_fname)


def boxplot_by_group(series_by_model, ylabel, title, fname):
    """
    Side-by-side box plots, one subplot per model.

    series_by_model: dict model -> dict group -> array-like values
    Returns the figure.
    """
    models = [m for m in MODEL_COLOR if m in series_by_model]
    n = len(models)
    if n == 0:
        raise ValueError('series_by_model is empty or has no recognised models')

    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 5), sharey=True)
        if n == 1:
            axes = [axes]
        for ax, model in zip(axes, models):
            groups = list(series_by_model[model].keys())
            bp_data = [np.asarray(series_by_model[model][g], float) for g in groups]
            bp = ax.boxplot(
                bp_data, patch_artist=True,
                medianprops=dict(color='black', lw=1.5),
                whiskerprops=dict(lw=0.8),
                capprops=dict(lw=0.8),
                flierprops=dict(marker='o', markersize=3, alpha=0.5),
            )
            for patch in bp['boxes']:
                patch.set_facecolor(MODEL_COLOR.get(model, '#888888'))
                patch.set_alpha(0.65)
            ax.set_xticklabels(groups, rotation=20, ha='right')
            ax.set_title(MODEL_DISPLAY.get(model, model))
        axes[0].set_ylabel(ylabel)
        fig.suptitle(title, y=1.01)
        plt.tight_layout()
        return _save(fig, fname)


# ---------------------------------------------------------------------------
# Phase-2 helper functions
# ---------------------------------------------------------------------------

def retrieval_metrics(query: dict, gallery: dict, sample_n: int = 1000, seed: int = 0) -> dict:
    """
    Cross-modal retrieval metrics (R@K, MedR, MRR).

    query, gallery: dicts raw_id -> np.float32 vector over the SAME id set (paired).
    For each query_i the correct gallery item is the same raw_id.

    If more than sample_n common ids, a random subset of sample_n is drawn (seed).
    Vectors are L2-normalised; cosine matrix built via matrix multiplication.

    Returns dict with keys: n, R@1, R@5, R@10, MedR, MRR.
    Call twice swapping query/gallery to get both V2T and T2V directions.
    """
    common = sorted(set(query) & set(gallery))
    if not common:
        return {'n': 0, 'R@1': np.nan, 'R@5': np.nan, 'R@10': np.nan, 'MedR': np.nan, 'MRR': np.nan}
    if len(common) > sample_n:
        rng = np.random.default_rng(seed)
        common = list(rng.choice(common, size=sample_n, replace=False))
        common.sort()
        # note: subsample taken (caller can see n < original)

    # Build matrices (n x d), L2-normalised
    def norm_mat(d, ids):
        mat = np.stack([d[i].astype(np.float32) for i in ids])
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        return mat / norms

    Q = norm_mat(query, common)   # (n, d)
    G = norm_mat(gallery, common) # (n, d)

    # Cosine similarity matrix (n x n)
    sim = Q @ G.T  # sim[i,j] = cos(query_i, gallery_j)

    n = len(common)
    ranks = []
    rr = []
    for i in range(n):
        # rank of correct item (diagonal) — 1-based
        row = sim[i]
        # number of gallery items with HIGHER similarity (ties broken by >=)
        rank = int(np.sum(row > row[i])) + 1  # 1-based rank
        ranks.append(rank)
        rr.append(1.0 / rank)

    ranks = np.array(ranks)
    return {
        'n':    n,
        'R@1':  float(np.mean(ranks <= 1)),
        'R@5':  float(np.mean(ranks <= 5)),
        'R@10': float(np.mean(ranks <= 10)),
        'MedR': float(np.median(ranks)),
        'MRR':  float(np.mean(rr)),
    }


def bootstrap_auc_ci(scores, labels, n: int = 2000, seed: int = 0, alpha: float = 0.05):
    """
    Bootstrap confidence interval for ROC-AUC.

    Returns (auc, lo, hi). Resamples indices with replacement; skips resamples
    where only one class is present.
    """
    from sklearn.metrics import roc_auc_score
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    ok = ~np.isnan(scores)
    scores, labels = scores[ok], labels[ok]
    if len(np.unique(labels)) < 2:
        return (np.nan, np.nan, np.nan)
    auc = float(roc_auc_score(labels, scores))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n):
        idx = rng.integers(0, len(scores), size=len(scores))
        lb, sb = labels[idx], scores[idx]
        if len(np.unique(lb)) < 2:
            continue
        boots.append(roc_auc_score(lb, sb))
    if not boots:
        return (auc, np.nan, np.nan)
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return (auc, lo, hi)


def paired_auc_diff_ci(scores_a, scores_b, labels, n: int = 2000, seed: int = 0, alpha: float = 0.05):
    """
    Bootstrap CI for the paired AUC difference (AUC_a − AUC_b).

    Both score vectors are over the same items with the same labels.
    Uses paired index resampling (bootstrap analogue of DeLong et al. 1988).
    Returns (diff, lo, hi).
    """
    from sklearn.metrics import roc_auc_score
    scores_a = np.asarray(scores_a, float)
    scores_b = np.asarray(scores_b, float)
    labels   = np.asarray(labels, int)
    ok = ~np.isnan(scores_a) & ~np.isnan(scores_b)
    scores_a, scores_b, labels = scores_a[ok], scores_b[ok], labels[ok]
    if len(np.unique(labels)) < 2:
        return (np.nan, np.nan, np.nan)
    diff_obs = float(roc_auc_score(labels, scores_a) - roc_auc_score(labels, scores_b))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n):
        idx = rng.integers(0, len(labels), size=len(labels))
        lb = labels[idx]
        if len(np.unique(lb)) < 2:
            continue
        da = roc_auc_score(lb, scores_a[idx]) - roc_auc_score(lb, scores_b[idx])
        boots.append(da)
    if not boots:
        return (diff_obs, np.nan, np.nan)
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return (diff_obs, lo, hi)


def roc_operating_point(scores, labels) -> dict:
    """
    ROC curve + Youden's J optimal operating point.

    Returns dict: {auc, thr_youden, tpr, fpr, sensitivity, specificity}.
    sensitivity = tpr at Youden point, specificity = 1 - fpr.
    """
    from sklearn.metrics import roc_curve, roc_auc_score
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    ok = ~np.isnan(scores)
    scores, labels = scores[ok], labels[ok]
    if len(np.unique(labels)) < 2:
        return {'auc': np.nan, 'thr_youden': np.nan, 'tpr': np.nan,
                'fpr': np.nan, 'sensitivity': np.nan, 'specificity': np.nan}
    auc = float(roc_auc_score(labels, scores))
    fpr_arr, tpr_arr, thr_arr = roc_curve(labels, scores)
    j = tpr_arr - fpr_arr
    best = int(np.argmax(j))
    return {
        'auc':         auc,
        'thr_youden':  float(thr_arr[best]),
        'tpr':         float(tpr_arr[best]),
        'fpr':         float(fpr_arr[best]),
        'sensitivity': float(tpr_arr[best]),
        'specificity': float(1.0 - fpr_arr[best]),
    }


def modality_gap(emb_a: dict, emb_b: dict, intra_n: int = 500, seed: int = 0) -> dict:
    """
    Modality gap between two embedding sets (Liang et al., NeurIPS 2022).

    emb_a, emb_b: dicts raw_id -> np.float32 vector.
    Restricts to common ids, L2-normalises each vector.

    Returns dict:
        gap_euclidean  — ||centroid_a − centroid_b||_2
        mean_inter_cos — mean cos(a_i, b_i) over paired common ids
        mean_intra_a   — mean pairwise cos within modality a (random 500-sample)
        mean_intra_b   — same for b
    """
    common = sorted(set(emb_a) & set(emb_b))
    if not common:
        return {'gap_euclidean': np.nan, 'mean_inter_cos': np.nan,
                'mean_intra_a': np.nan, 'mean_intra_b': np.nan}

    def norm_mat(d, ids):
        mat = np.stack([d[i].astype(np.float32) for i in ids])
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        return mat / norms

    A = norm_mat(emb_a, common)  # (n, d)
    B = norm_mat(emb_b, common)  # (n, d)

    ca = A.mean(axis=0)
    cb = B.mean(axis=0)
    gap_euc = float(np.linalg.norm(ca - cb))

    inter_cos = float(np.mean(np.sum(A * B, axis=1)))  # paired dot products (normalised)

    # intra: sample up to intra_n vectors, compute mean pairwise cos
    def mean_intra(M):
        n = len(M)
        if n <= 1:
            return np.nan
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=min(intra_n, n), replace=False)
        S = M[idx]
        # all-pairs cosine via matrix mult (already normalised)
        sim = S @ S.T
        # exclude diagonal
        mask = ~np.eye(len(S), dtype=bool)
        return float(sim[mask].mean())

    return {
        'gap_euclidean':  gap_euc,
        'mean_inter_cos': inter_cos,
        'mean_intra_a':   mean_intra(A),
        'mean_intra_b':   mean_intra(B),
    }


# ---------------------------------------------------------------------------
# Embedding-space geometry metrics
# ---------------------------------------------------------------------------

def _to_matrix(emb):
    """Accept dict {raw_id: vec} or 2-D np.array; return np.float32 2-D array."""
    if isinstance(emb, dict):
        return np.stack(list(emb.values())).astype(np.float32)
    return np.asarray(emb, dtype=np.float32)


def anisotropy(emb) -> float:
    """
    Mean cosine similarity between random pairs (isotropy measure).

    Uses the closed-form expression for the mean over all ordered pairs:
        (||sum of unit rows||^2 - N) / (N*(N-1))

    Reference: Ethayarajh (2019), "How Contextual are Contextualised Word
    Representations?", arXiv:1909.00512.

    Returns a float in [-1, 1].  ~0 => isotropic; near 1 => degenerate cone.
    """
    X = _to_matrix(emb)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    U = X / norms
    N = len(U)
    col_sum = U.sum(axis=0)
    return float((np.dot(col_sum, col_sum) - N) / (N * (N - 1)))


def effective_rank(emb) -> float:
    """
    Entropy-based effective rank of the embedding matrix.

    Centers columns, computes singular values s, normalises to a probability
    distribution p = s / sum(s), then returns exp(H(p)) where H is Shannon
    entropy (zero entries ignored).

    Reference: Roy & Vetterli (2007), "The effective rank: A measure of
    effective dimensionality", EUSIPCO 2007.

    Returns a float >= 1.
    """
    X = _to_matrix(emb).astype(np.float64)
    Xc = X - X.mean(axis=0)
    s = np.linalg.svd(Xc, compute_uv=False)
    s = s[s > 0]
    p = s / s.sum()
    h = -np.sum(p * np.log(p))
    return float(np.exp(h))


def intrinsic_dim_twonn(emb, sample: int = 2000, seed: int = 0) -> float:
    """
    TwoNN intrinsic dimensionality estimator.

    For each point finds the distances to its 1st (r1) and 2nd (r2) nearest
    neighbours (excluding self), forms mu = r2/r1, then fits a Pareto
    distribution through the origin on the log-log empirical CDF.

    Reference: Facco et al. (2017), "Estimating the intrinsic dimension of
    datasets by a minimal neighborhood information", Sci. Rep. 7:12140.

    Parameters
    ----------
    emb    : dict or 2-D array
    sample : maximum number of points to use (random subsample, for speed)
    seed   : RNG seed for subsampling

    Returns float d (intrinsic dimension estimate).
    """
    X = _to_matrix(emb).astype(np.float64)
    N = len(X)
    if N > sample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(N, size=sample, replace=False)
        X = X[idx]
        N = sample

    # Pairwise Euclidean distances via ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    sq_norms = np.sum(X ** 2, axis=1)              # (N,)
    sq_dist = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (X @ X.T)
    np.clip(sq_dist, 0, None, out=sq_dist)          # numerical safety
    dist = np.sqrt(sq_dist)                         # (N, N)
    np.fill_diagonal(dist, np.inf)

    sorted_d = np.sort(dist, axis=1)
    r1 = sorted_d[:, 0]
    r2 = sorted_d[:, 1]

    mu = r2 / r1
    valid = (mu > 1) & np.isfinite(mu)
    mu = mu[valid]
    mu = np.sort(mu)
    N_v = len(mu)
    if N_v < 2:
        return np.nan

    F = np.arange(1, N_v + 1) / N_v
    keep = F < 1.0
    x = np.log(mu[keep])
    y = -np.log(1.0 - F[keep])
    # Linear regression through the origin: d = sum(x*y) / sum(x*x)
    d = float(np.sum(x * y) / np.sum(x * x))
    return d


def alignment_uniformity(A, B, sample: int = 2000, seed: int = 0) -> dict:
    """
    Alignment and uniformity of paired embeddings.

    alignment  = mean ||a_i - b_i||^2 over positive pairs (lower => more aligned).
    uniformity = log(mean exp(-2 ||x_i - x_j||^2)) over random pairs drawn from
                 the pooled, L2-normalised set (lower => more uniform on the sphere).

    Reference: Wang & Isola (2020), "Understanding Contrastive Representation
    Learning through Alignment and Uniformity on the Hypersphere", ICML 2020,
    arXiv:2005.10242.

    Parameters
    ----------
    A, B   : paired dicts (matched by shared keys) or equal-length arrays
    sample : max pooled points for the uniformity term
    seed   : RNG seed

    Returns dict with keys 'alignment' and 'uniformity'.
    """
    if isinstance(A, dict) and isinstance(B, dict):
        common = sorted(set(A) & set(B))
        a_mat = np.stack([A[k].astype(np.float32) for k in common])
        b_mat = np.stack([B[k].astype(np.float32) for k in common])
    else:
        a_mat = np.asarray(A, dtype=np.float32)
        b_mat = np.asarray(B, dtype=np.float32)

    # L2-normalise
    def _norm(M):
        n = np.linalg.norm(M, axis=1, keepdims=True)
        n = np.where(n < 1e-9, 1.0, n)
        return M / n

    a_mat = _norm(a_mat)
    b_mat = _norm(b_mat)

    # Alignment: mean squared Euclidean distance between pairs
    alignment = float(np.mean(np.sum((a_mat - b_mat) ** 2, axis=1)))

    # Uniformity: pool, subsample, all-pairs kernel
    pooled = np.concatenate([a_mat, b_mat], axis=0)
    N_p = len(pooled)
    if N_p > sample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(N_p, size=sample, replace=False)
        pooled = pooled[idx]

    # ||x_i - x_j||^2 = 2 - 2 x_i.x_j  (unit vectors)
    sim = pooled @ pooled.T          # (n, n) cosine similarities
    sq_dist = 2.0 - 2.0 * sim       # (n, n)
    # upper-triangle (exclude diagonal and duplicate pairs)
    iu = np.triu_indices(len(pooled), k=1)
    sq_dists_pairs = sq_dist[iu]
    uniformity = float(np.log(np.mean(np.exp(-2.0 * sq_dists_pairs))))

    return {'alignment': alignment, 'uniformity': uniformity}


def linear_cka(X, Y) -> float:
    """
    Linear Centered Kernel Alignment (CKA) between representation matrices.

    X (N x D1) and Y (N x D2) must have the same N rows (same items).
    Columns of each are mean-centred.  The linear kernel is K = X X^T.

    cka = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)

    Reference: Kornblith et al. (2019), "Similarity of Neural Network
    Representations Revisited", ICML 2019, arXiv:1905.00414.

    Returns a float in [0, 1].
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    # ||Y^T X||_F^2
    YtX = Y.T @ X
    num = float(np.sum(YtX ** 2))
    # ||X^T X||_F
    XtX = X.T @ X
    denom_x = float(np.sqrt(np.sum(XtX ** 2)))
    YtY = Y.T @ Y
    denom_y = float(np.sqrt(np.sum(YtY ** 2)))
    if denom_x == 0 or denom_y == 0:
        return np.nan
    return float(num / (denom_x * denom_y))


def dist_real_vs_fake(real, fake, xlabel, title, fname):
    """
    Overlaid histogram + KDE for two distributions with AUC in the legend.

    real, fake: array-like of floats
    Returns the figure.
    """
    from scipy.stats import gaussian_kde

    real = np.asarray(real, float)
    fake = np.asarray(fake, float)
    auc = auc_fake(
        np.concatenate([fake, real]),
        np.concatenate([np.ones(len(fake)), np.zeros(len(real))]),
    )

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(7, 4))
        bins = np.linspace(
            min(real.min(), fake.min()),
            max(real.max(), fake.max()),
            40,
        )
        ax.hist(real, bins=bins, alpha=0.35, color='#4472C4', label='Real', density=True)
        ax.hist(fake, bins=bins, alpha=0.35, color='#ED7D31', label=f'Fake  (AUC={auc:.3f})', density=True)
        for arr, col in [(real, '#4472C4'), (fake, '#ED7D31')]:
            kde = gaussian_kde(arr[~np.isnan(arr)])
            xs = np.linspace(bins[0], bins[-1], 300)
            ax.plot(xs, kde(xs), color=col, lw=1.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Density')
        ax.set_title(title, pad=10)
        ax.legend()
        plt.tight_layout()
        return _save(fig, fname)
