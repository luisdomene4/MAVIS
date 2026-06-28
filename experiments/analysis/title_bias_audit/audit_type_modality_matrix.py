# -*- coding: utf-8 -*-
"""
Audit 3: fake-type x modality matrix.

The binary probe pools ALL fake types into one class (dominated by false_title,
76% of fakes). This script disaggregates: for each fake type t, probe
reals vs fakes-with-type-t, with every feature set.

Reading the matrix:
  - Diagonal (modality matched to the lie: title->false_title,
    transcript->false_speech, video->cgi/temporal_edit,
    relational->contradictory/unsupported) = where genuine signal could live.
  - Off-diagonal (e.g. predicting CGI from the title) has no causal path:
    any AUC > 0.5 there is the dataset-bias floor.
  - Genuine detection = diagonal excess over the off-diagonal floor.

Also reports:
  - n per type, co-occurrence-free ("pure") subsets where n allows.
  - TF-IDF rows (no embeddings) as the lexical-bias reference.

Outputs (same folder): audit_type_modality_results.json / _log.txt
"""
import sys, json, csv
from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
csv.field_size_limit(10**7)
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
import experiments.analysis.mavis_analysis as ma

MODELS = ["ge2", "qw2b", "qw8b"]   # WAVE skipped (token bug / regen pending)
TYPES  = ["false_title", "false_speech", "temporal_edit", "cgi", "contradictory", "unsupported"]
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
LOG = []


def log(msg=""):
    print(msg)
    LOG.append(str(msg))


def emb_pipe(high_dim=True):
    steps = [("scaler", StandardScaler())]
    if high_dim:
        steps.append(("pca", PCA(n_components=128, random_state=0)))
    steps.append(("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                            random_state=0, solver="lbfgs")))
    return Pipeline(steps)


def tfidf_pipe():
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, sublinear_tf=True,
                                  ngram_range=(1, 2), min_df=2)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=0)),
    ])


def auc_cv(pipe, X, y):
    proba = cross_val_predict(pipe, X, y, cv=CV, method="predict_proba")
    return float(roc_auc_score(y, proba[:, 1]))


def relational_features(vid, txt1, txt2):
    def l2n(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v
    v, t1, t2 = l2n(vid.astype(np.float64)), l2n(txt1.astype(np.float64)), l2n(txt2.astype(np.float64))
    return np.array([
        float(v @ t1), float(v @ t2), float(t1 @ t2),
        float(np.linalg.norm(v - t1)), float(np.linalg.norm(v - t2)),
        float(np.linalg.norm(t1 - t2)),
    ])


with open(REPO / "data/data.csv", encoding="utf-8") as f:
    meta = {r["video_id"]: r for r in csv.DictReader(f)}

out = {}

for model in MODELS:
    log(f"\n{'='*72}\nMODELO {model}\n{'='*72}")
    conn = ma.connect(ma.db_path("GroundLie360", model))
    vid_emb   = ma.load_globals(conn, "video")
    title_emb = ma.load_globals(conn, "text_title")
    tr_emb    = ma.load_globals(conn, "transcript")
    labels_df = ma.load_labels(conn)
    conn.close()

    ids = sorted(set(vid_emb) & set(title_emb) & set(tr_emb) & set(labels_df.index))
    labels_df = labels_df.loc[ids]
    valid = labels_df["binary_label"].notna()
    ids = [i for i in ids if valid[i]]
    df = labels_df.loc[ids].copy()

    y  = df["binary_label"].astype(int).to_numpy()
    V  = np.stack([vid_emb[i]   for i in ids]).astype(np.float64)
    T  = np.stack([title_emb[i] for i in ids]).astype(np.float64)
    Tr = np.stack([tr_emb[i]    for i in ids]).astype(np.float64)
    R  = np.stack([relational_features(vid_emb[i], title_emb[i], tr_emb[i]) for i in ids])
    titles      = df["title"].astype(str).tolist()
    transcripts = [meta[i]["video_transcript"] if i in meta else "" for i in ids]

    def col(c):
        return df[c].fillna(0).astype(int).to_numpy()

    type_mask = {t: (y == 1) & (col(t) == 1) for t in TYPES}
    n_types   = {t: int(type_mask[t].sum()) for t in TYPES}
    # "pure" = only this type, no other
    other_sum = {t: sum(col(u) for u in TYPES if u != t) for t in TYPES}
    pure_mask = {t: type_mask[t] & (other_sum[t] == 0) for t in TYPES}
    mask_real = y == 0

    log(f"N={len(ids)}  reals={mask_real.sum()}  | n por tipo (inclusivo / puro):")
    for t in TYPES:
        log(f"   {t:<14} {n_types[t]:>4} / {int(pure_mask[t].sum()):>4}")

    feature_sets = {
        "tfidf_title":      ("text", titles),
        "tfidf_transcript": ("text", transcripts),
        "title":            ("emb", T),
        "transcript":       ("emb", Tr),
        "video":            ("emb", V),
        "relational":       ("emb_low", R),
    }

    res = {"n_reals": int(mask_real.sum()),
           "n_per_type": n_types,
           "n_per_type_pure": {t: int(pure_mask[t].sum()) for t in TYPES},
           "matrix": {}, "matrix_pure": {}}

    header = f"{'tipo':<14}" + "".join(f"{fs:>18}" for fs in feature_sets)
    log("\n[Matriz inclusiva] AUC reals vs fakes-con-tipo-t:")
    log(header)
    for t in TYPES:
        res["matrix"][t] = {}
        row = f"{t:<14}"
        m = mask_real | type_mask[t]
        ys = y[m]
        for fs, (kind, X) in feature_sets.items():
            if kind == "text":
                auc = auc_cv(tfidf_pipe(), [x for x, mm in zip(X, m) if mm], ys)
            elif kind == "emb":
                auc = auc_cv(emb_pipe(), X[m], ys)
            else:
                auc = auc_cv(emb_pipe(high_dim=False), X[m], ys)
            res["matrix"][t][fs] = round(auc, 4)
            row += f"{auc:>18.4f}"
        log(row)

    log("\n[Matriz pura] AUC reals vs fakes-SOLO-tipo-t (n>=40):")
    log(header)
    for t in TYPES:
        if pure_mask[t].sum() < 40:
            log(f"{t:<14}  (n={int(pure_mask[t].sum())} insuficiente)")
            continue
        res["matrix_pure"][t] = {"n_fake": int(pure_mask[t].sum())}
        row = f"{t:<14}"
        m = mask_real | pure_mask[t]
        ys = y[m]
        for fs, (kind, X) in feature_sets.items():
            if kind == "text":
                auc = auc_cv(tfidf_pipe(), [x for x, mm in zip(X, m) if mm], ys)
            elif kind == "emb":
                auc = auc_cv(emb_pipe(), X[m], ys)
            else:
                auc = auc_cv(emb_pipe(high_dim=False), X[m], ys)
            res["matrix_pure"][t][fs] = round(auc, 4)
            row += f"{auc:>18.4f}"
        log(row)

    out[model] = res

(HERE / "audit_type_modality_results.json").write_text(
    json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
(HERE / "audit_type_modality_log.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
log("\nGuardado: audit_type_modality_results.json + audit_type_modality_log.txt")
