# -*- coding: utf-8 -*-
"""
Audit: why does the title separate fake/real so well in the GroundLie360 probe?

Tests
  T1  Reproduce the embedding title-probe (ge2) on the exact same ids/CV.
  T2  TF-IDF bag-of-words on the SAME titles + SAME CV (no embeddings).
  T3  Style-only features (length, punctuation, caps...) — no word identity.
  T4  Deduplicate titles -> does the embedding probe AUC drop?
  T5  Subsets: reals vs fakes WITH false_title  /  reals vs fakes WITHOUT
      false_title (their title is TRUE). High AUC in the second = style/topic
      bias, not falseness detection.
  T6  Most discriminative words (LogReg coefficients on TF-IDF).
  T7  Exact lexical markers (Snopes-claim style vs headline style).

Outputs (same folder):
  audit_title_probe_results.json  — all numbers
  audit_title_probe_log.txt       — full console log
"""
import sys, json, re
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
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
import experiments.analysis.mavis_analysis as ma

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
LOG_LINES = []


def log(msg=""):
    print(msg)
    LOG_LINES.append(str(msg))


def emb_pipe():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=128, random_state=0)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                   random_state=0, solver="lbfgs")),
    ])


def auc_cv(pipe, X, y, cv=CV):
    proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")
    return float(roc_auc_score(y, proba[:, 1]))


# ---------------------------------------------------------------- load (ge2)
conn = ma.connect(ma.db_path("GroundLie360", "ge2"))
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

y      = df["binary_label"].astype(int).to_numpy()
titles = df["title"].astype(str).tolist()
T      = np.stack([title_emb[i] for i in ids]).astype(np.float64)
log(f"N={len(ids)}  fakes={y.sum()}  reals={(1-y).sum()}")

out = {"n": len(ids), "n_fake": int(y.sum()), "n_real": int((1 - y).sum())}

# ---------------------------------------------------------------- T1 reproduce
auc_t1 = auc_cv(emb_pipe(), T, y)
log(f"\n[T1] Embedding title probe (ge2, mismo CV):        AUC={auc_t1:.4f}")
out["T1_emb_title_auc"] = round(auc_t1, 4)

# ---------------------------------------------------------------- T2 TF-IDF
tfidf_word = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, sublinear_tf=True,
                              ngram_range=(1, 2), min_df=2)),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                               random_state=0)),
])
auc_t2 = auc_cv(tfidf_word, titles, y)
log(f"[T2] TF-IDF palabras (sin embeddings):              AUC={auc_t2:.4f}")

tfidf_char = Pipeline([
    ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                               random_state=0)),
])
auc_t2c = auc_cv(tfidf_char, titles, y)
log(f"[T2] TF-IDF char 3-5grams:                          AUC={auc_t2c:.4f}")
out["T2_tfidf_word_auc"] = round(auc_t2, 4)
out["T2_tfidf_char_auc"] = round(auc_t2c, 4)

# ---------------------------------------------------------------- T3 style only
def style_feats(t):
    words = t.split()
    n_chars = max(len(t), 1)
    return [
        len(t), len(words),
        np.mean([len(w) for w in words]) if words else 0,
        t.count("!"), t.count("?"), t.count('"') + t.count("'"),
        t.count(","), t.count(":"), t.count("-"),
        sum(c.isdigit() for c in t) / n_chars,
        sum(c.isupper() for c in t) / n_chars,
        sum(1 for w in words if w.isupper() and len(w) > 1),
        int(bool(re.search(r"\b(19|20)\d\d\b", t))),
    ]

S = np.array([style_feats(t) for t in titles])
style_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                               random_state=0)),
])
auc_t3 = auc_cv(style_pipe, S, y)
log(f"[T3] Solo estilo (13 features, sin léxico):         AUC={auc_t3:.4f}")
out["T3_style_only_auc"] = round(auc_t3, 4)

# ---------------------------------------------------------------- T4 dedup
norm_titles = [re.sub(r"\s+", " ", t.strip().lower()) for t in titles]
seen, keep = set(), []
for k, nt in enumerate(norm_titles):
    if nt not in seen:
        seen.add(nt)
        keep.append(k)
keep = np.array(keep)
n_dup = len(ids) - len(keep)
auc_t4_emb  = auc_cv(emb_pipe(), T[keep], y[keep])
auc_t4_word = auc_cv(tfidf_word, [titles[k] for k in keep], y[keep])
log(f"\n[T4] Tras dedup ({n_dup} duplicados eliminados, N={len(keep)}):")
log(f"     Embedding title probe:                         AUC={auc_t4_emb:.4f}")
log(f"     TF-IDF palabras:                               AUC={auc_t4_word:.4f}")
out["T4_dedup_removed"] = int(n_dup)
out["T4_emb_auc_dedup"] = round(auc_t4_emb, 4)
out["T4_tfidf_auc_dedup"] = round(auc_t4_word, 4)

# ---------------------------------------------------------------- T5 subsets
ft = df["false_title"].fillna(0).astype(int).to_numpy()
mask_real        = y == 0
mask_fake_ft     = (y == 1) & (ft == 1)
mask_fake_noft   = (y == 1) & (ft == 0)
log(f"\n[T5] Subconjuntos (reals={mask_real.sum()}, fake+false_title={mask_fake_ft.sum()}, "
    f"fake sin false_title={mask_fake_noft.sum()}):")
out["T5"] = {}

for key, name, mask_fake in [
    ("fakes_con_false_title", "reals vs fakes CON false_title", mask_fake_ft),
    ("fakes_sin_false_title", "reals vs fakes SIN false_title (titulo VERDADERO)", mask_fake_noft),
]:
    m = mask_real | mask_fake
    ys = y[m]
    auc_e = auc_cv(emb_pipe(), T[m], ys)
    auc_w = auc_cv(tfidf_word, [t for t, mm in zip(titles, m) if mm], ys)
    log(f"     {name}:")
    log(f"       embeddings AUC={auc_e:.4f}   TF-IDF AUC={auc_w:.4f}   (n={m.sum()})")
    out["T5"][key] = {"emb_auc": round(auc_e, 4), "tfidf_auc": round(auc_w, 4), "n": int(m.sum())}

# per-type breakdown of cross-validated probe scores (full model)
proba_full = cross_val_predict(emb_pipe(), T, y, cv=CV, method="predict_proba")[:, 1]
log("\n[T5b] Score medio del probe (CV) por tipo de fake:")
log(f"      reals:                 {proba_full[mask_real].mean():.3f}")
out["T5b_mean_probe_score"] = {"reals": round(float(proba_full[mask_real].mean()), 4)}
for col in ["false_title", "false_speech", "temporal_edit", "cgi", "contradictory", "unsupported"]:
    m = (y == 1) & (df[col].fillna(0).astype(int).to_numpy() == 1)
    if m.sum():
        log(f"      fake {col:<14} (n={m.sum():>3}): {proba_full[m].mean():.3f}")
        out["T5b_mean_probe_score"][f"fake_{col}"] = round(float(proba_full[m].mean()), 4)
m = mask_fake_noft
log(f"      fake SIN false_title  (n={m.sum():>3}): {proba_full[m].mean():.3f}")
out["T5b_mean_probe_score"]["fake_sin_false_title"] = round(float(proba_full[m].mean()), 4)

# ---------------------------------------------------------------- T6 top words
vec = TfidfVectorizer(lowercase=True, sublinear_tf=True, ngram_range=(1, 2), min_df=2)
Xw = vec.fit_transform(titles)
clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=0).fit(Xw, y)
vocab = np.array(vec.get_feature_names_out())
order = np.argsort(clf.coef_[0])
top_fake = vocab[order[-20:]][::-1].tolist()
top_real = vocab[order[:20]].tolist()
log("\n[T6] Top-20 palabras hacia FAKE:")
log("     " + ", ".join(top_fake))
log("[T6] Top-20 palabras hacia REAL:")
log("     " + ", ".join(top_real))
out["T6_top_fake_words"] = top_fake
out["T6_top_real_words"] = top_real

# ---------------------------------------------------------------- T7 lexical markers
markers = {
    "termina_en_punto":       lambda t: t.rstrip().endswith("."),
    "contiene_shows":         lambda t: "shows" in t.lower(),
    "contiene_video":         lambda t: "video" in t.lower(),
    "empieza_claim_video":    lambda t: bool(re.match(r"^(A video|Video|This video|Footage)\b", t)),
    "contiene_dos_puntos":    lambda t: ": " in t,
    "contiene_trump":         lambda t: "trump" in t.lower(),
}
log("\n[T7] Marcadores léxicos exactos (subset analizado):")
out["T7_lexical_markers"] = {}
for name, fn in markers.items():
    hits = np.array([fn(t) for t in titles])
    pf = hits[y == 1].mean()
    pr = hits[y == 0].mean()
    log(f"      {name:<22} fakes={pf:6.1%}  reals={pr:6.1%}")
    out["T7_lexical_markers"][name] = {"fakes_pct": round(float(pf), 4),
                                       "reals_pct": round(float(pr), 4)}

# ---------------------------------------------------------------- save
(HERE / "audit_title_probe_results.json").write_text(
    json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
(HERE / "audit_title_probe_log.txt").write_text(
    "\n".join(LOG_LINES) + "\n", encoding="utf-8")
log("\nGuardado: audit_title_probe_results.json + audit_title_probe_log.txt")
