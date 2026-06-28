# -*- coding: utf-8 -*-
"""
Audit 2: does the Snopes-construction bias also affect the OTHER modalities
(video, transcript) and the relational features of the probe?

Runs the full validity battery for every model (ge2, qw2b, qw8b; WAVE excluded
pending token-bug regen), same ids/CV as the probe notebook.

Tests
  M0  Metadata-only probes (no content): duration, platform, year, all-meta.
      Model-independent (run once). High AUC = separable from pure metadata.
  M1  Video probe on subsets: reals vs fakes WITHOUT visual manipulation
      (cgi=0 & temporal_edit=0). If AUC barely drops, the video embedding is
      not detecting manipulation but source/style/topic.
  M2  Transcript probe vs TF-IDF on raw transcript text + subset without
      false_speech (spoken content is TRUE).
  M3  Concat WITHOUT title (V+Tr): how much of concat was the title?
  M4  Relational features validity: should work for cross-modal fakes
      (contradictory/unsupported), not for purely visual ones.
  M5  Title probe on subset without false_title (title is TRUE).

Outputs (same folder): audit_modality_bias_results.json / _log.txt
"""
import sys, json, csv, re
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


def auc_cv(pipe, X, y, cv=CV):
    proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")
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


def load_model_data(model):
    """Same id selection as the probe notebook for a given model."""
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
    return ids, df, y, V, T, Tr, R


# ---------------------------------------------------------------- shared data
with open(REPO / "data/data.csv", encoding="utf-8") as f:
    meta = {r["video_id"]: r for r in csv.DictReader(f)}

conn = ma.connect(ma.db_path("GroundLie360", "ge2"))
durations_all = dict(conn.execute("SELECT raw_id, duration_seconds FROM video_metadata"))
conn.close()

out = {}

# ---------------------------------------------------------------- M0 metadata (once, ge2 ids)
ids, df, y, V, T, Tr, R = load_model_data("ge2")
transcripts = [meta[i]["video_transcript"] if i in meta else "" for i in ids]
log(f"[M0] Probes SOLO con metadatos (sin contenido) — ids ge2, N={len(ids)}:")

dur = np.array([[durations_all.get(i, np.nan) or np.nan] for i in ids])
dur = np.nan_to_num(dur, nan=float(np.nanmean(dur)))
auc_dur = auc_cv(emb_pipe(high_dim=False), dur, y)
log(f"     duración del vídeo (1 feature):                AUC={auc_dur:.4f}")

platforms = [meta[i]["platform"].strip().lower() if i in meta else "?" for i in ids]
plat_vocab = sorted(set(platforms))
P = np.array([[int(p == v) for v in plat_vocab] for p in platforms], dtype=float)
auc_plat = auc_cv(emb_pipe(high_dim=False), P, y)
log(f"     plataforma one-hot ({len(plat_vocab)} cats):              AUC={auc_plat:.4f}")

years = []
for i in ids:
    m = re.search(r"(19|20)\d\d", meta[i]["video_date"] if i in meta else "")
    years.append(int(m.group()) if m else 0)
yr = np.array(years, dtype=float)
yr[yr == 0] = np.median(yr[yr > 0])
auc_year = auc_cv(emb_pipe(high_dim=False), yr.reshape(-1, 1), y)
log(f"     año del vídeo (1 feature):                     AUC={auc_year:.4f}")

has_tr = np.array([[float(len(t.strip()) > 0), float(len(t.split()))] for t in transcripts])
M_all = np.hstack([dur, P, yr.reshape(-1, 1), has_tr])
auc_meta = auc_cv(emb_pipe(high_dim=False), M_all, y)
log(f"     todos los metadatos juntos:                    AUC={auc_meta:.4f}")
out["M0"] = {"duration_auc": round(auc_dur, 4), "platform_auc": round(auc_plat, 4),
             "year_auc": round(auc_year, 4), "all_meta_auc": round(auc_meta, 4),
             "platforms": plat_vocab}

log("\n     Distribución de plataforma por clase:")
out["M0"]["platform_dist"] = {}
for v in plat_vocab:
    pv = np.array([p == v for p in platforms])
    f_pct, r_pct = pv[y == 1].mean(), pv[y == 0].mean()
    if max(f_pct, r_pct) >= 0.01:
        log(f"       {v or '(vacío)':<16} fakes={f_pct:6.1%}  reals={r_pct:6.1%}")
    out["M0"]["platform_dist"][v] = {"fakes": round(float(f_pct), 4),
                                     "reals": round(float(r_pct), 4)}

# TF-IDF transcript is embedding-independent: compute once on ge2 ids
tfidf = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, sublinear_tf=True,
                              ngram_range=(1, 2), min_df=2)),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=0)),
])
auc_tr_tfidf_global = auc_cv(tfidf, transcripts, y)
log(f"\n     TF-IDF del texto ASR (sin embeddings, ids ge2): AUC={auc_tr_tfidf_global:.4f}")
out["M0"]["transcript_tfidf_auc"] = round(auc_tr_tfidf_global, 4)

# ---------------------------------------------------------------- per-model battery
for model in MODELS:
    log(f"\n{'='*64}\nMODELO {model}\n{'='*64}")
    ids, df, y, V, T, Tr, R = load_model_data(model)
    transcripts = [meta[i]["video_transcript"] if i in meta else "" for i in ids]
    dur = np.array([[durations_all.get(i, np.nan) or np.nan] for i in ids])
    dur = np.nan_to_num(dur, nan=float(np.nanmean(dur)))
    log(f"N={len(ids)}  fakes={y.sum()}  reals={(1-y).sum()}")
    res = {"n": len(ids)}

    def col(c):
        return df[c].fillna(0).astype(int).to_numpy()

    mask_real     = y == 0
    visual_fake   = (y == 1) & ((col("cgi") == 1) | (col("temporal_edit") == 1))
    novis_fake    = (y == 1) & (col("cgi") == 0) & (col("temporal_edit") == 0)
    xmod_fake     = (y == 1) & ((col("contradictory") == 1) | (col("unsupported") == 1))
    noxmod_fake   = (y == 1) & ~((col("contradictory") == 1) | (col("unsupported") == 1))
    nospeech_fake = (y == 1) & (col("false_speech") == 0)
    noft_fake     = (y == 1) & (col("false_title") == 0)

    # M1 video
    log("\n[M1] Modalidad VIDEO:")
    auc_v_full = auc_cv(emb_pipe(), V, y)
    log(f"     full (reproduce notebook):                     AUC={auc_v_full:.4f}")
    res["M1"] = {"video_full_auc": round(auc_v_full, 4)}
    for key, name, mask_fake in [
        ("sin_manip_visual", "reals vs fakes SIN manipulación visual", novis_fake),
        ("con_manip_visual", "reals vs fakes CON manipulación visual (cgi/temporal)", visual_fake),
    ]:
        m = mask_real | mask_fake
        auc_e = auc_cv(emb_pipe(), V[m], y[m])
        auc_d = auc_cv(emb_pipe(high_dim=False), dur[m], y[m])
        log(f"     {name} (n_fake={mask_fake.sum()}):")
        log(f"       video AUC={auc_e:.4f}   (solo duración: {auc_d:.4f})")
        res["M1"][key] = {"video_auc": round(auc_e, 4), "duration_auc": round(auc_d, 4),
                          "n_fake": int(mask_fake.sum())}

    # M2 transcript
    log("\n[M2] Modalidad TRANSCRIPT:")
    auc_tr_full = auc_cv(emb_pipe(), Tr, y)
    log(f"     embedding full (reproduce notebook):           AUC={auc_tr_full:.4f}")
    m = mask_real | nospeech_fake
    auc_tr_nospeech = auc_cv(emb_pipe(), Tr[m], y[m])
    log(f"     reals vs fakes SIN false_speech (n_fake={nospeech_fake.sum()}):  AUC={auc_tr_nospeech:.4f}")
    res["M2"] = {"transcript_full_auc": round(auc_tr_full, 4),
                 "sin_false_speech_auc": round(auc_tr_nospeech, 4),
                 "n_fake_sin_false_speech": int(nospeech_fake.sum())}

    # M3 concat sin título
    log("\n[M3] CONCAT sin título (V+Tr):")
    auc_c_all = auc_cv(emb_pipe(), np.concatenate([V, T, Tr], axis=1), y)
    auc_c_vt  = auc_cv(emb_pipe(), np.concatenate([V, Tr], axis=1), y)
    log(f"     concat V+T+Tr (reproduce notebook):            AUC={auc_c_all:.4f}")
    log(f"     concat V+Tr (sin título):                      AUC={auc_c_vt:.4f}")
    res["M3"] = {"concat_full_auc": round(auc_c_all, 4),
                 "concat_no_title_auc": round(auc_c_vt, 4)}

    # M4 relational
    log("\n[M4] Features RELACIONALES (validez cross-modal):")
    auc_r_full = auc_cv(emb_pipe(high_dim=False), R, y)
    log(f"     full (reproduce notebook):                     AUC={auc_r_full:.4f}")
    res["M4"] = {"relational_full_auc": round(auc_r_full, 4)}
    for key, name, mask_fake in [
        ("fakes_crossmodal", "reals vs fakes cross-modal (contradictory/unsupported)", xmod_fake),
        ("fakes_no_crossmodal", "reals vs fakes NO cross-modal", noxmod_fake),
    ]:
        m = mask_real | mask_fake
        auc_r = auc_cv(emb_pipe(high_dim=False), R[m], y[m])
        log(f"     {name} (n_fake={mask_fake.sum()}): AUC={auc_r:.4f}")
        res["M4"][key] = {"relational_auc": round(auc_r, 4), "n_fake": int(mask_fake.sum())}

    # M5 title subset
    log("\n[M5] Modalidad TITLE (subset):")
    auc_t_full = auc_cv(emb_pipe(), T, y)
    m = mask_real | noft_fake
    auc_t_noft = auc_cv(emb_pipe(), T[m], y[m])
    log(f"     full (reproduce notebook):                     AUC={auc_t_full:.4f}")
    log(f"     reals vs fakes SIN false_title (n_fake={noft_fake.sum()}):   AUC={auc_t_noft:.4f}")
    res["M5"] = {"title_full_auc": round(auc_t_full, 4),
                 "sin_false_title_auc": round(auc_t_noft, 4),
                 "n_fake_sin_false_title": int(noft_fake.sum())}

    out[model] = res

# ---------------------------------------------------------------- save
(HERE / "audit_modality_bias_results.json").write_text(
    json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
(HERE / "audit_modality_bias_log.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
log("\nGuardado: audit_modality_bias_results.json + audit_modality_bias_log.txt")
