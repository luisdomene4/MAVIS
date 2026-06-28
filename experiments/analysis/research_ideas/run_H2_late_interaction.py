# -*- coding: utf-8 -*-
"""H2 — Late-interaction: comparar el título contra ESCENAS en vez del vídeo comprimido.

QUÉ HACE
  Para cada vídeo de GroundLie360 con segmentación de escenas, calcula el coseno
  del título contra CADA escena y resume el perfil en 5 estadísticas:
    max  = la escena que MÁS apoya el título (¿hay evidencia en algún sitio?)
    min  = la escena que MENOS lo apoya (¿algo lo contradice?)
    mean = coherencia media (≈ lo que ve el vector global)
    std  = dispersión entre escenas (¿el vídeo es homogéneo respecto al título?)
    range= max−min
  y las compara con el coseno del embedding GLOBAL (lo que usamos hasta ahora).

POR QUÉ
  El vector global promedia todas las escenas: si la señal de (in)coherencia es
  LOCAL (una escena concreta), el promedio la diluye (E1c: σ temática ≫ señal).
  Si las estadísticas de escena baten al global, la compresión era parte del
  problema. Controles: n_escenas solo (confound de duración/edición) y el suelo
  de sesgo léxico ya conocido (TF-IDF título 0.77).

EVALUACIÓN
  (a) AUC zero-shot de cada estadística (población real vs fake; sin entrenar).
      Orientación: para max/min/mean/global el score de "fake" es −cos (menos
      coherencia ⇒ más fake); para std/range/n_escenas es +stat.
  (b) Probe (LogReg, StratifiedKFold(5)) con las 5 stats vs con el global
      [cos, dist] vs combinado, + control n_escenas.
  (c) Validez por tipo de mentira: reals vs fakes-de-tipo-t para los 6 tipos
      (¿mejora en cross-modal/temporal_edit sin mejorar el suelo de las celdas
      sin camino causal?).

Salida: H2_results.json + consola.
"""
import sys, json
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import experiments.analysis.mavis_analysis as ma

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
CV = StratifiedKFold(5, shuffle=True, random_state=0)
TYPES = ["false_title", "false_speech", "temporal_edit", "cgi", "contradictory", "unsupported"]

# La DB principal de GE2 no tiene segmentos; están en un fichero aparte (subset n=100)
GE2_SEG_DB = REPO / "experiments/GroundLie360/google_embeddings2/results/groundlie_ge2_segments.db"


def probe_auc(X, y):
    pipe = Pipeline([("sc", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                                random_state=0))])
    proba = cross_val_predict(pipe, X, y, cv=CV, method="predict_proba")
    return float(roc_auc_score(y, proba[:, 1]))


results = {}
for model in ["qw2b", "qw8b", "ge2"]:
    main_db = ma.db_path("GroundLie360", model)
    seg_db = GE2_SEG_DB if model == "ge2" else main_db
    if not (main_db.exists() and seg_db.exists()):
        continue

    conn = ma.connect(main_db)
    vid_g = ma.load_globals(conn, "video")
    tit_g = ma.load_globals(conn, "text_title")
    lab = ma.load_labels(conn)
    conn.close()

    conn = ma.connect(seg_db)
    scenes = ma.load_segments(conn, "scene", "video")
    conn.close()

    ids = sorted(set(scenes) & set(vid_g) & set(tit_g) & set(lab.index))
    ids = [i for i in ids if lab.loc[i, "binary_label"] in (0, 1) and len(scenes[i]) >= 1]
    y = lab.loc[ids, "binary_label"].astype(int).to_numpy()

    rows = []
    for i in ids:
        t = tit_g[i]
        cs = np.array([ma.cos(s["vector"], t) for s in scenes[i]])
        rows.append([cs.max(), cs.min(), cs.mean(),
                     cs.std() if len(cs) > 1 else 0.0,
                     cs.max() - cs.min(), len(cs), ma.cos(vid_g[i], t),
                     float(np.linalg.norm(ma.l2(vid_g[i]) - ma.l2(t)))])
    M = np.array(rows)
    smax, smin, smean, sstd, srange, nsc, cglob, dglob = M.T

    res = {"n": len(ids), "n_fake": int(y.sum())}
    log = [f"\n=== {model} — N={len(ids)} (fakes={y.sum()}) ==="]

    # (a) zero-shot por estadística
    zs = {
        "global_cos(-)": roc_auc_score(y, -cglob),
        "scene_max(-)":  roc_auc_score(y, -smax),
        "scene_min(-)":  roc_auc_score(y, -smin),
        "scene_mean(-)": roc_auc_score(y, -smean),
        "scene_std(+)":  roc_auc_score(y, sstd),
        "scene_range(+)": roc_auc_score(y, srange),
        "n_scenes(+)":   roc_auc_score(y, nsc),
    }
    res["zero_shot"] = {k: round(float(v), 4) for k, v in zs.items()}
    log.append("[a] zero-shot AUC: " + "  ".join(f"{k}={v:.3f}" for k, v in zs.items()))

    # (b) probes
    F_scene = np.stack([smax, smin, smean, sstd, srange], axis=1)
    F_glob = np.stack([cglob, dglob], axis=1)
    probes = {
        "probe_scene_stats(5f)": probe_auc(F_scene, y),
        "probe_global_rel(2f)":  probe_auc(F_glob, y),
        "probe_combinado(7f)":   probe_auc(np.hstack([F_scene, F_glob]), y),
        "probe_n_scenes(1f)":    probe_auc(nsc.reshape(-1, 1), y),
    }
    res["probes"] = {k: round(v, 4) for k, v in probes.items()}
    log.append("[b] probes:        " + "  ".join(f"{k}={v:.3f}" for k, v in probes.items()))

    # (c) validez por tipo (probe scene-stats vs global-rel, reals vs fakes-de-tipo)
    res["por_tipo"] = {}
    log.append("[c] por tipo (scene-stats / global-rel / n_fake):")
    mask_real = y == 0
    for t in TYPES:
        tcol = lab.loc[ids, t].fillna(0).astype(int).to_numpy()
        m = mask_real | ((y == 1) & (tcol == 1))
        nf = int(((y == 1) & (tcol == 1)).sum())
        if nf < 25:
            log.append(f"      {t:<14} n_fake={nf} insuficiente")
            continue
        a_sc = probe_auc(F_scene[m], y[m])
        a_gl = probe_auc(F_glob[m], y[m])
        res["por_tipo"][t] = {"scene_stats": round(a_sc, 4), "global_rel": round(a_gl, 4),
                              "n_fake": nf}
        log.append(f"      {t:<14} {a_sc:.3f} / {a_gl:.3f}  (n={nf})")

    print("\n".join(log))
    results[model] = res

(HERE / "H2_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print("\nGuardado: H2_results.json")
