# -*- coding: utf-8 -*-
"""H5 — ¿Salva al umbral una normalización LOCAL de retrieval (CSLS / z-local)?

QUÉ HACE
  E1c demostró que restar la media GLOBAL de cos(V, corpus) no arregla el
  clasificador absoluto (0.565→0.567). Aquí se prueban las correcciones LOCALES
  estándar de retrieval, que corrigen hubness (puntos que son vecinos de todo):

    raw      : score = cos(V, T)                              [referencia E1b]
    CSLS(K)  : score = 2·cos(V,T) − rK(V) − rK(T)
               rK(V) = media de los K cosenos más altos de V contra el BANCO de
               títulos de train; rK(T) = ídem de T contra los vídeos de train.
               (Conneau et al. 2018 — corrige que un hub tenga cosenos altos
               "gratis" con todo.)
    z-local  : score = (cos(V,T) − μK(V)) / σK(V), con μK/σK calculados sobre
               los K títulos de train más cercanos a V (estandariza la escala
               local de cada vídeo).

  Protocolo idéntico a E1b: split 70/30 POR VÍDEO (estratificado por categoría),
  umbral que maximiza accuracy en train (regla `fake si score < thr`),
  evaluación en test, 50 repeticiones. Los bancos (títulos/vídeos de referencia)
  se construyen SOLO con el split de train (sin fuga).

  Diagnóstico de hubness: skewness de la k-ocurrencia (cuántas veces aparece
  cada título real en el top-10 de los vídeos). Skewness alta = hay hubs.

MÉTRICA
  accuracy media en test (50 reps) por score; + AUC agrupado de cada score.

Salida: H5_results.json + consola.
"""
import sys, json
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.stats import skew

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import experiments.analysis.mavis_analysis as ma

from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
K = 10
N_REPS, TRAIN_FRAC = 50, 0.7

idx = json.load(open(REPO / "experiments/FakeVV_testset/google_embeddings2/test_index.json",
                     encoding="utf-8"))
cat_by_id = {e["raw_id"]: e.get("category", "unknown") for e in idx}


def l2rows(M):
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n[n == 0] = 1
    return M / n


def fit_threshold(scores, labels):
    s = np.asarray(scores, float); l = np.asarray(labels, int)
    o = np.argsort(s, kind="mergesort")
    s, l = s[o], l[o]
    n, nf = len(s), int(l.sum())
    fk = np.concatenate([[0], np.cumsum(l)])
    k = np.arange(n + 1)
    correct = fk + ((n - nf) - (k - fk))
    b = int(np.argmax(correct))
    if b == 0:  return float(s[0] - 1e-9)
    if b == n:  return float(s[-1] + 1e-9)
    return float((s[b - 1] + s[b]) / 2)


def topk_mean(sim_rows, k):
    """Media de los k valores más altos de cada fila."""
    part = np.partition(sim_rows, -k, axis=1)[:, -k:]
    return part.mean(axis=1)


results = {}
for model in ma.MODELS_BY_DATASET["FakeVV"]:
    conn = ma.connect(ma.db_path("FakeVV", model))
    vid = ma.load_globals(conn, "video")
    trl = ma.load_globals(conn, "text_real")
    tfk = ma.load_globals(conn, "text_fake")
    conn.close()
    ids = sorted(set(vid) & set(trl) & set(tfk))
    n = len(ids)
    V  = l2rows(np.stack([vid[r] for r in ids]).astype(np.float64))
    Tr = l2rows(np.stack([trl[r] for r in ids]).astype(np.float64))
    Tf = l2rows(np.stack([tfk[r] for r in ids]).astype(np.float64))
    cats = [cat_by_id.get(r, "?") for r in ids]
    by_cat = defaultdict(list)
    for j, c in enumerate(cats):
        by_cat[c].append(j)

    C_r = V @ Tr.T            # cos(V_i, título_real_j): banco completo
    C_f_diag = np.einsum("ij,ij->i", V, Tf)
    C_r_diag = np.diag(C_r)

    # diagnóstico de hubness: k-ocurrencia de cada título en el top-10 de los vídeos
    top10 = np.argsort(-C_r, axis=1)[:, :10]
    kocc = np.bincount(top10.ravel(), minlength=n)
    hub_skew = float(skew(kocc))

    accs = {"raw": [], "csls": [], "zlocal": []}
    for rep in range(N_REPS):
        rng = np.random.default_rng(1000 + rep)
        tr_idx, te_idx = [], []
        for c, js in by_cat.items():
            js = list(js); rng.shuffle(js)
            cut = int(round(TRAIN_FRAC * len(js)))
            tr_idx += js[:cut]; te_idx += js[cut:]
        tr_idx, te_idx = np.array(tr_idx), np.array(te_idx)
        tr_mask = np.zeros(n, bool); tr_mask[tr_idx] = True

        # bancos SOLO de train: títulos reales y vídeos de train
        simV_bank = C_r[:, tr_mask].copy()           # cos(V_i, títulos train)
        # quitar el propio título del banco cuando i ∈ train
        own_col = np.cumsum(tr_mask) - 1
        for i in tr_idx:
            simV_bank[i, own_col[i]] = -np.inf
        rK_V = topk_mean(np.where(np.isfinite(simV_bank), simV_bank, -1.0), K)

        Vbank = V[tr_idx]
        rK_Tr = topk_mean(Tr @ Vbank.T, K)           # rK del título real
        rK_Tf = topk_mean(Tf @ Vbank.T, K)           # rK del título fake

        # z-local: μ y σ de los K títulos de train más cercanos a cada vídeo
        finite = np.where(np.isfinite(simV_bank), simV_bank, -1.0)
        partK = np.partition(finite, -K, axis=1)[:, -K:]
        muK, sdK = partK.mean(axis=1), partK.std(axis=1) + 1e-6

        scores = {
            "raw":    (C_r_diag, C_f_diag),
            "csls":   (2 * C_r_diag - rK_V - rK_Tr, 2 * C_f_diag - rK_V - rK_Tf),
            "zlocal": ((C_r_diag - muK) / sdK, (C_f_diag - muK) / sdK),
        }
        for name, (s_r, s_f) in scores.items():
            thr = fit_threshold(np.concatenate([s_r[tr_idx], s_f[tr_idx]]),
                                [0] * len(tr_idx) + [1] * len(tr_idx))
            te_s = np.concatenate([s_r[te_idx], s_f[te_idx]])
            te_l = np.array([0] * len(te_idx) + [1] * len(te_idx))
            accs[name].append(float(((te_s < thr).astype(int) == te_l).mean()))

    # AUC agrupado con bancos del split completo (orientativo)
    res = {"n": n, "hubness_skew_kocc": round(hub_skew, 3)}
    for name in accs:
        res[f"acc_{name}"] = round(float(np.mean(accs[name])), 4)
        res[f"acc_{name}_std"] = round(float(np.std(accs[name])), 4)
    results[model] = res
    print(f"[{model}] hub_skew={hub_skew:+.2f}  "
          f"acc raw={res['acc_raw']:.3f}  CSLS={res['acc_csls']:.3f}  "
          f"z-local={res['acc_zlocal']:.3f}")

(HERE / "H5_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print("\nGuardado: H5_results.json")
