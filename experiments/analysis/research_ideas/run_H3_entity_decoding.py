# -*- coding: utf-8 -*-
"""H3 — ¿El embedding de VÍDEO sabe qué entidad aparece, o solo de qué va el tema?

QUÉ HACE
  Para las entidades más frecuentes en los títulos REALES de FakeVV (p. ej.
  "trump", "china"...), entrena un probe lineal que intenta predecir, SOLO desde
  el embedding del vídeo, si esa entidad aparece en el título real del vídeo.

  - AUC_video : decodabilidad desde V (la medida central).
  - AUC_texto : mismo probe desde el embedding del título real (techo: ahí la
                palabra está literalmente).
  - AUC_perm  : control de selectividad (Hewitt & Liang): mismas etiquetas
                permutadas (3 permutaciones) → debe dar ≈0.5; garantiza que el
                probe no "inventa" señal.
  - Cruce con E1b: ¿la accuracy PAREADA de los pares cuya entidad es más
                decodificable es mayor? (correlación entre ambas, por entidad)

POR QUÉ
  Separa dos causas que hasta ahora iban juntas:
  (a) la identidad de la entidad NO sobrevive a la compresión del vídeo en un
      vector → ningún operador podría verificarla (límite de percepción);
  (b) la identidad SÍ está en V pero el coseno/probe par-a-par no la explota
      (límite del operador → arregla H1/H2).

Salida: H3_results.json + consola.
"""
import sys, json, re
from pathlib import Path
from collections import Counter

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import experiments.analysis.mavis_analysis as ma

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
CV = StratifiedKFold(5, shuffle=True, random_state=0)

idx = json.load(open(REPO / "experiments/FakeVV_testset/google_embeddings2/test_index.json",
                     encoding="utf-8"))
title_by_id = {e["raw_id"]: e["title"] for e in idx}

# entidades candidatas: tokens capitalizados frecuentes en títulos reales (proxy de NER)
STOP = {"the", "a", "an", "in", "on", "of", "to", "for", "and", "at", "with", "after",
        "new", "this", "is", "was", "are", "what", "how", "why", "video", "watch", "us"}
tok_counts = Counter()
for t in title_by_id.values():
    for w in re.findall(r"[A-Za-z']+", t):
        if w[0].isupper() and w.lower() not in STOP and len(w) > 2:
            tok_counts[w.lower()] += 1
ENTITIES = [w for w, c in tok_counts.most_common(60) if c >= 25][:12]
print("Entidades evaluadas (≥25 títulos):", ENTITIES, "\n")


def probe_auc(X, y, seed=0):
    pipe = Pipeline([("sc", StandardScaler()),
                     ("pca", PCA(n_components=128, random_state=0)),
                     ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                                random_state=seed))])
    proba = cross_val_predict(pipe, X, y, cv=CV, method="predict_proba")
    return float(roc_auc_score(y, proba[:, 1]))


results = {}
for model in ma.MODELS_BY_DATASET["FakeVV"]:
    conn = ma.connect(ma.db_path("FakeVV", model))
    vid = ma.load_globals(conn, "video")
    trl = ma.load_globals(conn, "text_real")
    tfk = ma.load_globals(conn, "text_fake")
    conn.close()
    ids = sorted(set(vid) & set(trl) & set(tfk) & set(title_by_id))
    V = np.stack([vid[r] for r in ids]).astype(np.float64)
    T = np.stack([trl[r] for r in ids]).astype(np.float64)
    titles_low = [title_by_id[r].lower() for r in ids]
    cos_r = np.array([ma.cos(vid[r], trl[r]) for r in ids])
    cos_f = np.array([ma.cos(vid[r], tfk[r]) for r in ids])
    paired_ok = (cos_r > cos_f).astype(int)

    rows, dec_aucs, pair_accs = {}, [], []
    rng = np.random.default_rng(0)
    for ent in ENTITIES:
        y = np.array([int(re.search(rf"\b{ent}\b", t) is not None) for t in titles_low])
        if y.sum() < 15 or y.sum() > len(y) - 15:
            continue
        auc_v = probe_auc(V, y)
        auc_t = probe_auc(T, y)
        auc_perm = float(np.mean([probe_auc(V, rng.permutation(y), seed=s) for s in range(3)]))
        pacc = float(paired_ok[y == 1].mean())   # acc pareada de los pares con esa entidad
        rows[ent] = {"n_pos": int(y.sum()), "auc_video": round(auc_v, 4),
                     "auc_texto": round(auc_t, 4), "auc_perm": round(auc_perm, 4),
                     "paired_acc_subset": round(pacc, 4)}
        dec_aucs.append(auc_v); pair_accs.append(pacc)
        print(f"[{model}] {ent:<10} n={y.sum():>3}  AUC_video={auc_v:.3f}  "
              f"AUC_texto={auc_t:.3f}  AUC_perm={auc_perm:.3f}  pairedAcc={pacc:.3f}")

    corr = float(np.corrcoef(dec_aucs, pair_accs)[0, 1]) if len(dec_aucs) > 2 else None
    results[model] = {"entities": rows,
                      "mean_auc_video": round(float(np.mean(dec_aucs)), 4),
                      "corr_decodabilidad_pairedacc": round(corr, 3) if corr is not None else None}
    print(f"[{model}] media AUC_video={np.mean(dec_aucs):.3f}  "
          f"corr(decodabilidad, paired acc)={corr:+.2f}\n")

(HERE / "H3_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print("Guardado: H3_results.json")
