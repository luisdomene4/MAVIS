# -*- coding: utf-8 -*-
"""H6 — Transferencia cross-dataset: ¿qué feature set generaliza entre datasets?

QUÉ HACE
  Entrena un clasificador real/fake en un dataset COMPLETO (p. ej. GroundLie360)
  y lo evalúa, sin tocarlo, en el otro (FakeVV) — y viceversa. Si un feature set
  aprende un ATAJO del dataset (estilo Snopes, entidades Biden), su AUC debe
  desplomarse a ~0.5 al cambiar de dataset; si aprende una señal de coherencia
  genuina, debe retener parte del AUC.

TAREA COMÚN (para que el clasificador sea transferible)
  Ejemplo = (embedding de vídeo V, embedding de título T) → ¿es un par fake?
  - GroundLie360: T = título del vídeo; y = binary_label (1 049 fake / 995 real).
  - FakeVV: cada vídeo aporta 2 ejemplos: (V, T_real)→0 y (V, T_fake)→1.

FEATURE SETS (idénticos en ambos datasets)
  video    = V                  → ¿transfiere el "aspecto" de los vídeos fake?
  title    = T                  → ¿transfiere el estilo/léxico del título fake?
  concat   = [V, T]             → techo de contenido
  relational = [cos(V,T), ||V−T||]  → SOLO coherencia (2 números)
  hadamard_full = V⊙T sin PCA   → similitud bilineal aprendida (Exp. 3b)

MÉTRICA
  AUC en el dataset de test (0.5 = azar). Referencias: AUC intra-dataset
  (entrenar y testear en el mismo, con CV) y zero-shot cos.

Salida: H6_results.json + tabla por consola.
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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

MODELS = ["ge2", "qw2b", "qw8b"]
HERE = Path(__file__).resolve().parent


def l2rows(M):
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n[n == 0] = 1
    return M / n


def load_groundlie(model):
    """(V, T, y, groups) para GroundLie360: un ejemplo por vídeo."""
    conn = ma.connect(ma.db_path("GroundLie360", model))
    vid = ma.load_globals(conn, "video")
    tit = ma.load_globals(conn, "text_title")
    lab = ma.load_labels(conn)
    conn.close()
    ids = sorted(set(vid) & set(tit) & set(lab.index))
    lab = lab.loc[ids]
    ids = [i for i in ids if lab.loc[i, "binary_label"] in (0, 1)]
    V = l2rows(np.stack([vid[i] for i in ids]).astype(np.float64))
    T = l2rows(np.stack([tit[i] for i in ids]).astype(np.float64))
    y = lab.loc[ids, "binary_label"].astype(int).to_numpy()
    return V, T, y, np.array(ids)


def load_fakevv(model):
    """(V, T, y, groups) para FakeVV: dos ejemplos (real/fake) por vídeo."""
    conn = ma.connect(ma.db_path("FakeVV", model))
    vid = ma.load_globals(conn, "video")
    trl = ma.load_globals(conn, "text_real")
    tfk = ma.load_globals(conn, "text_fake")
    conn.close()
    ids = sorted(set(vid) & set(trl) & set(tfk))
    V_rows, T_rows, y_rows, g_rows = [], [], [], []
    for r in ids:
        v = vid[r].astype(np.float64)
        for t, lab in [(trl[r], 0), (tfk[r], 1)]:
            V_rows.append(v)
            T_rows.append(t.astype(np.float64))
            y_rows.append(lab)
            g_rows.append(r)
    return (l2rows(np.stack(V_rows)), l2rows(np.stack(T_rows)),
            np.array(y_rows), np.array(g_rows))


def feature_sets(V, T):
    cos = np.einsum("ij,ij->i", V, T)
    dist = np.linalg.norm(V - T, axis=1)
    return {
        "video":         V,
        "title":         T,
        "concat":        np.concatenate([V, T], axis=1),
        "relational":    np.stack([cos, dist], axis=1),
        "hadamard_full": V * T,
    }, cos


def make_pipe(fs_name, dim):
    if fs_name == "hadamard_full":
        return Pipeline([("sc", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=3000, class_weight="balanced",
                                                    random_state=0, C=0.1))])
    steps = [("sc", StandardScaler())]
    if dim > 512:
        steps.append(("pca", PCA(n_components=128, random_state=0)))
    steps.append(("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                            random_state=0)))
    return Pipeline(steps)


results = {}
for model in MODELS:
    print(f"\n{'='*72}\nMODELO {model}\n{'='*72}")
    data = {"GL": load_groundlie(model), "FVV": load_fakevv(model)}
    fsets, coss, cvs = {}, {}, {}
    for ds, (V, T, y, g) in data.items():
        fsets[ds], coss[ds] = feature_sets(V, T)
        # CV intra-dataset: GroupKFold en FVV (2 ejemplos/vídeo), StratifiedKFold en GL
        cvs[ds] = (GroupKFold(5), g) if ds == "FVV" else (StratifiedKFold(5, shuffle=True, random_state=0), None)

    res = {"zs_cos": {ds: round(float(roc_auc_score(data[ds][2], -coss[ds])), 4) for ds in data}}
    for fs_name in ["video", "title", "concat", "relational", "hadamard_full"]:
        row = {}
        # referencia intra-dataset (CV)
        for ds in data:
            X, y = fsets[ds][fs_name], data[ds][2]
            cv, groups = cvs[ds]
            kw = dict(cv=cv, method="predict_proba")
            if groups is not None:
                kw["groups"] = groups
            proba = cross_val_predict(make_pipe(fs_name, X.shape[1]), X, y, **kw)
            row[f"intra_{ds}"] = round(float(roc_auc_score(y, proba[:, 1])), 4)
        # transferencia (train completo en A, test completo en B)
        for a, b in [("GL", "FVV"), ("FVV", "GL")]:
            Xa, ya = fsets[a][fs_name], data[a][2]
            Xb, yb = fsets[b][fs_name], data[b][2]
            pipe = make_pipe(fs_name, Xa.shape[1]).fit(Xa, ya)
            auc = float(roc_auc_score(yb, pipe.predict_proba(Xb)[:, 1]))
            row[f"{a}->{b}"] = round(auc, 4)
        res[fs_name] = row
        print(f"  {fs_name:<14} intra GL={row['intra_GL']:.3f}  intra FVV={row['intra_FVV']:.3f}  "
              f"GL->FVV={row['GL->FVV']:.3f}  FVV->GL={row['FVV->GL']:.3f}")
    print(f"  {'zero-shot cos':<14} GL={res['zs_cos']['GL']:.3f}  FVV={res['zs_cos']['FVV']:.3f}")
    results[model] = res

(HERE / "H6_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print("\nGuardado: H6_results.json")
