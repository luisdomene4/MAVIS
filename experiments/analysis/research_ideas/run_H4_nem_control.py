# -*- coding: utf-8 -*-
"""H4/N2 — Control de vídeo aleatorio para M3A B (NEM).

QUÉ HACE
  M3A B reporta que, dado un vídeo V y dos resúmenes (el real T y el NEM Tn con
  una entidad cambiada), cos(V,T) > cos(V,Tn) en el 76–93 % de los casos.
  La pregunta: ¿eso es coherencia CON EL VÍDEO, o el texto NEM simplemente
  "suena raro" y se aleja de todo? Tres mediciones:

  1. acc_own   = mean[ cos(V_propio, T) > cos(V_propio, Tn) ]   ← lo reportado
  2. acc_rand  = lo mismo con un VÍDEO DE OTRO ejemplo elegido al azar
                 (5 permutaciones). Si acc_rand ≈ acc_own, el vídeo no pinta
                 nada: la señal es un artefacto del texto. Si acc_rand ≈ 0.5,
                 la señal es video-grounded.
  3. tfidf_auc = ¿un clasificador de BOLSA DE PALABRAS distingue el resumen
                 real del NEM? (GroupKFold por vídeo). Mide si el generador
                 NEM deja huella léxica (entidades insertadas repetidas, como
                 el "Biden" de FakeVV).

  Complemento: cos(T, Tn) — cuánto mueve la manipulación el embedding de texto.

Salida: H4_results.json + consola.
"""
import sys, json
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import experiments.analysis.mavis_analysis as ma

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
NEM_SUBS = ["person", "location", "organization", "complete"]
N_PERMS = 5


def l2rows(M):
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n[n == 0] = 1
    return M / n


results = {}

# ---------- 1+2: pareado con vídeo propio vs vídeo aleatorio (por modelo) ----------
for model in ["qw2b", "qw8b"]:
    conn = ma.connect(ma.db_path("M3A", model))
    V = ma.load_globals(conn, "video")
    T = ma.load_globals(conn, "text_summary")
    res = {}
    for sub in NEM_SUBS:
        Tn = ma.load_globals(conn, f"text_nem_{sub}")
        ids = sorted(r for r in V if r in T and r in Tn)
        if not ids:
            continue
        Vm = l2rows(np.stack([V[r] for r in ids]).astype(np.float64))
        Tm = l2rows(np.stack([T[r] for r in ids]).astype(np.float64))
        Nm = l2rows(np.stack([Tn[r] for r in ids]).astype(np.float64))

        own_real = np.einsum("ij,ij->i", Vm, Tm)
        own_fake = np.einsum("ij,ij->i", Vm, Nm)
        acc_own = float((own_real > own_fake).mean())

        accs_rand = []
        rng = np.random.default_rng(0)
        for _ in range(N_PERMS):
            p = rng.permutation(len(ids))
            # evitar coincidencias identidad (vídeo propio) en la permutación
            p = np.where(p == np.arange(len(ids)), (p + 1) % len(ids), p)
            Vr = Vm[p]
            accs_rand.append(float((np.einsum("ij,ij->i", Vr, Tm) >
                                    np.einsum("ij,ij->i", Vr, Nm)).mean()))
        acc_rand = float(np.mean(accs_rand))
        cos_t_tn = float(np.einsum("ij,ij->i", Tm, Nm).mean())

        res[sub] = {"n": len(ids), "acc_own": round(acc_own, 4),
                    "acc_rand_video": round(acc_rand, 4),
                    "cos_T_Tn_mean": round(cos_t_tn, 4)}
        print(f"[{model}] NEM-{sub:<13} N={len(ids):>5}  acc_own={acc_own:.3f}  "
              f"acc_VIDEO-ALEATORIO={acc_rand:.3f}  cos(T,Tn)={cos_t_tn:.3f}")
    conn.close()
    results[model] = res
    print()

# ---------- 3: control léxico TF-IDF (independiente del encoder) ----------
import sqlite3
conn = sqlite3.connect(str(ma.db_path("M3A", "qw2b")))
summaries = dict(conn.execute("SELECT raw_id, summary FROM m3a_meta"))
nem_rows = conn.execute("SELECT raw_id, subtype, text FROM m3a_nem").fetchall()
conn.close()
nem_texts = {}
for rid, sub, txt in nem_rows:
    nem_texts.setdefault(sub, {})[rid] = txt

results["tfidf_text_control"] = {}
print("[control léxico] TF-IDF resumen real vs NEM (GroupKFold por vídeo):")
for sub in NEM_SUBS:
    d = nem_texts.get(sub, {})
    ids = sorted(r for r in d if r in summaries and summaries[r] and d[r])
    texts, y, g = [], [], []
    for r in ids:
        texts += [summaries[r], d[r]]
        y += [0, 1]
        g += [r, r]
    pipe = Pipeline([("tfidf", TfidfVectorizer(lowercase=True, sublinear_tf=True,
                                               ngram_range=(1, 2), min_df=2)),
                     ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                                random_state=0))])
    proba = cross_val_predict(pipe, texts, np.array(y), cv=GroupKFold(5),
                              groups=np.array(g), method="predict_proba")
    auc = float(roc_auc_score(y, proba[:, 1]))
    results["tfidf_text_control"][sub] = {"n_pairs": len(ids), "tfidf_auc": round(auc, 4)}
    print(f"   NEM-{sub:<13} n_pares={len(ids):>5}  TF-IDF AUC={auc:.4f}")

(HERE / "H4_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print("\nGuardado: H4_results.json")
