# -*- coding: utf-8 -*-
"""Heatmap of the fake-type x modality AUC matrix (3 models).

Black boxes mark the "matched" cells (modality semantically paired with the
lie type) — the only cells where genuine detection signal could live.
Reads audit_type_modality_results.json; writes fig_type_modality_matrix.png.
"""
import sys, json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
import experiments.analysis.mavis_analysis as ma

res = json.loads((HERE / "audit_type_modality_results.json").read_text(encoding="utf-8"))

MODELS = ["ge2", "qw2b", "qw8b"]
TYPES  = ["false_title", "false_speech", "temporal_edit", "cgi", "contradictory", "unsupported"]
FSETS  = ["tfidf_title", "tfidf_transcript", "title", "transcript", "video", "relational"]

TYPE_LABEL = {
    "false_title": "False title", "false_speech": "False speech",
    "temporal_edit": "Temporal edit", "cgi": "CGI",
    "contradictory": "Contradictory", "unsupported": "Unsupported",
}
FS_LABEL = {
    "tfidf_title": "TF-IDF\ntitle", "tfidf_transcript": "TF-IDF\ntranscript",
    "title": "Title\nemb.", "transcript": "Transcript\nemb.",
    "video": "Video\nemb.", "relational": "Relational",
}

# modality matched to each lie type (where genuine signal could live)
MATCHED = {
    "false_title":   ["tfidf_title", "title"],
    "false_speech":  ["tfidf_transcript", "transcript"],
    "temporal_edit": ["video"],
    "cgi":           ["video"],
    "contradictory": ["relational"],
    "unsupported":   ["relational"],
}

with plt.rc_context(ma._RC):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)

    for ax, model in zip(axes, MODELS):
        M = np.array([[res[model]["matrix"][t][fs] for fs in FSETS] for t in TYPES])
        im = ax.imshow(M, cmap="viridis", vmin=0.5, vmax=0.95, aspect="auto")

        for i, t in enumerate(TYPES):
            for j, fs in enumerate(FSETS):
                v = M[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if v < 0.78 else "black")
                if fs in MATCHED[t]:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor="black", linewidth=2.2))

        n_lab = [f"{TYPE_LABEL[t]} (n={res[model]['n_per_type'][t]})" for t in TYPES]
        ax.set_xticks(range(len(FSETS)), [FS_LABEL[f] for f in FSETS])
        ax.set_yticks(range(len(TYPES)), n_lab if model == "ge2" else [TYPE_LABEL[t] for t in TYPES])
        ax.set_title(f"{ma.MODEL_DISPLAY[model]} — AUC reals vs fakes with type", fontsize=11)
        ax.grid(False)

    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.01)
    cbar.set_label("ROC-AUC")
    fig.suptitle("GroundLie360 — Fake type × feature set probe matrix "
                 "(boxes = modality matched to the lie)", fontsize=12)

    out = HERE / "fig_type_modality_matrix.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out)
