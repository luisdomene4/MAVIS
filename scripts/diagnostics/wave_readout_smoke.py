"""
Smoke test: compara la extracción de embedding de vídeo de WAVE-7B de dos formas
para diagnosticar la anisotropia (~0.90) en `video`:

  OLD  -> texto crudo  `<|VIDEO|>\\n<prompt>`              (lo que hace el pipeline ahora)
  NEW  -> apply_chat_template terminando en `<|im_end|>`    (lo que se uso en entrenamiento)

Replica data_qwen.py:313-316 para el lado documento (vídeo):
  text = processor.apply_chat_template([{user, [video, text]}], add_generation_prompt=False)
  text = text.split("<|im_start|>user\\n")[-1].strip()   # termina en <|im_end|>

Mide el coseno medio entre pares de N vídeos distintos para cada variante.
Si NEW << OLD, el bug es la posicion de lectura (falta <|im_end|> + chat template).

NO escribe en ninguna DB.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "experiments"))

from run_m3a_wave import load_wave_model, _extract_video_frames, _downscale_frame, _run_wave

OLD_PROMPT = "Please describe the video content for fact-checking purposes."
TRAIN_PROMPT = "Please describe the video."   # exacto del entrenamiento (scripts/ret_*.json)


def emb_old(frames, model, processor):
    inputs = processor(text=[f"<|VIDEO|>\n{OLD_PROMPT}"], videos=[frames], return_tensors="pt")
    inputs = {k: (v.to(model.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    inputs["types"] = "video"; inputs["use_audio"] = False
    return _run_wave(model, inputs)


def emb_new(frames, model, processor, prompt=TRAIN_PROMPT):
    conv = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(conv, add_generation_prompt=False, tokenize=False)
    if isinstance(text, list):
        text = text[0]
    text = text.split("<|im_start|>user\n")[-1].strip()   # termina en <|im_end|>
    inputs = processor(text=[text], videos=[frames], return_tensors="pt")
    inputs = {k: (v.to(model.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    inputs["types"] = "video"; inputs["use_audio"] = False
    return _run_wave(model, inputs), text


def mean_pairwise_cos(mat):
    X = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
    S = X @ X.T
    iu = np.triu_indices(len(X), k=1)
    return float(S[iu].mean()), float(S[iu].min()), float(S[iu].max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--wave-repo", required=True)
    ap.add_argument("--beats-path", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--index-json", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--max-frames", type=int, default=16)
    ap.add_argument("--max-frame-side", type=int, default=256)
    ap.add_argument("--quantize", action="store_true")
    args = ap.parse_args()

    idx = json.load(open(args.index_json))
    paths, taken = [], 0
    for it in idx:
        p = Path(args.data_dir) / it["video_path"]
        if p.exists():
            paths.append(str(p)); taken += 1
        if taken >= args.n:
            break
    print(f"Vídeos: {len(paths)}")

    model, processor = load_wave_model(args.model_dir, args.wave_repo, args.beats_path, args.quantize)

    old_embs, new_embs = [], []
    shown = False
    for p in paths:
        frames = _extract_video_frames(p, args.max_frames)
        if args.max_frame_side > 0:
            frames = [_downscale_frame(f, args.max_frame_side) for f in frames]
        old_embs.append(emb_old(frames, model, processor))
        ne, txt = emb_new(frames, model, processor)
        new_embs.append(ne)
        if not shown:
            print("NEW input text (repr):", repr(txt)[:200], "...")
            print("  termina en <|im_end|>:", txt.rstrip().endswith("<|im_end|>"))
            shown = True

    old = np.vstack(old_embs); new = np.vstack(new_embs)
    om, omn, omx = mean_pairwise_cos(old)
    nm, nmn, nmx = mean_pairwise_cos(new)
    print("\n==== RESULTADO (coseno entre vídeos distintos; mas bajo = mas discriminativo) ====")
    print(f"OLD (crudo, sin <|im_end|>):  mean={om:.4f}  min={omn:.4f}  max={omx:.4f}")
    print(f"NEW (chat template + im_end): mean={nm:.4f}  min={nmn:.4f}  max={nmx:.4f}")
    print(f"\nMejora anisotropia: {om - nm:+.4f} (positivo = NEW discrimina mejor)")


if __name__ == "__main__":
    main()
