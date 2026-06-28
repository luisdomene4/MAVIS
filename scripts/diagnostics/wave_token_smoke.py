"""
WAVE token smoke test.

For ~8 GroundLie360 videos (temporal_edit=1) computes embeddings with the OLD
token (broken) and the NEW token (fixed) for three modalities:
  - video:       OLD=<image>  vs NEW=<video>
  - audio:       OLD=(no token, plain AUDIO_PROMPT)  vs NEW=<audio>\n{AUDIO_PROMPT}
  - audiovisual: OLD=<image>  vs NEW=<video>

Reports:
  (a) Mean pairwise cosine between DISTINCT videos (healthy != 1.0)
  (b) Mean cosine between ADJACENT scenes of each temporal_edit video (E3 proxy)
  (c) Per-modality OLD vs NEW comparison table

Usage (from repo root on cluster):
    python scripts/diagnostics/wave_token_smoke.py \
        --data-dir   data/GroundLie360 \
        --model-dir  src/models/WAVE-7B \
        --wave-repo  src/models/wave_repo \
        --beats-path src/models/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt \
        [--index-json experiments/GroundLie360/groundlie_index_filtered.json] \
        [--n-videos 8] \
        [--max-frames 8] \
        [--max-frame-side 224] \
        [--quantize]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import gc
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_INDEX = "experiments/GroundLie360/groundlie_index_filtered.json"

VIDEO_PROMPT = "Please describe the video content for fact-checking purposes."
AUDIO_PROMPT = "Please describe the audio content for fact-checking purposes."
AV_PROMPT    = "Please describe the audio-visual content for fact-checking purposes."


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="WAVE token smoke test")
    p.add_argument("--data-dir",       required=True)
    p.add_argument("--model-dir",      required=True)
    p.add_argument("--wave-repo",      required=True)
    p.add_argument("--beats-path",     required=True)
    p.add_argument("--index-json",     default=DEFAULT_INDEX)
    p.add_argument("--n-videos",       type=int, default=8,
                   help="Number of temporal_edit=1 videos to test")
    p.add_argument("--max-frames",     type=int, default=8)
    p.add_argument("--max-frame-side", type=int, default=224)
    p.add_argument("--quantize",       action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading (copied from run_groundlie360_wave.py)
# ---------------------------------------------------------------------------

def load_wave_model(model_dir, wave_repo, beats_path, quantize=False):
    if str(wave_repo) not in sys.path:
        sys.path.insert(0, str(wave_repo))

    from transformers import BitsAndBytesConfig
    from qwenvl.data.processing_qwen2_5_omni import Qwen2_5OmniProcessor
    from qwenvl.model.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniThinkerConfig
    from qwenvl.model.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration

    print(f"Loading WAVE-7B from {model_dir} ({'INT4' if quantize else 'FP16'})")
    model_config = Qwen2_5OmniThinkerConfig.from_pretrained(str(model_dir))
    model_config.train_classify = True
    model_config.classify_type = "all_layer"
    model_config.sim_temperature = 0.02
    model_config.audio_config.beats_path = str(beats_path)
    model_config.audio_config.beats_only = False

    bnb_config = None
    if quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["beats"],
        )

    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        str(model_dir),
        config=model_config,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=bnb_config,
    )
    beats_ckpt = torch.load(str(beats_path), map_location="cpu")
    model.beats.load_state_dict(beats_ckpt["model"])
    model.eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(str(model_dir))
    print("WAVE-7B ready")
    return model, processor


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def free_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _to_numpy(v):
    if isinstance(v, torch.Tensor):
        v = v.float().cpu().numpy()
    return np.array(v, dtype=np.float32)


def _run_wave(model, inputs_dict):
    with torch.no_grad():
        inputs_dict["pred_embeds"] = True
        outputs = model(**inputs_dict)
    return _to_numpy(outputs.mllm_embeds.squeeze(0))


def _extract_audio(video_path, sample_rate=16000):
    import av
    container = av.open(video_path)
    audio_streams = [s for s in container.streams if s.type == "audio"]
    if not audio_streams:
        container.close()
        return None, None
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=sample_rate)
    frames = []
    for frame in container.decode(audio=0):
        for rf in resampler.resample(frame):
            frames.append(rf.to_ndarray()[0])
    container.close()
    if not frames:
        return None, None
    return np.concatenate(frames).astype(np.float32), sample_rate


def _extract_video_frames(video_path, max_frames):
    import av
    try:
        import decord
        decord.bridge.set_bridge("numpy")
        vr = decord.VideoReader(video_path)
        total = len(vr)
        indices = np.linspace(0, total - 1, min(max_frames, total), dtype=int)
        return list(vr.get_batch(indices).asnumpy())
    except ImportError:
        pass
    frames = []
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        total = stream.frames or sum(1 for _ in container.decode(stream))
    indices = set(np.linspace(0, total - 1, min(max_frames, total), dtype=int))
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if i in indices:
                frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) >= len(indices):
                break
    return frames


def _downscale_frame(frame, max_side):
    if not max_side or max_side <= 0:
        return frame
    h, w = frame.shape[:2]
    if max(h, w) <= max_side:
        return frame
    scale = max_side / float(max(h, w))
    new_h, new_w = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    try:
        from PIL import Image
        img = Image.fromarray(frame).resize((new_w, new_h), Image.BICUBIC)
        return np.asarray(img).astype(np.uint8, copy=False)
    except Exception:
        import torch.nn.functional as F
        t = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float()
        t = F.interpolate(t, size=(new_h, new_w), mode="bilinear", align_corners=False)
        return t.clamp(0, 255).byte().squeeze(0).permute(1, 2, 0).numpy()


def embed_video(video_path, model, processor, max_frames, max_frame_side, token):
    """token: '<video>' (correct) or '<image>' (old/broken)"""
    frames = _extract_video_frames(video_path, max_frames)
    frames = [_downscale_frame(f, max_frame_side) for f in frames]
    inputs = processor(text=[f"{token}\n{VIDEO_PROMPT}"], videos=[frames], return_tensors="pt")
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    inputs["types"] = "video"
    inputs["use_audio"] = False
    emb = _run_wave(model, inputs)
    free_vram()
    return emb


def embed_audio(video_path, model, processor, token):
    """token: '<audio>' (correct) or None (old/broken — no token prefix)"""
    audio, sr = _extract_audio(video_path)
    if audio is None:
        return None
    text = f"<audio>\n{AUDIO_PROMPT}" if token == "<audio>" else AUDIO_PROMPT
    inputs = processor(text=[text], return_tensors="pt")
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    inputs["types"] = "audio"
    inputs["use_audio"] = True
    inputs["input_raw_wav"] = [torch.tensor(audio)]
    emb = _run_wave(model, inputs)
    free_vram()
    return emb


def embed_audiovisual(video_path, model, processor, max_frames, max_frame_side, token):
    """token: '<video>' (correct) or '<image>' (old/broken)"""
    audio, sr = _extract_audio(video_path)
    if audio is None:
        return None
    frames = _extract_video_frames(video_path, max_frames)
    frames = [_downscale_frame(f, max_frame_side) for f in frames]
    inputs = processor(text=[f"{token}\n{AV_PROMPT}"], videos=[frames], return_tensors="pt")
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    inputs["types"] = "video"
    inputs["use_audio"] = True
    inputs["input_raw_wav"] = [torch.tensor(audio)]
    emb = _run_wave(model, inputs)
    free_vram()
    return emb


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def mean_pairwise_cosine(embeddings):
    """Mean cosine between all DISTINCT pairs."""
    embs = np.stack(embeddings)  # (N, D)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    normed = embs / (norms + 1e-9)
    sim_matrix = normed @ normed.T
    n = len(embeddings)
    total, count = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            total += sim_matrix[i, j]
            count += 1
    return total / count if count > 0 else float("nan")


def mean_adjacent_scene_cosine(scene_embeddings_list):
    """
    scene_embeddings_list: list of lists, each inner list is scene embeddings for one video.
    Returns mean cosine between adjacent scenes across all videos.
    """
    sims = []
    for scene_embs in scene_embeddings_list:
        if len(scene_embs) < 2:
            continue
        for a, b in zip(scene_embs[:-1], scene_embs[1:]):
            a_n = a / (np.linalg.norm(a) + 1e-9)
            b_n = b / (np.linalg.norm(b) + 1e-9)
            sims.append(float(a_n @ b_n))
    return float(np.mean(sims)) if sims else float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    wave_repo = Path(args.wave_repo)
    beats_path = Path(args.beats_path)

    # Load index, pick n_videos temporal_edit=1 videos
    with open(args.index_json, encoding="utf-8") as f:
        index = json.load(f)
    candidates = [e for e in index if e.get("temporal_edit") == 1 and len(e.get("scene_frames", [])) >= 3]
    # Prefer variety: pick videos with different lengths of scene_frames
    candidates.sort(key=lambda e: -len(e.get("scene_frames", [])))
    videos = candidates[:args.n_videos]
    print(f"Selected {len(videos)} temporal_edit=1 videos with >=3 scenes.")
    for v in videos:
        print(f"  {v['raw_id']}  scenes={len(v['scene_frames'])}")

    # Build absolute video paths
    for v in videos:
        v["_abs_path"] = str(data_dir / v["video_path"])
        if not os.path.exists(v["_abs_path"]):
            print(f"  WARNING: file not found: {v['_abs_path']}")

    # Load model once
    model, processor = load_wave_model(model_dir, wave_repo, beats_path, args.quantize)

    mf = args.max_frames
    ms = args.max_frame_side

    # -------------------------------------------------------------------
    # Collect embeddings: global (per video) and per-scene (for E3 proxy)
    # -------------------------------------------------------------------
    results = {mod: {"old": [], "new": []} for mod in ["video", "audio", "av"]}
    scene_results = {"video": {"old": [], "new": []}}  # per-video lists of scene embs

    for idx, entry in enumerate(videos):
        vid = entry["_abs_path"]
        rid = entry["raw_id"]
        scene_frame_counts = entry["scene_frames"]  # list of int (frames per scene)
        n_scenes = len(scene_frame_counts)
        print(f"\n[{idx+1}/{len(videos)}] {rid}  ({n_scenes} scenes)")

        # --- VIDEO (global) ---
        print("  video OLD (<image>)...")
        try:
            old_vid = embed_video(vid, model, processor, mf, ms, "<image>")
            results["video"]["old"].append(old_vid)
        except Exception as e:
            print(f"    ERROR: {e}")
            old_vid = None

        print("  video NEW (<video>)...")
        try:
            new_vid = embed_video(vid, model, processor, mf, ms, "<video>")
            results["video"]["new"].append(new_vid)
        except Exception as e:
            print(f"    ERROR: {e}")
            new_vid = None

        # --- VIDEO SCENE EMBEDDINGS (E3 proxy) ---
        # Use evenly-spaced scene offsets to sample frames representative of each scene
        old_scene_embs, new_scene_embs = [], []
        total_frames_est = sum(scene_frame_counts)
        cumulative = 0
        print(f"  video scenes ({n_scenes})...")
        for s_idx, s_nframes in enumerate(scene_frame_counts):
            # Compute approximate time offset for scene midpoint
            mid_frame = cumulative + s_nframes // 2
            frac_start = cumulative / max(total_frames_est, 1)
            frac_end   = (cumulative + s_nframes) / max(total_frames_est, 1)
            cumulative += s_nframes

            # Extract a short clip around this scene by sampling full video
            # and keeping only the fraction window — approximate but works
            # without TransNetV2. Just embed the full video with offset hint
            # via sampling: we linspace within [frac_start, frac_end].
            try:
                import av as _av
                container = _av.open(vid)
                stream = container.streams.video[0]
                total_v = stream.frames or 1
                container.close()
                scene_start = int(frac_start * total_v)
                scene_end   = max(scene_start + 1, int(frac_end * total_v))

                # Read only frames in [scene_start, scene_end]
                scene_frames_raw = []
                with _av.open(vid) as c:
                    vs = c.streams.video[0]
                    for fi, frame in enumerate(c.decode(vs)):
                        if fi >= scene_start and fi < scene_end:
                            scene_frames_raw.append(frame.to_ndarray(format="rgb24"))
                        if fi >= scene_end:
                            break
                if not scene_frames_raw:
                    continue
                # Subsample to max_frames
                idxs = np.linspace(0, len(scene_frames_raw)-1, min(mf, len(scene_frames_raw)), dtype=int)
                scene_frames_sampled = [_downscale_frame(scene_frames_raw[i], ms) for i in idxs]

                inputs_old = processor(text=[f"<image>\n{VIDEO_PROMPT}"], videos=[scene_frames_sampled], return_tensors="pt")
                inputs_old = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs_old.items()}
                inputs_old["types"] = "video"
                inputs_old["use_audio"] = False
                old_scene_embs.append(_run_wave(model, inputs_old))
                free_vram()

                inputs_new = processor(text=[f"<video>\n{VIDEO_PROMPT}"], videos=[scene_frames_sampled], return_tensors="pt")
                inputs_new = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs_new.items()}
                inputs_new["types"] = "video"
                inputs_new["use_audio"] = False
                new_scene_embs.append(_run_wave(model, inputs_new))
                free_vram()

            except Exception as e:
                print(f"    scene {s_idx} ERROR: {e}")

        if old_scene_embs:
            scene_results["video"]["old"].append(old_scene_embs)
        if new_scene_embs:
            scene_results["video"]["new"].append(new_scene_embs)

        # --- AUDIO ---
        print("  audio OLD (no token)...")
        try:
            old_aud = embed_audio(vid, model, processor, token=None)
            if old_aud is not None:
                results["audio"]["old"].append(old_aud)
        except Exception as e:
            print(f"    ERROR: {e}")

        print("  audio NEW (<audio>)...")
        try:
            new_aud = embed_audio(vid, model, processor, token="<audio>")
            if new_aud is not None:
                results["audio"]["new"].append(new_aud)
        except Exception as e:
            print(f"    ERROR: {e}")

        # --- AUDIOVISUAL ---
        print("  audiovisual OLD (<image>)...")
        try:
            old_av = embed_audiovisual(vid, model, processor, mf, ms, "<image>")
            if old_av is not None:
                results["av"]["old"].append(old_av)
        except Exception as e:
            print(f"    ERROR: {e}")

        print("  audiovisual NEW (<video>)...")
        try:
            new_av = embed_audiovisual(vid, model, processor, mf, ms, "<video>")
            if new_av is not None:
                results["av"]["new"].append(new_av)
        except Exception as e:
            print(f"    ERROR: {e}")

    # -------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("WAVE TOKEN SMOKE TEST — RESULTS")
    print("=" * 70)
    print(f"{'Metric':<45} {'OLD (broken)':>14} {'NEW (fixed)':>12}")
    print("-" * 70)

    def report_pairwise(mod_key, label):
        old_embs = results[mod_key]["old"]
        new_embs = results[mod_key]["new"]
        if len(old_embs) >= 2:
            old_sim = mean_pairwise_cosine(old_embs)
        else:
            old_sim = float("nan")
        if len(new_embs) >= 2:
            new_sim = mean_pairwise_cosine(new_embs)
        else:
            new_sim = float("nan")
        print(f"{label:<45} {old_sim:>14.6f} {new_sim:>12.6f}")
        return old_sim, new_sim

    print("\n(a) Mean pairwise cosine between DISTINCT videos (want: old~1.0, new<<1.0)")
    report_pairwise("video", "  Video global (pairwise)")
    report_pairwise("audio", "  Audio global (pairwise)")
    report_pairwise("av",    "  Audiovisual global (pairwise)")

    print("\n(b) Mean adjacent-scene cosine — E3 proxy (want: old~1.0, new<<1.0)")
    for token_key, label in [("old", "  Video scenes OLD (<image>)"), ("new", "  Video scenes NEW (<video>)")]:
        val = mean_adjacent_scene_cosine(scene_results["video"][token_key])
        print(f"{label:<45} {val:>14.6f}")

    print("\n(c) Audio token effect (same as (a) audio row above)")
    old_n = len(results["audio"]["old"])
    new_n = len(results["audio"]["new"])
    print(f"  Audio embeddings computed: OLD={old_n}  NEW={new_n}")

    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDE")
    print("  OLD broken token: pairwise cosine near 1.0 = all embeddings collapsed")
    print("  NEW fixed token:  pairwise cosine << 1.0 (e.g. 0.3-0.8) = discriminative")
    print("  Adjacent scenes:  OLD should be ~1.0000; NEW should be <0.9999 or lower")
    print("  If NEW is still ~1.0: report to Opus — model may behave differently than expected")
    print("=" * 70)


if __name__ == "__main__":
    main()
