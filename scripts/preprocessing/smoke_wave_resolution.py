#!/usr/bin/env python3
"""
WAVE Resolution Smoke Test
===========================
Tests WAVE-7B video embeddings at different frame resolutions to find the
highest resolution that fits in ~12 GB VRAM (shard:12000) without OOM.

For each resolution, processes --limit videos and reports:
  - VRAM peak usage
  - Whether processing succeeded or OOM'd
  - E1 accuracy (sim(video, real_title) > sim(video, fake_title))
  - Average margin

This tells us whether re-running the full pipeline at higher resolution is worth it.
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


def parse_args():
    p = argparse.ArgumentParser(description="WAVE resolution smoke test")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--wave-repo", required=True)
    p.add_argument("--beats-path", required=True)
    p.add_argument("--limit", type=int, default=5, help="Number of videos to test per resolution")
    return p.parse_args()


def vram_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated(0) / 1024**2
    return 0


def vram_peak_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated(0) / 1024**2
    return 0


def free_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    wave_repo = Path(args.wave_repo)
    model_dir = Path(args.model_dir)
    beats_path = Path(args.beats_path)

    # --- Load model (quantized, same as production) ---
    print("=" * 70)
    print("WAVE RESOLUTION SMOKE TEST")
    print("=" * 70)

    if str(wave_repo) not in sys.path:
        sys.path.insert(0, str(wave_repo))

    from transformers import BitsAndBytesConfig
    from qwenvl.data.processing_qwen2_5_omni import Qwen2_5OmniProcessor
    from qwenvl.model.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniThinkerConfig
    from qwenvl.model.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["beats"],
    )

    model_config = Qwen2_5OmniThinkerConfig.from_pretrained(str(model_dir))
    model_config.train_classify = True
    model_config.classify_type = "all_layer"
    model_config.sim_temperature = 0.02
    model_config.audio_config.beats_path = str(beats_path)
    model_config.audio_config.beats_only = False

    print(f"Loading WAVE-7B (INT4 quantized)...")
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        str(model_dir),
        config=model_config,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=bnb_config,
    )

    print("Loading BEATs checkpoint...")
    beats_ckpt = torch.load(str(beats_path), map_location="cpu")
    print("Loading BEATs state dict...")
    model.beats.load_state_dict(beats_ckpt["model"])
    print("Setting model to eval...")
    model.eval()

    print("Loading processor...")
    processor = Qwen2_5OmniProcessor.from_pretrained(str(model_dir))

    print(f"Model loaded. VRAM: {vram_mb():.0f} MB")
    model_vram = vram_mb()

    # --- Load test index ---
    with open(data_dir / "test_index.json", encoding="utf-8") as f:
        test_index = json.load(f)

    # Find available videos
    available = []
    for item in test_index:
        raw_id = item["raw_id"]
        vpath = data_dir / f"test_video/{raw_id}.mp4"
        if not vpath.exists():
            vpath = data_dir / f"{raw_id}.mp4"
        if vpath.exists():
            item["_abs_video"] = str(vpath)
            available.append(item)
        if len(available) >= args.limit:
            break

    print(f"Test videos: {len(available)}")
    if not available:
        print("ERROR: No videos found")
        sys.exit(1)

    # --- Helper functions ---
    def _to_numpy(v):
        if isinstance(v, torch.Tensor):
            v = v.float().cpu().numpy()
        return np.array(v, dtype=np.float32)

    def _normalize(v):
        v = _to_numpy(v)
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-9 else v

    def cosine_sim(a, b):
        an = _normalize(a)
        bn = _normalize(b)
        return float(np.dot(an, bn))

    def get_text_embedding(text, prefix=""):
        full = f"{prefix}{text}" if prefix else text
        inputs = processor(
            text=[full], return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        inputs["types"] = "text"
        with torch.no_grad():
            inputs["pred_embeds"] = True
            outputs = model(**inputs)
        return _to_numpy(outputs.mllm_embeds.squeeze(0))

    def _extract_frames(video_path, max_frames=16):
        """Extract frames using decord (fast) or PyAV fallback."""
        try:
            import decord
            decord.bridge.set_bridge("numpy")
            vr = decord.VideoReader(video_path)
            total = len(vr)
            if total > max_frames:
                indices = np.linspace(0, total - 1, max_frames, dtype=int)
            else:
                indices = np.arange(total)
            return list(vr.get_batch(indices).asnumpy())
        except ImportError:
            import av
            def _count(path):
                with av.open(path) as c:
                    s = c.streams.video[0]
                    if s.frames and s.frames > 0:
                        return s.frames
                    return sum(1 for _ in c.decode(s))
            total = _count(video_path)
            if total == 0:
                raise ValueError(f"No frames in {video_path}")
            if total > max_frames:
                indices = set(np.linspace(0, total - 1, max_frames, dtype=int))
            else:
                indices = set(np.arange(total))
            frames = []
            with av.open(video_path) as container:
                stream = container.streams.video[0]
                for i, frame in enumerate(container.decode(stream)):
                    if i in indices:
                        frames.append(frame.to_ndarray(format="rgb24"))
                    if len(frames) >= len(indices):
                        break
            return frames

    def _downscale(frame, max_side):
        if max_side is None or max_side <= 0:
            return frame
        h, w = frame.shape[:2]
        if max(h, w) <= max_side:
            return frame
        scale = max_side / float(max(h, w))
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        from PIL import Image
        img = Image.fromarray(frame)
        img = img.resize((new_w, new_h), Image.BICUBIC)
        return np.asarray(img).astype(np.uint8)

    def get_video_embedding(video_path, max_frames=8, max_side=None):
        frames = _extract_frames(video_path, max_frames)
        if max_side and max_side > 0:
            frames = [_downscale(f, max_side) for f in frames]
        # Report frame dimensions
        if frames:
            h, w = frames[0].shape[:2]
            print(f"      Frame size: {w}x{h}, {len(frames)} frames")

        inputs = processor(
            text=["<image>\nPlease describe the video."],
            videos=[frames],
            return_tensors="pt",
        )
        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        inputs["types"] = "video"
        inputs["use_audio"] = False

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            inputs["pred_embeds"] = True
            outputs = model(**inputs)

        emb = _to_numpy(outputs.mllm_embeds.squeeze(0))
        peak = vram_peak_mb()
        free_vram()
        return emb, peak

    # --- Pre-compute text embeddings (same for all resolutions) ---
    print("\n--- Pre-computing text embeddings ---")
    text_embs = {}
    for item in available:
        rid = item["raw_id"]
        text_embs[rid] = {
            "real": get_text_embedding(item["title"]),
            "fake": get_text_embedding(item["fake_title"]),
        }
    print(f"  Done: {len(text_embs)} pairs")

    # --- Test each resolution ---
    RESOLUTIONS = [
        ("224px (current)", 224),
        ("384px", 384),
        ("512px", 512),
        ("768px", 768),
        ("native (no limit)", 0),
    ]

    print("\n" + "=" * 70)
    print("RESOLUTION TEST RESULTS")
    print("=" * 70)

    for res_name, max_side in RESOLUTIONS:
        print(f"\n{'─' * 60}")
        print(f"Testing: {res_name} (max_side={max_side})")
        print(f"{'─' * 60}")

        correct = 0
        total = 0
        margins = []
        peak_vram = 0
        oom = False

        for item in available:
            rid = item["raw_id"]
            vpath = item["_abs_video"]
            try:
                emb_v, peak = get_video_embedding(vpath, max_frames=8, max_side=max_side if max_side > 0 else None)
                peak_vram = max(peak_vram, peak)

                sim_real = cosine_sim(emb_v, text_embs[rid]["real"])
                sim_fake = cosine_sim(emb_v, text_embs[rid]["fake"])
                margin = sim_real - sim_fake
                margins.append(margin)

                is_correct = sim_real > sim_fake
                if is_correct:
                    correct += 1
                total += 1

                status = "✅" if is_correct else "❌"
                print(f"    {status} {rid[:12]}… real={sim_real:.4f} fake={sim_fake:.4f} margin={margin:+.4f} peak={peak:.0f}MB")

            except torch.cuda.OutOfMemoryError:
                print(f"    💥 OOM on {rid[:12]}… — skipping this and higher resolutions")
                oom = True
                free_vram()
                break
            except Exception as e:
                print(f"    ⚠️  Error {rid[:12]}…: {e}")
                free_vram()

        # Summary
        if total > 0:
            acc = correct / total
            avg_margin = np.mean(margins)
            std_margin = np.std(margins)
            print(f"\n  📊 SUMMARY [{res_name}]:")
            print(f"     Accuracy:   {acc:.2%} ({correct}/{total})")
            print(f"     Avg margin: {avg_margin:+.6f}")
            print(f"     Std margin: {std_margin:.6f}")
            print(f"     Peak VRAM:  {peak_vram:.0f} MB (model baseline: {model_vram:.0f} MB)")
            if peak_vram > 11500:
                print(f"     ⚠️  CLOSE TO 12GB LIMIT")
        else:
            print(f"\n  📊 SUMMARY [{res_name}]: No videos processed")

        if oom:
            print(f"  💥 OOM at this resolution — stopping higher tests")
            break

        free_vram()

    print("\n" + "=" * 70)
    print("SMOKE TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
