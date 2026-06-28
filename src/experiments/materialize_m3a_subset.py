"""
MAVIS — M3A subset materializer.

Walks the stratified-ordered index (m3a_index_full_ordered.json) IN ORDER and, for each
video until --target kept ones: extracts the mp4 from its outlet zip, ffprobes
duration + audio, keeps it if <=120s, and attaches the texts (Summary, NEM, MTG). Writes
a GroundLie360-compatible index `m3a_index_<N>.json` consumed by the run_m3a_* scripts.

Crash-safe & incremental: a probe cache (m3a_probe_cache.json) records duration/has_audio
per video so re-runs (e.g. larger --target) never re-probe; rejected (>120s) videos are
deleted to save space but stay cached. Kept videos live at
    <m3a-dir>/videos/<outlet>/<id>.mp4

Run on the cluster:
    python src/experiments/materialize_m3a_subset.py \
        --m3a-dir   data/M3A \
        --index     experiments/M3A/m3a_index_full_ordered.json \
        --target    2000 \
        [--ffprobe  ~/miniconda3/envs/tfg2/bin/ffprobe] \
        [--max-dur  120]

Output:
    experiments/M3A/m3a_index_<target>.json   -- list of materialized entries
    experiments/M3A/m3a_probe_cache.json      -- {raw_id: [duration, has_audio]}
"""

import argparse
import json
import os
import subprocess
import zipfile

NEM_TYPES = ["Person", "Location", "Organization", "Complete"]


def ffprobe_duration_audio(ffprobe, path):
    """Return (duration_seconds | None, has_audio bool)."""
    try:
        out = subprocess.run(
            [ffprobe, "-v", "quiet", "-print_format", "json", "-show_streams", path],
            capture_output=True, text=True, timeout=60,
        )
        streams = json.loads(out.stdout).get("streams", [])
        dur, has_audio, has_video = None, False, False
        for s in streams:
            if s.get("codec_type") == "video":
                has_video = True
                if dur is None and s.get("duration"):
                    dur = float(s["duration"])
            if s.get("codec_type") == "audio":
                has_audio = True
        # fall back to container duration if no stream duration
        if dur is None:
            fmt_dur = json.loads(out.stdout).get("format", {}).get("duration")
            if fmt_dur:
                dur = float(fmt_dur)
        return (dur if has_video else None), has_audio
    except Exception:
        return None, False


class TextStore:
    """Lazily loads + caches per-outlet Summary / NEM / MTG json dicts."""

    def __init__(self, m3a_dir):
        self.m3a = m3a_dir
        self._summary = {}
        self._nem = {t: {} for t in NEM_TYPES}
        self._mtg = {}

    def _load(self, zippath, inner_outlet):
        with zipfile.ZipFile(zippath) as z:
            name = next((n for n in z.namelist()
                         if os.path.basename(n) == f"{inner_outlet}.json"), None)
            if name is None:
                return {}
            with z.open(name) as f:
                return json.load(f)

    def summary(self, outlet, vid):
        if outlet not in self._summary:
            self._summary[outlet] = self._load(
                os.path.join(self.m3a, "Text", "Summary.zip"), outlet)
        return self._summary[outlet].get(vid)

    def nem(self, ntype, outlet, vid):
        if outlet not in self._nem[ntype]:
            self._nem[ntype][outlet] = self._load(
                os.path.join(self.m3a, "NEM", f"{ntype}.zip"), outlet)
        return self._nem[ntype][outlet].get(vid)

    def mtg(self, outlet, vid):
        if outlet not in self._mtg:
            self._mtg[outlet] = self._load(
                os.path.join(self.m3a, "MTG", "Evaluation set", "model-generated.zip"), outlet)
        return self._mtg[outlet].get(vid)


def extract_video(m3a_dir, outlet, vid, dest):
    """Extract a single mp4 from its outlet zip to dest. Returns True on success."""
    zippath = os.path.join(m3a_dir, "Video", f"{outlet}.zip")
    inner = f"{outlet}/{vid}.mp4"
    try:
        with zipfile.ZipFile(zippath) as z:
            with z.open(inner) as src, open(dest, "wb") as out:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
        return True
    except Exception as exc:
        print(f"  extract FAIL {outlet}/{vid}: {exc}", flush=True)
        if os.path.exists(dest):
            os.remove(dest)
        return False


def main():
    ap = argparse.ArgumentParser(description="Materialize an ordered <=120s M3A subset")
    ap.add_argument("--m3a-dir", required=True)
    ap.add_argument("--index", required=True, help="m3a_index_full_ordered.json")
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument("--ffprobe", default="ffprobe")
    ap.add_argument("--max-dur", type=float, default=120.0)
    args = ap.parse_args()

    out_dir = os.path.dirname(args.index)
    videos_root = os.path.join(args.m3a_dir, "videos")
    cache_path = os.path.join(out_dir, "m3a_probe_cache.json")
    out_path = os.path.join(out_dir, f"m3a_index_{args.target}.json")

    with open(args.index, encoding="utf-8") as f:
        ordered = json.load(f)
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    texts = TextStore(args.m3a_dir)

    kept = []
    scanned = 0
    for entry in ordered:
        if len(kept) >= args.target:
            break
        scanned += 1
        rid, outlet, vid = entry["raw_id"], entry["outlet"], entry["id"]
        dest = os.path.join(videos_root, outlet, f"{vid}.mp4")
        os.makedirs(os.path.dirname(dest), exist_ok=True)

        if rid in cache:
            dur, has_audio = cache[rid]
        else:
            if not extract_video(args.m3a_dir, outlet, vid, dest):
                cache[rid] = [None, False]
                continue
            dur, has_audio = ffprobe_duration_audio(args.ffprobe, dest)
            cache[rid] = [dur, has_audio]
            if not (dur is not None and 0 < dur <= args.max_dur):
                if os.path.exists(dest):
                    os.remove(dest)  # reject: reclaim space, keep cache
            if len(cache) % 200 == 0:
                json.dump(cache, open(cache_path, "w"))

        if not (dur is not None and 0 < dur <= args.max_dur):
            continue

        # kept: ensure file present (could have been pruned earlier)
        if not os.path.exists(dest):
            if not extract_video(args.m3a_dir, outlet, vid, dest):
                continue

        out_entry = dict(entry)
        out_entry["video_path"] = os.path.join("videos", outlet, f"{vid}.mp4")
        out_entry["duration_seconds"] = round(dur, 3)
        out_entry["has_audio"] = bool(has_audio)
        out_entry["has_transcript"] = bool(has_audio)
        out_entry["summary"] = texts.summary(outlet, vid)
        out_entry["nem_texts"] = {
            t.lower(): texts.nem(t, outlet, vid)
            for t in NEM_TYPES if entry["nem"][t.lower()]
        }
        out_entry["mtg_text"] = texts.mtg(outlet, vid) if entry["mtg"] else None
        kept.append(out_entry)

        if len(kept) % 100 == 0:
            print(f"  kept {len(kept)}/{args.target} (scanned {scanned})", flush=True)

    json.dump(cache, open(cache_path, "w"))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False)

    n_audio = sum(1 for e in kept if e["has_audio"])
    print(f"\nDONE: kept {len(kept)} (scanned {scanned}, reject rate "
          f"{100*(scanned-len(kept))/max(scanned,1):.1f}%), with audio {n_audio}")
    print(f"  index -> {out_path}")
    print(f"  cache -> {cache_path} ({len(cache)} probed)")
    print(f"  videos -> {videos_root}/<outlet>/<id>.mp4")


if __name__ == "__main__":
    main()
