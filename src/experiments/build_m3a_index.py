"""
MAVIS — M3A index builder.

Builds the FULL stratified-ordered index over the M3A text-image-audio-video subset
(182k pristine videos, 60 outlets). The index is independent of the final subset size:
it is *ordered* so that any prefix is representative (round-robin by outlet + scarce
fakes front-loaded). `materialize_m3a_subset.py` later walks this order, extracts the
mp4s, filters <=120s and attaches the texts.

This script is CHEAP: it only reads zip central directories + the small json zips
(Summary, NEM, MM, MTG, Data statistics). It does NOT extract videos or run ffprobe.

Run on the cluster (where data/M3A lives):
    python src/experiments/build_m3a_index.py \
        --m3a-dir    data/M3A \
        --out        experiments/M3A/m3a_index_full_ordered.json \
        [--seed 0]

Output: a JSON list of entries, each:
    {
      "raw_id": "bbcnews/2021-06-23_16-09-08_UTC",   # unique <outlet>/<id>
      "outlet": "bbcnews", "id": "2021-06-23_16-09-08_UTC",
      "topic": "Environment", "sentiment": "neutral", "geography": "UK",
      "nem":  {"person": true, "location": false, "organization": true, "complete": true},
      "mtg":  false,
      "mm_sources": {"text": "usatoday/...", "image": "...", "audio": "...", "video": "..."}
    }
Texts (summary, NEM, MTG) are intentionally NOT stored here — they are attached for the
selected <=120s subset by materialize_m3a_subset.py to keep this index light.
"""

import argparse
import glob
import json
import os
import random
import zipfile
from collections import defaultdict

# ---------------------------------------------------------------------------
# Outlet -> geography (approx., aligned with Xu et al. 2024 OOD regions:
# ~34 North America, 6 Asia, etc.). Used for stratification / OOD cuts.
# ---------------------------------------------------------------------------
GEOGRAPHY = {
    # North America (USA + Canada)
    "abcnews": "North America", "apnews": "North America", "bloombergbusiness": "North America",
    "buzzfeednews2": "North America", "cbcnews": "North America", "cbsnews": "North America",
    "cnbc": "North America", "cnn": "North America", "denverpost": "North America",
    "foxnews": "North America", "globalnews": "North America", "globeandmail": "North America",
    "huffpost": "North America", "latimes": "North America", "msnbc": "North America",
    "nbcla": "North America", "nbcnews": "North America", "newshour": "North America",
    "newsweek": "North America", "npr": "North America", "nypost": "North America",
    "nytimes": "North America", "pbs": "North America", "politico": "North America",
    "theatlantic": "North America", "theintercept": "North America", "time": "North America",
    "usatoday": "North America", "vicenews": "North America", "voxdotcom": "North America",
    "washingtonpost": "North America", "wsj": "North America", "wearebreitbart": "North America",
    "foreignpolicymag": "North America",
    # UK
    "bbcnews": "UK", "dailymail": "UK", "financialtimes": "UK", "guardian": "UK",
    "reuters": "UK", "skynews": "UK", "telegraph": "UK", "the.independent": "UK",
    "thesun": "UK", "theeconomist": "UK",
    # Europe (non-UK)
    "dwnews": "Europe", "euronews.tv": "Europe", "france24_en": "Europe",
    # Australia
    "abcnews_au": "Australia", "heraldsunphoto": "Australia", "the.australian": "Australia",
    # Asia
    "cctv": "Asia", "the_hindu": "Asia", "timesofindia": "Asia", "philippinestar": "Asia",
    "thestaronline": "Asia", "straits_times": "Asia",
    # Middle East
    "aljazeeraenglish": "Middle East", "thenationalnews.com": "Middle East",
    "middleeasteye": "Middle East",
    # International
    "unitednations": "International",
}

NEM_TYPES = ["Person", "Location", "Organization", "Complete"]
MM_TYPES = ["Text", "Image", "Audio", "Video"]


def load_peroutlet_keys(zippath):
    """Per-outlet json zip ({id: ...}) -> dict outlet -> set(ids)."""
    out = {}
    with zipfile.ZipFile(zippath) as z:
        for n in z.namelist():
            if not n.endswith(".json"):
                continue
            outlet = os.path.basename(n)[:-5]
            with z.open(n) as f:
                out[outlet] = set(json.load(f).keys())
    return out


def load_global_map(zippath, inner):
    """Global json ({outlet/id: source}) -> dict."""
    with zipfile.ZipFile(zippath) as z:
        with z.open(inner) as f:
            return json.load(f)


def main():
    ap = argparse.ArgumentParser(description="Build M3A full stratified-ordered index")
    ap.add_argument("--m3a-dir", required=True, help="Path to data/M3A")
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    m3a = args.m3a_dir
    rng = random.Random(args.seed)

    # 1) Universe of T-I-A-V video ids per outlet (zip central dir only)
    per_outlet_ids = {}
    for zp in sorted(glob.glob(os.path.join(m3a, "Video", "*.zip"))):
        outlet = os.path.basename(zp)[:-4]
        with zipfile.ZipFile(zp) as z:
            ids = [n.split("/")[-1][:-4] for n in z.namelist() if n.endswith(".mp4")]
        per_outlet_ids[outlet] = set(ids)
    n_videos = sum(len(v) for v in per_outlet_ids.values())
    print(f"Video universe: {n_videos} across {len(per_outlet_ids)} outlets")

    # 2) Summary keys (must exist to use the text modality)
    summary = load_peroutlet_keys(os.path.join(m3a, "Text", "Summary.zip"))

    # 3) NEM presence per outlet
    nem = {t: load_peroutlet_keys(os.path.join(m3a, "NEM", f"{t}.zip")) for t in NEM_TYPES}

    # 4) MTG presence per outlet
    mtg = load_peroutlet_keys(os.path.join(m3a, "MTG", "Evaluation set", "model-generated.zip"))

    # 5) MM source maps (global, keyed outlet/id)
    mm = {t: load_global_map(os.path.join(m3a, "MM", "Imagebind.zip"),
                             f"Imagebind/{t}-changed.json") for t in MM_TYPES}

    # 6) topic / sentiment (global, keyed outlet/id)
    topic = json.load(open(os.path.join(m3a, "Data statistics", "topic.json")))
    sentiment = json.load(open(os.path.join(m3a, "Data statistics", "sentiment.json")))

    # 7) Build entries (only ids that have a Summary)
    geo_missing = set()
    no_summary = 0
    per_outlet_entries = defaultdict(list)
    for outlet, ids in per_outlet_ids.items():
        sum_ids = summary.get(outlet, set())
        geo = GEOGRAPHY.get(outlet)
        if geo is None:
            geo_missing.add(outlet)
            geo = "Other"
        for vid in ids:
            if vid not in sum_ids:
                no_summary += 1
                continue
            rid = f"{outlet}/{vid}"
            nem_flags = {t.lower(): (vid in nem[t].get(outlet, set())) for t in NEM_TYPES}
            entry = {
                "raw_id": rid,
                "outlet": outlet,
                "id": vid,
                "topic": topic.get(rid),
                "sentiment": sentiment.get(rid),
                "geography": geo,
                "nem": nem_flags,
                "mtg": vid in mtg.get(outlet, set()),
                "mm_sources": {t.lower(): mm[t].get(rid) for t in MM_TYPES},
            }
            per_outlet_entries[outlet].append(entry)

    if geo_missing:
        print(f"WARNING: no geography for outlets: {sorted(geo_missing)}")
    print(f"Dropped {no_summary} videos without a Summary text")

    # 8) Within each outlet: front-load scarce fakes (MTG > NEM-Organization > rest),
    #    random tiebreak (seeded) for reproducibility.
    for outlet, entries in per_outlet_entries.items():
        for e in entries:
            e["_r"] = rng.random()
        entries.sort(key=lambda e: (
            0 if e["mtg"] else 1,
            0 if e["nem"]["organization"] else 1,
            e["_r"],
        ))
        for e in entries:
            del e["_r"]

    # 9) Round-robin interleave across outlets (uniform-by-outlet in any prefix).
    outlets = sorted(per_outlet_entries.keys())
    max_len = max(len(v) for v in per_outlet_entries.values())
    ordered = []
    for rank in range(max_len):
        for outlet in outlets:
            lst = per_outlet_entries[outlet]
            if rank < len(lst):
                ordered.append(lst[rank])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(ordered, f, ensure_ascii=False)
    print(f"Wrote {len(ordered)} entries -> {args.out}")

    # 10) Quick sanity on a prefix
    for pref in (600, 2000):
        sub = ordered[:pref]
        outl = defaultdict(int)
        for e in sub:
            outl[e["outlet"]] += 1
        n_mtg = sum(1 for e in sub if e["mtg"])
        n_org = sum(1 for e in sub if e["nem"]["organization"])
        print(f"  prefix {pref}: outlets covered {len(outl)}/60, "
              f"min/outlet {min(outl.values())}, max/outlet {max(outl.values())}, "
              f"MTG {n_mtg}, NEM-Org {n_org}")


if __name__ == "__main__":
    main()
