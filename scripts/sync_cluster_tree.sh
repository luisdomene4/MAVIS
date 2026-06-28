#!/bin/bash
# Generate file tree of repo, collapsing repetitive media dirs into summaries.
# Run from repo root on cluster: bash scripts/sync_cluster_tree.sh
#
# Rules:
#   - Video dirs (>5 files) -> "[438 videos: mp4, webm]"
#   - Image dirs (>10 files) -> "[200 frames: jpg, png]"
#   - Everything else listed individually (model weights, JSONs, CSVs, configs)
#   - Skip .git, __pycache__, .pyc, node_modules

OUTPUT="docs/cluster_tree.md"

{
    echo "# Cluster File Tree"
    echo ""
    echo "_Generated: $(date '+%Y-%m-%d %H:%M') on $(hostname)_"
    echo ""
    echo '```'
    python3 scripts/gentree.py .
    echo '```'
} > "$OUTPUT"

echo "Wrote: $OUTPUT"
echo "Next: git add $OUTPUT && git commit -m 'update cluster tree' && git push"
