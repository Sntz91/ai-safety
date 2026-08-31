#!/bin/bash
set -e

echo "--- Splitting standalone dataset: bhx ---"
PYTHONPATH=. python ai_safety/data/create/misc/splits.py --dataset bhx --out-dir splits --val-ratio 0.10 --test-ratio 0.10

echo "--- Splitting paired datasets: rsna & sinoct ---"
PYTHONPATH=. python ai_safety/data/create/misc/splits.py --dataset rsna sinoct \
    --mapping /run/media/tobias/backup/data/sinoct_rsna_mapping.csv \
    --out-dir splits --val-ratio 0.10 --test-ratio 0.10

echo "--- All splits created successfully in splits/ directory! ---"
