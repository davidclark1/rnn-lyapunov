#!/bin/bash
# Pixel comparison of regenerated figures with reference PDFs (e.g. the paper's Overleaf figures/ directory).
#   tools/compare_figures.sh <reference_dir> [new_dir=figures]
# Renders both at 150 dpi (pdftoppm) and prints the number of differing pixels (ImageMagick `compare -metric AE`);
# writes figures/compare/<name>_diff.png for inspection.
ref=$1; new=${2:-figures}; out=$new/compare; mkdir -p "$out"
for f in "$new"/*.pdf; do
  n=$(basename "$f" .pdf); [ -f "$ref/$n.pdf" ] || { echo "$n: no reference"; continue; }
  pdftoppm -r 150 -png -singlefile "$f" "$out/${n}_new"; pdftoppm -r 150 -png -singlefile "$ref/$n.pdf" "$out/${n}_ref"
  ae=$(compare -metric AE "$out/${n}_new.png" "$out/${n}_ref.png" "$out/${n}_diff.png" 2>&1)
  echo "$n: differing pixels = $ae"
done
