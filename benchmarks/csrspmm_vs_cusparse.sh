#!/usr/bin/env bash
set -euo pipefail

OUT_FILE="csrspmm_vs_cusparse.csv"

echo "name,M,N,K,density,sparsity,lib,algo,gflops" > "$OUT_FILE"

CONFIGS=(
  "small  512   1024  64"
  "medium 1024  2048  128"
  "large  4096  4096  128"
  "XL     8192  4096  256"
)

DENSITIES=(0.01 0.05 0.10 0.20)
SPARSITIES=(0.99 0.95 0.90 0.80)

CSR_ALGOS=("Naive" "WarpPerRow" "WarpPerRowSmem" "WarpPerRowFp4" "WarpPerRowSmemFp4")


get_gflops () {
  awk '/GFLOP/ {
    for (i = 1; i <= NF; i++) {
      if ($i ~ /GFLOP/) {
        num = $(i-1)
        gsub(/[()]/, "", num)
        print num
        exit
      }
    }
  }'
}



for cfg in "${CONFIGS[@]}"; do
  read -r NAME M N K <<<"$cfg"

  for idx in "${!DENSITIES[@]}"; do
    DENS="${DENSITIES[$idx]}"
    SPAR="${SPARSITIES[$idx]}"

    echo "=== $NAME  M=$M N=$N K=$K  density=$DENS (sparsity=$SPAR) ==="

    # custom CSR kernels
    for ALGO in "${CSR_ALGOS[@]}"; do
      echo "Running $ALGO"
      OUT=$(./bench_csrspmm -M "$M" -N "$N" -K "$K" -s "$SPAR" -a 1.0 -b 0.0 -algo "$ALGO")
      GFLOPS=$(printf '%s\n' "$OUT" | get_gflops)
      echo "$NAME,$M,$N,$K,$DENS,$SPAR,csrspmm,$ALGO,$GFLOPS" >> "$OUT_FILE"
    done

    # cuSPARSE baseline
    echo "Running cuSPARSE"
    OUT=$(./bench_cusparse -M "$M" -N "$N" -K "$K" -s "$SPAR" -a 1.0 -b 0.0)
    GFLOPS=$(printf '%s\n' "$OUT" | get_gflops)
    echo "$NAME,$M,$N,$K,$DENS,$SPAR,cuSPARSE,csrmm,$GFLOPS" >> "$OUT_FILE"
  done
done

echo "Finished. Results saved to $OUT_FILE"

