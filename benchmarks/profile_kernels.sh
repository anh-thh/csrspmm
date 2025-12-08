#!/bin/bash

M=8192
N=4096
K=256
S=0.90

ALGOS=("Naive" "WarpPerRow" "WarpPerRowSmem" "WarpPerRowFp4" "WarpPerRowSmemFp4")

OUTDIR="./ncu_profiles"
if [ ! -d "$OUTDIR" ]; then
    mkdir -p "$OUTDIR"
    echo "Create $OUTDIR"
else
    echo "directory $OUTDIR already exists"
fi

echo ""

for algo in "${ALGOS[@]}"; do
    echo "Profiling $algo ..."

    LOGFILE="$OUTDIR/profile_${algo}.log"

    ncu --set full \
        --launch-skip 500 \
        --launch-count 1 \
        ./bench_csrspmm \
            -M $M -N $N -K $K -s $S -algo $algo \
        > "$LOGFILE" 2>&1

    echo "Profiling $algo done."
    echo ""
done

echo "All profiles completed."

