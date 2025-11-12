#!/bin/bash
# NOTE: You can change the code to run the benchmark

TEST_PROGRAM=./test_csr_spmm
BENCHMARK_PROGRAM=./bench_csr_spmm
SPARSITY=0.7
# ALPHA=1.0
BETA=0.5
ALGO=naive

# test 
$TEST_PROGRAM -algo "$ALGO"

for SPARSITY in 0.7; do  
    for i in {3..10}; do
        M=$((2**i))
        echo "$BENCHMARK_PROGRAM -M $M -s $SPARSITY -a $ALPHA -b $BETA -algo $ALGO"
        $BENCHMARK_PROGRAM -M "$M" -s "$SPARSITY" -a "$ALPHA" -b "$BETA" -algo "$ALGO"
    done
done
