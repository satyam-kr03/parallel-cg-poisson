#!/bin/bash
set -euo pipefail

mkdir -p results/scaling

for N in 512 1024 2048; do
  for T in 1 2 4 8 16; do
    export OMP_NUM_THREADS=$T
    for P in 1 2 4 8; do
      mpirun -n $P ./cg_solver --grid $N --tol 1e-8 --mode mpi \
        | tee results/scaling/N${N}_T${T}_P${P}.log
    done
  done
done
