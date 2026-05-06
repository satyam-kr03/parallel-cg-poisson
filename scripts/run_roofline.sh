#!/bin/bash
set -euo pipefail

N=${1:-1024}
mkdir -p results/roofline

ncu --metrics \
  sm__cycles_elapsed.avg,\
  dram__bytes_read.sum,\
  dram__bytes_write.sum,\
  sm__sass_thread_inst_executed_op_dfma_pred_on.sum \
  --target-processes all \
  ./cg_solver --grid $N --mode cuda \
  > results/roofline/ncu_N${N}.txt
