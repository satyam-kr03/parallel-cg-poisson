# parallel-cg-poisson
A parallel Conjugate Gradient (CG) solver for sparse linear systems arising from the finite-difference discretization of the 2D Poisson equation.

## Build

```bash
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Run

```bash
# Serial baseline
./cg_solver --grid 512 --mode serial

# CUDA SpMV + CPU CG
./cg_solver --grid 1024 --mode cuda

# MPI (CPU SpMV)
mpirun -n 4 ./cg_solver --grid 1024 --mode mpi

# Hybrid (single-rank CUDA path)
mpirun -n 1 ./cg_solver --grid 1024 --mode hybrid
```

## Tests

```bash
./test_cg
./test_spmv
```
