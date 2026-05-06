#include "cg_solver.h"

#ifdef ENABLE_CUDA
#include "cuda_utils.cuh"
#include "spmv_csr.cuh"
#include "poisson_setup.h"
#include "perf_utils.h"

#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

static void compute_row_range(int N, int rank, int size, int *row_start, int *row_end) {
    int base = N / size;
    int rem = N % size;
    int start = rank * base + (rank < rem ? rank : rem);
    int rows = base + (rank < rem ? 1 : 0);
    *row_start = start;
    *row_end = start + rows;
}

int cg_solve_mpi_cuda(int N, double tol, int max_iter, int verbose) {
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size > 1) {
        if (rank == 0) {
            fprintf(stderr, "MPI+CUDA path currently supports only 1 rank; falling back to CPU.\n");
        }
        return -1;
    }

    CSRMatrix A;
    int n = N * N;
    double *b = (double *)malloc((size_t)n * sizeof(double));
    double *x = (double *)calloc((size_t)n, sizeof(double));
    if (!b || !x) {
        free(b);
        free(x);
        return -1;
    }
    if (build_poisson_csr(N, &A, b) != 0) {
        free(b);
        free(x);
        return -1;
    }

    DeviceCSR dA;
    if (device_csr_alloc(&dA, &A) != 0) {
        csr_free(&A);
        free(b);
        free(x);
        return -1;
    }

    double *r = (double *)malloc((size_t)n * sizeof(double));
    double *p = NULL;
    double *Ap = NULL;
    cudaMallocHost((void **)&p, (size_t)n * sizeof(double));
    cudaMallocHost((void **)&Ap, (size_t)n * sizeof(double));
    if (!r || !p || !Ap) {
        csr_free(&A);
        device_csr_free(&dA);
        free(b);
        free(x);
        free(r);
        if (p) cudaFreeHost(p);
        if (Ap) cudaFreeHost(Ap);
        return -1;
    }

    for (int i = 0; i < n; i++) {
        r[i] = b[i];
        p[i] = r[i];
    }

    double rr0 = dot_omp(r, r, n);
    if (rr0 == 0.0) {
        csr_free(&A);
        device_csr_free(&dA);
        free(b);
        free(x);
        free(r);
        cudaFreeHost(p);
        cudaFreeHost(Ap);
        return 0;
    }

    double *d_p = NULL;
    double *d_Ap = NULL;
    cudaMalloc((void **)&d_p, (size_t)n * sizeof(double));
    cudaMalloc((void **)&d_Ap, (size_t)n * sizeof(double));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpyAsync(d_p, p, (size_t)n * sizeof(double), cudaMemcpyHostToDevice, stream);

    double rr = rr0;
    double t0 = now_seconds();
    int iter = 0;

    for (; iter < max_iter; iter++) {
        spmv_csr_vector(&dA, d_p, d_Ap, stream);
        cudaMemcpyAsync(Ap, d_Ap, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        double pAp = dot_omp(p, Ap, n);
        if (pAp == 0.0) {
            break;
        }
        double alpha = rr / pAp;
        axpy_omp(x, p, alpha, n);
        axpy_omp(r, Ap, -alpha, n);

        double rr_new = dot_omp(r, r, n);
        double rel = sqrt(rr_new / rr0);
        if (verbose && (iter % 10 == 0 || rel < tol)) {
            printf("iter %d, rel_res=%.6e\n", iter, rel);
        }
        if (rel < tol) {
            rr = rr_new;
            break;
        }

        double beta = rr_new / rr;
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int i = 0; i < n; i++) {
            p[i] = r[i] + beta * p[i];
        }
        rr = rr_new;
        cudaMemcpyAsync(d_p, p, (size_t)n * sizeof(double), cudaMemcpyHostToDevice, stream);
    }

    double t1 = now_seconds();
    if (verbose) {
        double rel = sqrt(rr / rr0);
        printf("CG done in %d iterations, rel_res=%.6e, time=%.3f s\n",
               iter, rel, t1 - t0);
    }

    cudaStreamDestroy(stream);
    cudaFree(d_p);
    cudaFree(d_Ap);
    cudaFreeHost(p);
    cudaFreeHost(Ap);
    free(r);
    csr_free(&A);
    device_csr_free(&dA);
    free(b);
    free(x);
    return iter;
}
#endif
