#include "cg_solver.h"

#ifdef ENABLE_CUDA
#include "cuda_utils.cuh"
#include "spmv_csr.cuh"
#include "perf_utils.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static void spmv_csr_cpu(const CSRMatrix *A, const double *x, double *y) {
    for (int i = 0; i < A->n; i++) {
        double sum = 0.0;
        for (int jj = A->row_ptr[i]; jj < A->row_ptr[i + 1]; jj++) {
            sum += A->values[jj] * x[A->col_idx[jj]];
        }
        y[i] = sum;
    }
}

int cg_solve_cuda(const CSRMatrix *A, const double *b, double *x,
                  double tol, int max_iter, int verbose) {
    int n = A->n;
    DeviceCSR dA;
    if (device_csr_alloc(&dA, A) != 0) {
        return -1;
    }

    double *r = (double *)malloc((size_t)n * sizeof(double));
    double *p = NULL;
    double *Ap = NULL;
    cudaMallocHost((void **)&p, (size_t)n * sizeof(double));
    cudaMallocHost((void **)&Ap, (size_t)n * sizeof(double));
    if (!r || !p || !Ap) {
        free(r);
        if (p) cudaFreeHost(p);
        if (Ap) cudaFreeHost(Ap);
        device_csr_free(&dA);
        return -1;
    }

    spmv_csr_cpu(A, x, Ap);
    for (int i = 0; i < n; i++) {
        r[i] = b[i] - Ap[i];
        p[i] = r[i];
    }

    double rr0 = dot_omp(r, r, n);
    if (rr0 == 0.0) {
        free(r);
        cudaFreeHost(p);
        cudaFreeHost(Ap);
        device_csr_free(&dA);
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
    device_csr_free(&dA);
    return iter;
}
#else
int cg_solve_cuda(const CSRMatrix *A, const double *b, double *x,
                  double tol, int max_iter, int verbose) {
    (void)A;
    (void)b;
    (void)x;
    (void)tol;
    (void)max_iter;
    (void)verbose;
    return -1;
}
#endif
