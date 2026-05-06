#include "cuda_utils.cuh"
#include "poisson_setup.h"
#include "spmv_csr.cuh"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static void spmv_cpu(const CSRMatrix *A, const double *x, double *y) {
    for (int i = 0; i < A->n; i++) {
        double sum = 0.0;
        for (int jj = A->row_ptr[i]; jj < A->row_ptr[i + 1]; jj++) {
            sum += A->values[jj] * x[A->col_idx[jj]];
        }
        y[i] = sum;
    }
}

int main(void) {
    int N = 32;
    int n = N * N;

    CSRMatrix A;
    double *b = (double *)malloc((size_t)n * sizeof(double));
    if (!b) {
        return 1;
    }
    if (build_poisson_csr(N, &A, b) != 0) {
        free(b);
        return 1;
    }

    double *x = (double *)malloc((size_t)n * sizeof(double));
    double *y_cpu = (double *)malloc((size_t)n * sizeof(double));
    double *y_gpu = (double *)malloc((size_t)n * sizeof(double));
    if (!x || !y_cpu || !y_gpu) {
        free(b);
        free(x);
        free(y_cpu);
        free(y_gpu);
        csr_free(&A);
        return 1;
    }

    for (int i = 0; i < n; i++) {
        x[i] = sin(0.1 * (double)i);
    }

    spmv_cpu(&A, x, y_cpu);

    DeviceCSR dA;
    if (device_csr_alloc(&dA, &A) != 0) {
        free(b);
        free(x);
        free(y_cpu);
        free(y_gpu);
        csr_free(&A);
        return 1;
    }

    double *d_x = NULL;
    double *d_y = NULL;
    cudaMalloc((void **)&d_x, (size_t)n * sizeof(double));
    cudaMalloc((void **)&d_y, (size_t)n * sizeof(double));
    cudaMemcpy(d_x, x, (size_t)n * sizeof(double), cudaMemcpyHostToDevice);

    spmv_csr_vector(&dA, d_x, d_y, 0);
    cudaMemcpy(y_gpu, d_y, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost);

    double max_err = 0.0;
    for (int i = 0; i < n; i++) {
        double err = fabs(y_cpu[i] - y_gpu[i]);
        if (err > max_err) {
            max_err = err;
        }
    }

    printf("max_spmv_error=%.6e\n", max_err);

    cudaFree(d_x);
    cudaFree(d_y);
    device_csr_free(&dA);
    free(b);
    free(x);
    free(y_cpu);
    free(y_gpu);
    csr_free(&A);

    return (max_err < 1e-10) ? 0 : 1;
}
