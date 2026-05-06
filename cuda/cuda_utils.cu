#include "cuda_utils.cuh"

#include <stdio.h>

int device_csr_alloc(DeviceCSR *dA, const CSRMatrix *hA) {
    if (!dA || !hA) {
        return -1;
    }
    dA->n = hA->n;
    dA->nnz = hA->nnz;

    if (cudaMalloc((void **)&dA->d_values, (size_t)hA->nnz * sizeof(double)) != cudaSuccess) {
        return -1;
    }
    if (cudaMalloc((void **)&dA->d_col_idx, (size_t)hA->nnz * sizeof(int)) != cudaSuccess) {
        return -1;
    }
    if (cudaMalloc((void **)&dA->d_row_ptr, (size_t)(hA->n + 1) * sizeof(int)) != cudaSuccess) {
        return -1;
    }

    if (cudaMemcpy(dA->d_values, hA->values, (size_t)hA->nnz * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
        return -1;
    }
    if (cudaMemcpy(dA->d_col_idx, hA->col_idx, (size_t)hA->nnz * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
        return -1;
    }
    if (cudaMemcpy(dA->d_row_ptr, hA->row_ptr, (size_t)(hA->n + 1) * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
        return -1;
    }
    return 0;
}

int device_csr_free(DeviceCSR *dA) {
    if (!dA) {
        return -1;
    }
    if (dA->d_values) cudaFree(dA->d_values);
    if (dA->d_col_idx) cudaFree(dA->d_col_idx);
    if (dA->d_row_ptr) cudaFree(dA->d_row_ptr);
    dA->d_values = NULL;
    dA->d_col_idx = NULL;
    dA->d_row_ptr = NULL;
    dA->n = 0;
    dA->nnz = 0;
    return 0;
}
