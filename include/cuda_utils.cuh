#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <cuda_runtime.h>
#include <stdio.h>
#include "csr_matrix.h"

typedef struct {
    double *d_values;
    int *d_col_idx;
    int *d_row_ptr;
    int n;
    int nnz;
} DeviceCSR;

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return -1; \
    } \
} while (0)

int device_csr_alloc(DeviceCSR *dA, const CSRMatrix *hA);
int device_csr_free(DeviceCSR *dA);

#endif
