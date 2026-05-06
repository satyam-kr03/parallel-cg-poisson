#include "spmv_csr.cuh"

#include <cuda_runtime.h>

__global__ void spmv_csr_scalar_kernel(const double *values, const int *col_idx,
                                      const int *row_ptr, const double *x,
                                      double *y, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    double sum = 0.0;
    for (int j = row_ptr[row]; j < row_ptr[row + 1]; j++) {
        sum += values[j] * x[col_idx[j]];
    }
    y[row] = sum;
}

__global__ void spmv_csr_vector_kernel(const double *values, const int *col_idx,
                                      const int *row_ptr, const double *x,
                                      double *y, int n) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    if (warp_id >= n) return;

    double sum = 0.0;
    int row_start = row_ptr[warp_id];
    int row_end = row_ptr[warp_id + 1];
    for (int j = row_start + lane; j < row_end; j += 32) {
        sum += values[j] * x[col_idx[j]];
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        y[warp_id] = sum;
    }
}

void spmv_csr_scalar(const DeviceCSR *dA, const double *d_x, double *d_y, cudaStream_t stream) {
    int block = 256;
    int grid = (dA->n + block - 1) / block;
    spmv_csr_scalar_kernel<<<grid, block, 0, stream>>>(dA->d_values, dA->d_col_idx,
                                                       dA->d_row_ptr, d_x, d_y, dA->n);
}

void spmv_csr_vector(const DeviceCSR *dA, const double *d_x, double *d_y, cudaStream_t stream) {
    int block = 256;
    int grid = (dA->n * 32 + block - 1) / block;
    spmv_csr_vector_kernel<<<grid, block, 0, stream>>>(dA->d_values, dA->d_col_idx,
                                                       dA->d_row_ptr, d_x, d_y, dA->n);
}
