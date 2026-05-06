#ifndef SPMV_CSR_CUH
#define SPMV_CSR_CUH

#include <cuda_runtime.h>
#include "cuda_utils.cuh"

void spmv_csr_scalar(const DeviceCSR *dA, const double *d_x, double *d_y, cudaStream_t stream);
void spmv_csr_vector(const DeviceCSR *dA, const double *d_x, double *d_y, cudaStream_t stream);

#endif
