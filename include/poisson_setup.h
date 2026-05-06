#ifndef POISSON_SETUP_H
#define POISSON_SETUP_H

#include "csr_matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

int build_poisson_csr(int N, CSRMatrix *A, double *rhs);
int build_poisson_csr_local(int N, int row_start, int row_end,
                            CSRMatrix *A_local, double *rhs_local);

#ifdef __cplusplus
}
#endif

#endif
