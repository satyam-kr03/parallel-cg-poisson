#include "poisson_setup.h"

#include <math.h>
#include <stdlib.h>

static int count_row_nnz(int i, int j, int N) {
    int nnz = 1;
    if (i > 0) nnz++;
    if (i < N - 1) nnz++;
    if (j > 0) nnz++;
    if (j < N - 1) nnz++;
    return nnz;
}

int build_poisson_csr(int N, CSRMatrix *A, double *rhs) {
    int n = N * N;
    int nnz = 5 * n - 4 * N;
    if (csr_alloc(A, n, nnz) != 0) {
        return -1;
    }

    const double h = 1.0 / (double)(N + 1);
    const double pi = acos(-1.0);

    int offset = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int row = i * N + j;
            A->row_ptr[row] = offset;

            if (i > 0) {
                A->values[offset] = -1.0;
                A->col_idx[offset] = (i - 1) * N + j;
                offset++;
            }
            if (j > 0) {
                A->values[offset] = -1.0;
                A->col_idx[offset] = i * N + (j - 1);
                offset++;
            }

            A->values[offset] = 4.0;
            A->col_idx[offset] = row;
            offset++;

            if (j < N - 1) {
                A->values[offset] = -1.0;
                A->col_idx[offset] = i * N + (j + 1);
                offset++;
            }
            if (i < N - 1) {
                A->values[offset] = -1.0;
                A->col_idx[offset] = (i + 1) * N + j;
                offset++;
            }

            double x = (double)(i + 1) * h;
            double y = (double)(j + 1) * h;
            double f = 2.0 * pi * pi * sin(pi * x) * sin(pi * y);
            rhs[row] = h * h * f;
        }
    }
    A->row_ptr[n] = offset;

    return (offset == nnz) ? 0 : -1;
}

int build_poisson_csr_local(int N, int row_start, int row_end,
                            CSRMatrix *A_local, double *rhs_local) {
    int local_rows = row_end - row_start;
    int local_n = local_rows * N;
    int nnz = 0;

    for (int i = row_start; i < row_end; i++) {
        for (int j = 0; j < N; j++) {
            nnz += count_row_nnz(i, j, N);
        }
    }

    if (csr_alloc(A_local, local_n, nnz) != 0) {
        return -1;
    }

    const double h = 1.0 / (double)(N + 1);
    const double pi = acos(-1.0);

    int offset = 0;
    for (int i = row_start; i < row_end; i++) {
        for (int j = 0; j < N; j++) {
            int local_row = (i - row_start) * N + j;
            int global_row = i * N + j;
            A_local->row_ptr[local_row] = offset;

            if (i > 0) {
                A_local->values[offset] = -1.0;
                A_local->col_idx[offset] = (i - 1) * N + j;
                offset++;
            }
            if (j > 0) {
                A_local->values[offset] = -1.0;
                A_local->col_idx[offset] = i * N + (j - 1);
                offset++;
            }

            A_local->values[offset] = 4.0;
            A_local->col_idx[offset] = global_row;
            offset++;

            if (j < N - 1) {
                A_local->values[offset] = -1.0;
                A_local->col_idx[offset] = i * N + (j + 1);
                offset++;
            }
            if (i < N - 1) {
                A_local->values[offset] = -1.0;
                A_local->col_idx[offset] = (i + 1) * N + j;
                offset++;
            }

            double x = (double)(i + 1) * h;
            double y = (double)(j + 1) * h;
            double f = 2.0 * pi * pi * sin(pi * x) * sin(pi * y);
            rhs_local[local_row] = h * h * f;
        }
    }
    A_local->row_ptr[local_n] = offset;

    return (offset == nnz) ? 0 : -1;
}
