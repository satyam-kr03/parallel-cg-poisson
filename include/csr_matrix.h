#ifndef CSR_MATRIX_H
#define CSR_MATRIX_H

#include <stdlib.h>

typedef struct {
    int n;
    int nnz;
    double *values;
    int *col_idx;
    int *row_ptr;
} CSRMatrix;

static inline int csr_alloc(CSRMatrix *A, int n, int nnz) {
    A->n = n;
    A->nnz = nnz;
    A->values = (double *)malloc((size_t)nnz * sizeof(double));
    A->col_idx = (int *)malloc((size_t)nnz * sizeof(int));
    A->row_ptr = (int *)malloc((size_t)(n + 1) * sizeof(int));
    if (!A->values || !A->col_idx || !A->row_ptr) {
        return -1;
    }
    return 0;
}

static inline void csr_free(CSRMatrix *A) {
    if (!A) return;
    free(A->values);
    free(A->col_idx);
    free(A->row_ptr);
    A->values = NULL;
    A->col_idx = NULL;
    A->row_ptr = NULL;
    A->n = 0;
    A->nnz = 0;
}

#endif
