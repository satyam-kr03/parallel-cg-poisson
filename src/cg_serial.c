#include "cg_solver.h"
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

int cg_solve_serial(const CSRMatrix *A, const double *b, double *x,
                    double tol, int max_iter, int verbose) {
    int n = A->n;
    double *r = (double *)malloc((size_t)n * sizeof(double));
    double *p = (double *)malloc((size_t)n * sizeof(double));
    double *Ap = (double *)malloc((size_t)n * sizeof(double));
    if (!r || !p || !Ap) {
        free(r);
        free(p);
        free(Ap);
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
        free(p);
        free(Ap);
        return 0;
    }

    double rr = rr0;
    double t0 = now_seconds();
    int iter = 0;

    for (; iter < max_iter; iter++) {
        spmv_csr_cpu(A, p, Ap);
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

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int i = 0; i < n; i++) {
            p[i] = r[i] + (rr_new / rr) * p[i];
        }

        rr = rr_new;
    }

    double t1 = now_seconds();
    if (verbose) {
        double rel = sqrt(rr / rr0);
        printf("CG done in %d iterations, rel_res=%.6e, time=%.3f s\n",
               iter, rel, t1 - t0);
    }

    free(r);
    free(p);
    free(Ap);
    return iter;
}
