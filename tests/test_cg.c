#include "cg_solver.h"
#include "poisson_setup.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    int N = 32;
    int n = N * N;
    double tol = 1e-8;
    int max_iter = 5000;

    CSRMatrix A;
    double *b = (double *)malloc((size_t)n * sizeof(double));
    double *x = (double *)calloc((size_t)n, sizeof(double));
    if (!b || !x) {
        free(b);
        free(x);
        return 1;
    }
    if (build_poisson_csr(N, &A, b) != 0) {
        free(b);
        free(x);
        return 1;
    }

    int iters = cg_solve_serial(&A, b, x, tol, max_iter, 0);
    if (iters < 0) {
        csr_free(&A);
        free(b);
        free(x);
        return 1;
    }

    double h = 1.0 / (double)(N + 1);
    double pi = acos(-1.0);
    double max_err = 0.0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double xcoord = (double)(i + 1) * h;
            double ycoord = (double)(j + 1) * h;
            double u_exact = sin(pi * xcoord) * sin(pi * ycoord);
            double err = fabs(x[i * N + j] - u_exact);
            if (err > max_err) {
                max_err = err;
            }
        }
    }

    printf("max_error=%.6e\n", max_err);

    csr_free(&A);
    free(b);
    free(x);

    return (max_err < 1e-3) ? 0 : 1;
}
