#include "cg_solver.h"
#include "poisson_setup.h"

#ifdef ENABLE_MPI
#include <mpi.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *prog) {
    printf("Usage: %s --grid N --mode serial|cuda|mpi|hybrid [--tol T] [--max-iter K]\n", prog);
}

int main(int argc, char **argv) {
    int N = 512;
    double tol = 1e-8;
    int max_iter = 10000;
    const char *mode = "serial";
    int verbose = 1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--grid") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--tol") == 0 && i + 1 < argc) {
            tol = atof(argv[++i]);
        } else if (strcmp(argv[i], "--max-iter") == 0 && i + 1 < argc) {
            max_iter = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            mode = argv[++i];
        } else if (strcmp(argv[i], "--quiet") == 0) {
            verbose = 0;
        } else if (strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    if (strcmp(mode, "mpi") == 0 || strcmp(mode, "hybrid") == 0) {
#ifdef ENABLE_MPI
        MPI_Init(&argc, &argv);
        int use_cuda = (strcmp(mode, "hybrid") == 0) ? 1 : 0;
        int iters = cg_solve_mpi(N, tol, max_iter, verbose, use_cuda);
        MPI_Finalize();
        return (iters < 0) ? 1 : 0;
#else
        fprintf(stderr, "MPI support not enabled. Reconfigure with -DENABLE_MPI=ON.\n");
        return 1;
#endif
    }

    CSRMatrix A;
    int n = N * N;
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

    int iters = -1;
    if (strcmp(mode, "serial") == 0) {
        iters = cg_solve_serial(&A, b, x, tol, max_iter, verbose);
    } else if (strcmp(mode, "cuda") == 0) {
        iters = cg_solve_cuda(&A, b, x, tol, max_iter, verbose);
    } else {
        usage(argv[0]);
    }

    csr_free(&A);
    free(b);
    free(x);
    return (iters < 0) ? 1 : 0;
}
