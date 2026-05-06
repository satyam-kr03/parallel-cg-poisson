#ifndef CG_SOLVER_H
#define CG_SOLVER_H

#include "csr_matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

int cg_solve_serial(const CSRMatrix *A, const double *b, double *x,
                    double tol, int max_iter, int verbose);

int cg_solve_cuda(const CSRMatrix *A, const double *b, double *x,
                  double tol, int max_iter, int verbose);

int cg_solve_mpi(int N, double tol, int max_iter, int verbose, int use_cuda);

double dot_omp(const double *a, const double *b, int n);
void axpy_omp(double *y, const double *x, double alpha, int n);
void copy_omp(double *y, const double *x, int n);

#ifdef __cplusplus
}
#endif

#endif
