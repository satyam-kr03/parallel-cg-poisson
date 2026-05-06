#include "cg_solver.h"

#ifdef _OPENMP
#include <omp.h>
#endif

double dot_omp(const double *a, const double *b, int n) {
    double sum = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:sum)
#endif
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

void axpy_omp(double *y, const double *x, double alpha, int n) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < n; i++) {
        y[i] += alpha * x[i];
    }
}

void copy_omp(double *y, const double *x, int n) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < n; i++) {
        y[i] = x[i];
    }
}
