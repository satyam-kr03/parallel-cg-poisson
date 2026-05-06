#include "cg_solver.h"
#include "poisson_setup.h"
#include "perf_utils.h"

#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

static void compute_row_range(int N, int rank, int size, int *row_start, int *row_end) {
    int base = N / size;
    int rem = N % size;
    int start = rank * base + (rank < rem ? rank : rem);
    int rows = base + (rank < rem ? 1 : 0);
    *row_start = start;
    *row_end = start + rows;
}

static double get_p_value(int col, int local_start, int local_end, int N,
                          const double *p_local, const double *halo_top,
                          const double *halo_bottom) {
    int local_start_idx = local_start * N;
    int local_end_idx = local_end * N;
    if (col >= local_start_idx && col < local_end_idx) {
        return p_local[col - local_start_idx];
    }
    int top_start = (local_start - 1) * N;
    int bottom_start = local_end * N;
    if (halo_top && col >= top_start && col < top_start + N) {
        return halo_top[col - top_start];
    }
    if (halo_bottom && col >= bottom_start && col < bottom_start + N) {
        return halo_bottom[col - bottom_start];
    }
    return 0.0;
}

static void spmv_local(const CSRMatrix *A_local, const double *p_local,
                       int row_start, int row_end, int N,
                       const double *halo_top, const double *halo_bottom,
                       double *Ap_local) {
    int local_n = A_local->n;
    for (int i = 0; i < local_n; i++) {
        double sum = 0.0;
        for (int jj = A_local->row_ptr[i]; jj < A_local->row_ptr[i + 1]; jj++) {
            int col = A_local->col_idx[jj];
            double x = get_p_value(col, row_start, row_end, N, p_local, halo_top, halo_bottom);
            sum += A_local->values[jj] * x;
        }
        Ap_local[i] = sum;
    }
}

int cg_solve_mpi(int N, double tol, int max_iter, int verbose, int use_cuda) {
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

#ifdef ENABLE_CUDA
    if (use_cuda && size == 1) {
        CSRMatrix A;
        int n = N * N;
        double *b = (double *)malloc((size_t)n * sizeof(double));
        double *x = (double *)calloc((size_t)n, sizeof(double));
        if (!b || !x) {
            free(b);
            free(x);
            return -1;
        }
        if (build_poisson_csr(N, &A, b) != 0) {
            free(b);
            free(x);
            return -1;
        }
        int iters = cg_solve_cuda(&A, b, x, tol, max_iter, verbose && rank == 0);
        csr_free(&A);
        free(b);
        free(x);
        return iters;
    }
#endif

    if (use_cuda && size > 1) {
        if (rank == 0) {
            fprintf(stderr, "MPI+CUDA for multiple ranks is not implemented; using CPU SpMV.\n");
        }
    }
#ifndef ENABLE_CUDA
    (void)use_cuda;
#endif

    int row_start = 0;
    int row_end = 0;
    compute_row_range(N, rank, size, &row_start, &row_end);

    int local_rows = row_end - row_start;
    int local_n = local_rows * N;

    CSRMatrix A_local;
    double *b_local = (double *)malloc((size_t)local_n * sizeof(double));
    double *x_local = (double *)calloc((size_t)local_n, sizeof(double));
    double *r_local = (double *)malloc((size_t)local_n * sizeof(double));
    double *p_local = (double *)malloc((size_t)local_n * sizeof(double));
    double *Ap_local = (double *)malloc((size_t)local_n * sizeof(double));

    if (!b_local || !x_local || !r_local || !p_local || !Ap_local) {
        free(b_local);
        free(x_local);
        free(r_local);
        free(p_local);
        free(Ap_local);
        return -1;
    }

    if (build_poisson_csr_local(N, row_start, row_end, &A_local, b_local) != 0) {
        free(b_local);
        free(x_local);
        free(r_local);
        free(p_local);
        free(Ap_local);
        return -1;
    }

    double *halo_top = NULL;
    double *halo_bottom = NULL;
    if (row_start > 0) {
        halo_top = (double *)malloc((size_t)N * sizeof(double));
    }
    if (row_end < N) {
        halo_bottom = (double *)malloc((size_t)N * sizeof(double));
    }

    for (int i = 0; i < local_n; i++) {
        r_local[i] = b_local[i];
        p_local[i] = r_local[i];
    }

    double rr_local = dot_omp(r_local, r_local, local_n);
    double rr0 = 0.0;
    MPI_Allreduce(&rr_local, &rr0, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (rr0 == 0.0) {
        csr_free(&A_local);
        free(b_local);
        free(x_local);
        free(r_local);
        free(p_local);
        free(Ap_local);
        free(halo_top);
        free(halo_bottom);
        return 0;
    }

    double rr = rr0;
    double t0 = now_seconds();
    int iter = 0;

    double t_comm = 0.0;
    double t_spmv = 0.0;
    double t_dot = 0.0;
    double t_axpy = 0.0;
    double t_allreduce = 0.0;

    for (; iter < max_iter; iter++) {
        MPI_Request reqs[4];
        int req_count = 0;
        double t_comm_start = now_seconds();
        if (row_start > 0) {
            MPI_Irecv(halo_top, N, MPI_DOUBLE, rank - 1, 100, MPI_COMM_WORLD, &reqs[req_count++]);
            MPI_Isend(p_local, N, MPI_DOUBLE, rank - 1, 101, MPI_COMM_WORLD, &reqs[req_count++]);
        }
        if (row_end < N) {
            MPI_Irecv(halo_bottom, N, MPI_DOUBLE, rank + 1, 101, MPI_COMM_WORLD, &reqs[req_count++]);
            MPI_Isend(p_local + (local_rows - 1) * N, N, MPI_DOUBLE, rank + 1, 100, MPI_COMM_WORLD, &reqs[req_count++]);
        }
        if (req_count > 0) {
            MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);
        }
        t_comm += now_seconds() - t_comm_start;

        double t_spmv_start = now_seconds();
        spmv_local(&A_local, p_local, row_start, row_end, N, halo_top, halo_bottom, Ap_local);
        t_spmv += now_seconds() - t_spmv_start;

        double t_dot_start = now_seconds();
        double pAp_local = dot_omp(p_local, Ap_local, local_n);
        t_dot += now_seconds() - t_dot_start;
        double t_allreduce_start = now_seconds();
        double pAp = 0.0;
        MPI_Allreduce(&pAp_local, &pAp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        t_allreduce += now_seconds() - t_allreduce_start;
        if (pAp == 0.0) {
            break;
        }
        double alpha = rr / pAp;

        double t_axpy_start = now_seconds();
        axpy_omp(x_local, p_local, alpha, local_n);
        axpy_omp(r_local, Ap_local, -alpha, local_n);
        t_axpy += now_seconds() - t_axpy_start;

        t_dot_start = now_seconds();
        double rr_new_local = dot_omp(r_local, r_local, local_n);
        t_dot += now_seconds() - t_dot_start;
        t_allreduce_start = now_seconds();
        double rr_new = 0.0;
        MPI_Allreduce(&rr_new_local, &rr_new, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        t_allreduce += now_seconds() - t_allreduce_start;
        double rel = sqrt(rr_new / rr0);

        if (rank == 0 && verbose && (iter % 10 == 0 || rel < tol)) {
            printf("iter %d, rel_res=%.6e\n", iter, rel);
        }
        if (rel < tol) {
            rr = rr_new;
            break;
        }

        double beta = rr_new / rr;
        t_axpy_start = now_seconds();
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int i = 0; i < local_n; i++) {
            p_local[i] = r_local[i] + beta * p_local[i];
        }
        t_axpy += now_seconds() - t_axpy_start;
        rr = rr_new;
    }

    double t1 = now_seconds();
    if (rank == 0 && verbose) {
        double rel = sqrt(rr / rr0);
        printf("CG done in %d iterations, rel_res=%.6e, time=%.3f s\n",
               iter, rel, t1 - t0);
        printf("Profile breakdown (s): comm=%.6f, spmv=%.6f, dot=%.6f, axpy=%.6f, allreduce=%.6f\n",
               t_comm, t_spmv, t_dot, t_axpy, t_allreduce);
    }

    csr_free(&A_local);
    free(b_local);
    free(x_local);
    free(r_local);
    free(p_local);
    free(Ap_local);
    free(halo_top);
    free(halo_bottom);
    return iter;
}
