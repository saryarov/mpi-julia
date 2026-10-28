#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define NITER 50000
#define WARMUP 100

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int sizes[] = {64, 256, 1024, 4096, 16384, 65536, 262144, 1048576};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    // ==========================================
    // 1. ТЕСТ BROADCAST
    // ==========================================
    if (rank == 0) {
        printf("\n--- C Benchmark: Bcast (%d iterations, %d processes) ---\n", NITER, nprocs);
        printf("%10s | %12s | %12s | %8s\n", "size(B)", "regular (us)", "persist (us)", "speedup");
        printf("--------------------------------------------------\n");
    }

    for (int i = 0; i < num_sizes; i++) {
        int n_elements = sizes[i] / sizeof(double);
        double *buf = (double*)malloc(n_elements * sizeof(double));
        if (!buf) { MPI_Abort(MPI_COMM_WORLD, 1); }
        for (int j = 0; j < n_elements; j++) buf[j] = (double)j;

        // --- Обычный Bcast ---
        for (int w = 0; w < WARMUP; w++) MPI_Bcast(buf, n_elements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        double t_start_reg = MPI_Wtime();
        for (int iter = 0; iter < NITER; iter++) {
            MPI_Bcast(buf, n_elements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
        double t_reg = MPI_Wtime() - t_start_reg;

        // --- Персистентный Bcast ---
        MPI_Request preq;
        MPI_Bcast_init(buf, n_elements, MPI_DOUBLE, 0, MPI_COMM_WORLD, MPI_INFO_NULL, &preq);
        for (int w = 0; w < WARMUP; w++) { MPI_Start(&preq); MPI_Wait(&preq, MPI_STATUS_IGNORE); }
        MPI_Barrier(MPI_COMM_WORLD);
        double t_start_pers = MPI_Wtime();
        for (int iter = 0; iter < NITER; iter++) {
            MPI_Start(&preq);
            MPI_Wait(&preq, MPI_STATUS_IGNORE);
        }
        double t_pers = MPI_Wtime() - t_start_pers;
        MPI_Request_free(&preq);

        if (rank == 0) {
            double speedup = t_reg / t_pers;
            printf("%10d | %12.2f | %12.2f | %7.2fx\n", sizes[i], (t_reg / NITER) * 1e6, (t_pers / NITER) * 1e6, speedup);
        }
        free(buf);
    }

    // ==========================================
    // 2. ТЕСТ ALLREDUCE
    // ==========================================
    if (rank == 0) {
        printf("\n--- C Benchmark: Allreduce (%d iterations, %d processes) ---\n", NITER, nprocs);
        printf("%10s | %12s | %12s | %8s\n", "size(B)", "regular (us)", "persist (us)", "speedup");
        printf("--------------------------------------------------\n");
    }

    for (int i = 0; i < num_sizes; i++) {
        int n_elements = sizes[i] / sizeof(double);
        double *sendbuf = (double*)malloc(n_elements * sizeof(double));
        double *recvbuf = (double*)malloc(n_elements * sizeof(double));
        if (!sendbuf || !recvbuf) { MPI_Abort(MPI_COMM_WORLD, 1); }
        for (int j = 0; j < n_elements; j++) sendbuf[j] = (double)(j + rank);

        // --- Обычный Allreduce ---
        for (int w = 0; w < WARMUP; w++) MPI_Allreduce(sendbuf, recvbuf, n_elements, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        double t_start_reg = MPI_Wtime();
        for (int iter = 0; iter < NITER; iter++) {
            MPI_Allreduce(sendbuf, recvbuf, n_elements, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        }
        double t_reg = MPI_Wtime() - t_start_reg;

        // --- Персистентный Allreduce ---
        MPI_Request preq_ar;
        MPI_Allreduce_init(sendbuf, recvbuf, n_elements, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, MPI_INFO_NULL, &preq_ar);
        for (int w = 0; w < WARMUP; w++) { MPI_Start(&preq_ar); MPI_Wait(&preq_ar, MPI_STATUS_IGNORE); }
        MPI_Barrier(MPI_COMM_WORLD);
        double t_start_pers = MPI_Wtime();
        for (int iter = 0; iter < NITER; iter++) {
            MPI_Start(&preq_ar);
            MPI_Wait(&preq_ar, MPI_STATUS_IGNORE);
        }
        double t_pers = MPI_Wtime() - t_start_pers;
        MPI_Request_free(&preq_ar);

        if (rank == 0) {
            double speedup = t_reg / t_pers;
            printf("%10d | %12.2f | %12.2f | %7.2fx\n", sizes[i], (t_reg / NITER) * 1e6, (t_pers / NITER) * 1e6, speedup);
        }
        free(sendbuf);
        free(recvbuf);
    }

    MPI_Finalize();
    return 0;
}