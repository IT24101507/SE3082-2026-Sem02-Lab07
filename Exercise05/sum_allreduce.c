#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000000

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int chunk_size = N / size;

    int *array = NULL;
    int *local_chunk = malloc(chunk_size * sizeof(int));

    if (rank == 0) {
        array = malloc(N * sizeof(int));
        for (int i = 0; i < N; i++)
            array[i] = i + 1;
        printf("Root filled array with values 1 to %d\n", N);
    }

    double start = MPI_Wtime();

    MPI_Scatter(array, chunk_size, MPI_INT,
                local_chunk, chunk_size, MPI_INT,
                0, MPI_COMM_WORLD);

    long long local_sum = 0;
    for (int i = 0; i < chunk_size; i++)
        local_sum += local_chunk[i];

    int start_idx = rank * chunk_size;
    printf("  Rank %d: summed indices [%d, %d) => local_sum = %lld\n",
           rank, start_idx, start_idx + chunk_size, local_sum);

    /* ALLREDUCE: every process gets the total (no root parameter) */
    long long total_sum = 0;
    MPI_Allreduce(&local_sum, &total_sum, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    double pct = 100.0 * (double)local_sum / (double)total_sum;
    printf("  Rank %d: local_sum = %lld, total = %lld, contribution = %.2f%%\n",
           rank, local_sum, total_sum, pct);

    if (rank == 0) {
        double elapsed = MPI_Wtime() - start;
        long long expected = (long long)N * (N + 1) / 2;
        printf("\n[Allreduce] Total sum = %lld\n", total_sum);
        printf("[Allreduce] Expected    = %lld\n", expected);
        printf("[Allreduce] Correct?    = %s\n", total_sum == expected ? "YES" : "NO");
        printf("[Allreduce] Time        = %.4f sec\n", elapsed);
    }

    free(local_chunk);
    if (rank == 0) free(array);
    MPI_Finalize();
    return 0;
}
