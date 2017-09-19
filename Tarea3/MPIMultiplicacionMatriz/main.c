#include <stdio.h>
#include <mpi.h>

int main()
{
    MPI_Init(NULL, NULL);

    double local_A[]; /* local 2D array A (local_m rows x n colums)stored as 1D array*/
    double local_x[]; /* local vector x with local_n elements*/
    double local_y[]; /* resultant local vector y = A.x with local_m elements*/
    int local_m; /* the number of rows in A assigned per process.*/
    int n; /* the number of elements in vector x*/
    int local_n; /* the number of elements in x assigned per process*/

    double *x;
    int local_i, j;

    x = malloc(n*sizeof(double));
    MPI_Allgather(local_x, local_n, MPI_DOUBLE, x, local_n, MPI_DOUBLE, MPI_COMM_WORLD);

    for(local_i = 0; local_i < local_m; local_i++){
        local_y[local_i] = 0.0;

        for(j = 0; j < n; j++)
            local_y[local_i] += local_A[local_i*n+j] * x[j];
    }

    free(x);

    MPI_Finalize();

    return 0;
}
