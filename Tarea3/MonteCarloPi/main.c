#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#define SEED 35791246

int main(int argc, char** argv)
{
    int numberOfProcesses;
    int myRank;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    int iterations = 100000;
    double x, y, z, pi;
    int i;
    int count = 0;
    int final_count = 0;
    int subset_size = iterations / (numberOfProcesses - 1);

    srand(SEED);

    if(myRank != 0){
        printf("[Process %d] Begin \n", myRank);

        for(i = 0; i < subset_size; i++){
            x = (double)rand() / RAND_MAX;
            y = (double)rand() / RAND_MAX;
            z = x*x + y*y;

            if(z <= 1) count++;
        }

        printf("[Process %d] End \n", myRank);
    }

    MPI_Reduce(&count, &final_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if(myRank == 0){
        pi = (double)final_count/iterations*4;

        printf("\n# of trials per process = %d \n", subset_size);
        printf("# of trials = %d , estimate of pi is %g \n", iterations, pi);
    }

    MPI_Finalize();

    return 0;
}
