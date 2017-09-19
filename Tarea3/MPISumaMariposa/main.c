#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv)
{
    int numberOfProcesses;
    int myRank;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    int i, j;
    int myElement = 1;
    int myFinalElement = 1;
    int elementReciv;
    int jump = 2;

    while(jump < numberOfProcesses){
        for(i = 0; i < numberOfProcesses; i+=jump){
            for(j = i; j < (jump/2); j++){
                if(myRank == j+1){
                    MPI_Recv(&elementReciv, 1, MPI_INT, j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    myFinalElement += elementReciv;
                }

                if(myRank == j)
                    MPI_Send(&myElement, 1, MPI_INT, j+1, 0, MPI_COMM_WORLD);

                // --------------------------------------------------

                if(myRank == j){
                    MPI_Recv(&elementReciv, 1, MPI_INT, j+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    myFinalElement += elementReciv;

                    myElement = myFinalElement;
                    break;
                }

                if(myRank == j+1){
                    MPI_Send(&myElement, 1, MPI_INT, j, 0, MPI_COMM_WORLD);

                    myElement = myFinalElement;
                    break;
                }
            }
        }

        jump = jump * 2;
    }

    printf("[Process %d] myElement = %d \n", myRank, myElement);

    MPI_Finalize();

    return 0;
}
