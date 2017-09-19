#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv)
{
    int numberOfProcesses;
    int myRank;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    int i;
    int sendTo, recivFrom;
    int myElement = 1;
    int elementRecv;

    int lastProcess = numberOfProcesses;

    int beginRank = lastProcess / 2;
    int endRank = lastProcess - 1;
    lastProcess = lastProcess / 2;

    while(beginRank != endRank){
        for(i = 0; i < beginRank; i++){
            if(myRank == i){
                recivFrom = i + beginRank;
                printf("[Process %d] recivFrom = %d \n", myRank, recivFrom);
                MPI_Recv(&elementRecv, 1, MPI_INT, recivFrom, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                myElement += elementRecv;

                break;
            }
        }

        for(i = beginRank; i <= endRank; i++){
            if(myRank == i){
                sendTo = i - beginRank;
                printf("[Process %d] sendTo = %d \n", myRank, sendTo);
                MPI_Send(&myElement, 1, MPI_INT, sendTo, 0, MPI_COMM_WORLD);

                break;
            }
        }

        beginRank = lastProcess / 2;
        endRank = lastProcess - 1;
        lastProcess = lastProcess / 2;
    }

    if(myRank == 0)
        printf("Sum = %d \n", myElement);

    MPI_Finalize();

    return 0;
}
