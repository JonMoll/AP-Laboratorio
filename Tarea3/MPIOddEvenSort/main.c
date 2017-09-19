#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv){
    int numberOfProcesses;
    int myRank;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    int elements[] = {5, 1, 14, 13, 11, 11, 18, 62};
    int recvElements[1];
    int i;

    MPI_Scatter(&elements, 1, MPI_INT, &recvElements, 1, MPI_INT, 0, MPI_COMM_WORLD);

    for(i = 0; i < numberOfProcesses; i++){
        if(i%2 == 0){
            if(myRank%2 == 0){
                /*printf("[Process %d] old recvElements[0] %d \n", myRank, recvElements[0]);
                printf("[Process %d] old recvElements[1] %d \n", myRank, recvElements[1]);*/

                MPI_Send(&recvElements[0], 1, MPI_INT, myRank+1, 0, MPI_COMM_WORLD);
                MPI_Recv(&recvElements[1], 1, MPI_INT, myRank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                /*printf("[Process %d] new recvElements[0] %d \n", myRank, recvElements[0]);
                printf("[Process %d] new recvElements[1] %d \n \n", myRank, recvElements[1]);*/

                if(recvElements[1] < recvElements[0])
                    recvElements[0] = recvElements[1];
            }
            else{
                MPI_Recv(&recvElements[1], 1, MPI_INT, myRank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(&recvElements[0], 1, MPI_INT, myRank-1, 0, MPI_COMM_WORLD);

                if(recvElements[1] > recvElements[0])
                    recvElements[0] = recvElements[1];
            }
        }
        else{
            if(myRank%2 != 0 && myRank != (numberOfProcesses-1)){
                MPI_Send(&recvElements[0], 1, MPI_INT, myRank+1, 0, MPI_COMM_WORLD);
                MPI_Recv(&recvElements[1], 1, MPI_INT, myRank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                if(recvElements[1] < recvElements[0])
                    recvElements[0] = recvElements[1];
            }
            else if(myRank != 0 && myRank != (numberOfProcesses-1)){
                MPI_Recv(&recvElements[1], 1, MPI_INT, myRank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(&recvElements[0], 1, MPI_INT, myRank-1, 0, MPI_COMM_WORLD);

                if(recvElements[1] > recvElements[0])
                    recvElements[0] = recvElements[1];
            }
        }
    }

    MPI_Gather(&recvElements[0], 1, MPI_INT, &elements[0], 1, MPI_INT, 0, MPI_COMM_WORLD);

    if(myRank == 0){
        for(i = 0; i < numberOfProcesses; i++)
            printf("%d \n", elements[i]);
    }

    MPI_Finalize();

    return 0;
}
