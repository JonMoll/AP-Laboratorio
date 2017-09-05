#include <stdio.h>
#include <string.h>
#include <mpi.h>

int main(int argc, char** argv){
	int comm_sz;
	int my_rank;
	int limite = 10;
	int i = 0;

	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	if(my_rank == 0){
		i++;
		MPI_Send(&i, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
		printf("Proceso %d : i = %d \n", my_rank, i);
	}

	while(i < limite){
		MPI_Recv(&i, 1, MPI_INT, (my_rank+1)%2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		i++;
		printf("Proceso %d : i = %d \n", my_rank, i);
		MPI_Send(&i, 1, MPI_INT, (my_rank+1)%2, 0, MPI_COMM_WORLD);
	}

	MPI_Finalize();

	return 0;
}