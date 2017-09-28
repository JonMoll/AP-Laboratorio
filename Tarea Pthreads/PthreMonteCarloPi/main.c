#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define SEED 35791246

int thread_count;
int subset_size;
int count[1024];

void *MonteCarloPi(void *rank){
    long my_rank = (long) rank;

    double x, y, z;
    int i;

    count[my_rank] = 0;

    for(i = 0; i < subset_size; i++){
        x = (double)rand() / RAND_MAX;
        y = (double)rand() / RAND_MAX;
        z = x*x + y*y;

        if(z <= 1)
            count[my_rank] = count[my_rank] + 1;
    }

    printf("[Process %d] End \n", my_rank);

    return NULL;
}

int main(int argc, char* argv[]){
    srand(SEED);

    long thread;

    pthread_t *thread_handles;

    thread_count = strtol(argv[1], NULL, 10);

    thread_handles = malloc(thread_count * sizeof(pthread_t));

    int iterations = 800000;
    subset_size = iterations / thread_count;

    for(thread = 0; thread < thread_count; thread++)
        pthread_create(&thread_handles[thread], NULL, MonteCarloPi, (void*) thread);

    for(thread = 0; thread < thread_count; thread++)
        pthread_join(thread_handles[thread], NULL);

    free(thread_handles);

    // JUNTANDO TODOS LOS RESULTADOS
    int i;
    int final_count = 0;

    for(i = 0; i < thread_count; i++)
        final_count += count[i];

    // MOSTRANDO EL RESULTADO
    double pi = (double)final_count/iterations*4;

    printf("\n# of trials per process = %d \n", subset_size);
    printf("# of trials = %d , estimate of pi is %g \n", iterations, pi);

    return 0;
}
