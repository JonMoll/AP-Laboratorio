#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include <time.h>
#include <stdlib.h>

int main(int argc, char *argv[]){
    int thread_count = strtol(argv[1], NULL, 10);

    int n = 80000;
    int a[n];
    int phase, i, tmp;

    // RELLENANDO VECTOR
    srand(time(NULL));

    for(i = 0; i < n; i++)
        a[i] = rand();

    clock_t begin = clock();

    #pragma omp parallel num_threads(thread_count) \
    default(none) shared(a, n) private(i, tmp, phase)

    for(phase = 0; phase < n; phase++){
        if(phase%2 == 0){
            #pragma omp for

            for(i = 1; i < n; i+=2){
                if(a[i-1] > a[i]){
                    tmp = a[i-1];
                    a[i-1] = a[i];
                    a[i] = tmp;
                }
            }
        }
        else{
            #pragma omp for

            for(i = 1; i < n-1; i+=2){
                if(a[i] > a[i+1]){
                    tmp = a[i+1];
                    a[i+1] = a[i];
                    a[i] = tmp;
                }
            }
        }
    }

    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

    printf("time: %f sec\n", time_spent);

    /*printf("Array:");

    for(i = 0; i < n; i++)
        printf(" %d ", a[i]);

    printf("\n");*/

    return 0;
}
