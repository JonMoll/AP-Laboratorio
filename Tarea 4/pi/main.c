#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include <time.h>

int main(int argc, char *argv[]){
    int thread_count = strtol(argv[1], NULL, 10);

    int k;
    int n = 80000;

    /*printf("n: ");
    scanf("%d", &n);*/

    double factor = 1.0;
    double sum = 0.0;

    clock_t begin = clock();

    #pragma omp parallel for num_threads(thread_count) \
    reduction(+:sum) private(factor)

    for(k = 0; k < n; k++){
        if(k%2 == 0)
            factor = 1.0;
        else
            factor = -1.0;

        sum += factor/(2*k+1);
    }

    double pi_approx = 4.0*sum;

    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

    printf("time: %f sec\n", time_spent);

    printf("Pi approx: %e\n", pi_approx);

    return 0;
}
