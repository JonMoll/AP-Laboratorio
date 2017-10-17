#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include <time.h>
#include <stdlib.h>

int main(int argc, char *argv[]){
    int thread_count = strtol(argv[1], NULL, 10);

    int i, j;

    int m = 8;
    int n = 80000;

    int A[m][n];
    int x[n];

    // RELLENANDO MATRIZ Y VECTOR
    srand(time(NULL));

    for(i = 0; i < m; i++){
        for(j = 0; j < n; j++)
            A[i][j] = rand();
    }

    for(i = 0; i < n; i++)
        x[i] = rand();

    int y[m];

    clock_t begin = clock();

    #pragma omp parallel for num_threads(thread_count) \
    default(none) private(i, j) shared(A, x, y, m, n)

    for(i = 0; i < m; i++){
        y[i] = 0;

        for(j = 0; j < n; j++)
            y[i] += A[i][j] * x[j];
    }

    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

    /*printf("matrix y: ");

    for(i = 0; i < m; i++)
        printf("%d ", y[i]);

    printf("\n");*/

    printf("time: %f sec\n", time_spent);

    return 0;
}
