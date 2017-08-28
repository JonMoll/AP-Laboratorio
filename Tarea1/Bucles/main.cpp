#include <iostream>

#include <stdlib.h> // srand, rand
#include <time.h> // time

using namespace std;

int main()
{
    srand(time(NULL));
    int MAX = 10;

    double A[MAX][MAX], x[MAX], y[MAX];

    // RELLENANDO LA MATRIZ Y LOS ARRAYS
    for(int i = 0; i < MAX; i++){
        x[i] = rand() % 100;
        y[i] = rand() % 100;

        for(int j = 0; j < MAX; j++)
            A[i][j] = rand() % 100;
    }

    // BUCLE 1
    for(int i = 0; i < MAX; i++)
        for(int j = 0; j < MAX; j++)
            y[i] += A[i][j]* x[j];

    // BUCLE 2
    for(int j = 0; j < MAX; j++)
        for(int i = 0; i < MAX; i++)
            y[i] += A[i][j]* x[j];

    cout << y[0] << endl;

    return 0;
}
