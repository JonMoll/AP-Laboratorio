#include "CMatriz.h"

int main()
{
    srand(time(NULL));

    int filas_a = 4;
    int columnas_a = 2;
    int filas_b = 2;
    int columnas_b = 6;

    CMatriz a(filas_a, columnas_a, false);
    CMatriz b(filas_b, columnas_b, false);
    CMatriz r(filas_a, columnas_b, true); // RESULTADO

    a.Imprimir();
    b.Imprimir();

    // MULTIPLICACION DE MATRICES CLASICA ==================================================
    for(int i = 0; i < filas_a; i++){
        for(int j = 0; j < columnas_b; j++){
            int acumulado = 0;

            for(int k = 0; k < columnas_a; k++)
                acumulado += a.Obtener(i, k) * b.Obtener(k, j);

            r.Establecer(i, j, acumulado);
        }
    }

    r.Imprimir();
    r.Vaciar();
    r.Imprimir();

    // MULTIPLICACION DE MATRICES POR BLOQUES ==================================================
    int matriz_cuadrada = 2; // EL TAMANO DE LOS BLOQUES
    CMatriz acumulado(matriz_cuadrada, matriz_cuadrada, true);

    for(int i = 0; i < filas_a; i += matriz_cuadrada){
        for(int j = 0; j < columnas_b; j += matriz_cuadrada){
            acumulado.Vaciar();

            for(int k = 0; k < columnas_a; k += matriz_cuadrada){
                // MULTIPLICACION DE MATRICES CLASICA
                for(int l = i, ii = 0; ii < matriz_cuadrada; l++, ii++){
                    for(int m = j, jj = 0; jj < matriz_cuadrada; m++, jj++){
                        int acumulado_unitario = 0;

                        for(int n = k; n < k + matriz_cuadrada; n++)
                            acumulado_unitario += a.Obtener(l, n) * b.Obtener(n, m);

                        acumulado_unitario += acumulado.Obtener(ii, jj);
                        acumulado.Establecer(ii, jj, acumulado_unitario);
                    }
                }
            }

            // GUARDANDO ACUMULADO
            for(int k = i, ii = 0; ii < matriz_cuadrada; k++, ii++){
                for(int l = j, jj = 0; jj < matriz_cuadrada; l++, jj++)
                    r.Establecer(k, l, acumulado.Obtener(ii, jj));
            }
        }
    }

    r.Imprimir();

    return 0;
}
