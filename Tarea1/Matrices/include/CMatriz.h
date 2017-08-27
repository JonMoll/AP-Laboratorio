#ifndef CMATRIZ_H
#define CMATRIZ_H

#include <iostream>
#include <stdlib.h> // srand, rand
#include <time.h> // time

using namespace std;

class CMatriz{
    private:
        int** m_matriz;

    public:
        int m_filas;
        int m_columnas;

        void Rellenar();
        void Vaciar();

        int Obtener(int fila, int columna);
        void Establecer(int fila, int columna, int elemento);
        void Imprimir();

        CMatriz(int filas, int columnas, bool vacia);
        virtual ~CMatriz();
};

#endif
