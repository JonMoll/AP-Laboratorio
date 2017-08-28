#include "CMatriz.h"

#define ALEATORIO_MAXIMO 100;

CMatriz::CMatriz(int filas, int columnas, bool vacia){
    m_filas = filas;
    m_columnas = columnas;

    m_matriz = new int* [filas];

    for(int i = 0; i < filas; i++)
        m_matriz[i] = new int [columnas];

    if(vacia) Vaciar();
    else Rellenar();
}

CMatriz::~CMatriz(){
    for(int i = 0; i < m_filas; i++)
        delete [] m_matriz[i];

    delete [] m_matriz;
}

void CMatriz::Rellenar(){
    for(int i = 0; i < m_filas; i++){
        for(int j = 0; j < m_columnas; j++)
            m_matriz[i][j] = rand() % ALEATORIO_MAXIMO;
    }
}

void CMatriz::Vaciar(){
    for(int i = 0; i < m_filas; i++){
        for(int j = 0; j < m_columnas; j++)
            m_matriz[i][j] = 0;
    }
}

int CMatriz::Obtener(int fila, int columna){
    return m_matriz[fila][columna];
}

void CMatriz::Establecer(int fila, int columna, int elemento){
    m_matriz[fila][columna] = elemento;
}

void CMatriz::Imprimir(){
    for(int i = 0; i < m_filas; i++){
        for(int j = 0; j < m_columnas; j++)
            cout << m_matriz[i][j] << "  ";

        cout << endl;
    }

    cout << endl;
}
