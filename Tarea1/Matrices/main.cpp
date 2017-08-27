#include "CMatriz.h"

CMatriz MultiplicacionClasica(CMatriz a, CMatriz b){
    CMatriz r(a.m_filas, b.m_columnas, true);

    return r;
}

int main()
{
    srand(time(NULL));

    CMatriz a(2, 4, true);
    a.Imprimir();

    return 0;
}
