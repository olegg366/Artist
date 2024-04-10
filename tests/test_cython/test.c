#include <stdlib.h> 

__declspec(dllexport) void foobar(const int m, const int n, const 
double **x, double **y) 
{ 
    size_t i, j; 
    for(i=0; i<m; i++) 
        for(j=0; j<n; j++) 
            y[i][j] = x[i][j]; 
} 