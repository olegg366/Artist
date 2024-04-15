#include <stdlib.h> 
#include <stdio.h>

void foobar(int m, int n, double **x) 
{ 
    for(int i = 0; i < m; i++) 
    {    
        for(int j = 0; j < n; j++) 
        {    
            printf("%f ", x[i][j]);
        }
        printf("\n");
    }
} 