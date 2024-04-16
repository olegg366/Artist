#include <stdlib.h> 
#include <stdio.h>

void foobar(int m, int n, int l, double ***x) 
{ 
    for(int i = 0; i < m; i++) 
    {    
        for(int j = 0; j < n; j++) 
        {    
            for (int k = 0; k < l; k++)
                printf("%f ", x[i][j][k]);
        }
        printf("\n");
    }
} 