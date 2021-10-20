#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <stdlib.h>
#include <time.h>
#include <cstdlib>
#include "omp.h"

int main(int argc, char **argv) {
    // инициализация матриц
    int num_threads = std::stoi(argv[1]);
    int N = std::stoi(argv[2]);

    omp_set_num_threads(num_threads);
    
    int array1[N][N];   
    int array2[N][N];     
    srand(time(NULL));
    for(int i = 0; i < N; i++)  
        for(int j = 0; j < N; j++){
            array1[i][j] = 1 + rand() % 10000;
            array2[i][j] = 1 + rand() % 10000;
        }
    
    std::cout << "Matrix 1 is \n";
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
            std::cout << array1[i][j] << " ";
        std::cout << std::endl;
    }    

    std::cout << "Matrix 2 is \n";
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
            std::cout << array2[i][j] << " ";
        std::cout << std::endl;
    }

    double start = omp_get_wtime();

    int **res = (int**)malloc(sizeof(int*)*N);;
    for(int i = 0; i < N; i++) {
        res[i] = (int*)malloc(sizeof(int)*N);
    }
    omp_set_num_threads(num_threads);
    int i, j, k;
    #pragma omp parallel for shared(array1, array2, res) private(i, j, k)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            res[i][j] = 0;
            for (k = 0; k < N; k++) {
                res[i][j] += (array1[i][k] * array2[k][j]);
            }
        }
    }
    
    double finish = omp_get_wtime();
    std::cout << "Result is " << res[0][0] << "\nTime spend " << finish - start << std::endl;
    free(res);

    // создание метода перемножения матриц без распараллеливания
    res = (int**)malloc(sizeof(int*)*N);
    for(int i = 0; i < N; i++) {
        res[i] = (int*)malloc(sizeof(int)*N);
    }

    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            res[i][j] = 0;
            for(int k = 0; k < N; k++) {
                res[i][j] += (array1[i][k] * array2[k][j]);
            }
        }
    }
    finish = omp_get_wtime();
    std::cout << "Result without parralel is " << res[0][0] << "\nTime spend " << finish - start << std::endl;
    free(res);
    return 0;
}
