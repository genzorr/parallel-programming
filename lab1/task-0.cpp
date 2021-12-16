#include <cstdio>
#include <cmath>
#include <omp.h>
#include <iostream>

#define ISIZE 10000
#define JSIZE 10000

void writeArray(double **a, char *name)
{
    FILE *ff;
    ff = fopen(name, "w");
    for (int i = 0; i < ISIZE; i++)
    {
        for (int j = 0; j < JSIZE; j++)
            fprintf(ff, "%.7e ", a[i][j]);
        fprintf(ff,"\n");
    }
    fclose(ff);
}

void taskSequential(double **a)
{
    int i, j;

    for (i = 0; i < ISIZE; i++)
    {
        for (j = 0; j < JSIZE; j++)
            a[i][j] = 10*i +j;
    }

    for (i = 0; i<ISIZE; i++)
    {
        for (j = 0; j < JSIZE; j++)
            a[i][j] = sin(0.00001 * a[i][j]);
    }
}

void taskParallel(double **a, int nThreads)
{
#pragma omp parallel for num_threads(nThreads)
    for (int i = 0; i < ISIZE; i++)
    {
        for (int j = 0; j < JSIZE; j++)
            a[i][j] = 10*i +j;
    }

#pragma omp parallel for num_threads(nThreads)
    for (int i = 0; i<ISIZE; i++)
    {
        for (int j = 0; j < JSIZE; j++)
            a[i][j] = sin(0.00001 * a[i][j]);
    }
}

int main(int argc, char **argv)
{
    double start = 0., finish = 0.;
    std::cout.precision(4);
    std::cout << "Lab 1, task 0" << std::scientific << "\n";

    double **a = (double **)calloc(ISIZE, sizeof(*a));
    for (int i = 0; i < JSIZE; i++)
        a[i] = (double *)calloc(JSIZE, sizeof(double));

    // Timings
    FILE *ff;
    ff = fopen("times.txt", "w");

    // Run sequential program
    start = omp_get_wtime();
    taskSequential(a);
    finish = omp_get_wtime();
    std::cout << "Sequential, time: " << finish - start << "s.\n";
    fprintf(ff, "%d\t%.4e\n", 1, finish - start);
//    writeArray(a, "result-0-seq.txt");

    // Run parallel programs
    for (int nThreads = 2; nThreads <= 6; nThreads += 2)
    {
        start = omp_get_wtime();
        taskParallel(a, nThreads);
        finish = omp_get_wtime();
        std::cout << "Parallel, nThreads = " << nThreads << ", time: " << finish - start << "s.\n";
        fprintf(ff, "%d\t%.4e\n", nThreads, finish - start);

//        if (nThreads == 6)
//            writeArray(a, "result-0-par.txt");
    }
    fclose(ff);

    for (int i = 0; i < JSIZE; i++)
        free(a[i]);
    free(a);

    return 0;
}