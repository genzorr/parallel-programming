#include <cstdio>
#include <cmath>
#include <omp.h>
#include <iostream>

#define ISIZE 1000
#define JSIZE 1000

void writeArray(double a[ISIZE][JSIZE], char *name)
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

void taskSequential(double a[ISIZE][JSIZE])
{
    for (int i = 0; i < ISIZE; i++)
    {
        for (int j = 0; j < JSIZE; j++)
            a[i][j] = 10*i +j;
    }

    for (int i = 1; i < ISIZE; i++)
    {
        for (int j = 3; j < JSIZE - 1; j++)
            a[i][j] = sin(0.00001 * a[i - 1][j - 3]);
    }
}

void taskParallel(double a[ISIZE][JSIZE], int nThreads)
{
#pragma omp parallel for num_threads(nThreads)
    for (int i = 0; i < ISIZE; i++)
    {
        for (int j = 0; j < JSIZE; j++)
            a[i][j] = 10*i +j;
    }

    for (int i = 1; i < ISIZE; i++)
    {
    #pragma omp parallel for num_threads(nThreads)
        for (int j = 3; j < JSIZE - 1; j++)
            a[i][j] = sin(0.00001 * a[i - 1][j - 3]);
    }
}

int main(int argc, char **argv)
{
    double start = 0., finish = 0.;
    std::cout.precision(4);
    std::cout << "Lab 1, task 1, true dependence" << std::scientific << "\n";
    double a[ISIZE][JSIZE];

    // Timings
    FILE *ff;
    ff = fopen("times.txt", "w");

    // Run sequential program
    start = omp_get_wtime();
    taskSequential(a);
    finish = omp_get_wtime();
    std::cout << "Sequential, time: " << finish - start << "s.\n";
    fprintf(ff, "%d\t%.4e\n", 1, finish - start);
    writeArray(a, "result-1-seq.txt");

    // Run parallel programs
    for (int nThreads = 2; nThreads <= 6; nThreads += 2)
    {
        start = omp_get_wtime();
        taskParallel(a, nThreads);
        finish = omp_get_wtime();
        std::cout << "Parallel, nThreads = " << nThreads << ", time: " << finish - start << "s.\n";
        fprintf(ff, "%d\t%.4e\n", nThreads, finish - start);

        if (nThreads == 6)
            writeArray(a, "result-1-par.txt");
    }
    fclose(ff);

    return 0;
}
