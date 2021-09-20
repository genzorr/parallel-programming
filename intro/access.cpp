#include <cstdio>
#include <omp.h>

int main(int argc, char **argv)
{
    int x = 0;
#pragma omp parallel for ordered default(none) shared(x)
    for (int i = 0; i < omp_get_num_threads(); i++)
    {
        int tid = omp_get_thread_num();
    #pragma omp ordered
        {
            x += 1;
            printf("Thread %d, x_prev = %d, x = %d\n", tid, x-1, x);
        }
    }
    return 0;
}
