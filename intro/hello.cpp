#include <cstdio>
#include <omp.h>

int main()
{
    int id;
#pragma omp parallel default(none) private(id)
    {
        id = omp_get_thread_num();
        printf("Hello, World from %d thread\n", id);
    }
    return 0;
}