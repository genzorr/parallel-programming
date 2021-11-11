#include <cstdio>
#include <omp.h>

int x = 123;
int y = 123;
#pragma omp threadprivate(x)

int main(int argc, char **argv)
{
    omp_set_dynamic(0);

    printf("Run without copyin clause.\n");
#pragma omp parallel default(none) private(y)
    printf("Thread %d: n = %d\n", omp_get_thread_num(), y);

    printf("\nRun with copyin clause.\n");
#pragma omp parallel default(none) copyin(x)
    printf("Thread %d: n = %d\n", omp_get_thread_num(), x);
    return 0;
}
