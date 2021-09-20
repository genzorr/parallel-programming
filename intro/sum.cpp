#include <iostream>
#include <omp.h>

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cout << "Pass the number of elements in sum.";
        return -1;
    }

    int N = atoi(argv[1]);
    double sum = 0;

#pragma omp parallel for default(none) shared(N) reduction(+:sum)
    for (int i = 1; i <= N; i++)
    {
        sum += 1. / i;
    }
    std::cout << "Sum = " << sum;
    return 0;
}
