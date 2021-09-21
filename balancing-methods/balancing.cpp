#include <iostream>
#include <fstream>
#include <vector>
#include <omp.h>

const int threadNum = 4;
const int iterNum = 65;

void printThreadUsage(int tNum, int iNum, std::vector<int> &iterations, std::string &name)
{
    std::cout << "Results for " << name << " scheduling.\n";
    for (int tid = 0; tid < tNum; tid++)
    {
        std::cout << tid << ": ";
        for (int i = 0; i < iNum; i++)
        {
            if (iterations.at(i) != tid)
                std::cout << " ";
            else
                std::cout << "*";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main(int argc, char **argv)
{
    omp_set_num_threads(threadNum);
    std::vector<int> iterations(iterNum, -1);
    std::string name;

    // Default scheduling
    name = "default";
#pragma omp parallel for default(none) shared(iterNum, iterations)
    for (int i = 0; i < iterNum; i++)
    {
        iterations[i] = omp_get_thread_num();
    }
    printThreadUsage(threadNum, iterNum, iterations, name);

    // Static scheduling
    name = "static, 1";
#pragma omp parallel for schedule(static, 1) default(none) shared(iterNum, iterations)
    for (int i = 0; i < iterNum; i++)
    {
        iterations[i] = omp_get_thread_num();
    }
    printThreadUsage(threadNum, iterNum, iterations, name);

    name = "static, 4";
#pragma omp parallel for schedule(static, 4) default(none) shared(iterNum, iterations)
    for (int i = 0; i < iterNum; i++)
    {
        iterations[i] = omp_get_thread_num();
    }
    printThreadUsage(threadNum, iterNum, iterations, name);

    // Dynamic scheduling
    name = "dynamic, 1";
#pragma omp parallel for schedule(monotonic:dynamic, 1) default(none) shared(iterNum, iterations)
    for (int i = 0; i < iterNum; i++)
    {
        iterations[i] = omp_get_thread_num();
    }
    printThreadUsage(threadNum, iterNum, iterations, name);

    name = "dynamic, 4";
#pragma omp parallel for ordered schedule(dynamic, 4) default(none) shared(iterNum, iterations)
    for (int i = 0; i < iterNum; i++)
    {
        iterations[i] = omp_get_thread_num();
    }
    printThreadUsage(threadNum, iterNum, iterations, name);

    // Guided scheduling
    name = "guided, 1";
#pragma omp parallel for schedule(guided, 1) default(none) shared(iterNum, iterations)
    for (int i = 0; i < iterNum; i++)
    {
        iterations[i] = omp_get_thread_num();
    }
    printThreadUsage(threadNum, iterNum, iterations, name);

    name = "guided, 4";
#pragma omp parallel for schedule(guided, 4) default(none) shared(iterNum, iterations)
    for (int i = 0; i < iterNum; i++)
    {
        iterations[i] = omp_get_thread_num();
    }
    printThreadUsage(threadNum, iterNum, iterations, name);

    return 0;
}
