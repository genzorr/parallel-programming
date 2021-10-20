#include <iostream>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <string>
#include <fstream>

const uint16_t maxN = 2048;
typedef int64_t data_t;
data_t m1cached[maxN * maxN];
data_t m2cached[maxN * maxN];

void sequentialMultiply(uint16_t N, const data_t *m1, const data_t *m2, data_t *res)
{
    for (uint16_t i = 0; i < N; i++)
    {
        uint32_t iOff = i * N;
        for (uint16_t j = 0; j < N; j++)
        {
            data_t sum = 0;
            res[iOff + j] = 0;
            for (uint16_t k = 0; k < N; k++)
            {
                sum += (m1[iOff + k] * m2[k * N + j]);
            }
            res[iOff + j] = sum;
        }
    }
}

void parallelMultiply(uint16_t N, const data_t *m1, const data_t *m2, data_t *res)
{
#pragma omp parallel for default(none) shared(m1, m2, res, N)
    for (uint16_t i = 0; i < N; i++)
    {
        uint32_t iOff = i * N;
        for (uint16_t j = 0; j < N; j++)
        {
            data_t sum = 0;
            res[iOff + j] = 0;
            for (uint16_t k = 0; k < N; k++)
            {
                sum += (m1[iOff + k] * m2[k * N + j]);
            }
            res[iOff + j] = sum;
        }
    }
}

void convertToCached(uint16_t N, const data_t *m1, const data_t *m2, data_t *m1Cached, data_t *m2Cached)
{
#pragma omp parallel for default(none) shared(N, m1, m2, m1Cached, m2Cached)
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            m1Cached[i * N + j] = m1[i * N + j];
            m2Cached[j * N + i] = m2[i * N + j];
        }
    }
}

void parallelOptimizedMultiply(uint16_t N, const data_t *m1, const data_t *m2, data_t *res)
{
    convertToCached(N, m1, m2, m1cached, m2cached);

#pragma omp parallel for default(none) shared(m1cached, m2cached, res, N)
    for (uint16_t i = 0; i < N; i++)
    {
        uint32_t iOff = i * N;
        for (uint16_t j = 0; j < N; j++)
        {
            data_t sum = 0;
            uint32_t jOff = j * N;
            res[iOff + j] = 0;
        #pragma omp simd
            for (uint16_t k = 0; k < N; k++)
            {
                sum += (m1cached[iOff + k] * m2cached[jOff + k]);
            }
            res[iOff + j] = sum;
        }
    }
}

void writeMatToFile(uint16_t N, data_t *res, const char *name)
{
    std::ofstream file(name);
    if (!file.is_open())
    {
        std::cout << "Can't write " << name << " to file.";
        return;
    }

    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
            file << res[i * N + j] << " ";
        file << std::endl;
    }
}

int main(int argc, char **argv)
{
    uint16_t num_threads = std::stoi(argv[1]);
    uint16_t N = std::stoi(argv[2]);
    omp_set_num_threads(num_threads);

    // Allocate memory for matrices
    uint32_t mSize = N * N;
    data_t *m1 = (data_t *)calloc(mSize, sizeof(*m1));
    data_t *m2 = (data_t *)calloc(mSize, sizeof(*m2));
    data_t *res = (data_t *)calloc(mSize, sizeof(*res));

    // Fill matrices with random values
    srand(time(nullptr));
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            m1[i * N + j] = 1 + rand() % 10000;
            m2[i * N + j] = 1 + rand() % 10000;
        }
    }
    writeMatToFile(N, m1, "m1.dat");
    writeMatToFile(N, m2, "m2.dat");

    double start = 0., finish = 0.;

    // Sequential algorithm
    start = omp_get_wtime();
    sequentialMultiply(N, m1, m2, res);
    finish = omp_get_wtime();
    std::cout << "Time spent (sequential algorithm) " << finish - start << "s." << std::endl;
    writeMatToFile(N, res, "sequential.dat");

    // Parallel algorithm
    start = omp_get_wtime();
    parallelMultiply(N, m1, m2, res);
    finish = omp_get_wtime();
    std::cout << "Time spent (parallel algorithm) " << finish - start << "s." << std::endl;
    writeMatToFile(N, res, "parallel.dat");

    // Parallel algorithm
    start = omp_get_wtime();
    parallelOptimizedMultiply(N, m1, m2, res);
    finish = omp_get_wtime();
    std::cout << "Time spent (optimized parallel algorithm) " << finish - start << "s." << std::endl;
    writeMatToFile(N, res, "parallel-opt.dat");

    free(res);
    free(m2);
    free(m1);
    return 0;
}
