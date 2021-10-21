#include <iostream>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <string>
#include <fstream>

const uint16_t maxN = 2048;
typedef uint64_t data_t;
data_t m1cached[maxN * maxN];
data_t m2cached[maxN * maxN];

data_t *initMatrix(uint16_t size)
{
    data_t *m = (data_t *)calloc(size * size, sizeof(*m));
    return m;
}

data_t *add(data_t *M1, data_t *M2, int n)
{
    data_t *temp = initMatrix(n);
    for(int i = 0; i < n; i++)
    {
        uint32_t in = i * n;
        for (int j = 0; j < n; j++)
            temp[in + j] = M1[in + j] + M2[in + j];
    }
    return temp;
}

data_t *subtract(data_t *M1, data_t *M2, int n)
{
    data_t *temp = initMatrix(n);
    for(int i = 0; i < n; i++)
    {
        uint32_t in = i * n;
        for (int j = 0; j < n; j++)
            temp[in + j] = M1[in + j] - M2[in + j];
    }
    return temp;
}

data_t *addPar(data_t *M1, data_t *M2, int n)
{
    data_t *temp = initMatrix(n);
    for (int i = 0; i < n; i++)
    {
        uint32_t in = i * n;
#pragma omp simd
        for (int j = 0; j < n; j++)
            temp[in + j] = M1[in + j] + M2[in + j];
    }
    return temp;
}

data_t *subtractPar(data_t *M1, data_t *M2, int n)
{
    data_t *temp = initMatrix(n);
    for (int i = 0; i < n; i++)
    {
        uint32_t in = i * n;
#pragma omp simd
        for (int j = 0; j < n; j++)
            temp[in + j] = M1[in + j] - M2[in + j];
    }
    return temp;
}

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

void sequentialOptimizedMultiply(uint16_t N, const data_t *m1, const data_t *m2, data_t *res)
{
    convertToCached(N, m1, m2, m1cached, m2cached);

    for (uint16_t i = 0; i < N; i++)
    {
        uint32_t iOff = i * N;
        for (uint16_t j = 0; j < N; j++)
        {
            data_t sum = 0;
            uint32_t jOff = j * N;
            res[iOff + j] = 0;
            for (uint16_t k = 0; k < N; k++)
            {
                sum += (m1cached[iOff + k] * m2cached[jOff + k]);
            }
            res[iOff + j] = sum;
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

data_t *strassenSequentialMultiply(uint16_t n, const data_t *m1, const data_t *m2)
{
    if (n == 64)
    {
        data_t *res = initMatrix(n);
        sequentialOptimizedMultiply(n, m1, m2, res);
        return res;
    }

    data_t *C = initMatrix(n);
    int k = n/2;

    data_t *A11 = initMatrix(k);
    data_t *A12 = initMatrix(k);
    data_t *A21 = initMatrix(k);
    data_t *A22 = initMatrix(k);
    data_t *B11 = initMatrix(k);
    data_t *B12 = initMatrix(k);
    data_t *B21 = initMatrix(k);
    data_t *B22 = initMatrix(k);

    for(int i=0; i<k; i++)
    {
        for (int j = 0; j < k; j++)
        {
            A11[i * k + j] = m1[i * n + j];
            A12[i * k + j] = m1[i * n + k + j];
            A21[i * k + j] = m1[(k + i) * n + j];
            A22[i * k + j] = m1[(k + i) * n + k + j];
            B11[i * k + j] = m2[i * n + j];
            B12[i * k + j] = m2[i * n + k + j];
            B21[i * k + j] = m2[(k + i) * n + j];
            B22[i * k + j] = m2[(k + i) * n + k + j];
        }
    }

    data_t *P1 = strassenSequentialMultiply(k, A11, subtract(B12, B22, k));
    data_t *P2 = strassenSequentialMultiply(k, add(A11, A12, k), B22);
    data_t *P3 = strassenSequentialMultiply(k, add(A21, A22, k), B11);
    data_t *P4 = strassenSequentialMultiply(k, A22, subtract(B21, B11, k));
    data_t *P5 = strassenSequentialMultiply(k, add(A11, A22, k), add(B11, B22, k));
    data_t *P6 = strassenSequentialMultiply(k, subtract(A12, A22, k), add(B21, B22, k));
    data_t *P7 = strassenSequentialMultiply(k, subtract(A11, A21, k), add(B11, B12, k));

    data_t *C11 = subtract(add(add(P5, P4, k), P6, k), P2, k);
    data_t *C12 = add(P1, P2, k);
    data_t *C21 = add(P3, P4, k);
    data_t *C22 = subtract(subtract(add(P5, P1, k), P3, k), P7, k);

    for(int i=0; i < k; i++)
    {
        for (int j = 0; j < k; j++)
        {
            C[i * n + j] = C11[i * k + j];
            C[i * n + j + k] = C12[i * k + j];
            C[(k + i) * n + j] = C21[i * k + j];
            C[(k + i) * n + k + j] = C22[i * k + j];
        }
    }

    free(A11);
    free(A12);
    free(A21);
    free(A22);
    free(B11);
    free(B12);
    free(B21);
    free(B22);
    free(P1);
    free(P2);
    free(P3);
    free(P4);
    free(P5);
    free(P6);
    free(P7);
    free(C11);
    free(C12);
    free(C21);
    free(C22);

    return C;
}

data_t *strassenParallelMultiply(uint16_t n, const data_t *m1, const data_t *m2)
{
    if (n == 64)
    {
        data_t *res = initMatrix(n);
        parallelOptimizedMultiply(n, m1, m2, res);
        return res;
    }

    data_t *C = initMatrix(n);
    int k = n/2;

    data_t *A11 = initMatrix(k);
    data_t *A12 = initMatrix(k);
    data_t *A21 = initMatrix(k);
    data_t *A22 = initMatrix(k);
    data_t *B11 = initMatrix(k);
    data_t *B12 = initMatrix(k);
    data_t *B21 = initMatrix(k);
    data_t *B22 = initMatrix(k);

#pragma omp parallel for
    for (int i = 0; i < k; i++)
    {
        uint32_t in = i * n;
        uint32_t ik = i * k;
#pragma omp simd
        for (int j = 0; j < k; j++)
        {
            A11[ik + j] = m1[in + j];
            A12[ik + j] = m1[in + k + j];
            A21[ik + j] = m1[(k + i) * n + j];
            A22[ik + j] = m1[(k + i) * n + k + j];
            B11[ik + j] = m2[in + j];
            B12[ik + j] = m2[in + k + j];
            B21[ik + j] = m2[(k + i) * n + j];
            B22[ik + j] = m2[(k + i) * n + k + j];
        }
    }

    data_t *P1, *P2, *P3, *P4, *P5, *P6, *P7, *C11, *C12, *C21, *C22;
#pragma omp parallel sections
    {
        P1 = strassenParallelMultiply(k, A11, subtractPar(B12, B22, k));
#pragma omp section
        P2 = strassenParallelMultiply(k, addPar(A11, A12, k), B22);
#pragma omp section
        P3 = strassenParallelMultiply(k, addPar(A21, A22, k), B11);
#pragma omp section
        P4 = strassenParallelMultiply(k, A22, subtractPar(B21, B11, k));
#pragma omp section
        P5 = strassenParallelMultiply(k, addPar(A11, A22, k), addPar(B11, B22, k));
#pragma omp section
        P6 = strassenParallelMultiply(k, subtractPar(A12, A22, k), addPar(B21, B22, k));
#pragma omp section
        P7 = strassenParallelMultiply(k, subtractPar(A11, A21, k), addPar(B11, B12, k));
    }

#pragma omp parallel sections
    {
        C11 = subtractPar(addPar(addPar(P5, P4, k), P6, k), P2, k);
#pragma omp section
        C12 = addPar(P1, P2, k);
#pragma omp section
        C21 = addPar(P3, P4, k);
#pragma omp section
        C22 = subtractPar(subtractPar(addPar(P5, P1, k), P3, k), P7, k);
    }

#pragma omp parallel for
    for (int i = 0; i < k; i++)
    {
        uint32_t in = i * n;
        uint32_t ik = i * k;
#pragma omp simd
        for (int j = 0; j < k; j++)
        {
            C[in + j] = C11[ik + j];
            C[in + j + k] = C12[ik + j];
            C[(k + i) * n + j] = C21[ik + j];
            C[(k + i) * n + k + j] = C22[ik + j];
        }
    }

    free(A11);
    free(A12);
    free(A21);
    free(A22);
    free(B11);
    free(B12);
    free(B21);
    free(B22);
    free(P1);
    free(P2);
    free(P3);
    free(P4);
    free(P5);
    free(P6);
    free(P7);
    free(C11);
    free(C12);
    free(C21);
    free(C22);

    return C;
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
    data_t *m1 = initMatrix(N);
    data_t *m2 = initMatrix(N);
    data_t *res = initMatrix(N);

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
    std::cout << "Time spent (sequential) " << finish - start << "s." << std::endl;
    writeMatToFile(N, res, "sequential.dat");

    // Parallel optimized algorithm
    start = omp_get_wtime();
    sequentialOptimizedMultiply(N, m1, m2, res);
    finish = omp_get_wtime();
    std::cout << "Time spent (optimized sequential) " << finish - start << "s." << std::endl;
    writeMatToFile(N, res, "sequential-opt.dat");

    // Parallel algorithm
    start = omp_get_wtime();
    parallelMultiply(N, m1, m2, res);
    finish = omp_get_wtime();
    std::cout << "Time spent (parallel) " << finish - start << "s." << std::endl;
    writeMatToFile(N, res, "parallel.dat");

    // Parallel optimized algorithm
    start = omp_get_wtime();
    parallelOptimizedMultiply(N, m1, m2, res);
    finish = omp_get_wtime();
    std::cout << "Time spent (optimized parallel) " << finish - start << "s." << std::endl;
    writeMatToFile(N, res, "parallel-opt.dat");

    free(res);

    // Strassen optimized sequential
    start = omp_get_wtime();
    res = strassenSequentialMultiply(N, m1, m2);
    finish = omp_get_wtime();
    std::cout << "Time spent (Strassen optimized sequential) " << finish - start << "s." << std::endl;
    writeMatToFile(N, res, "strassen-sequential-opt.dat");
    free(res);

    // Strassen optimized parallel
    start = omp_get_wtime();
    res = strassenParallelMultiply(N, m1, m2);
    finish = omp_get_wtime();
    std::cout << "Time spent (Strassen optimized parallel) " << finish - start << "s." << std::endl;
    writeMatToFile(N, res, "strassen-parallel-opt.dat");

    free(res);
    free(m2);
    free(m1);
    return 0;
}
