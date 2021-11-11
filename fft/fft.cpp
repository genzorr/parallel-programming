#include <iostream>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <string>
#include <fstream>
#include <cmath>
#include <complex>

typedef double data_t;

void writeSignalToFile(std::complex<data_t> *signal, uint32_t nSamples, data_t varInit, data_t varStep, const char* name)
{
    std::ofstream file(name);
    if (!file.is_open())
    {
        std::cout << "Can't write " << name << " to file.";
        return;
    }

    data_t varValue = varInit;
    for (uint32_t i = 0; i < nSamples; i++)
    {
        file << varValue << "\t" << signal[i].real() << "\t" << signal[i].imag() << std::endl;
        varValue += varStep;
    }
    file.close();
}

void generateSineWave(std::complex<data_t> *signal, uint32_t nSamples, data_t ampl, data_t freq, data_t dt, data_t noise)
{
    auto time = (data_t)0;
    for (uint32_t i = 0; i < nSamples; i++)
    {
        signal[i] = ampl * (sin(2 * M_PI * freq * time) + noise * ((data_t)rand() - (data_t)RAND_MAX/2) / (data_t)RAND_MAX);
        time += dt;
    }
}

void sequentialFFTRecursion(std::complex<data_t> *x, unsigned int N)
{
    // Check if it is splitted enough
    if (N <= 1)
        return;

    // Split even and odd
    std::complex<data_t> *odd = (std::complex<data_t> *)calloc(N/2, sizeof(*odd));
    std::complex<data_t> *even = (std::complex<data_t> *)calloc(N/2, sizeof(*even));
    for (int i = 0; i < N / 2; i++)
    {
        even[i] = x[i*2];
        odd[i] = x[i*2+1];
    }

    // Split on tasks
    sequentialFFTRecursion(even, N / 2);
    sequentialFFTRecursion(odd, N / 2);

    // Calculate DFT
    for (int k = 0; k < N / 2; k++)
    {
        std::complex<data_t> t = exp(std::complex<data_t>(0, -2 * M_PI * k / N)) * odd[k];
        x[k] = even[k] + t;
        x[N / 2 + k] = even[k] - t;
    }

    free(odd);
    free(even);
}

void recursiveSeqFFT(std::complex<data_t> *x_in, std::complex<data_t> *x_out, unsigned int N)
{
    data_t nrm = data_t(1) / (data_t)N;
    for (int i = 0; i < N; i++)
        x_out[i] = nrm * x_in[i];
    // Start recursion
    sequentialFFTRecursion(x_out, N);
}

void parallelFFTRecursion(std::complex<data_t> *x, unsigned int N)
{
    // Check if it is splitted enough
    if (N <= 1)
        return;

    // Split even and odd
    std::complex<data_t> *odd = (std::complex<data_t> *)calloc(N/2, sizeof(*odd));
    std::complex<data_t> *even = (std::complex<data_t> *)calloc(N/2, sizeof(*even));
#pragma omp parallel for
    for (int i = 0; i < N / 2; i++)
    {
        even[i] = x[i*2];
        odd[i] = x[i*2+1];
    }

    // Split on tasks
    parallelFFTRecursion(even, N / 2);
    parallelFFTRecursion(odd, N / 2);

    // Calculate DFT
#pragma omp parallel for
    for (int k = 0; k < N / 2; k++)
    {
        std::complex<data_t> t = exp(std::complex<data_t>(0, -2 * M_PI * k / N)) * odd[k];
        x[k] = even[k] + t;
        x[N / 2 + k] = even[k] - t;
    }

    free(odd);
    free(even);
}

void recursiveParFFT(std::complex<data_t> *x_in, std::complex<data_t> *x_out, unsigned int N)
{
    data_t nrm = data_t(1) / (data_t)N;
#pragma omp parallel for
    for (int i = 0; i < N; i++)
        x_out[i] = nrm * x_in[i];
    // Start recursion
    parallelFFTRecursion(x_out, N);
}

// Finds offset to MSB
int findMSB(int x)
{
    int p = 0;
    while (x > 1)
    {
        x >>= 1;
        ++p;
    }
    return p;
}

// Performs bit reverse for integer
int bitr(uint32_t x, int nb)
{
    x = ( x               << 16) | ( x               >> 16);
    x = ((x & 0x00FF00FF) <<  8) | ((x & 0xFF00FF00) >>  8);
    x = ((x & 0x0F0F0F0F) <<  4) | ((x & 0xF0F0F0F0) >>  4);
    x = ((x & 0x33333333) <<  2) | ((x & 0xCCCCCCCC) >>  2);
    x = ((x & 0x55555555) <<  1) | ((x & 0xAAAAAAAA) >>  1);

    return ((x >> (32 - nb)) & (0xFFFFFFFF >> (32 - nb)));
}

void iterativeSeqFFT(std::complex<data_t> *xi, std::complex<data_t> *xo, unsigned int N)
{
    int msb = findMSB(N);
    data_t nrm = data_t(1) / (data_t)N;

    // Rearrange data using bit reversal indexes
    for (int j = 0; j < N; ++j)
        xo[j] = nrm * xi[bitr(j, msb)];

    // FFT passes; each pass combines pairs of blocks of size 2^i
    for (int i = 0; i < msb; ++i)
    {
        int bm = 1 << i; // butterfly mask
        int bw = 2 << i; // butterfly width
        data_t ang = data_t(1) * M_PI / data_t(bm); // precomputation

        // FFT butterflies
        for (int j = 0; j < (N / 2); ++j)
        {
            int i1 = ((j >> i) << (i + 1)) + j % bm; // left wing
            int i2 = i1 ^ bm;                        // right wing
            std::complex<data_t> z1 = std::polar(data_t(1), ang * data_t(i1 ^ bw)); // left wing rotation
            std::complex<data_t> z2 = std::polar(data_t(1), ang * data_t(i2 ^ bw)); // right wing rotation
            std::complex<data_t> tmp = xo[i1];

            xo[i1] += z1 * xo[i2];
            xo[i2] =  tmp + z2 * xo[i2];
        }
    }
}

void iterativeParFFT(std::complex<data_t> *xi, std::complex<data_t> *xo, unsigned int cnt)
{
    int msb = findMSB(cnt);
    data_t nrm = data_t(1) / (data_t)cnt;

    // Rearrange data using bit reversal indexes
#pragma omp parallel for
    for (int j = 0; j < cnt; ++j)
        xo[j] = nrm * xi[bitr(j, msb)];

    // FFT passes; each pass combines pairs of blocks of size 2^i
    for (int i = 0; i < msb; ++i)
    {
        int bm = 1 << i; // butterfly mask
        int bw = 2 << i; // butterfly width
        data_t ang = data_t(1) * M_PI / data_t(bm); // precomputation

        // FFT butterflies
    #pragma omp parallel for
        for (int j = 0; j < (cnt/2); ++j)
        {
            int i1 = ((j >> i) << (i + 1)) + j % bm; // left wing
            int i2 = i1 ^ bm;                        // right wing
            std::complex<data_t> z1 = std::polar(data_t(1), ang * data_t(i1 ^ bw)); // left wing rotation
            std::complex<data_t> z2 = std::polar(data_t(1), ang * data_t(i2 ^ bw)); // right wing rotation
            std::complex<data_t> tmp = xo[i1];

            xo[i1] += z1 * xo[i2];
            xo[i2] =  tmp + z2 * xo[i2];
        }
    }
}

int main(int argc, char **argv)
{
    srand((unsigned)time(nullptr));
    uint16_t num_threads = std::stoi(argv[1]);
    omp_set_num_threads(num_threads);

    data_t ampl = 1;
    data_t freq = 2000;
    data_t fs = 12000;
    auto dt = (data_t)1 / fs;
    data_t noise = 1;
    double start = 0., finish = 0.;

    std::cout.precision(4);
    std::cout << "Implementation of Cooley-Tukey FFT algorithm" << std::scientific << "\n\n";

    uint32_t samples[7] = {16384, 32768, 65536, 131072, 262144, 524288, 1048576};
    for (auto nSamples : samples)
    {
        std::complex<data_t> *signal = (std::complex<data_t> *)calloc(nSamples, sizeof(*signal));
        std::complex<data_t> *signal_fft = (std::complex<data_t> *)calloc(nSamples, sizeof(*signal_fft));

        data_t df = fs / (data_t)nSamples;

        // Generate sine signal with noise and save to file
        generateSineWave(signal, nSamples, ampl, freq, dt, noise);
        std::string name = "sine-" + std::to_string(nSamples) + ".dat";
        writeSignalToFile(signal, nSamples, 0, dt, name.c_str());

        // Recursive sequential FFT algorithm
        start = omp_get_wtime();
        recursiveSeqFFT(signal, signal_fft, nSamples);
        finish = omp_get_wtime();
        std::cout << "N = " << nSamples << ", recursive sequential\t" << finish - start << "s." << std::endl;

//        name = "fft-rec-seq-" + std::to_string(nSamples) + ".dat";
//        writeSignalToFile(signal_fft, nSamples, -fs/2, df, name.c_str());

        // Recursive parallel FFT algorithm
        start = omp_get_wtime();
        recursiveParFFT(signal, signal_fft, nSamples);
        finish = omp_get_wtime();
        std::cout << "N = " << nSamples << ", recursive parallel  \t" << finish - start << "s." << std::endl;

//        name = "fft-rec-par-" + std::to_string(nSamples) + ".dat";
//        writeSignalToFile(signal_fft, nSamples, -fs/2, df, name.c_str());

        // Iterative sequential FFT algorithm
        start = omp_get_wtime();
        iterativeSeqFFT(signal, signal_fft, nSamples);
        finish = omp_get_wtime();
        std::cout << "N = " << nSamples << ", iterative sequential\t" << finish - start << "s." << std::endl;

//        name = "fft-iter-seq-" + std::to_string(nSamples) + ".dat";
//        writeSignalToFile(signal_fft, nSamples, -fs/2, df, name.c_str());

        // Iterative parallel FFT algorithm
        start = omp_get_wtime();
        iterativeParFFT(signal, signal_fft, nSamples);
        finish = omp_get_wtime();
        std::cout << "N = " << nSamples << ", iterative parallel  \t" << finish - start << "s." << std::endl;
        std::cout << std::endl;

        name = "fft-iter-par-" + std::to_string(nSamples) + ".dat";
        writeSignalToFile(signal_fft, nSamples, -fs/2, df, name.c_str());

        free(signal);
        free(signal_fft);
    }

    return 0;
}
