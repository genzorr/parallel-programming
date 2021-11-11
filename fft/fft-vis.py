import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

samples = [16384, 1048576]#, 32768, 65536, 131072, 262144, 524288, 1048576]

for nSamples in samples:
    dfRaw = pd.read_csv(f'cmake-build-release/sine-{nSamples}.dat', header=None, sep='\t')
    dfRaw.columns = ['t', 'real', 'imag']
    dfRaw['signal'] = dfRaw['real']# + 1j*dfRaw['imag']

    dfFFT = pd.read_csv(f'cmake-build-release/fft-iter-par-{nSamples}.dat', header=None, sep='\t')
    dfFFT.columns = ['t', 'real', 'imag']
    dfFFT['signal'] = dfFFT['real'] + 1j*dfFFT['imag']

    fig, axs = plt.subplots(2)
    fig.suptitle(f'FFT for sine wave with {nSamples} samples')
    axs[0].plot(dfRaw['t'], dfRaw['signal'], color='blue')
    axs[0].set_xlabel('t, s')
    axs[0].set_ylabel('signal')

    axs[1].plot(dfFFT['t'], np.fft.fftshift(abs(dfFFT['signal'])), color='blue')
    axs[1].set_xlabel('f, Hz')
    axs[1].set_ylabel('FFT')

plt.show()