import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(f'cmake-build-release/times.txt', header=None, sep='\t')
df.columns = ['n', 't']

fig, axs = plt.subplots(1)
fig.suptitle(f'Timings')
axs.plot(df['n'], df['t'], color='blue')
axs.set_ylabel('t, s')
axs.set_xlabel('Number of threads')

plt.show()