import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import IPython
from scipy.signal import spectrogram, lfilter, freqz, tf2zpk


def read_file():
    s, fs = sf.read('xkanko00.wav')
    s = s[:48000]
    t = np.arange(s.size) / fs
    plt.figure(figsize=(6,3))
    plt.plot(t, s)
    plt.gca().set_xlabel('$t[s]$')
    plt.gca().set_title('Zvukový signál')

    plt.tight_layout()


if __name__ == '__main__':
    read_file()
