import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import pandas
import IPython
from scipy.signal import spectrogram, lfilter, freqz, tf2zpk


def read_file():
    s, fs = sf.read('xkanko00.wav')
    size = s.size
    print("delka vstupniho souboru je", size, "hz")
    print("vzorkovaci frekvence je", fs, "hz")
    x = s
    s = s[:size]
    print("minimum je", s.min())
    print("maximum je", s.max())
    t = np.arange(size) / fs
    plt.figure(figsize=(6, 3))
    plt.plot(t, s)
    plt.gca().set_xlabel('$t[s]$')
    plt.gca().set_title('Zvukový signál')
    plt.gca().grid(alpha=0.5, linestyle='--')
    plt.tight_layout()
    plt.show()
    return fs, x


def normalize_signal(s, fs):
    an = np.mean(s)
    print("stredni hodnota ja", an)
    t = np.arange(s.size) / fs
    plt.figure(figsize=(6, 3))
    plt.plot(t, s)
    plt.gca().set_xlabel('$t[s]$')
    plt.gca().set_title('Ustredeny a normalizovany signal')
    plt.gca().grid(alpha=0.5, linestyle='--')
    plt.tight_layout()
    plt.show()
    return s


def div_signal(s, fs):
    s = [s[i: i + 1024] for i in range(0, len(s), 512)]
    t = np.arange(len(s[0])) / fs
    plt.figure(figsize=(6, 3))
    plt.plot(t, s[15])
    plt.gca().set_xlabel('$t[s]$')
    plt.gca().set_title('Signal v prvnim useku')
    plt.gca().grid(alpha=0.5, linestyle='--')
    plt.tight_layout()
    plt.show()
    return s


def dft_calc(s, fs):
    x = np.asarray(s, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    dftx = np.dot(M, x)
    plt.figure(figsize=(6, 3))
    r = np.abs(dftx)[:1024//2+1]**2
    f_axis = np.arange(513)
    plt.plot(f_axis/1024*fs, r)
    plt.gca().set_xlabel('Hz')
    plt.gca().set_title('DFT')
    plt.gca().grid(alpha=0.5, linestyle='--')
    plt.tight_layout()
    plt.show()
    return dftx


if __name__ == '__main__':
    fr, x = read_file()
    x = normalize_signal(x, fr)
    s = div_signal(x, fr)
    tx = dft_calc(s[15], fr)



