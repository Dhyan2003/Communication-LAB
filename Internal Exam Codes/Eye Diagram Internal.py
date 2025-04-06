#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

# ---------- SRRC Pulse Generator ----------
def srrc_pulse(Tsym, beta, L, Nsym):
    t = np.arange(-Nsym/2, Nsym/2 + 1/L, 1/L)
    p = np.zeros_like(t)

    for i in range(len(t)):
        ti = t[i]
        if ti == 0.0:
            p[i] = 1.0 - beta + (4 * beta / np.pi)
        elif abs(ti) == Tsym / (4 * beta):
            p[i] = (beta / np.sqrt(2)) * ((1 + 2/np.pi) * np.sin(np.pi/(4*beta)) +
                                          (1 - 2/np.pi) * np.cos(np.pi/(4*beta)))
        else:
            numerator = np.sin(np.pi * ti * (1 - beta) / Tsym) +                         4 * beta * ti * np.cos(np.pi * ti * (1 + beta) / Tsym) / Tsym
            denominator = np.pi * ti * (1 - (4 * beta * ti / Tsym) ** 2) / Tsym
            p[i] = numerator / denominator

    # Normalize pulse energy
    p = p / np.sqrt(np.sum(p**2))
    return p

# ---------- Parameters ----------
num_bits = 10000
bits = np.random.randint(0, 2, num_bits)
symbols = 2 * bits - 1  # BPSK: 0 -> -1, 1 -> 1

L = 8          # Oversampling factor
Tsym = 1       # Symbol duration
Nsym = 8       # Filter span (in symbols)
beta = 0.25    # Roll-off factor
SNR_dBs = [-20, 10]

# Generate SRRC pulse
pulse = srrc_pulse(Tsym, beta, L, Nsym)

# Upsample and shape
upsampled = np.zeros(len(symbols) * L)
upsampled[::L] = symbols
shaped_signal = convolve(upsampled, pulse, mode='same')

# ---------- Loop over SNR ----------
for SNR in SNR_dBs:
    # Add AWGN
    noise_power = 1 / (2 * 10**(SNR / 10))
    noise = np.sqrt(noise_power) * np.random.randn(len(shaped_signal))
    received_signal = shaped_signal + noise

    # Matched filtering
    matched = convolve(received_signal, pulse, mode='same')

    # ---------- Eye Diagram ----------
    nTraces = 100
    nSamples = 3 * L
    segment = matched[:nTraces * nSamples]
    traces = segment.reshape(nTraces, nSamples)

    plt.figure(figsize=(8, 5))
    for trace in traces:
        plt.plot(trace, color='blue', alpha=0.5)

    plt.title(f"Eye Diagram | SNR = {SNR} dB | β = {beta}")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


# In[ ]:




