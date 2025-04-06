#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc  # For theoretical BER

# Parameters
num_bits = 10000
EbN0_dB = np.arange(0, 11, 1)  # 0 to 10 dB
EbN0_linear = 10 ** (EbN0_dB / 10)

def add_awgn_noise(signal, snr_dB):
    snr_linear = 10**(snr_dB / 10)
    power_signal = np.mean(np.abs(signal) ** 2)
    noise_power = power_signal / snr_linear
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
    return signal + noise

def bpsk_mod(bits):
    return 2 * bits - 1

def bpsk_demod(signal):
    return (signal.real >= 0).astype(int)

def qpsk_mod(bits):
    bits = bits.reshape((-1, 2))
    mapping = {
        (0, 0): 1 + 1j,
        (0, 1): -1 + 1j,
        (1, 1): -1 - 1j,
        (1, 0): 1 - 1j
    }
    symbols = np.array([mapping[tuple(b)] for b in bits]) / np.sqrt(2)
    return symbols

def qpsk_demod(symbols):
    symbols = symbols * np.sqrt(2)
    bits = np.zeros((len(symbols), 2), dtype=int)
    for i, s in enumerate(symbols):
        if np.real(s) > 0 and np.imag(s) > 0:
            bits[i] = [0, 0]
        elif np.real(s) < 0 and np.imag(s) > 0:
            bits[i] = [0, 1]
        elif np.real(s) < 0 and np.imag(s) < 0:
            bits[i] = [1, 1]
        elif np.real(s) > 0 and np.imag(s) < 0:
            bits[i] = [1, 0]
    return bits.reshape(-1)

# BER Results
ber_bpsk = []
ber_qpsk = []

# Random bits
bits = np.random.randint(0, 2, num_bits)
bits_qpsk = np.append(bits, 0) if num_bits % 2 != 0 else bits.copy()

for snr in EbN0_dB:
    # BPSK
    bpsk_symbols = bpsk_mod(bits)
    rx_bpsk = add_awgn_noise(bpsk_symbols, snr)
    bpsk_out = bpsk_demod(rx_bpsk)
    ber_bpsk.append(np.mean(bits != bpsk_out))

    # QPSK
    qpsk_symbols = qpsk_mod(bits_qpsk)
    rx_qpsk = add_awgn_noise(qpsk_symbols, snr)
    qpsk_out = qpsk_demod(rx_qpsk)[:num_bits]
    ber_qpsk.append(np.mean(bits != qpsk_out))

# Theoretical BER
ber_theory = 0.5 * erfc(np.sqrt(EbN0_linear))

# Plotting
plt.figure(figsize=(10, 6))
plt.semilogy(EbN0_dB, ber_bpsk, 'o-', label='BPSK Simulated')
plt.semilogy(EbN0_dB, ber_qpsk, 's-', label='QPSK Simulated')
# plt.semilogy(EbN0_dB, ber_theory, 'k--', label='Theoretical (BPSK/QPSK)')
plt.title('BER vs Eb/N0 for BPSK and QPSK (AWGN Channel)')
plt.xlabel('Eb/N0 (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.grid(True, which='both')
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:




