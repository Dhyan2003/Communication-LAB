#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

print("Name: Dhyan Navneeth V")
print("Roll No: 25")
print("Dept: ECE")

# Generate 10,000 random bits
num_bits = 10000
bits = np.random.randint(0, 2, num_bits)

# BPSK modulation: 0 -> -1, 1 -> +1
symbols = 2 * bits - 1

# Define SNR range (in dB)
snr_db_range = np.arange(0, 11, 0.5)
ber_simulated = []

target_ber = 1e-3
min_snr_found = False

for snr_db in snr_db_range:
    snr_linear = 10 ** (snr_db / 10)
    noise_std = np.sqrt(1 / (2 * snr_linear))
    
    # AWGN noise
    noise = noise_std * np.random.randn(num_bits)
    
    # Received signal
    received = symbols + noise
    
    # Detection
    detected = (received > 0).astype(int)
    
    # Calculate BER
    errors = np.sum(detected != bits)
    ber = errors / num_bits
    ber_simulated.append(ber)

    if not min_snr_found and ber <= target_ber:
        print(f"✅ Minimum SNR required to achieve BER ≤ 0.001 is: {snr_db} dB")
        min_snr_found = True

# Theoretical BER for BPSK
ber_theoretical = 0.5 * erfc(np.sqrt(10 ** (snr_db_range / 10)))

# Plot BER vs SNR
plt.figure(figsize=(8, 6))
plt.semilogy(snr_db_range, ber_simulated, 'go', label='Simulated BER')
plt.semilogy(snr_db_range, ber_theoretical, 'b-', label='Theoretical BER')
plt.axhline(y=target_ber, color='r', linestyle='--', label='Target BER = 0.001')
plt.xlabel('SNR (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.title('BER vs SNR for BPSK over AWGN')
plt.grid(True, which='both')
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:




