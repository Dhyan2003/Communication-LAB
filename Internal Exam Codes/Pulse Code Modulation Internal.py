#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import matplotlib.pyplot as plt

# Signal parameters
frequency = 10  # 10 Hz sine wave
sampling_rate = 800  # At least 10x the signal frequency (Nyquist)
duration = 1  # seconds
t = np.linspace(0, duration, int(sampling_rate * duration))
signal = 2 + np.sin(2 * np.pi * frequency * t)  # Sine wave with DC offset

# DSP processor settings
bit_depths = [8, 4]
SQNR_results = []
reconstructed_signals = []

for bits in bit_depths:
    L = 2 ** bits  # Number of quantization levels
    min_val = np.min(signal)
    max_val = np.max(signal)
    
    # Normalize signal to 0 to L-1
    signal_norm = (signal - min_val) / (max_val - min_val) * (L - 1)
    
    # Quantization
    quantized = np.round(signal_norm)
    
    # Dequantization
    reconstructed = quantized / (L - 1) * (max_val - min_val) + min_val
    reconstructed_signals.append(reconstructed)

    # SQNR calculation
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean((signal - reconstructed) ** 2)
    SQNR = 10 * np.log10(signal_power / noise_power)
    SQNR_results.append(SQNR)

# Print SQNR results
print(f"SQNR (8-bit processor): {SQNR_results[0]:.2f} dB")
print(f"SQNR (4-bit processor): {SQNR_results[1]:.2f} dB")
print(f"Difference in SQNR: {SQNR_results[0] - SQNR_results[1]:.2f} dB")

# Plot zoomed-in signal
plt.figure(figsize=(10, 4))
plt.plot(t, signal, label='Original Signal', color='black')
plt.plot(t, reconstructed_signals[1], label='Reconstructed (4-bit)', color='red')

# Zoom into the first 0.2 seconds and amplitude range
plt.xlim(0, 0.2)
plt.ylim(1, 3)

plt.title('Zoomed-in View: Signal vs Reconstructed (4-bit)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:




