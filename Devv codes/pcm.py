"Date : 13/02/2025"


#            Pulse Coded Modulation
#                   -----------------------

# Generate the following raised sine wave by sampling it at four times the Nyquist rate
# s(t) = (mod(r, 5) + 1)((1 + cos(8πt))/2)

# where mod(r, 5) is the reminder when your roll number r is divided by 5. Quantize the samples
# of s(t) with L = 4, 8, 16, 32, 64 where L is the number of quantization levels.
# (a) Compute the signal to quantization noise ratio and plot it against N, where N = log2(L)
# is the number of bits used for quantization.
# (b) Generate the PCM modulated output for L = 32 using binary encoding.





import numpy as np
import matplotlib.pyplot as plt

roll_number = 23    
mod_r = roll_number % 5 + 1
sampling_rate = 8000
duration = 1
t = np.linspace(0, duration, int(sampling_rate * duration))


s_t = mod_r * (1 + np.cos(8 * np.pi * t)) / 2

L_values = [4, 8, 16, 32, 64]
N_values = np.log2(L_values)


SQNR_values = []
quantized_signals = []

for L in L_values:
    delta = (np.max(s_t) - np.min(s_t)) / L
    quantized_signal = np.round(s_t / delta) *delta
    quantized_signals.append(quantized_signal)

    quantization_error = s_t - quantized_signal
    signal_power = np.mean(s_t**2)
    noise_power = np.mean(quantization_error**2)
    SQNR = 10 * np.log10(signal_power / noise_power)
    SQNR_values.append(SQNR)

# Plot SQNR vs N
plt.figure()
plt.plot(N_values, SQNR_values,marker='o')
plt.xlabel('Number of bits (N)')
plt.ylabel('Signal to Quantization Noise Ratio (dB)')
plt.title('SQNR vs Number of bits for Quantization')
plt.grid(True)


z = 32
delta = (np.max(s_t) - np.min(s_t)) / z

# Normalize signal into 0 to 31 range
levels = np.round((s_t) / delta).astype(int)
plt.figure(figsize=(12, 5))
plt.step(t, levels, label="PCM Output", color='b')

plt.xlabel("Time (s)")
plt.ylabel("Quantization Level (Binary Encoded)")
plt.title("PCM Modulated Output (L=32, 5-bit encoding)")
plt.legend()
plt.grid(True)

plt.show()


import numpy as np
import matplotlib.pyplot as plt
sampling_rate=8000
duration=1
r=23
t=np.linspace(0,duration,int(sampling_rate*duration))
s_t=((r%5)+1)*(1+np.cos(np.pi*8*t))/2

L_values=[4,8,16,32,64]
N_values=np.log2(L_values)
SQNR_values=[]
quantized_values=[]

for l in L_values:
    delta=(np.max(s_t)-np.min(s_t))/l
    quantized_signal=np.round(s_t/delta)*delta
    quantized_values.append(quantized_signal)

    q_error=s_t-quantized_signal
    signal_power=np.mean(s_t**2)
    noise_power=np.mean(q_error**2)
    sqnr=10*np.log10(signal_power/noise_power)
    SQNR_values.append(sqnr)

plt.plot(N_values,SQNR_values,marker='o')
plt.grid(True)
plt.xlabel('NUM OF BITS')
plt.ylabel('SQNR')
plt.show()

