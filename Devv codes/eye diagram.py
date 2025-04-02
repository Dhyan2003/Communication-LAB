import numpy as np
import cv2
import matplotlib.pyplot as plt

# Step 1: Read the image
image = cv2.imread('cameraman.png', cv2.IMREAD_GRAYSCALE)
print("Name:Devika A M \nS6 ECE \nRoll no :23")

# Step 2: Convert pixel values to bits and map to BPSK symbols
bits = np.unpackbits(image)
bpsk_symbols = 2 * bits - 1  # Map 0 -> -1, 1 -> 1

# Step 3: Pulse shaping parameters
L = 4  # Upsampling factor
Tsym = 1  # Symbol duration
Nsym = 8  # Filter length in symbols
t = np.arange(-Nsym/2, Nsym/2 + 1/L, 1/L)
p = np.sinc(t)  # Simple sinc filter

# Upsampling
upsampled_symbols = np.zeros(L * len(bpsk_symbols))
upsampled_symbols[::L] = bpsk_symbols

# Pulse shaping
shaped_signal = np.convolve(upsampled_symbols, p, mode='same')

# Step 4: Add Gaussian noise (Es/N0 as SNR, Es=1)


SNR_db= [-10,-5,5,10]
for SNR in SNR_db: # Set SNR in dB
    noise_power = 1 / (2 * (10 ** (SNR / 10)))
    noise = np.sqrt(noise_power) * (np.random.randn(len(shaped_signal)) + 1j * np.random.randn(len(shaped_signal)))
    received_signal = shaped_signal + noise

    # Step 5: Matched filtering
    g = p[::-1]  # g[n] = p[-n]
    matched_output = np.convolve(received_signal, g, mode='same')

    #   Step 6: Downsampling and demapping
    downsampled_output = matched_output[::L]
    demapped_bits = (downsampled_output.real > 0).astype(int)

    # Step 7: Convert bits to pixel values
    received_image = np.packbits(demapped_bits).reshape(image.shape)





# Step 9: Plot Eye Diagram
    nSamples = 3 * L
    nTraces = 100
    samples = downsampled_output[:nSamples * nTraces].reshape(nTraces, nSamples)

    plt.figure(figsize=(8, 5))

    beta = 0.25
    for trace in samples:
        plt.plot(trace,  alpha=0.7)

    plt.title(f'Eye Diagram (SNR={SNR} dB, Beta={beta})')
    plt.xlim(4, 6)
    plt.grid(True)
    plt.show()
    plt.figure(figsize=(8, 5))
    beta = 0.5
    for trace in samples:
        plt.plot(trace,  alpha=0.7)
    plt.title(f'Eye Diagram (SNR={SNR} dB, Beta={beta})')
    plt.xlim(4, 6)


    plt.grid(True)
    plt.show()

