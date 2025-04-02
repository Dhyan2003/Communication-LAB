

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.special import erfc

# Step 1: Read the image
image = cv2.imread('cameraman.png', cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (256, 256))
pixels = image.flatten()
BER=[]
BER_theory=[]
#  Convert pixel values to binary
binary = [bin(i)[2:].zfill(8) for i in pixels]  # 8-bit binary representation

# Convert binary strings to bit array
bit_array = [int(bit) for byte in binary for bit in byte]

# Map bits to BPSK symbols
bpsk_map = np.array([1 if b == 0 else -1 for b in bit_array])  # 0 → +1, 1 → -1f

# Define SNR and generate noise
SNR_dB = [i for i in range(-10,11)]
Ps = 1  # Signal Power
  # Noise Power
plt.figure(figsize=(8, 4))

plt.scatter(bpsk_map, np.zeros(len(bpsk_map)), color='red', label="SNR in  dB")
plt.title("transmitted")
plt.xlabel("In-Phase")
plt.ylabel("Quadrature")
plt.grid()
plt.show()

for i in SNR_dB:
    Pn = Ps / (10**(i / 10))
# Generate complex Gaussian noise
    awgn = np.sqrt(Pn / 2) * np.random.randn(len(bpsk_map)) + 1j * np.random.randn(len(bpsk_map))

# Add noise to signal
    Rx = bpsk_map + awgn

# Constellation Diagram
    plt.scatter(np.real(Rx), np.imag(Rx), color='blue', label="Received Symbols")
    plt.title("BPSK Constellation (With Noise)")
    plt.xlabel("In-Phase")
    plt.ylabel("Quadrature")
    plt.grid()
    plt.show()
#  BPSK Demodulation
    Map = np.array([1 if x.real < 0 else 0 for x in Rx])

#Convert bits back to pixel values
    bit_chunks = np.array_split(Map, len(Map) // 8)  # Split into 8-bit chunks
    reconstructed_pixels = np.array([int("".join(map(str, chunk)), 2) for chunk in bit_chunks])
#  Reconstruct the Image
    reconstructed_image = reconstructed_pixels.reshape(256, 256)

# Save and Display the Image
    filename = f"reconstructed_image_{i}.png"
    cv2.imwrite(filename, reconstructed_image)
    plt.imshow(reconstructed_image, cmap='gray')
    plt.title(f"Reconstructed Image for {i} dB")
    plt.show()

# Calculate BER

    bit_errors = np.sum(np.array(bit_array) != np.array(Map))
    bit_error_rate = bit_errors / len(bit_array)
    print(f"Bit Error Rate (BER): {bit_error_rate}")
    theoretical_BER=(0.5)*erfc(np.sqrt(10**(i/10)))
    BER.append(bit_error_rate)
    BER_theory.append(theoretical_BER)

plt.figure(figsize=(8, 4))



plt.semilogy(SNR_dB,BER,label="BER simulated")
plt.semilogy(SNR_dB,BER_theory,label="BER Theoretical")
plt.legend()
plt.show()
