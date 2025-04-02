import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.special import erfc

print("Name: Krishna Deepak\nRoll No: 43\nDepartment: ECE\n")


# Function to generate MPSK symbols
def generate_mpsk_symbols(M):
    angles = 2 * np.pi * np.arange(M) / M
    symbols = np.exp(1j * angles)
    return symbols


# Function to modulate bits to MPSK
def modulate(bits, M):
    k = int(np.log2(M))
    pad_size = (k - len(bits) % k) % k  # Calculate padding size
    bits = np.pad(bits, (0, pad_size), 'constant')  # Pad bits if needed
    symbol_indices = bits.reshape(-1, k).dot(2 ** np.arange(k)[::-1])
    symbols = generate_mpsk_symbols(M)
    return symbols[symbol_indices]


# Function to add AWGN noise
def add_awgn(signal, SNR_dB):
    SNR = 10 ** (SNR_dB / 10)
    noise_std = np.sqrt(1 / (2 * SNR))
    noise = noise_std * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
    return signal + noise


# Function to demodulate received symbols
def demodulate(received, M):
    symbols = generate_mpsk_symbols(M)
    distances = np.abs(received[:, None] - symbols[None, :])
    detected_indices = np.argmin(distances, axis=1)
    k = int(np.log2(M))
    return ((detected_indices[:, None] & (1 << np.arange(k)[::-1])) > 0).astype(int).flatten()


# Function to calculate SER and BER
def calculate_errors(bits, detected_bits):
    bit_errors = np.sum(bits != detected_bits[:len(bits)])  # Ignore padded bits
    BER = bit_errors / len(bits)
    SER = 1 - (1 - BER) ** np.log2(len(np.unique(detected_bits)))
    return BER, SER


# Read the image and convert to bits
image = cv2.imread("cameraman.png", cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError("Image file not found. Please check the file path.")
image = image.astype(np.uint8)  # Ensure correct data type
bits = np.unpackbits(image)

M_values = [2, 4, 8]
SNR_range = np.arange(-10, 11, 1)  # SNR from -10 to 10 dB

# Plot BER and SER separately
plt.figure(figsize=(12, 6))
for M in M_values:
    BER_values, SER_values = [], []
    for SNR_dB in SNR_range:
        modulated_signal = modulate(bits, M)
        noisy_signal = add_awgn(modulated_signal, SNR_dB)
        detected_bits = demodulate(noisy_signal, M)
        BER, SER = calculate_errors(bits, detected_bits)
        BER_values.append(BER)
        SER_values.append(SER)

    plt.subplot(1, 2, 1)
    plt.plot(SNR_range, BER_values, marker='o', label=f"Simulated BER (M={M})")
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.title("BER vs SNR")

    plt.subplot(1, 2, 2)
    plt.plot(SNR_range, SER_values, marker='o', label=f"Simulated SER (M={M})")
    plt.xlabel("SNR (dB)")
    plt.ylabel("SER")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.title("SER vs SNR")

plt.tight_layout()
plt.show()

# Display image reconstruction
plt.figure(figsize=(15, 5))

# Display the original image
plt.subplot(1, len(M_values) + 1, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis("off")

# Reconstruct and display images for different M values
for i, M in enumerate(M_values, 2):
    modulated_signal = modulate(bits, M)
    noisy_signal = add_awgn(modulated_signal, 5)  # Moderate SNR for visualization
    detected_bits = demodulate(noisy_signal, M)
    reconstructed_image = np.packbits(detected_bits[:len(bits)]).reshape(image.shape)

    plt.subplot(1, len(M_values) + 1, i)
    plt.imshow(reconstructed_image, cmap='gray')
    plt.title(f"Reconstructed (M={M})")
    plt.axis("off")

plt.tight_layout()
plt.show()