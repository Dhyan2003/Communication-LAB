from numpy import *
from matplotlib.pyplot import *
import cv2
from scipy.special import erfc  # For theoretical BER calculation

# Read the image using cv2
image = cv2.imread("D:/cameraman.png", cv2.IMREAD_GRAYSCALE)

# Convert each pixel to 8-bit binary and flatten to a single bitstream
binary_bits = np.unpackbits(image).astype(np.int8)  # Convert to int8 for correct BPSK mapping
print(binary_bits[0:20])
# BPSK Mapping: 0 → +1, 1 → -1
bpsk_symbols = 1 - 2 * binary_bits  # 0 → +1, 1 → -1

# Initialize SNR range and BER lists
snr_db_range = range(-10, 11, 1)  # SNR values from -10 dB to 10 dB
ber_simulated = []  # List to store simulated BER values
ber_theoretical = []  # List to store theoretical BER values

# Iterate over SNR values
for snr_db in snr_db_range:
    # Convert SNR from dB to linear scale
    snr_linear = 10**(snr_db / 10)

    # Calculate signal power
    signal_power = np.mean(np.abs(bpsk_symbols)**2)

    # Calculate noise power
    noise_power = signal_power / snr_linear

    # Generate complex Gaussian noise
    noise = np.sqrt(noise_power / 2) * (    np.random.randn(len(bpsk_symbols)) + 1j * np.random.randn(len(bpsk_symbols)))

    # Transmit signal with noise
    received_signal = bpsk_symbols + noise

    # Decision rule: Nearest point mapping
    # Threshold at 0: Values >= 0 → +1, Values < 0 → -1
    decoded_symbols = np.where(received_signal.real >= 0, +1, -1)

    # Convert decoded BPSK symbols to binary bits
    binary_bits_decoded = np.where(decoded_symbols == +1, 0, 1)

    # Count the number of bit changes (errors)
    num_bit_changes = np.sum(binary_bits_decoded != binary_bits)

    # Calculate BER (Bit Error Rate)
    ber = num_bit_changes / len(binary_bits)
    ber_simulated.append(ber)

    # Calculate theoretical BER for BPSK
    ber_theory = 0.5 * erfc(np.sqrt(snr_linear))
    ber_theoretical.append(ber_theory)

    # Print results for the current SNR
    print(f"SNR (dB): {snr_db}, Simulated BER: {ber:.6f}, Theoretical BER: {ber_theory:.6f}")

    # Reconstruct the image from the decoded binary bits
    decoded_binary_values_8bit = ["".join(map(str, binary_bits_decoded[i:i+8])) for i in range(0, len(binary_bits_decoded), 8)]
    decoded_pixel_values = np.array([int(b, 2) for b in decoded_binary_values_8bit if len(b) == 8], dtype=np.uint8)
    decoded_image = decoded_pixel_values.reshape(image.shape)

    # Plot the reconstructed image and constellation diagram for the current SNR
    figure(figsize=(12, 6))

    # Subplot 1: Reconstructed Image
    subplot(1, 2, 1)
    imshow(decoded_image, cmap='gray')
    title(f"Reconstructed Image (SNR={snr_db} dB)")
    axis("off")

    # Subplot 2: Constellation Diagram
    subplot(1, 2, 2)
    scatter(received_signal.real, received_signal.imag, marker='o', color='b', s=1)
    title(f"BPSK Constellation Diagram (SNR={snr_db} dB)")
    xlabel("In-Phase (I)")
    ylabel("Quadrature (Q)")
    axhline(0, color='gray', lw=0.5)
    axvline(0, color='gray', lw=0.5)
    grid(True)

    # Show the plots for the current SNR
    tight_layout()
    show()

# Plot BER vs SNR
figure(figsize=(10, 6))
semilogy(snr_db_range, ber_simulated, 'o-', label="Simulated BER")
semilogy(snr_db_range, ber_theoretical, 's--', label="Theoretical BER")
title("BER vs SNR for BPSK Modulation")
xlabel("SNR (dB)")
ylabel("Bit Error Rate (BER)")
grid(True, which="both", linestyle="--", linewidth=0.5)
legend()
show()