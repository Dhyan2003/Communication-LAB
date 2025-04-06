# Import necessary libraries
from numpy import *  # Import all functions from numpy
from matplotlib.pyplot import *  # Import all functions from matplotlib.pyplot
import cv2  # Import OpenCV for image processing
from google.colab import files  # Import files module for uploading files in Google Colab
from scipy.special import erfc  # Import complementary error function for theoretical BER calculation

# Upload an image file using Google Colab's file upload feature
image = files.upload()

# Extract the filename of the uploaded image
filename = list(image.keys())[0]

# Read the uploaded image in grayscale mode
image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

# Convert the grayscale image into binary bits
binary_bits = np.unpackbits(image).astype(np.int8)

# Print the first 20 binary bits for verification
print(binary_bits[0:20])

# Map binary bits to BPSK symbols: 0 → +1, 1 → -1
bpsk_symbols = 1 - 2 * binary_bits

# Define the range of SNR (Signal-to-Noise Ratio) values in dB
snr_values = range(-10, 11, 1)

# Initialize lists to store simulated and theoretical BER values
ber_simulated = []
ber_theory = []

# Loop through each SNR value
for snr_db in snr_values:
    # Convert SNR from dB to linear scale
    snr_linear = 10 ** (snr_db / 10)

    # Calculate the signal power (mean squared value of BPSK symbols)
    signal_power = np.mean(np.abs(bpsk_symbols) ** 2)

    # Calculate the noise power based on the signal power and SNR
    noise_power = signal_power / snr_linear

    # Generate complex Gaussian noise with the calculated noise power
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(len(bpsk_symbols)) + 1j * np.random.randn(len(bpsk_symbols))
    )

    # Add noise to the BPSK symbols to simulate the received signal
    received_signal = bpsk_symbols + noise

    # Apply the decision rule to decode the received signal
    # Threshold at 0: Values >= 0 → +1, Values < 0 → -1
    decoded_symbols = np.where(received_signal.real >= 0, +1, -1)

    # Convert decoded BPSK symbols back to binary bits
    binary_bits_decoded = np.where(decoded_symbols == +1, 0, 1)

    # Count the number of bit errors (differences between transmitted and received bits)
    num_bit_changes = np.sum(binary_bits_decoded != binary_bits)

    # Calculate the simulated BER (Bit Error Rate)
    ber = num_bit_changes / len(binary_bits)
    ber_simulated.append(ber)

    # Calculate the theoretical BER for BPSK using the complementary error function
    ber_theor = 0.5 * erfc(np.sqrt(snr_linear))
    ber_theory.append(ber_theor)

    # Print the results for the current SNR
    print(f"SNR (dB): {snr_db}, Simulated BER: {ber:.6f}, Theoretical BER: {ber_theor:.6f}")

    # Reconstruct the image from the decoded binary bits
    decoded_image = packbits(binary_bits_decoded[:len(binary_bits_decoded)], bitorder='big')
    decoded_image = decoded_image.reshape(image.shape)

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
semilogy(snr_values, ber_simulated, 'o-', label="Simulated BER")
semilogy(snr_values, ber_theory, 's--', label="Theoretical BER")
title("BER vs SNR for BPSK Modulation")
xlabel("SNR (dB)")
ylabel("Bit Error Rate (BER)")
grid(True, which="both", linestyle="--", linewidth=0.5)
legend()
show()