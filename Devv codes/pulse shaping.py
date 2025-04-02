


"PULSE SHAPING CODE"

import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.special import erfc


# Constants
SNR_dB = 10
L = 4
T_sym = 1
N_sym = 8
beta = 0.9
t = np.arange(-N_sym/2, N_sym/2, 1/L)

def im_to_array(im):
    im = cv2.imread(im, cv2.IMREAD_GRAYSCALE)
    return im

def array_to_im(file_name, arr):
    arr = [arr[i: i+8] for i in range(0, len(arr), 8)]
    pixel_arr = []
    for pixel_list in arr:
      pixel = 0
      for bit in pixel_list:
        pixel = (pixel << 1) | bit
      pixel_arr.append(pixel)
    arr = np.array(pixel_arr)

    cv2.imwrite(file_name, arr.reshape(shape))

def bpsk_modulation(arr_1D):
    binary_strings = [format(pixel, '08b') for pixel in arr_1D]
    bit_stream = np.array([int(bit) for bit_str in binary_strings for bit in bit_str])
    bit_encode = np.where(bit_stream == 0, 1, -1)
    return bit_encode, bit_stream

def upsampler(bit_stream, L):
    upsampled_bit_stream = []
    for i in range(len(bit_stream) - 1):
        upsampled_bit_stream.append(bit_stream[i])
        upsampled_bit_stream.extend([0] * L)
    upsampled_bit_stream.append(bit_stream[-1])

    return np.array(upsampled_bit_stream)

def pulse_shaping_filter(t, T_sym, beta):
  # Square Root Raised Cosine Filter
  if t == 0:
    return (1 / np.sqrt(T_sym)) * (1 - beta + (4 * beta / np.pi))

  if np.abs(t) == T_sym / (4 * beta):
    term_1= (1 + 2 / np.pi)*np.sin(np.pi/(4*beta))
    term_2= (1 - 2 / np.pi)*np.cos(np.pi/(4*beta))

    return (beta / np.sqrt(2*T_sym)) * (term_1 + term_2)
  C = 4 * beta * t / T_sym
  num_sin = np.sin(np.pi * t * (1 - beta) / T_sym)
  num_cos = np.cos(np.pi * t * (1 + beta) / T_sym)

  # Avoid division by zero
  den = (np.pi * t / T_sym) * (1 - (C ** 2))
  den = np.where(den == 0, 1e-10, den)  # Replace zero with a small number

  num = num_sin + C * num_cos
  return (1/np.sqrt(T_sym)) * (num / den)

def apply_AWGN(snr_db, bit_stream):
    var = 1
    noise_power = 1 / (10 ** (snr_db / 10))
    n = np.sqrt(noise_power / 2) * (np.random.normal(0, var, len(bit_stream)) + 1j * np.random.normal(0, var, len(bit_stream)))
    return bit_stream + n

def matched_filter(impulse_response: np.ndarray) -> np.ndarray:
    matched_filter = np.flip(impulse_response)
    return matched_filter

def bpsk_demodulation(bit_stream):
    bit_decode = np.where(np.real(bit_stream) > 0, 0, 1)
    return bit_decode

def bit_error_rate(original_bits, demodulated_bits):
    return np.sum(original_bits != demodulated_bits) / len(original_bits)

def theoretical_ber(snr_db: float) -> float:
    snr = 10 ** (snr_db / 10)
    return 0.5 * erfc(np.sqrt(snr))


arr = im_to_array('cameraman.png')
def calculate_ber(arr, L, T_sym, beta, N_sym):
    orig_bits = arr.flatten()
    bpsk_arr, original_bits = bpsk_modulation(orig_bits)
    upsampled_output = upsampler(bpsk_arr, L)
    t = np.arange(-N_sym/2, N_sym/2, 1/L)
    impulse_response = np.array([pulse_shaping_filter(tx, T_sym, beta) for tx in t])
    snr_values = np.arange(-10, 11, 1)
    theoretical_ber_values = []
    ber_values = []

    for snr_db in snr_values:
        transmitted_signal = np.convolve(upsampled_output, impulse_response)
        received_signal = apply_AWGN(snr_db, transmitted_signal)

        # Matched filtering
        matched_filter_response = matched_filter(impulse_response)
        convolved_output = np.convolve(received_signal, matched_filter_response)

        # Downsampling and demodulation
        filt_delay = int((len(impulse_response)-1)/2)
        downsampled_output = convolved_output[2*filt_delay+1:-(2*filt_delay + 1):L+1]/L
        demodulated_bits = bpsk_demodulation(downsampled_output)


        # Calculate BER
        ber = bit_error_rate(original_bits, demodulated_bits)
        ber_values.append(ber)
        theoretical_ber_values.append(theoretical_ber(snr_db))

    return snr_values, ber_values, theoretical_ber_values, impulse_response

snr_values, ber_values, theoretical_ber_values, impulse_response = calculate_ber(arr, L, T_sym, beta, N_sym)

plt.figure(figsize=(6, 5))
plt.semilogy(snr_values, ber_values, marker='o', label='Bit Error Rate')
plt.plot(snr_values, theoretical_ber_values, marker='x', label='Theoretical BER')
plt.title('BER vs SNR')
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()