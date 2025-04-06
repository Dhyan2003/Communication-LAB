from numpy import *
import cv2
from matplotlib.pyplot import *
from scipy.signal import convolve
from google.colab import files

image =files.upload()

filename = list(image.keys())[0]
image = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)

bitstream = unpackbits(image.flatten())
symbols = where(bitstream == 0,-1,1)

def srrc_pulse(Tsym, beta, L, Nsym):
    """Generates a Square-Root Raised Cosine (SRRC) pulse while handling singularities."""
    t = np.arange(-Nsym / 2, Nsym / 2, 1 / L)
    p = np.zeros_like(t)
    
    for i in range(len(t)):
        if t[i] == 0:
            p[i] = (1 - beta + 4 * beta / np.pi) / np.sqrt(Tsym)
        elif abs(t[i]) == Tsym / (4 * beta):
            p[i] = (beta / np.sqrt(2 * Tsym)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            num = (np.sin(np.pi * t[i] * (1 - beta) / Tsym) +
                   4 * beta * t[i] / Tsym * np.cos(np.pi * t[i] * (1 + beta) / Tsym))
            denom = (np.pi * t[i] / Tsym * (1 - (4 * beta * t[i] / Tsym) ** 2))
            p[i] = num / denom
    
    return p / np.sqrt(np.sum(p ** 2))

def upsampler_filter(symbols,L,pulse):
  upsampled = zeros(len(symbols)*L)
  upsampled[::L] = symbols
  return convolve(upsampled,pulse,mode='full')

def awgn(snr_db,signal):
  snr_linear = 10**(snr_db/10)
  noise_power = 1/(2*snr_linear)
  noise = sqrt(noise_power) * (random.randn(*signal.shape))
  return signal + noise
def downsample_demodulate(signal,pulse,bit_len):
  matched_output = convolve(signal,pulse,mode='full')
  delay = (len(pulse)-1)//2
  sampled = matched_output[2*delay+1::L]
  sampled_output = where(sampled>=0,1,-1)
  return sampled_output[:bit_len]


Tsym,L,beta,Nsym = 1,4,0.3,8
pulse = srrc_pulse(Tsym,beta,L,Nsym)
transmitted_signal = upsampler_filter(symbols,L,pulse)

snr_db_range = arange(-10,21,5)
ber_values = []

for snr in snr_db_range:
  received_signal = awgn(snr,transmitted_signal)
  downsampled_demodulated = downsample_demodulate(received_signal,pulse,len(bitstream))

  recovered_bits = (downsampled_demodulated == 1).astype(uint8)
  recovered_bits = pad(recovered_bits,(0,8-len(recovered_bits) %8 ),mode='constant')[:len(bitstream)]
  reconstructed_image = packbits(recovered_bits).reshape(image.shape)

  errors  = sum(recovered_bits != bitstream)
  ber = errors/len(bitstream)
  ber_values.append(ber)

    # Plot the reconstructed image
  figure()
  imshow(reconstructed_image, cmap='gray')
  title(f'Reconstructed Image at SNR={snr} dB')
  axis('off')
  show()
    # Plot SNR vs BER Curve
figure()
semilogy(snr_db_range, ber_values, 'o-', label="Simulated BER")
xlabel("SNR (dB)")
ylabel("Bit Error Rate (BER)")
title("SNR vs BER Curve")
grid(True, which='both')
legend()
show()