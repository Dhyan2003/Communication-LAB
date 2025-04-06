from numpy import *
from matplotlib.pyplot import *
from google.colab import files 
from scipy.special import erfc
import cv2



image = files.upload()

filename = list(image.keys())[0]

image = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
imshow(image, cmap='gray')
show()
M = [2,4,8]
snr_db_range = arange(-10,11,1)

def modulate(bitstream,M):
  k = int(log2(M))
  
  bitstream = bitstream.reshape(-1,k)
  symbols = array([int("".join(map(str,b)),2) for b in bitstream])
  angle = 2*pi*symbols/M
  return cos(angle)+1j*sin(angle)

def add_noise(signal,snr,bps):
  noise_std = sqrt(1/(2*bps*snr))
  noise = noise_std * (random.randn(*signal.shape) + 1j*random.randn(*signal.shape))
  return signal+noise

def decode(bits,M):
  angles = angle(bits)
  decoded_symbols = round((angles/(2*pi))*M) %M
  return decoded_symbols.astype(int)

def theoretical_ber(M, snr_db):
    k = np.log2(M)
    return erfc(np.sqrt(k * 10**(snr_db / 10)) / np.sqrt(2)) / k

def theoretical_ser(M, snr_db):
    return 2 * erfc(np.sqrt(2 * 10**(snr_db / 10)) * np.sin(np.pi / M))

BER_res = {}
SER_res = {}
for M_values in M:
  bits_per_symbol = int(log2(M_values))
  bitstream = unpackbits(image.flatten())
  bitstream = bitstream[:len(bitstream)-(len(bitstream)%bits_per_symbol)]
  transmitted_symbols = modulate(bitstream,M_values)
  BER_sim = []
  SER_sim = []
    
  for snr_db in snr_db_range:
    snr_linear = 10**(snr_db/10)
    received_symbols = add_noise(transmitted_symbols,snr_linear,bits_per_symbol)
    decoded_symbols = decode(received_symbols,M_values)
    decoded_bits = array([list(binary_repr(s,width = bits_per_symbol)) for s in decoded_symbols]).astype(int).flatten()

    bit_errors = sum(decoded_bits[:len(bitstream)] != bitstream)
    symbol_errors = sum(decoded_symbols[:len(transmitted_symbols)] != round((angle(transmitted_symbols) / (2 * pi)) * M_values) % M_values)

    BER_sim.append(bit_errors/len(bitstream))

    SER_sim.append(symbol_errors/len(transmitted_symbols))
    figure(figsize = (10,5))
    subplot(1,2,1)
    scatter(received_symbols.real,received_symbols.imag,s=1)

    reconstructed_bits = decoded_bits[:len(image.flatten()) * 8]
    reconstructed_image = packbits(reconstructed_bits)[:len(image.flatten())].reshape(image.shape)


    subplot(1,2,2)
    imshow(reconstructed_image,cmap = 'grey')
    show()
    
  BER_res[M_values] = BER_sim
  SER_res[M_values] = SER_sim
  
  figure(figsize=(8, 5))
  semilogy(snr_db_range,BER_res[M_values])
  semilogy(snr_db_range, BER_res[M_values], marker='o', linestyle='-', label=f'Practical BER M={M_values}')
  semilogy(snr_db_range, [theoretical_ber(M_values, snr) for snr in snr_db_range], linestyle='--', label=f'Theoretical BER M={M_values}')
  semilogy(snr_db_range, SER_res[M_values], marker='s', linestyle='-', label=f'Practical SER M={M_values}')
  semilogy(snr_db_range, [theoretical_ser(M_values, snr) for snr in snr_db_range], linestyle='--', label=f'Theoretical SER M={M_values}')
    
  xlabel('SNR (dB)')
  ylabel('Error Rate')
  title(f'BER & SER for M={M_values} over AWGN Channel')
  grid(True, which='both')
  legend()
  show()
figure(figsize=(10, 6))
for M in M_values:
    semilogy(snr_db_range, BER_res[M], marker='o', linestyle='-', label=f'BER M={M}')
    semilogy(snr_db_range, SER_res[M], marker='s', linestyle='--', label=f'SER M={M}')
    
xlabel('SNR (dB)')
ylabel('Error Rate')
title('Combined BER and SER for MPSK over AWGN Channel')
grid(True, which='both')
legend()
show()
