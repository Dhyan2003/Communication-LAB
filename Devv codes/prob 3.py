#PROBABILITY AND RANDOM VARIABLES
#Plotting histograms of random variables X,Y Z where Z=X+Y
#Plot of PDF is Gaussian in nature
#Plotting the PMFs of X and Y
#To determine the mean and variance of Z

#Authors- Sreelakshmi Ajit, Devika A M, Parvathy Narayan H, Keshav Nair
#Date- 13/02/2025

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

print("Student Name- Devika A M" )
print("Student Department- ECE")

r = 23  #roll number
sigma2 = 1  # Variance
sigma = np.sqrt(sigma2)  # Standard deviation

# Generate random variables
n_samples = 10000
X = np.random.normal(r, sigma, n_samples)
Y = np.random.normal(r, sigma, n_samples)
Z = X + Y

# Plot histograms
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.hist(X, bins=50, density=True, alpha=0.7, color='b', label='X')
plt.hist(Y, bins=50, density=True, alpha=0.7
         , color='g', label='Y')
x_vals = np.linspace(r - 4*sigma, r + 4*sigma, 100)
plt.plot(x_vals, stats.norm.pdf(x_vals, r, sigma))
plt.title("Histogram of X and Y")
plt.legend()

# Plot Cumulative Distribution Function (CDF)
def plot_cdf(data, color, label):
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    plt.plot(sorted_data, cdf, color=color, label=label)
    plt.xlabel("Values")
    plt.ylabel("Cumulative Probability")
    plt.title("CDF of X and Y")
    plt.legend()

plt.subplot(1, 3, 2)
plot_cdf(X, 'b', 'X')
plot_cdf(Y, 'g', 'Y')
plt.subplot(1, 3, 2)
plot_cdf(X, 'b', 'X')
plot_cdf(Y, 'g', 'Y')

# Plot histogram of Z
plt.subplot(1, 3, 3)
plt.hist(Z, bins=50,density=True, alpha=0.6, color='b', label='Z')
z_vals = np.linspace(2*r - 4*np.sqrt(2), 2*r + 4*np.sqrt(2), 100)
plt.plot(z_vals, stats.norm.pdf(z_vals, 2*r, np.sqrt(2*sigma2)), label='PDF')
plt.title("Histogram of Z")
plt.legend()
plt.show()

# Calculate mean and variance of Z
mean_Z = np.mean(Z)
var_Z = np.var(Z)
print(f"Mean of Z: {mean_Z}")
print(f"Variance of Z: {var_Z}")