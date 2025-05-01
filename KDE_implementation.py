from sklearn.neighbors import KernelDensity
import numpy as np
import matplotlib.pyplot as plt

# Generate 1D Gaussian-like data
np.random.seed(42)
X_kde = np.concatenate([np.random.normal(loc=0, scale=1, size=300),
                        np.random.normal(loc=5, scale=0.5, size=100)])[:, np.newaxis]

# Fit KDE model
kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X_kde)

# Evaluation of density on a grid
x_d = np.linspace(-4, 9, 1000)[:, np.newaxis]
log_dens = kde.score_samples(x_d)

# Plot the density
plt.figure(figsize=(7, 4))
plt.fill_between(x_d[:, 0], np.exp(log_dens), color="orange", alpha=0.6)
plt.title('Kernel Density Estimation (1D)')
plt.xlabel('Value')
plt.ylabel('Density')
plt.grid(True)
plt.show()






