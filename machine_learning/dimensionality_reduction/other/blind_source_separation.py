"""
===================================
FastICA for Blind source separation
===================================
Task: estimating sources from noisy data (noisy measurements).
Imagine 3 instruments playing simultaneously and 3 microphones recording the mixed signals.
ICA is used to recover the sources ie. what is played by each instrument.
Importantly, PCA fails at recovering our `instruments` since the related signals reflect non-Gaussian processes.

https://scikit-learn.org/stable/auto_examples/decomposition/plot_ica_blind_source_separation.html
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from sklearn.decomposition import FastICA, PCA

#####################################

# Generate sample data
np.random.seed(0)
n_samples = 2000
time = np.linspace(0, 8, n_samples)

s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal

S = np.c_[s1, s2, s3]
S += 0.2 * np.random.normal(size=S.shape)  # Add noise

S /= S.std(axis=0)  # Standardize data
# Mix data
A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
X = np.dot(S, A.T)  # Generate observations

#####################################

# Reconstruct signals

# Compute ICA
ica = FastICA(n_components=3)
S_ = ica.fit_transform(X)
# We can `prove` that the ICA model applies by reverting the un-mixing:
A_ = ica.mixing_  # Get estimated mixing matrix
assert np.allclose(X, np.dot(S_, A_.T) + ica.mean_)

# Compute PCA (for comparison)
#   Reconstruct signals based on orthogonal components
pca = PCA(n_components=3)
H = pca.fit_transform(X)

#####################################

# Plot results

plt.figure(figsize=(10, 6))

models = [X, S, S_, H]
names = ['Observations (mixed signal)',
         'True Sources',
         'ICA recovered signals',
         'PCA recovered signals']
colors = ['red', 'steelblue', 'orange']

for i, (model, name) in enumerate(zip(models, names), start=1):
    plt.subplot(4, 1, i)  # rows, columns, position
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color)

plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)
plt.show()
