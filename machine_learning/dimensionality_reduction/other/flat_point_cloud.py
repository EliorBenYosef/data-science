from sklearn.decomposition import PCA

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#####################################

# Create the data - a point cloud which is very flat in one direction

e = np.exp(1)
np.random.seed(4)


def pdf(x):
    """
    Probability Density Function?
    """
    return 0.5 * (stats.norm(scale=0.25 / e).pdf(x)
                  + stats.norm(scale=4 / e).pdf(x))


y = np.random.normal(scale=0.5, size=(30000))
x = np.random.normal(scale=0.5, size=(30000))
z = np.random.normal(scale=0.1, size=len(x))

density = pdf(x) * pdf(y)
pdf_z = pdf(5 * z)

density *= pdf_z

a = x + y
b = 2 * y
c = a - b + z

norm = np.sqrt(a.var() + b.var())
a /= norm
b /= norm

#####################################

X = np.c_[a, b, c]

# Using: scipy.linalg.svd -
# _, explained_variance_ratio, V = SVD(X, full_matrices=False)

# PCA - chooses a direction that is not flat:
pca = PCA(n_components=3)
pca.fit(X)
explained_variance_ratio = pca.explained_variance_ratio_
V = pca.components_

x_axis, y_axis, z_axis = 3 * V.T

x_plane = np.r_[x_axis[:2], - x_axis[1::-1]]
y_plane = np.r_[y_axis[:2], - y_axis[1::-1]]
z_plane = np.r_[z_axis[:2], - z_axis[1::-1]]

x_plane.shape = (2, 2)
y_plane.shape = (2, 2)
z_plane.shape = (2, 2)


#####################################

# Plot the figures

def plot_figs(fig_num, elev, azim):
    fig = plt.figure(fig_num, figsize=(8, 6))
    plt.clf()
    ax = Axes3D(fig, elev=elev, azim=azim, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.scatter(a[::10], b[::10], c[::10], c=density[::10], marker='+', alpha=.4)
    ax.plot_surface(x_plane, y_plane, z_plane)
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])


plot_figs(1, elev=-40, azim=-80)
plot_figs(2, elev=30, azim=20)

plt.show()
