import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.neighbors import NearestNeighbors, NearestCentroid


class Visualizer:

    def __init__(self, X, sample_label, x_labels, y=None, clss_labels=None):
        self.X = X
        self.D = X.shape[1]

        self.y = y
        self.centroids = get_centroids(X, y) if y is not None else None
        self.clss_labels = clss_labels

        self.x_labels = x_labels

        self.sample_label = sample_label

        # self.colors_bold = ['red', 'blue', 'green', 'cyan', 'magenta']
        # # self.colors_bold = ('#FF0000', '#00FF00', '#0000FF')
        self.colors_bold = ('#0000FF', '#FF0000', '#00FF00',
                            '#00FFFF', '#FF00FF', '#FFFF00')
        self.colors_light = ('#AAAAFF', '#FFAAAA', '#AAFFAA',
                             '#AAFFFF', '#FFAAFF', '#FFFFAA')

        self.fig_num = 1

    def visualize_results(self, clusterer, centroids_pred, model_name, present_centroids=True):
        if self.D == 2:
            self.visualize_results_2D(clusterer, centroids_pred, model_name, present_centroids)
        elif self.D == 3:
            self.visualize_results_3D(clusterer, centroids_pred, model_name, present_centroids)

    def visualize_results_2D(self, clusterer, centroids_pred, model_name, present_centroids):
        """
        Visualizes the clusters
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        y_pred = clusterer.labels_
        if self.centroids is not None:
            y_pred = self.reorder_y_pred(y_pred, centroids_pred)

        if self.y is not None:
            y_combined = np.c_[y_pred, self.y]
            for i in set(y_pred):
                for j in set(self.y):
                    mask = np.all(y_combined == [i, j], axis=1)
                    ax.scatter(self.X[mask, 0], self.X[mask, 1],
                               color='k' if i == -1 else self.colors_light[i], edgecolor=self.colors_bold[j], s=50)
        else:
            if -1 in set(y_pred):
                outliers_mask = y_pred == -1
                ax.scatter(self.X[outliers_mask, 0], self.X[outliers_mask, 1],
                           c='k', label='outliers')  # s=50
                # to inverse a mask: ~mask / np.logical_not(mask)
                ax.scatter(self.X[~outliers_mask, 0], self.X[~outliers_mask, 1],
                           c=y_pred[~outliers_mask], cmap='rainbow')  # s=50
            else:
                ax.scatter(self.X[:, 0], self.X[:, 1],
                           c=y_pred, cmap='rainbow')  # s=50

        if present_centroids:
            ax.scatter(centroids_pred[:, 0], centroids_pred[:, 1],
                       edgecolor='k', c='white', s=50, label='Centroids')

        ax.legend()
        # ax.legend(loc='best', shadow=False, scatterpoints=1)
        ax.set_xlabel(self.x_labels[0])
        ax.set_ylabel(self.x_labels[1])
        ax.set_title(f'{model_name} - {self.sample_label} Clusters')
        # plt.show()

    # def plot_3d_fig(
    def visualize_results_3D(self, clusterer, centroids_pred, model_name, present_centroids,
                             elev=20, azim=-150):  # elev=48, azim=134
        fig = plt.figure(self.fig_num, figsize=(8, 6))
        self.fig_num += 1
        plt.clf()
        ax = Axes3D(fig, elev=elev, azim=azim, auto_add_to_figure=False)  # rect=[0, 0, .95, 1],
        fig.add_axes(ax)

        y_pred = clusterer.labels_
        if self.centroids is not None:
            y_pred = self.reorder_y_pred(y_pred, centroids_pred)

        if self.y is not None:
            y_combined = np.c_[y_pred, self.y]
            for i in set(y_pred):
                for j in set(self.y):
                    mask = np.all(y_combined == [i, j], axis=1)
                    ax.scatter(self.X[mask, 0], self.X[mask, 1], self.X[mask, 2],
                               color='k' if i == -1 else self.colors_light[i], edgecolor=self.colors_bold[j], s=50)
        else:
            if -1 in set(y_pred):
                outliers_mask = y_pred == -1
                ax.scatter(self.X[outliers_mask, 0], self.X[outliers_mask, 1], self.X[outliers_mask, 2],
                           c='k', label='outliers', edgecolor='k', s=50)
                # to inverse a mask: ~mask / np.logical_not(mask)
                ax.scatter(self.X[~outliers_mask, 0], self.X[~outliers_mask, 1], self.X[~outliers_mask, 2],
                           c=y_pred[~outliers_mask], cmap='rainbow', edgecolor='k', s=50)
            else:
                ax.scatter(self.X[:, 0], self.X[:, 1], self.X[:, 2],
                           c=y_pred, cmap='rainbow', edgecolor='k', s=50)

        if present_centroids:
            ax.scatter(centroids_pred[:, 0], centroids_pred[:, 1], centroids_pred[:, 2],
                       edgecolor='k', c='k', s=50, label='Centroids')

        if self.y is not None and self.clss_labels is not None:
            for label in self.clss_labels:
                mask = self.y == self.clss_labels.index(label)
                ax.text3D(self.X[mask, 0].mean(),
                          self.X[mask, 1].mean(),
                          self.X[mask, 2].mean() + 2, label,
                          horizontalalignment='center',
                          bbox=dict(alpha=.2, edgecolor='w', facecolor='w'))

        ax.set_xlabel(self.x_labels[0])
        ax.set_ylabel(self.x_labels[1])
        ax.set_zlabel(self.x_labels[2])
        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])

        ax.set_title(f'{model_name} - {self.sample_label} Clusters')

        # ax.dist = 12

    def reorder_y_pred(self, y_pred, centroids_pred):
        """
        Reorders the predicted labels to match the target labels order
        """
        indices = np.arange(len(list(centroids_pred)))
        visited_indices = []
        for i, c_target in enumerate(list(self.centroids)):
            dist = []
            for j, c_pred in enumerate(list(centroids_pred)):
                dist.append(np.inf if j in visited_indices else euclidean(c_pred, c_target))
            index = dist.index(min(dist))
            visited_indices.append(index)
            indices[index] = i
        if -1 in y_pred:
            y_pred_new = y_pred.copy()
            outliers_mask = y_pred == -1
            y_pred_new[~outliers_mask] = np.choose(y_pred[~outliers_mask], indices).astype(np.int)
        else:
            y_pred_new = np.choose(y_pred, indices).astype(np.int)
        return y_pred_new

    @staticmethod
    def show_results():
        plt.show()


def plot_k_dist(X, n_neighbors, dist_elbow_value):
    """
    Used to determine the optimal distance (eps in case of DBSCAN), via the Elbow method.
    """
    # Nearest Neighbors:
    nn_learner = NearestNeighbors(n_neighbors=n_neighbors)
    nn_learner.fit(X)

    # k-distance:
    distances, indices = nn_learner.kneighbors(X)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]

    # eps visualization graph - the optimal eps is at the point of maximum curvature (the Elbow method).
    plt.plot(distances)  # color='#D15E14'

    plt.axhline(y=dist_elbow_value, color='k', linestyle='--')  # xmin=0.05, xmax=0.95  # the Elbow method line
    # plt.xlim(1500, 1700)  # Zoom in plot
    plt.title('k-distance plot')
    plt.xlabel('Neighboring Samples')  # Clustered Samples
    plt.ylabel('Distance')  # 'eps' (for DBSCAN)
    plt.show()


def get_centroids(X, y):
    classifier = NearestCentroid(metric='euclidean')
    classifier.fit(X, y)
    centroids = classifier.centroids_
    return centroids
