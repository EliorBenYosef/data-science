import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial.distance import euclidean


class Visualizer:

    def __init__(self, X, y_label, x1_label, x2_label, x3_label=None,
                 y=None, centroids=None, clss_labels=None):
        self.X = X
        self.y = y
        self.centroids = centroids
        self.clss_labels = clss_labels

        self.x1_label = x1_label
        self.x2_label = x2_label
        self.x3_label = x3_label

        self.clustering = y_label

        # self.colors_bold = ['red', 'blue', 'green', 'cyan', 'magenta']
        # # self.colors_bold = ('#FF0000', '#00FF00', '#0000FF')
        self.colors_bold = ('#0000FF', '#FF0000', '#00FF00',
                            '#00FFFF', '#FF00FF', '#FFFF00')
        self.colors_light = ('#AAAAFF', '#FFAAAA', '#AAFFAA',
                             '#AAFFFF', '#FFAAFF', '#FFFFAA')

    def visualize_results_2D(self, clusterer, centroids, model_name):
        """
        Visualizes the clusters
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(self.X[:, 0], self.X[:, 1], c=clusterer.labels_, cmap='rainbow')  # s=50
        # for i in range(self.n_clusters):
        #     ax.scatter(self.X[y == i, 0], self.X[y == i, 1],
        #                c=self.colors_bold[i], s=50, label=f'Cluster {i + 1}')  # 50
        ax.scatter(centroids[:, 0], centroids[:, 1], edgecolor='k', c='white', s=50, label='Centroids')
        ax.legend()
        # ax.legend(loc='best', shadow=False, scatterpoints=1)
        ax.set_xlabel(self.x1_label)
        ax.set_ylabel(self.x2_label)
        ax.set_title(f'{model_name} - Clusters of {self.clustering}')
        plt.show()

    # def plot_3d_fig(
    def visualize_results_3D(self, clusterer, centroids, model_name, fig_num,
                             elev=20, azim=-150):  # elev=48, azim=134
        fig = plt.figure(fig_num, figsize=(8, 6))
        plt.clf()

        ax = Axes3D(fig, elev=elev, azim=azim, auto_add_to_figure=False)  # rect=[0, 0, .95, 1],
        fig.add_axes(ax)

        # Reorder the predicted labels to match the target labels order.
        indices = []
        for i, c_target in enumerate(list(self.centroids)):
            dist = []
            for c_pred in list(centroids):
                dist.append(euclidean(c_pred, c_target))
            index = dist.index(min(dist))
            indices.append(index)
        y_pred = clusterer.labels_
        y_pred = np.choose(y_pred, indices).astype(np.int)

        if self.y is not None:
            y_combined = np.c_[y_pred, self.y]
            for i in np.unique(y_pred):
                for j in np.unique(self.y):
                    mask = np.all(y_combined == [i, j], axis=1)
                    ax.scatter(self.X[mask, 0], self.X[mask, 1], self.X[mask, 2],
                               color=self.colors_light[i], edgecolor=self.colors_bold[j], s=50)
        else:
            ax.scatter(self.X[:, 0], self.X[:, 1], self.X[:, 2],
                       c=y_pred, cmap='rainbow', edgecolor='k', s=50)

        ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
                   edgecolor='k', c='k', s=50, label='Centroids')

        if self.y is not None and self.clss_labels is not None:
            for label in self.clss_labels:
                mask = self.y == self.clss_labels.index(label)
                ax.text3D(self.X[mask, 0].mean(),
                          self.X[mask, 1].mean(),
                          self.X[mask, 2].mean() + 2, label,
                          horizontalalignment='center',
                          bbox=dict(alpha=.2, edgecolor='w', facecolor='w'))

        ax.set_xlabel(self.x1_label)
        ax.set_ylabel(self.x2_label)
        ax.set_zlabel(self.x3_label)
        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])

        ax.set_title(f'{model_name} - Clusters of {self.clustering}')

        # ax.dist = 12

    @staticmethod
    def show_results():
        plt.show()
