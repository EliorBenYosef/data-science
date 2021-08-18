from sklearn import datasets
from data_tools.data import ClusteringDataSets, get_X_from_sklearn_dataset
from models_clustering import K_means, Hierarchical
from sklearn.neighbors import NearestCentroid
# from sklearn.datasets import make_blobs
from utils import Visualizer
import matplotlib.pyplot as plt


def bivariate_clustering():
    dataset = ClusteringDataSets()
    dataset.get_Mall_Customers(indices=[-2, -1])  # [3, 4]. Use only two features

    # # Create dataset with 3 random cluster centers and 1000 datapoints
    # X, y = make_blobs(n_samples=1000, centers=3, n_features=2, shuffle=True, random_state=31)
    # x1_label, x2_label = '', ''

    visualizer = Visualizer(dataset.X, dataset.y_label, *dataset.x_labels)

    km = K_means()
    clusterer_km, centroids_km = km.k_means(dataset.X, max_clusters=10, plot=True)
    visualizer.visualize_results_2D(clusterer_km, centroids_km, model_name='K-Means')

    hc = Hierarchical(affinity='euclidean', linkage='ward')
    hc.plot_dendrogram(dataset.X, xlabel=dataset.y_label)
    # Manually specify n_clusters after observing the dendrogram:
    clusterer_hc, centroids_hc = hc.hierarchical(dataset.X, n_clusters=5)
    visualizer.visualize_results_2D(clusterer_hc, centroids_hc, model_name='Hierarchical')


def trivariate_clustering():
    dataset = datasets.load_iris()
    X, x_labels = get_X_from_sklearn_dataset(dataset, indices=[3, 0, 2])  # Use only three features
    y = dataset.target
    y_label = 'Iris type'
    clss_labels = list(dataset.target_names)

    classifier = NearestCentroid()
    classifier.fit(X, y)
    centroids = classifier.centroids_

    visualizer = Visualizer(X, y_label, x_labels[0], x_labels[1], x_labels[2], y, centroids, list(clss_labels))

    km = K_means()
    clusterer_km, centroids_km = km.k_means(X, n_clusters=3, plot=True)
    visualizer.visualize_results_3D(clusterer_km, centroids_km, model_name='K-Means', fig_num=1)

    hc = Hierarchical(affinity='euclidean', linkage='ward')
    # Manually specify n_clusters after observing the dendrogram:
    clusterer_hc, centroids_hc = hc.hierarchical(X, n_clusters=3)
    visualizer.visualize_results_3D(clusterer_hc, centroids_hc, model_name='Hierarchical', fig_num=2)

    plt.show()


def multivariate_clustering():
    dataset = ClusteringDataSets()
    dataset.get_Mall_Customers()

    km = K_means()
    clusterer_km, centroids_km = km.k_means(dataset.X, max_clusters=50, plot=True)

    hc = Hierarchical(affinity='euclidean', linkage='ward')
    hc.plot_dendrogram(dataset.X, xlabel=dataset.y_label)
    # manually specify n_clusters after observing the dendrogram:
    clusterer_hc, centroids_hc = hc.hierarchical(dataset.X, n_clusters=16)


if __name__ == '__main__':
    bivariate_clustering()
    trivariate_clustering()
    multivariate_clustering()
