from sklearn import datasets
from data_tools.data import ClusteringDataSets, get_X_from_sklearn_dataset
from models_clustering import CentroidKMeans, ConnectivityHierarchical, DensityDBSCAN
from utils import Visualizer, plot_k_dist
import matplotlib.pyplot as plt


# from sklearn.datasets import make_blobs
# # Create dataset with 3 random cluster centers and 1000 datapoints
# X, y = make_blobs(n_samples=1000, centers=3, n_features=2, shuffle=True, random_state=31)
# x_labels = ['', '']
# y_label =


def clustering_analysis(X, max_clusters, eps_opt, min_samples_opt=None, euclidean_dist_threshold=None):
    # to get n_clusters for K-Means:
    km = CentroidKMeans(X)
    km.get_n_clusters_opt(max_clusters)

    # to get n_clusters for Hierarchical Clustering:
    hc = ConnectivityHierarchical(X, affinity='euclidean', linkage='ward')
    hc.plot_dendrogram(euclidean_dist_threshold=euclidean_dist_threshold)

    # to get eps for DBSCAN:
    if min_samples_opt is None:
        plot_k_dist(X, n_neighbors=DensityDBSCAN.get_min_samples(X), dist_elbow_value=eps_opt)
    else:
        plot_k_dist(X, n_neighbors=min_samples_opt, dist_elbow_value=eps_opt)


def clustering(X, sample_label, x_labels, eps, min_samples=None,
               n_clusters=None, y=None, clss_labels=None):

    D = X.shape[1]
    if y is not None:
        n_clusters = len(set(y))

    km = CentroidKMeans(X)
    clusterer_km, centroids_km = km.k_means(n_clusters=n_clusters)

    hc = ConnectivityHierarchical(X, affinity='euclidean', linkage='ward')
    clusterer_hc, centroids_hc = hc.hierarchical(n_clusters=n_clusters)

    dbscan = DensityDBSCAN(X)
    clusterer_dbscan, centroids_dbscan = dbscan.db_scan(eps, min_samples)

    if D == 2 or D == 3:
        visualizer = Visualizer(X, sample_label, x_labels, y, clss_labels)
        visualizer.visualize_results(clusterer_km, centroids_km, 'K-Means')
        visualizer.visualize_results(clusterer_hc, centroids_hc, 'Hierarchical')
        visualizer.visualize_results(clusterer_dbscan, centroids_dbscan, 'DBSCAN', present_centroids=False)
        plt.show()


#####################################

def bivariate_clustering():
    dataset = ClusteringDataSets()
    dataset.get_Mall_Customers(indices=[-2, -1])  # [3, 4]. Use only two features
    X, sample_label, x_labels = dataset.X, dataset.sample_label, dataset.x_labels

    min_samples = 4
    clustering_analysis(X, max_clusters=10, euclidean_dist_threshold=5, eps_opt=0.2, min_samples_opt=min_samples)
    clustering(X, sample_label, x_labels, n_clusters=5, eps=0.35, min_samples=min_samples)  # eps = 0.3645242131210


def trivariate_clustering():
    dataset = datasets.load_iris()
    X, x_labels = get_X_from_sklearn_dataset(dataset, indices=[3, 0, 2])  # Use only three features
    sample_label = 'Iris'

    min_samples = 7
    clustering_analysis(X, max_clusters=10, euclidean_dist_threshold=7, eps_opt=0.2236, min_samples_opt=min_samples)
    clustering(X, sample_label, x_labels, n_clusters=2, eps=0.6, min_samples=min_samples)  # eps=0.4


def multivariate_clustering():
    dataset = ClusteringDataSets()
    dataset.get_Mall_Customers()
    X, sample_label, x_labels = dataset.X, dataset.y_label, dataset.x_labels

    min_samples = 6
    clustering_analysis(X, max_clusters=50, euclidean_dist_threshold=5, eps_opt=0.55, min_samples_opt=min_samples)
    clustering(X, sample_label, x_labels, n_clusters=16, eps=0.55, min_samples=min_samples)


#####################################

def bivariate_clustering_with_targets():
    dataset = datasets.load_iris()
    X, x_labels = get_X_from_sklearn_dataset(dataset, indices=[3, 0])  # Use only 2 features
    sample_label = 'Iris'
    clss_labels = list(dataset.target_names)
    y = dataset.target

    # plot_k_dist(X, n_neighbors=DensityDBSCAN.get_min_samples(X), dist_elbow_value=0.2)

    clustering(X, sample_label, x_labels, eps=0.3, min_samples=5, y=y, clss_labels=clss_labels)


def trivariate_clustering_with_targets():
    dataset = datasets.load_iris()
    X, x_labels = get_X_from_sklearn_dataset(dataset, indices=[3, 0, 2])  # Use only 3 features
    sample_label = 'Iris'
    clss_labels = list(dataset.target_names)
    y = dataset.target

    # plot_k_dist(X, n_neighbors=DensityDBSCAN.get_min_samples(X), dist_elbow_value=0.2)

    clustering(X, sample_label, x_labels, eps=0.4, min_samples=4, y=y, clss_labels=clss_labels)


def multivariate_clustering_with_targets():
    dataset = datasets.load_iris()
    X, x_labels = get_X_from_sklearn_dataset(dataset)
    sample_label = 'Iris'
    clss_labels = list(dataset.target_names)
    y = dataset.target

    # plot_k_dist(X, n_neighbors=DensityDBSCAN.get_min_samples(X), dist_elbow_value=0.2)

    clustering(X, sample_label, x_labels, eps=0.4, min_samples=5, y=y, clss_labels=clss_labels)


#####################################

if __name__ == '__main__':
    bivariate_clustering()
    trivariate_clustering()
    multivariate_clustering()
    bivariate_clustering_with_targets()
    trivariate_clustering_with_targets()
    multivariate_clustering_with_targets()


