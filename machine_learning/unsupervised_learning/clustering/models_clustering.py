"""
https://medium.com/@masarudheena/4-best-ways-to-find-optimal-number-of-clusters-for-clustering-with-python-code-706199fa957c
https://towardsdatascience.com/cheat-sheet-to-implementing-7-methods-for-selecting-optimal-number-of-clusters-in-python-898241e1d6ad

- K-Means
- K-Medoids
- Fuzzy
    - Fuzzy C-Means
- Support-Vector Clustering (SVC)
- Spectral
- Hierarchical
- Expectation-Maximization (EM)
- Mean shift
- Density-based
- Gaussian Mixture Models (GMMs)
"""

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import cdist


class K_means:
    """
    K-Means Clustering
    Note that K-Means performs better than Hierarchical Clustering on large datasets.

    Initialization methods:
    - random - not recommended (random initialization trap).
        remember to set the seed - np.random.seed(#).
    - k-means++ - an  that enables a correct initialization of the centroids in K-Means,
        and avoids the random initialization trap.

    n_init - number of time the k-means algorithm will be run with different (random?) centroid seeds.
        The final results will be the best output of n_init consecutive runs in terms of inertia.
        only when init='random' ?

    https://medium.com/analytics-vidhya/how-to-determine-the-optimal-k-for-k-means-708505d204eb
    https://www.analyticsvidhya.com/blog/2021/05/k-mean-getting-the-optimal-number-of-clusters/
    """

    def __init__(self, init='k-means++'):
        self.init = init
        self.n_clusters_opt = 0

    def k_means(self, X, max_clusters=10, n_clusters=None, plot=False):
        if n_clusters is not None:
            clusterer = KMeans(n_clusters=n_clusters, init=self.init, random_state=0)
        else:
            self.elbow_method(X, max_clusters, plot)
            self.silhouette_method(X, max_clusters, plot)
            clusterer = KMeans(n_clusters=self.n_clusters_opt, init=self.init, random_state=0)

        clusterer.fit(X)
        # y_pred = clusterer.fit_predict(X)
        # y_pred = clusterer.labels_

        centroids = clusterer.cluster_centers_

        return clusterer, centroids

    def elbow_method(self, X, max_clusters, plot):
        """
        The Elbow Method is more of a decision rule.
        Determines the optimal number of clusters.
        Within Cluster Sum of Squares (WCSS) - measures the squared average distance of all the points within a cluster
            to the cluster centroid.

        https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/
        https://blog.cambridgespark.com/how-to-determine-the-optimal-number-of-clusters-for-k-means-clustering-14f27070048f
        https://www.codecademy.com/learn/machine-learning/modules/dspath-clustering/cheatsheet
        """
        # distortions = []
        wcss = []  # inertias
        wcss_diff = []
        wcss_diff_ratio = []

        for k in range(1, max_clusters + 1):
            k_means = KMeans(n_clusters=k, init=self.init, random_state=0)
            k_means.fit(X)

            # distortions.append(sum(np.min(cdist(X, k_means.cluster_centers_, metric='euclidean'), axis=1)) / X.shape[0])
            wcss.append(k_means.inertia_)
            if k > 1:
                diff = k_means.inertia_ - wcss[k - 2]
                if diff >= 0:
                    diff = wcss_diff[-1]
                wcss_diff.append(diff)
            if k > 2:
                wcss_diff_ratio.append(wcss_diff[k - 2] / wcss_diff[k - 3])

        wcss_diff_ratio_min = sorted(wcss_diff_ratio)[:3]
        print(f'Optimal clusters numbers (Elbow calculation): '
              f'{wcss_diff_ratio.index(wcss_diff_ratio_min[0]) + 2}, '
              f'{wcss_diff_ratio.index(wcss_diff_ratio_min[1]) + 2}, '
              f'{wcss_diff_ratio.index(wcss_diff_ratio_min[2]) + 2}')

        if plot:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(range(1, max_clusters + 1), wcss)
            ax.set_xlabel('Clusters Number (k)')
            ax.set_ylabel('WCSS')
            ax.set_title('The Elbow Method')
            plt.show()

    def silhouette_method(self, X, max_clusters, plot):
        """
        The Silhouette method is a metric used for validation while clustering.
        Thus, it can be used in combination with the Elbow Method.
        Therefore, both methods are not alternatives to each other for finding the optimal K.
            Rather they are tools to be used together for a more confident decision.
        Determines the optimal number of clusters.

        https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
        """
        sil = []
        # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
        for k in range(2, max_clusters + 1):
            k_means = KMeans(n_clusters=k, init=self.init, random_state=0)
            k_means.fit(X)
            sil.append(silhouette_score(X, k_means.labels_, metric='euclidean'))

        self.n_clusters_opt = sil.index(max(sil)) + 2
        print(f'Optimal clusters number (Silhouette metric): {self.n_clusters_opt}')

        if plot:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(range(2, max_clusters + 1), sil)
            ax.set_xlabel('Clusters Number (k)')
            ax.set_ylabel('Silhouette Score')
            ax.set_title('The Silhouette Method')
            plt.show()


class Hierarchical:
    """
    Hierarchical Clustering

    There are two types of hierarchical clustering:
    Agglomerative - bottom-up approach. Data points are clustered starting with individual data points.
    Divisive - top-down approach. All the data points are treated as one big cluster and the clustering process
        involves dividing the one big cluster into several small clusters.

    linkage - the clustering technique.
        'ward' - the method of minimum variance (withing-cluster variance). the most recommended method.
            minimizes the points variance inside the clusters.
            results in clusters with points that don't vary too much (with low variance).

    https://stackabuse.com/hierarchical-clustering-with-python-and-scikit-learn
    """

    def __init__(self, affinity='euclidean', linkage='ward'):
        self.affinity = affinity
        self.linkage = linkage

    def hierarchical(self, X, n_clusters):
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, affinity=self.affinity, linkage=self.linkage)
        clusterer.fit(X)
        # y_pred = clusterer.fit_predict(X)
        # y_pred = clusterer.labels_

        classifier = NearestCentroid()
        classifier.fit(X, clusterer.labels_)
        centroids = classifier.centroids_

        return clusterer, centroids

    def plot_dendrogram(self, X, xlabel):
        """
        Plots the dendrogram, which is used to deduce the optimal number of clusters.
        The chosen distance / dissimilarity threshold (where to place the horizontal line in the dendrogram)
            determines the number of clusters.
        For getting the optimal number of clusters:
            1. extend all the horizontal lines in the dendrograms.
            2. identify the longest vertical lines.
            2. place the threshold (horizontal line) so it crosses the longest vertical lines.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        sch.dendrogram(sch.linkage(X, method=self.linkage))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f'{self.affinity} distances')
        ax.set_title('Dendrogram')
        plt.show()
