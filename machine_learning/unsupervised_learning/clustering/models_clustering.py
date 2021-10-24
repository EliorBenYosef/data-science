"""
A "clustering" is essentially a set of such clusters, usually containing all objects in the dataset.
Additionally, it may specify the relationship of the clusters to each other, for example,
a hierarchy of clusters embedded in each other.

Clustering types:
- Centroid-based Clustering - K-Means, K-Medoids, Fuzzy C-Means
- Connectivity-based Clustering - Hierarchical Clustering
- Density-based Clustering - Density-based Spatial Clustering of Applications with Noise (DBSCAN),
    Ordering Points to Identify the Clustering Structure (OPTICS), Mean Shift
- Distribution-based Clustering - Gaussian Mixture Models (GMMs) utilize the Expectation-Maximization (EM) algorithm
    which uses multivariate normal distributions.
- Grid-based Clustering - STING, CLIQUE

https://scikit-learn.org/stable/modules/clustering.html
https://medium.com/@masarudheena/4-best-ways-to-find-optimal-number-of-clusters-for-clustering-with-python-code-706199fa957c
https://towardsdatascience.com/cheat-sheet-to-implementing-7-methods-for-selecting-optimal-number-of-clusters-in-python-898241e1d6ad
"""

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import cdist
import numpy as np
from utils import get_centroids


class CentroidKMeans:
    """
    K-Means Clustering
    Centroid-based Clustering

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

    def __init__(self, X, init='k-means++'):
        self.X = X
        self.init = init

    def k_means(self, n_clusters=None):
        clusterer = KMeans(n_clusters=n_clusters, init=self.init, random_state=0)
        clusterer.fit(self.X)
        # y_pred = clusterer.fit_predict(X)
        # y_pred = clusterer.labels_

        centroids = clusterer.cluster_centers_

        return clusterer, centroids

    def get_n_clusters_opt(self, max_clusters, plot=True):
        self.elbow_method(max_clusters, plot)
        self.silhouette_analysis_method(max_clusters, plot)

    def elbow_method(self, max_clusters, plot):
        """
        The Elbow Method is more of a decision rule.
        Determines the optimal number of clusters.
        Within Cluster Sum of Squares (WCSS) - measures:
            the squared average distance of all the samples within a cluster to the cluster centroid.
            (the sum of squared distances of samples to their closest cluster center).
            also referred to as "distortion"?

        https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/
        https://blog.cambridgespark.com/how-to-determine-the-optimal-number-of-clusters-for-k-means-clustering-14f27070048f
        https://www.codecademy.com/learn/machine-learning/modules/dspath-clustering/cheatsheet
        """
        min_clusters = 2

        # distortions = []
        wcss = []  # inertias
        wcss_diff = []
        wcss_diff_ratio = []

        for k in range(min_clusters - 1, max_clusters + 1):
            k_means = KMeans(n_clusters=k, init=self.init, random_state=0)
            k_means.fit(self.X)

            # distortions.append(sum(np.min(cdist(X, k_means.cluster_centers_, metric='euclidean'), axis=1)) / X.shape[0])
            wcss.append(k_means.inertia_)
            if k > min_clusters - 1:
                diff = k_means.inertia_ - wcss[k - 2]
                if diff >= 0:
                    diff = wcss_diff[-1]
                wcss_diff.append(diff)
            if k > min_clusters:
                wcss_diff_ratio.append(wcss_diff[k - 2] / wcss_diff[k - 3])

        wcss_diff_ratio_min = sorted(wcss_diff_ratio)[:3]
        n_clusters_opt_1 = wcss_diff_ratio.index(wcss_diff_ratio_min[0]) + min_clusters
        n_clusters_opt_2 = wcss_diff_ratio.index(wcss_diff_ratio_min[1]) + min_clusters
        n_clusters_opt_3 = wcss_diff_ratio.index(wcss_diff_ratio_min[2]) + min_clusters

        print(f'Optimal clusters numbers (Elbow calculation): '
              f'{n_clusters_opt_1}, {n_clusters_opt_2}, {n_clusters_opt_3}')

        if plot:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(range(min_clusters - 1, max_clusters + 1), wcss)
            # the Elbow method lines:
            plt.axvline(x=n_clusters_opt_1, color='green', linestyle='--')
            plt.axvline(x=n_clusters_opt_2, color='yellow', linestyle='--')
            plt.axvline(x=n_clusters_opt_3, color='red', linestyle='--')
            ax.set_xlabel('Clusters Number (k)')
            ax.set_ylabel('WCSS')
            ax.set_title('The Elbow Method')
            plt.show()

    def silhouette_analysis_method(self, max_clusters, plot):
        """
        The Silhouette Analysis method is a metric used for validation while clustering.
        Thus, it can be used in combination with the Elbow Method.
        Therefore, both methods are not alternatives to each other for finding the optimal K.
            Rather they are tools to be used together for a more confident decision.
        Determines the optimal number of clusters.

        This method calculates the average silhouette value for each sample in the cluster,
            this value represents how similar a sample is to its own cluster.
            The range of this measure from -1 to 1.
                A value of 1 means the sample is far away from the neighboring clusters.
                A negative value refers to samples that might have been assigned to the wrong cluster.

        On the Silhouette plot:
            each row represents one sample in the scatter plot
            and the X-axis refers to silhouette coefficient value.
            The red line indicates the average silhouette coefficient value for all samples in clusters.

        The cluster that has a high silhouette coefficient value is the best to choose.

        https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
        """
        # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2:
        min_clusters = 2

        sil_avg = []
        for k in range(min_clusters, max_clusters + 1):
            k_means = KMeans(n_clusters=k, init=self.init, random_state=0)
            k_means.fit(self.X)
            sil_avg.append(silhouette_score(self.X, k_means.labels_, metric='euclidean'))

        n_clusters_opt = sil_avg.index(max(sil_avg)) + min_clusters
        print(f'Optimal clusters number (Silhouette metric): {n_clusters_opt}')

        if plot:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(range(min_clusters, max_clusters + 1), sil_avg)
            plt.axvline(x=n_clusters_opt, color='green', linestyle='--')
            ax.set_xlabel('Clusters Number (k)')
            ax.set_ylabel('Silhouette Score (sil_avg)')
            ax.set_title('The Silhouette Analysis Method')
            plt.show()


class ConnectivityHierarchical:
    """
    Hierarchical Clustering
    Connectivity-based Clustering

    There are two types of hierarchical clustering:
    Agglomerative - bottom-up approach. Samples are clustered starting with individual samples.
    Divisive - top-down approach. All the samples are treated as one big cluster and the clustering process
        involves dividing the one big cluster into several small clusters.

    linkage - the clustering technique.
        'ward' - the method of minimum variance (withing-cluster variance). the most recommended method.
            minimizes the samples variance inside the clusters.
            results in clusters with samples that don't vary too much (with low variance).

    https://stackabuse.com/hierarchical-clustering-with-python-and-scikit-learn
    """

    def __init__(self, X, affinity='euclidean', linkage='ward'):
        self.X = X
        self.affinity = affinity
        self.linkage = linkage

    def hierarchical(self, n_clusters):
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, affinity=self.affinity, linkage=self.linkage)
        clusterer.fit(self.X)
        # y_pred = clusterer.fit_predict(X)
        # y_pred = clusterer.labels_

        centroids = get_centroids(self.X, clusterer.labels_)

        return clusterer, centroids

    def plot_dendrogram(self, euclidean_dist_threshold=None):
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
        Z = sch.linkage(self.X, method=self.linkage)
        if euclidean_dist_threshold is None:
            sch.dendrogram(Z)  # color_threshold_def = 0.7 * max(Z[:, 2])
        else:
            sch.dendrogram(Z, color_threshold=euclidean_dist_threshold)
        ax.set_xlabel('Samples')
        ax.set_ylabel(f'{str.title(self.affinity)} Distances')
        ax.set_title('Dendrogram')
        plt.show()


class DensityDBSCAN:
    """
    Density-based Spatial Clustering of Applications with Noise (DBSCAN)
    Density-based Clustering

    works well for detecting outliers and anomalies in the dataset.

    In DBSCAN, there are no centroids, and clusters are formed by linking nearby samples to one another.
    DBSCAN doesn't require specifying the number of clusters, it's detects it according to the .

    DBSCAN detects the following sample types:
    * Core samples - samples within eps of a core sample, that meet the min_samples criteria.
        clusterer.core_sample_indices_
    * Border samples - samples within eps of a core sample, but don't meet the min_samples criteria.
    * Noise samples (outliers) - sample that aren't Core / Border.

    https://medium.com/nearist-ai/dbscan-clustering-tutorial-dd6a9b637a4b
    https://medium.com/@agarwalvibhor84/lets-cluster-data-points-using-dbscan-278c5459bee5
    """

    def __init__(self, X):
        self.X = X

    def db_scan(self, eps, min_samples=None):
        """
        DBSCAN requires specifying 2 important parameters (eps, min_samples) which influence the decision of
        whether two nearby samples should be linked into the same cluster.
        This affects the resulting number of clusters (and that's the reason DBSCAN doesn't require specifying it)

        DBSCAN doesn't require specifying the number of clusters, it's good at detecting it itself.
        Note that evern the method returns centroids, in DBSCAN there are no "real" centroids,
            and clusters are formed by linking nearby samples to one another.

        :param min_samples: min_samples_for_a_core_cluster
            the minimal number of neighboring samples that is needed to form a cluster of core samples.
            Determined based on:
            * domain knowledge
            * the dataset size

        :param eps: max_dist - the max distance that determines a sample’s neighbor.
            Samples are neighbors if:
                dist(p1,p2) <= eps.
            the smaller the eps, the more clusters & outlier samples?
        """
        if min_samples is None:
            min_samples = self.get_min_samples(self.X)

        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        clusterer.fit(self.X)
        # y_pred = clusterer.fit_predict(X)
        y_pred = clusterer.labels_

        n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)

        core_samples_mask = np.zeros_like(y_pred, dtype=bool)
        core_samples_mask[clusterer.core_sample_indices_] = True

        # n_noise is the number of outlier samples (marked with the label -1)
        n_noise = list(y_pred).count(-1)
        outlier_samples_mask = np.zeros_like(y_pred, dtype=bool)
        outlier_samples_mask[y_pred == -1] = True

        outliers_mask = y_pred == -1
        # to inverse a mask: ~mask / np.logical_not(mask)
        centroids = get_centroids(self.X[~outliers_mask], clusterer.labels_[~outliers_mask])

        return clusterer, centroids

    @staticmethod
    def get_min_samples(X):
        """
        Returns the minimal number of neighboring samples that is needed to form a cluster of core samples
        (min_samples_for_a_core_cluster), according to a rule of thumb regarding the dataset size.
        """
        dataset_size = X.shape[0]
        D = X.shape[1]  # D - the dataset's (features) dimensionality

        if dataset_size < 1000:  # Smaller dataset
            min_samples = D + 1
        else:  # Larger dataset
            min_samples = D * 2
        return min_samples


