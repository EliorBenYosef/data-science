"""
Dimensionality Reduction techniques
Perform pattern recognition in high (d-)dimensional data,
    in order to project it into a smaller (k-)dimensional subspace while retaining most of the information.
Each Component is a new feature / dimension.

Used for:
- Visualization (2\3 dimensions):
    - n_components=2 for 2D visualization
    - n_components=3 for 3D visualization
- Data Compression (variance retention value, % of variance retention):
    - n_components=0.99 for 99% variance retention
    - n_components=0.95 for 95% variance retention

whiten - When True the components_ vectors are multiplied by the square root of n_samples and then divided by the
    singular values to ensure uncorrelated outputs with unit component-wise variances.
    Whitening will remove some information from the transformed signal (the relative variance scales of the components)
    but can sometime improve the predictive accuracy of the downstream estimators
    by making their data respect some hard-wired assumptions.

types:

- Type 1 - Feature Selection:
    - Stepwise Regression (Forward Selection, Backward Elimination, Bidirectional Elimination)
        (see in PracticalML LinReg)
    - Score Comparison
    - ...

- Type 2 - Feature Extraction:
    - Principal Component Analysis (PCA)
        - PCA
        - Kernel PCA (K-PCA)
    - Discriminant Analysis
        - Linear Discriminant Analysis (LDA)
        - Quadratic Discriminant Analysis (QDA)
        - Gaussian Discriminant Analysis (GDA)
    - Independent Component Analysis (ICA)
        - Fast ICA (F-ICA)
    - Singular Value Decomposition (SVD)
    - Latent Variable Model (LVM) - ???
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from machine_learning.supervised_learning.classification.models_classification import ClassificationModels
from machine_learning.supervised_learning.classification.utils import Visualizer
from sklearn import datasets
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import seaborn as sns

from sklearn.decomposition import PCA, KernelPCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from scipy.linalg import svd as SVD

colors_bold = ('#0000FF', '#FF0000', '#00FF00',
               '#00FFFF', '#FF00FF', '#FFFF00')


class DimensionReducer:

    def __init__(self, X_train, y_train, X_test, y_test, y_label, clss_labels, n_components):
        """
        :param clss_labels:
        :param n_components: int - when used for visualization (=2 for 2D / =3 for 3D),
            float (<1) - when used for Data Compression (variance retention value: =0.99 for 99%, =0.80 for 80%, etc.)
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.y_label = y_label
        self.clss_labels = clss_labels

        print('Original number of features:', X_train.shape[1], '\n')

        self.n_components = n_components

        self.X_trans = {}

    def get_n_components_by_variance_retention(self, fitted_model, goal_variance: float):
        total_variance = 0.0
        n_components = 0
        for explained_variance in fitted_model.explained_variance_ratio_:
            total_variance += explained_variance
            n_components += 1
            if total_variance >= goal_variance:
                break
        return n_components

    def print_num_of_features(self, X_train_trans, model_name):
        print(f'{model_name} - Number of features with {self.n_components} variance:', X_train_trans.shape[1])

    def print_components_variance_percentage(self, fitted_model, model_name):
        if hasattr(fitted_model, 'explained_variance_ratio_'):
            print(f'{model_name} components variance percentage:', fitted_model.explained_variance_ratio_)
            # # often useful in selecting components and estimating the dimensionality of your space:
            # explained_variance_cumulative_proportion = np.cumsum(fitted_model.explained_variance_ratio_)
        print()

    def show_scree_plot(self, fitted_model):
        """
        Scree plot - a line plot of the eigenvalues of factors or principal components in an analysis.
        The scree plot is used to determine / choose the number of:
            - Factors to retain, in an Exploratory Factor Analysis (EFA)
            - Principal Components to keep, in a Principal Component Analysis (PCA)
        This is done either via:
            - the desired Variance Retention value (80% is min?) (determined via the accumulated explained variance)
            - the Elbow Method (determined via the individual explained variance)
        """
        sns.color_palette("YlOrBr", as_cmap=True)

        plt.figure(figsize=(15, 8))

        # plot bars of individual explained variance:
        pd.Series(fitted_model.explained_variance_ratio_).plot(kind="bar", alpha=0.7)

        # Calculate the amount of variance explained added by
        acc_exp_var = []  # accumulated explained variance, for each additional component
        exp_var_total = 0
        for explained_variance in fitted_model.explained_variance_ratio_:
            exp_var_total += explained_variance
            acc_exp_var.append(exp_var_total)
        # plot line of accumulated explained variance:
        pd.Series(acc_exp_var).plot(marker="o", alpha=0.7)

        plt.xlabel("Principle Components", fontsize="x-large")
        plt.ylabel("Percentage Variance Explained", fontsize="x-large")
        plt.title("Scree Plot", fontsize="xx-large")
        plt.show()

    def visualize(self):
        n_clss = len(np.unique(self.y_test))

        for dimen_reducer_name, (X_train, X_test) in self.X_trans.items():

            if self.n_components == 2:
                plt.figure()
                plt.scatter(X_test[:, 0], X_test[:, 1],
                            c=self.y_test, cmap=ListedColormap(colors_bold[:n_clss]),
                            # alpha=.8, lw=2,
                            zorder=10, edgecolor='k', s=20)  # s - plot_symbol_size
                patches = [Patch(color=c, label=l) for c, l in zip(colors_bold[:n_clss], self.clss_labels)]
                plt.legend(handles=patches, loc='best', shadow=False, scatterpoints=1)
                var_name = dimen_reducer_name[-3:-1]
                plt.xlabel(f'{var_name}1')
                plt.ylabel(f'{var_name}2')
                plt.title(f'{dimen_reducer_name} - Predicted "{self.y_label}" label')

            elif self.n_components == 3:
                pass
                # TODO: complete

        plt.show()

    def classify_and_visualize(self):
        for dimen_reducer_name, (X_train, X_test) in self.X_trans.items():
            classification_models = ClassificationModels(X_train, self.y_train, X_test, self.y_test)
            classification_models.log_reg()
            classification_models.print_models_performance(dimen_reducer_name)

            if self.n_components == 2:
                var_name = dimen_reducer_name[-3:-1]
                visualizer = Visualizer(X_test, self.y_test, f'{var_name}1', f'{var_name}2', self.y_label,
                                        self.clss_labels)
                for classifier_name, classifier in classification_models.classifiers.items():
                    visualizer.visualize_results_2D(classifier, dimen_reducer_name + ' + ' + classifier_name)
                visualizer.show_results()
            elif self.n_components == 3:
                pass
                # TODO: complete

    def pca(self):
        """
        Principal Component Analysis (PCA) - unsupervised linear transformation technique
        https://setosa.io/ev/principal-component-analysis/

        The projection is done while retaining most of the information.
        Finds the principal component axes that maximize the variance (the directions of maximum variance).
        Identifies the combination of attributes (principal components / directions of greatest variance in the
            dataset's feature space) that account for the most variance in the data.
        Detects correlation between variables. strong correlation --> dimensionality reduction.
        Weakness: PCA is highly affected by outliers in the data.

        Algorithm steps:
        1. Standardize the data.
        2. Obtain the Eigenvectors and Eigenvalues from the covariance matrix or correlation matrix,
            or perform Singular Vector Decomposition.
        3. Sort eigenvalues in descending order and choose the k eigenvectors that correspond to the k largest eigenvalues
            where k is the number of dimensions of the new feature subspace (k<=d).
        4. Construct the projection matrix W from the selected k eigenvectors.
        5. Transform the original dataset X via W to obtain a k-dimensional feature subspace Y.
        """
        model_name = 'PCA'
        model = PCA(n_components=self.n_components)
        X_train_trans = model.fit_transform(self.X_train)
        X_test_trans = model.transform(self.X_test)

        if self.n_components < 1:  # n_components=0.99, whiten=True
            self.print_num_of_features(X_train_trans, model_name)
        self.print_components_variance_percentage(model, model_name)

        self.X_trans[model_name] = (X_train_trans, X_test_trans)

    def k_pca(self, kernel='rbf'):
        """
        Kernel PCA (K-PCA) - unsupervised nonlinear transformation technique
        """
        model_name = 'K-PCA'
        model = KernelPCA(n_components=self.n_components, kernel=kernel)
        X_train_trans = model.fit_transform(self.X_train)
        X_test_trans = model.transform(self.X_test)

        if self.n_components < 1:
            self.print_num_of_features(X_train_trans, model_name)
        self.print_components_variance_percentage(model, model_name)

        self.X_trans[model_name] = (X_train_trans, X_test_trans)

    def lda(self):
        """
        Linear Discriminant Analysis (LDA) - supervised linear transformation technique
        used as a data pre-processing step for classification tasks.

        The projection is done while retaining most of the information + maintaining the class-discriminatory information.
        Finds the principal component axes that maximize the class-separation.
        Identifies attributes that account for the most variance between classes.

        Algorithm steps:
        1. Compute the d-dimensional mean vectors for the different classes from the dataset.
        2. Compute the scatter matrices (in-between-class and within-class scatter matrix).
        3. Compute the eigenvectors (e_1,e_2,...,e_d) and corresponding eigenvalues (λ_1,λ_2,...,λ_d)
            for the scatter matrices.
        4. Sort the eigenvectors by decreasing eigenvalues and choose k eigenvectors with the largest eigenvalues
            to form a d×k dimensional matrix WW (where every column represents an eigenvector).
        5. Use this d×k eigenvector matrix to transform the samples onto the new subspace.
            This can be summarized by the matrix multiplication: Y=X×W
            (where X is a n×d-dimensional matrix representing the n samples,
            and y are the transformed n×k-dimensional samples in the new subspace).
        """
        model_name = 'LDA'

        if self.n_components < 1:
            n_components = self.get_n_components_by_variance_retention(
                fitted_model=LDA().fit(self.X_train, self.y_train),
                goal_variance=self.n_components)
        else:
            n_components = self.n_components
        model = LDA(n_components=n_components)
        X_train_trans = model.fit_transform(self.X_train, self.y_train)
        X_test_trans = model.transform(self.X_test)

        if self.n_components < 1:
            self.print_num_of_features(X_train_trans, model_name)
        self.print_components_variance_percentage(model, model_name)

        self.X_trans[model_name] = (X_train_trans, X_test_trans)

    def qda(self):
        """
        Quadratic Discriminant Analysis (QDA)
        """
        pass

    def gda(self):
        """
        Gaussian Discriminant Analysis (GDA)
        """
        pass

    def f_ica(self):
        """
        Fast Independent Component Analysis (F-ICA) - unsupervised
        """
        if self.n_components < 1:
            return

        model_name = 'F-ICA'
        model = FastICA(n_components=self.n_components)
        X_train_trans = model.fit_transform(self.X_train)
        X_test_trans = model.transform(self.X_test)

        # A_ = model.mixing_  # Get estimated mixing matrix

        if self.n_components < 1:
            self.print_num_of_features(X_train_trans, model_name)
        self.print_components_variance_percentage(model, model_name)

        self.X_trans[model_name] = (X_train_trans, X_test_trans)

    def svd(self):
        """
        Singular Value Decomposition (SVD)

        _, explained_variance_ratio, V = SVD(X, full_matrices=False)
        """
        pass

    def som(self):
        """
        Self-Organizing Maps (SOM)
        """
        pass

    def all(self, kernel='rbf'):
        self.pca()
        self.lda()
        self.k_pca(kernel)
        self.f_ica()


if __name__ == '__main__':
    # df = pd.read_csv('../../datasets/per_field/usl/dimensionality_reduction/Wine.csv')
    # X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
    # y_label = df.columns.values[-1]
    # clss_labels = ['S1', 'S2', 'S3']

    dataset = datasets.load_iris()
    X, y = dataset.data, dataset.target
    y_label = 'Iris type'
    clss_labels = dataset.target_names

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # # Feature Scaling  # not recommended, negatively impacts PCA & K-PCA
    # sc = StandardScaler()
    # X_train_sc = sc.fit_transform(X_train.astype(float))
    # X_test_sc = sc.transform(X_test.astype(float))

    dimen_reducer = DimensionReducer(X_train, y_train, X_test, y_test, y_label, clss_labels,
                                     n_components=2)
    dimen_reducer.all()
    dimen_reducer.visualize()
    # dimen_reducer.classify_and_visualize()
