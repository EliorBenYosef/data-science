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

whiten - relevant in PCA, F-ICA
    When True the components_ vectors are multiplied by the square root of n_samples and then divided by the
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
                                                Linear              Non-Linear
    - Principal Component Analysis (PCA)        PCA                 Kernel PCA (K-PCA)
    - Discriminant Analysis (DA)                Linear DA (LDA)     Quadratic DA (QDA), Gaussian DA (GDA)
    - Independent Component Analysis (ICA)      Fast ICA (F-ICA)
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

from sklearn.decomposition import PCA, KernelPCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from scipy.linalg import svd as SVD

colors_bold = ('#0000FF', '#FF0000', '#00FF00',
               '#00FFFF', '#FF00FF', '#FFFF00')


class DimensionReducer:

    def __init__(self, X_train, y_train, X_test, y_test, y_label, clss_labels):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.y_label = y_label
        self.clss_labels = clss_labels

        print('Original number of features:', X_train.shape[1])

        self.X_trans = {}

    @staticmethod
    def get_n_components_by_variance_retention(fitted_model, goal_variance: float):
        total_variance = 0.0
        n_components = 0
        for explained_variance in fitted_model.explained_variance_ratio_:
            total_variance += explained_variance
            n_components += 1
            if total_variance >= goal_variance:
                break
        return n_components

    @staticmethod
    def plot_scree(fitted_model, model_name, show=False):
        """
        Scree plot - a line plot of the eigenvalues of factors or principal components in an analysis.
        The scree plot is used to determine / choose the number of:
            - Factors to retain, in an Exploratory Factor Analysis (EFA)
            - Principal Components to keep, in a Principal Component Analysis (PCA)
        This is done either via:
            - the desired Variance Retention value (80% is min?) (determined via the accumulated explained variance)
            - the Elbow Method (determined via the individual explained variance)
        """
        explained_variances = fitted_model.explained_variance_ratio_

        plt.figure(figsize=(12, 8))

        # plot bars of individual explained variance:
        ax = pd.Series(explained_variances).plot(kind='bar', alpha=0.7)
        for x, p in enumerate(ax.patches):
            y = p.get_height()
            ax.annotate(f'{y:.5f}', (x, y),
                        textcoords="offset points",  # how to position the text
                        xytext=(0, 10),  # distance from text to points (x,y)
                        ha='center')  # horizontal alignment can be left, right or center

        # Calculate the amount of variance explained added by
        acc_exp_var = []  # accumulated explained variance, for each additional component
        exp_var_total = 0
        for explained_variance in fitted_model.explained_variance_ratio_:
            exp_var_total += explained_variance
            acc_exp_var.append(exp_var_total)

        # plot line of accumulated explained variance:
        pd.Series(acc_exp_var).plot(marker='o', alpha=0.7)
        for x, y in enumerate(acc_exp_var):
            if x != 0:
                plt.annotate(f'{y:.5f}', (x, y),
                             textcoords="offset points",  # how to position the text
                             xytext=(0, 10),  # distance from text to points (x,y)
                             ha='center')  # horizontal alignment can be left, right or center

        plt.axhline(y=0.9, color='k', linestyle='--')
        plt.axhline(y=0.95, color='k', linestyle='--')
        plt.axhline(y=1, color='k', linestyle='--')

        plt.xlabel('Principle Components', fontsize='x-large')
        plt.ylabel('Explained Variance %', fontsize='x-large')
        plt.title(f'Scree Plot - {model_name}', fontsize='xx-large')
        # plt.savefig(f'results/screeplot_{model_name}.png')
        if show:
            plt.show()

    def plot_scree_all(self, show=False):
        DimensionReducer.plot_scree(PCA().fit(self.X_train), 'PCA', show)
        DimensionReducer.plot_scree(LDA().fit(self.X_train, self.y_train), 'LDA', show)

    def visualize(self, show=False):
        n_clss = len(np.unique(self.y_test))

        for model_name, (n_components_title, X_train, X_test) in self.X_trans.items():

            n_components = X_train.shape[1]
            if n_components == 1 or n_components == 2:
                plt.figure()
                plt.scatter(X_test[:, 0], X_test[:, 1] if n_components == 2 else np.zeros(X_test.shape[0]),
                            c=self.y_test, cmap=ListedColormap(colors_bold[:n_clss]),
                            # alpha=.8, lw=2,
                            zorder=10, edgecolor='k', s=20)  # s - plot_symbol_size
                patches = [Patch(color=c, label=l) for c, l in zip(colors_bold[:n_clss], self.clss_labels)]
                plt.legend(handles=patches, loc='best', shadow=False, scatterpoints=1)
                var_name = model_name[-3:-1]
                plt.xlabel(f'{var_name}1')
                if n_components == 2:
                    plt.ylabel(f'{var_name}2')
                else:
                    plt.yticks([])
                plt.title(f'{model_name} - {n_components_title} '
                          + ('Variance Retention' if n_components_title < 1 else 'Components')
                          + f' - Predicted "{self.y_label}" label')
                # plt.savefig(f'results/dimenreduc_{n_components_title}_{model_name}.png')

            elif n_components == 3:
                pass  # TODO: complete 3D

        if show:
            plt.show()

    # def classify_and_visualize(self):
    #     for model_name, (n_components_title, X_train, X_test) in self.X_trans.items():
    #         classification_models = ClassificationModels(X_train, self.y_train, X_test, self.y_test)
    #         classification_models.log_reg()
    #         classification_models.print_models_performance(model_name)
    #
    #         n_components = X_train.shape[1]
    #         if n_components == 2:
    #             var_name = model_name[-3:-1]
    #             visualizer = Visualizer(X_test, self.y_test, f'{var_name}1', f'{var_name}2', self.y_label,
    #                                     self.clss_labels)
    #             for classifier_name, classifier in classification_models.classifiers.items():
    #                 visualizer.visualize_results_2D(classifier, model_name + ' + ' + classifier_name)
    #             visualizer.show_results()
    #         elif n_components == 3:
    #             pass
    #             # TODO: complete

    @staticmethod
    def print_results(fitted_model, model_name, n_components, X_train_trans):
        if n_components < 1:
            print(f'{model_name} components with {n_components} variance: {X_train_trans.shape[1]}')

        if hasattr(fitted_model, 'explained_variance_ratio_'):
            print(f'{model_name} components variance percentage:', fitted_model.explained_variance_ratio_)
            # # often useful in selecting components and estimating the dimensionality of your space:
            # explained_variance_cumulative_proportion = np.cumsum(fitted_model.explained_variance_ratio_)

    """
    LinearModels
    """

    def pca(self, n_components):
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
        model = PCA(n_components=n_components)  # n_components can be <1
        X_train_trans = model.fit_transform(self.X_train)
        X_test_trans = model.transform(self.X_test)
        self.print_results(model, model_name, n_components, X_train_trans)
        self.X_trans[model_name] = (n_components, X_train_trans, X_test_trans)

    def lda(self, n_components):
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
        model = LDA(n_components=n_components if n_components >= 1 else self.get_n_components_by_variance_retention(
                fitted_model=LDA().fit(self.X_train, self.y_train), goal_variance=n_components))
        X_train_trans = model.fit_transform(self.X_train, self.y_train)
        X_test_trans = model.transform(self.X_test)
        self.print_results(model, model_name, n_components, X_train_trans)
        self.X_trans[model_name] = (n_components, X_train_trans, X_test_trans)

    def f_ica(self, n_components):
        """
        Fast Independent Component Analysis (F-ICA) - unsupervised linear transformation technique
        probabilistic method.
        """
        model_name = 'F-ICA'
        if n_components < 1:
            return
        model = FastICA(n_components=n_components)
        X_train_trans = model.fit_transform(self.X_train)
        X_test_trans = model.transform(self.X_test)
        # A_ = model.mixing_  # Get estimated mixing matrix
        self.print_results(model, model_name, n_components, X_train_trans)
        self.X_trans[model_name] = (n_components, X_train_trans, X_test_trans)

    """
    NonLinearModels
    """

    def k_pca(self, n_components, kernel='rbf'):
        """
        Kernel PCA (K-PCA) - unsupervised nonlinear transformation technique
        """
        model_name = 'K-PCA'
        model = KernelPCA(n_components=n_components, kernel=kernel)
        X_train_trans = model.fit_transform(self.X_train)
        X_test_trans = model.transform(self.X_test)
        self.print_results(model, model_name, n_components, X_train_trans)
        self.X_trans[model_name] = (n_components, X_train_trans, X_test_trans)

    def qda(self, n_components):
        """
        Quadratic Discriminant Analysis (QDA)
        """
        pass

    def gda(self, n_components):
        """
        Gaussian Discriminant Analysis (GDA)
        """
        pass

    """
    To be completed
    """

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

    """
    AllModels
    """

    def all_linear(self, n_components):
        self.pca(n_components)
        self.lda(n_components)
        self.f_ica(n_components)

    def all_nonlinear(self, n_components, kernel='rbf'):
        self.k_pca(n_components, kernel)

    def all(self, n_components, kernel='rbf', show=False):
        """
        :param n_components: int - when used for visualization (=2 for 2D / =3 for 3D),
            float (<1) - when used for Data Compression (variance retention value: =0.99 for 99%, =0.80 for 80%, etc.)
        :param kernel: for the K-PCA
        :param show: show plots
        :return:
        """
        if n_components < 1:
            print('\n', f'~~ {n_components} variance retention ~~')
        else:
            print('\n', f'~~ {n_components} components ~~')
        self.all_linear(n_components)
        self.all_nonlinear(n_components, kernel)
        self.visualize(show)
        # self.classify_and_visualize()


#########################################

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

    dimen_reducer = DimensionReducer(X_train, y_train, X_test, y_test, y_label, clss_labels)
    dimen_reducer.plot_scree_all()
    dimen_reducer.all(n_components=0.95)  # variance retention
    dimen_reducer.all(n_components=2)  # n_components
