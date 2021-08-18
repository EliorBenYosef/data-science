import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import pylab as pl
from matplotlib import cm
import pandas.plotting as pdplt
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA

import numpy as np
from matplotlib.patches import Patch


def analyze_df(df):
    print(df.shape, '\n')  # (59, 7)
    print(df.head(), '\n')


def present_data(X, y):
    print('data shape: ', X.shape)  # (elements, features)
    # print('number of classes: ', len(np.unique(self.y)))  # np.bincount(self.y)
    unique, counts = np.unique(y, return_counts=True)
    print('number of classes: ', len(unique))
    print('number of elements in each class: ', counts)


def analyze_cat_var(df, cat_var, name):
    """
    Analyzes a single categorical variable
    :param df: the entire DataFrame object
    :param cat_var: the categorical variable's column name
    """
    print(df[cat_var].unique(), '\n')

    print(df.groupby(cat_var).size(), '\n')

    # Count plot (for a single feature = pandas Series object):
    plot = sns.countplot(x=df[cat_var], label='Count')
    plot.figure.savefig(f'results/{name}_count.png')
    plt.show()


def analyze_num_vars(df, cat_var, name):
    """
    Analyzes all the numerical variables
    """
    # statistical_summary.png (of the numerical variables):
    print(df.describe(), '\n')

    # Pair plot of all the different features against each other:
    plot = sns.pairplot(df, hue=cat_var, height=2.5)
    plot.fig.savefig(f'results/{name}_pair.png')
    plt.show()

    # Box plot (for each numerical input variable):
    #   (It looks like perhaps color score has a near Gaussian distribution)
    df.plot(kind='box', subplots=True, layout=(2, 2),
            sharex=False, sharey=False, figsize=(9, 9),
            title='Box Plot for each numerical input variable')
    plt.savefig(f'results/{name}_boxplot.png')
    plt.show()

    # Histogram (for each numerical input variable):
    df.hist(bins=30, figsize=(9, 9))
    pl.suptitle('Histogram for each numerical input variable')
    plt.savefig(f'results/{name}_hist.png')
    plt.show()


def plot_scatter_matrix(x, y, name):
    """
    Scatter matrix:
    Some pairs of attributes are correlated (mass and width).
    This suggests a high correlation and a predictable relationship.
    """
    cmap = cm.get_cmap('gnuplot')
    plot = pdplt.scatter_matrix(x, c=y, marker='o', s=40,
                                hist_kwds={'bins': 15}, figsize=(9, 9), cmap=cmap)
    plt.suptitle('Scatter-matrix for each input variable')
    plt.savefig(f'results/{name}_scatter_matrix.png')
    plt.show()


def plot_2D_data(x, y, feature_names, clss_labels, name):
    """
    2D plot of the 2 features
    """
    colors_tab10 = ('#1f77b4', '#ff7f0e', '#2ca02c', '#d62727', '#9467bd', '#8c564b')
    plt.figure(1, figsize=(8, 6))
    for i, clss in enumerate(clss_labels):
        plt.scatter(*x[y == i].T, color=colors_tab10[i], label=clss, edgecolor='k')
    # plt.legend(loc=2)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.xlim(x[:, 0].min() - .5, x[:, 0].max() + .5)
    plt.ylim(x[:, 1].min() - .5, x[:, 1].max() + .5)
    plt.savefig(f'results/{name}_pair_single.png')
    plt.show()


def plot_3D_data(x, y, name):
    """
    3D plot of the first 3 PCA dimensions (to getter a better understanding of interaction of the dimensions)
    """
    fig = plt.figure(2, figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110, auto_add_to_figure=False)
    fig.add_axes(ax)
    X_reduced = PCA(n_components=3).fit_transform(x)
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,
               cmap='Set1', edgecolor='k', s=40)
    ax.set_title("First three PCA directions")
    ax.set_xlabel("1st eigenvector")
    ax.set_ylabel("2nd eigenvector")
    ax.set_zlabel("3rd eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    plt.savefig(f'results/{name}_PCA_3D.png')
    plt.show()


#########################################

def analyze_fruits():
    """
    fruit_data_with_colors dataset.
    59 elements (rows): apple 19, lemon 16, mandarin 5, orange 19.
    7 features (columns): fruit_label, fruit_name, fruit_subtype, mass, width, height, color_score.
    4 classes ('fruit_name'): apple, lemon, mandarin, orange.

    see results/statistical_summary.png:
        the numerical values do not have the same scale.
        We will need to apply scaling to the test set that we computed for the training set.

    https://towardsdatascience.com/solving-a-simple-classification-problem-with-python-fruits-lovers-edition-d20ab6b071d2
    https://github.com/susanli2016/Machine-Learning-with-Python/blob/master/Solving%20A%20Simple%20Classification%20Problem%20with%20Python.ipynb
    """
    df = pd.read_table('../datasets/per_type/txt/fruits_data.txt')
    name = 'fruits'

    analyze_df(df)
    analyze_cat_var(df, 'fruit_name', name)
    analyze_num_vars(df.drop('fruit_label', axis=1), 'fruit_name', name)

    x_df = df[['mass', 'width', 'height', 'color_score']].astype(float)
    indices = [1, 2]  # Use only two features
    x = x_df.values[:, indices]

    y = df['fruit_name'].values
    le = LabelEncoder()
    y = le.fit_transform(y)

    plot_scatter_matrix(x_df, y, name)

    feature_names = x_df.columns.values[indices[0]], x_df.columns.values[indices[1]]
    plot_2D_data(x, y, feature_names, le.classes_, name)

    plot_3D_data(x_df, y, name)

    return x_df, y


def analyze_iris():
    df = sns.load_dataset('iris')
    name = 'iris'

    # analyze_df(df)
    # analyze_cat_var(df, df.columns.values[-1], name)
    # analyze_num_vars(df, df.columns.values[-1], name)

    x_df = df[df.columns.values[:-1]].astype(float)
    indices = [0, 1]  # Use only two features
    x = x_df.values[:, indices]

    y = df[df.columns.values[-1]].values
    le = LabelEncoder()
    y = le.fit_transform(y)

    # plot_scatter_matrix(x_df, y, name)

    feature_names = x_df.columns.values[indices[0]], x_df.columns.values[indices[1]]
    plot_2D_data(x, y, feature_names, le.classes_, name)

    plot_3D_data(x_df, y, name)

    return x_df, y


if __name__ == '__main__':
    # analyze_fruits()
    analyze_iris()

    # iris = datasets.load_iris()
    # plot_2D_data(iris.data[:, :2], iris.target, iris.feature_names[:2])  # we only take the first two features.
    # plot_3D_data(iris.data, iris.target, iris.feature_names[:2])
