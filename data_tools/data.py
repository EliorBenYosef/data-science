"""
sklearn.datasets:
dataset.DESCR - returns a details dexcription of the dataset.

Data Pre-Processing
"""

import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_X_from_sklearn_dataset(dataset, indices=None):
    if indices is not None:
        X = dataset.data[:, indices]
        x_labels = np.array(dataset.feature_names)[indices]
    else:
        X = dataset.data
        x_labels = dataset.feature_names
    return X, x_labels


def get_X_from_pandas_dataset(df, indices=None):
    if indices is not None:
        X = df.values[:, indices]
        x_labels = df.columns.values[indices]
        # X = df.iloc[:, indices].values
        # x_labels = df.iloc[:, indices].columns.values
    else:
        X = df.values[:, :-1]
        x_labels = df.columns.values[:-1]
        # X = df.iloc[:, :-1].values
        # x_labels = df.iloc[:, :-1].columns.values
    return X, x_labels


class ClassificationDataSets:

    def __init__(self):
        self.X, self.y = None, None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.X_train_sc, self.X_test_sc = None, None
        self.transformers = []

        self.x_labels = None
        self.y_label = None
        self.sample_label = None
        self.clss_labels = None

    def get_iris(self, indices=None):
        """
        150 iris observations.
        4 features: Sepal Length, Sepal Width, Petal Length, Petal Width.
        3 iris types (Setosa, Versicolour, Virginica).

        # X.shape = (150, 4) = (observations, features)
        # np.unique(y) = [0,1,2] = 3 different irises types

        https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html
        """
        dataset = datasets.load_iris()
        self.X, self.x_labels = get_X_from_sklearn_dataset(dataset, indices)
        self.y = dataset.target
        self.y_label = 'Iris Type'
        self.sample_label = 'Iris'
        self.clss_labels = dataset.target_names

        # Note: how to select a subset of features (0,1) and classes (all but #2).
        # #   in order to have a dataset with: 2 dimensions (features) & 2 classes.
        # X = X[y != 2, :2]
        # y = y[y != 2]

        # Dataset Splitting
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.25, random_state=0)

        # Feature Scaling
        sc = ColumnTransformer(
            transformers=[('standardization', StandardScaler(), [i for i in range(self.X.shape[1])])],
            remainder='passthrough')
        self.X_train_sc = sc.fit_transform(self.X_train.astype(float))
        self.X_test_sc = sc.transform(self.X_test.astype(float))

        self.transformers = [sc]

        self.sc_x1, self.sc_x2 = StandardScaler(), StandardScaler()
        self.sc_x1.fit(self.X_train[:, np.newaxis, 0].astype(float))
        self.sc_x2.fit(self.X_train[:, np.newaxis, 1].astype(float))

    def get_logreg_simple(self):
        """
        bivariate_binary_classification
        """
        df = pd.read_csv('../../../datasets/per_field/sl/clss/logreg_simple.txt',
                         header=None, names=['Exam #1', 'Exam #2', 'Admittance'])

        self.X, self.y = df.iloc[:, :-1].values, df.iloc[:, -1].values
        self.x_labels, self.y_label = df.columns.values[:-1], df.columns.values[-1]
        self.sample_label = 'Student'
        self.clss_labels = np.array(['Rejected', 'Accepted'])

        # Dataset Splitting
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.25, random_state=0)

        # Feature Scaling
        sc = ColumnTransformer(
            transformers=[('standardization', StandardScaler(), [i for i in range(self.X.shape[1])])],
            remainder='passthrough')
        self.X_train_sc = sc.fit_transform(self.X_train.astype(float))
        self.X_test_sc = sc.transform(self.X_test.astype(float))

        self.transformers = [sc]

        self.sc_x1, self.sc_x2 = StandardScaler(), StandardScaler()
        self.sc_x1.fit(self.X_train[:, np.newaxis, 0].astype(float))
        self.sc_x2.fit(self.X_train[:, np.newaxis, 1].astype(float))

    def get_Social_Network_Ads(self):
        """
        bivariate_binary_classification
        """
        df = pd.read_csv('../../../datasets/per_field/sl/clss/Social_Network_Ads.csv')

        self.X, self.y = df.iloc[:, :-1].values, df.iloc[:, -1].values
        self.x_labels, self.y_label = df.columns.values[:-1], df.columns.values[-1]
        self.sample_label = 'Viewer'
        self.clss_labels = np.array(['No', 'Yes'])

        # Dataset Splitting
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.25, random_state=0)

        # Feature Scaling
        sc = ColumnTransformer(
            transformers=[('standardization', StandardScaler(), [i for i in range(self.X.shape[1])])],
            remainder='passthrough')
        self.X_train_sc = sc.fit_transform(self.X_train.astype(float))
        self.X_test_sc = sc.transform(self.X_test.astype(float))

        self.transformers = [sc]

        self.sc_x1, self.sc_x2 = StandardScaler(), StandardScaler()
        self.sc_x1.fit(self.X_train[:, np.newaxis, 0].astype(float))
        self.sc_x2.fit(self.X_train[:, np.newaxis, 1].astype(float))

    def get_churn_modelling(self):
        """
        multivariate_binary_classification (10 features)
        """
        df = pd.read_csv('../../../datasets/per_field/sl/clss/Churn_Modelling.csv')
        self.X = df.iloc[:, 3:-1].values
        self.y = df.iloc[:, -1].values

        # Encoding categorical data
        ct = ColumnTransformer(
            transformers=[('nominal', OneHotEncoder(), [1]),  # Country (france, germany, spain)
                          ('ordinal', OrdinalEncoder(), [2, 7, 8])],  # Gender (female, male), HasCrCard, IsActiveMember
            remainder='passthrough')
        self.X = ct.fit_transform(self.X)

        # Dataset Splitting
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=0)

        # Feature Scaling
        # columns: CreditScore, Age, TenureInBank, Balance, NumOfProducts, EstimatedSalary
        sc = ColumnTransformer(
            transformers=[('standardization', StandardScaler(), [i for i in range(6, self.X.shape[1])])],  # [6, 7, 8, 9, 10, 11]
            remainder='passthrough')
        self.X_train_sc = sc.fit_transform(self.X_train.astype(float))
        self.X_test_sc = sc.transform(self.X_test.astype(float))

        self.transformers = [ct, sc]

    def get_Breast_Cancer_Wisconsin(self):
        """
        multivariate_binary_classification (9 features)
        """
        df = pd.read_csv('../../../datasets/per_field/sl/clss/Breast_Cancer_Wisconsin.csv')
        self.X, self.y = df.iloc[:, 1:-1].values, df.iloc[:, -1].values  # Y - Benign (2) \ Malignant (4)

        # Dataset Splitting
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.25, random_state=0)

        # Feature Scaling
        sc = ColumnTransformer(
            transformers=[('standardization', StandardScaler(), [i for i in range(self.X.shape[1])])],
            remainder='passthrough')
        self.X_train_sc = sc.fit_transform(self.X_train.astype(float))
        self.X_test_sc = sc.transform(self.X_test.astype(float))

        self.transformers = [sc]

    @staticmethod
    def get_digits(show_as_image=True):
        """
        1797 hand-written digits images.
        8x8 image (dataset.images) / 64 pixels array (dataset.data).
        10 digit types (0-9).

        https://scikit-learn.org/stable/auto_examples/datasets/plot_digits_last_image.html
        plt.figure(1, figsize=(3, 3))
        plt.imshow(dataset.images[-1], cmap=plt.cm.gray_r, interpolation='nearest')
        plt.show()
        """
        dataset = datasets.load_digits()
        if show_as_image:
            X = dataset.images  # data.shape = (1797, 8, 8) = (images, image height, image width)
        else:
            X = dataset.data  # X.shape = (1797, 64) = (images, pixels array)
        y = dataset.target  # np.unique(y) = [0,1,2,3,4,5,6,7,8,9] = 10 different digits types
        return X, y

    @staticmethod
    def get_fruits():
        """
        fruit_data_with_colors dataset.
        59 elements (rows): apple 19, lemon 16, mandarin 5, orange 19.
        7 features (columns): fruit_label, fruit_name, fruit_subtype, mass, width, height, color_score.
        4 classes ('fruit_name'): apple, lemon, mandarin, orange.
        """
        df = pd.read_table('../../../datasets/per_type/txt/fruits_data.txt')
        X = df[['mass', 'width', 'height', 'color_score']].astype(float)
        y = df['fruit_label']
        return X, y


class RegressionDataSets:

    def __init__(self):
        self.X, self.y = None, None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.X_train_sc, self.X_test_sc = None, None
        self.transformers = []

        self.x_labels = None
        self.y_label = None
        self.sample_label = None
        self.clss_labels = None

    def get_diabetes(self, indices=None):
        """
        442 patients measurements.
        10 features: physiological variables (age, sex, weight, blood pressure).
        An indication of disease progression after one year.

        x.shape = (442, 10)  = (patients, physiological variables)
        np.unique(y) = [0,1,2,3,4,5,6,7,8,9] = an indication of disease progression after one year
        """
        dataset = datasets.load_diabetes()
        self.X, self.x_labels = get_X_from_sklearn_dataset(dataset, indices)
        self.y = dataset.target
        self.y_label = 'Diabetes progression after 1Y'
        self.sample_label = 'Patient'

        # Dataset Splitting
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=0)

    def get_Salary_Data(self):
        """
        univariate
        """
        df = pd.read_csv('../../../datasets/per_field/sl/reg/Salary_Data.csv')
        self.X, self.y = df.iloc[:, :-1].values, df.iloc[:, -1].values
        self.x_labels, self.y_label = df.columns.values[:-1], df.columns.values[-1]
        # self.sample_label =

        # Dataset Splitting
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=0)

    def get_linreg_simple(self):
        """
        univariate
        """
        df = pd.read_csv('../../../datasets/per_field/sl/reg/linreg_simple.txt',
                         header=None, names=['Population', 'Profit'])
        self.X, self.y = df.iloc[:, :-1].values, df.iloc[:, -1].values
        self.x_labels, self.y_label = df.columns.values[:-1], df.columns.values[-1]
        # self.sample_label =

        # Dataset Splitting
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=0)

    def get_Position_Salaries(self):
        """
        univariate
        """
        df = pd.read_csv('../../../datasets/per_field/sl/reg/Position_Salaries.csv')
        self.X, self.y = df.iloc[:, 1].values[:, np.newaxis, ], df.iloc[:, -1].values
        self.x_labels, self.y_label = [df.columns.values[1]], df.columns.values[-1]
        # self.sample_label =

        # Dataset Splitting
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=0)

    def get_Combined_Cycle_Power_Plant(self, indices=None):
        """
        multivariate (4 features)
        """
        df = pd.read_csv('../../../datasets/per_field/sl/reg/Combined_Cycle_Power_Plant.csv')
        # df = pd.read_excel('../../../datasets/per_field/sl/reg/Combined_Cycle_Power_Plant.xlsx')
        self.X, self.x_labels = get_X_from_pandas_dataset(df, indices)
        self.y = df.iloc[:, -1].values
        self.y_label = df.columns.values[-1]
        # self.sample_label =

        # Dataset Splitting
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=0)

    def get_Startups(self, indices=None):
        """
        multivariate (4 features)
        """
        df = pd.read_csv('../../../datasets/per_field/sl/reg/Startups.csv')
        self.X, self.x_labels = get_X_from_pandas_dataset(df, indices)
        self.y = df.iloc[:, -1].values
        self.y_label = df.columns.values[-1]
        # self.sample_label =

        # Encoding categorical data:
        ct = ColumnTransformer(
            transformers=[('encoder', OneHotEncoder(), [3])],
            remainder='passthrough')
        self.X = ct.fit_transform(self.X)

        # Dataset Splitting
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=0)


class ClusteringDataSets:

    def __init__(self):
        self.X = None

        self.x_labels = None
        self.y_label = None
        self.sample_label = None

    def get_Mall_Customers(self, indices=None):
        df = pd.read_csv('../../../datasets/per_field/usl/clustering/Mall_Customers.csv')
        self.x_labels = df.columns.values[indices] if indices is not None else df.columns.values[1:-1]
        self.y_label = 'Customer Group'
        self.sample_label = 'Customer'

        # Encoding categorical data & Feature Scaling
        ct = ColumnTransformer(
            transformers=[('nominal', OneHotEncoder(), ['Sex']),
                          ('standardization', StandardScaler(), ['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])],
            remainder='passthrough')
        self.X = ct.fit_transform(df.iloc[:, 1:])
        if indices is not None:
            self.X = self.X[:, indices]
