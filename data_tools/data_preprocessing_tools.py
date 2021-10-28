"""
Data Pre-Processing
"""

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


def handle_missing_data(df, num_v_i, cat_v_i):
    """
    Note that the index of dependent variable (y) is -1 (the last column in the dataset).
    :param df: the entire DataFrame object
    :param num_v_i: the indices of the numerical dependant variables
    :param cat_v_i: the indices of the categorical dependant variables
    """
    rows_with_missing_data = df.shape[0] - df.dropna().shape[0]
    if not rows_with_missing_data:
        X = df.iloc[:, :-1].values  # features (independent variables) matrix
        y = df.iloc[:, -1].values  # dependant variable vector
    else:
        # take the rows where the dependant variable is not NA (remove missing dependant variable values):
        df = df[df.iloc[:, -1].notna()]

        rows_with_missing_data = df.shape[0] - df.dropna().shape[0]
        if rows_with_missing_data / df.shape[0] <= 0.01:
            # Option 1: removing data points with missing variable\s
            #   can be done if the percentage of removed data points is not that large (<1%)
            df = df.dropna()
            X = df.iloc[:, :-1].values  # features (independent variables) matrix
            y = df.iloc[:, -1].values  # dependant variable vector
        else:
            X = df.iloc[:, :-1].values  # features (independent variables) matrix
            y = df.iloc[:, -1].values  # dependant variable vector

            # Option 2: replacing missing (independent) variables with the column's:
            #   numerical variables - mean \ median \ most_frequent value \ a constant value
            #   another method:
            #       df.num_var = df.num_var.fillna(0)
            #       df.cat_var = df.cat_var.fillna('Other')
            #       df.date_time_var = df.date_time_var.fillna('N/A')  # N/A mean DateTime not available
            if num_v_i:
                num_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
                num_imputer.fit(X[:, num_v_i])
                X[:, num_v_i] = num_imputer.transform(X[:, num_v_i])
            #   categorical variables - most_frequent value \ a constant value
            if cat_v_i:
                cat_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
                cat_imputer.fit(X[:, cat_v_i])
                X[:, cat_v_i] = cat_imputer.transform(X[:, cat_v_i])
    return df, X, y


def encode_categorical_data(df, X, y, v_i, num_v_i, cat_v_i):
    """
    Encoding Categorical (non-numerical) Data into dummy variables / indicator variables.
    can be easily done with:
        df = pd.get_dummies(df)  # Convert categorical variable into dummy/indicator variables

    Encoding of Independent Variables:
        OrdinalEncoder - Index (scalar) Encoding:
            when the order matters / binary categories encoding (0\1)
            equivalent to Label Encoding, but accepts 2D instead of 1D, so it can be used on input features (X).
        OneHotEncoder - One-Hot (vector) Encoding:
            when there isn't a numerical order between the categories and there are more than 2 categories.
            will replace each categorical column with n numerical columns (where n is the number of categories),
                AKA "dummy variables", that will be placed at the BEGINNING (the first columns) of the data..
            better than labeling as: 0,1,2... so there's not a numerical order between the categories
            remainder='passthrough' - enables keeping the non-transformed columns

    Encoding the Dependent Variable (can be applied to a single column only - the target column!):
        LabelEncoder - Index (scalar) Encoding

    note that the specific columns ([0], [1, 2]) are hard-coded here.

    :param df: the entire DataFrame object
    :param v_i: the types of all the variables
    :param num_v_i: the indices of the numerical dependant variables
    :param cat_v_i: the indices of the categorical dependant variables
    """
    if cat_v_i:
        ct = ColumnTransformer(
            transformers=[('nominal', OneHotEncoder(), [0]),
                          ('ordinal', OrdinalEncoder(), [1, 2])],
            remainder='passthrough')
        X = ct.fit_transform(X)

        # updating the indices of the numerical dependant variables:
        num_v_i_added = num_v_i.copy()
        for i in [0]:  # considering only the one hot encoding columns
            dummy_vars = len(list(pd.unique(df[df.columns.values[i]])))
            for j, _ in enumerate(num_v_i):
                if i < num_v_i[j]:
                    num_v_i_added[j] += dummy_vars - 1
                else:
                    num_v_i_added[j] += dummy_vars
        num_v_i = num_v_i_added

    if v_i[-1] == np.object:
        le = LabelEncoder()
        y = le.fit_transform(y)
        # clss = le.classes_

    return X, y, num_v_i


def split_data_using_np(X, y, test_ratio=.9):
    """
    Split data to train & test sets without scaling.
    """
    np.random.seed(0)
    n_samples = len(X)
    indices = np.random.permutation(n_samples)  # A random permutation, to split the data randomly
    # X_train = X[indices[:-size]]
    # y_train = y[indices[:-size]]
    # X_test = X[indices[-size:]]
    # y_test = y[indices[-size:]]
    X_train = X[indices[:int(test_ratio * n_samples)]]
    y_train = y[indices[:int(test_ratio * n_samples)]]
    X_test = X[indices[int(test_ratio * n_samples):]]
    y_test = y[indices[int(test_ratio * n_samples):]]

    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test, y_train, y_test, num_v_i, v_i=None):
    """
    Feature Scaling
        Normalization (Normalizer) - scales a feature's values to [0,1]
        Standardization (StandardScaler) - scales a feature's values to [-3,3]

    Required for non-linear models
    Not required for linear classification models,
        but will improve the training performance and therefore the final predictions

    not applied to dummy variables (i.e. one-hot encoding columns)
    applied only for SOME of the ML models
        Not: LinReg, Decision Tree, Random Forest.
        Yes: SVR
    applied separately to the train & test sets to prevent information leakage,
        since the test set shouldn't affect the training.
    :param v_i: the types of all the variables
    :param num_v_i: the indices of the numerical dependant variables
    """
    sc_X = StandardScaler()
    X_train, X_test = X_train.astype(float), X_test.astype(float)  # SC requires a 2D float array
    X_train[:, num_v_i] = sc_X.fit_transform(X_train[:, num_v_i])  # fits the SC to the training set's mean & std
    X_test[:, num_v_i] = sc_X.transform(X_test[:, num_v_i])  # transform the test set with the same mean & std

    # scaler = MinMaxScaler()
    # self.x_train = scaler.fit_transform(self.x_train)
    # self.x_test = scaler.transform(self.x_test)

    if v_i is not None and v_i[-1] != np.object:
        sc_y = StandardScaler()
        y_train, y_test = y_train.astype(float)[:, np.newaxis], y_test.astype(float)[:, np.newaxis]
        y_train = sc_y.fit_transform(y_train)
        y_test = sc_y.transform(y_test)

    return X_train, X_test, y_train, y_test


def preprocess_dataframe(df, test_size=0.2, perform_feature_scaling=False):
    # Getting the indices of the numerical & categorical independent values
    v_i = [v_type for i, v_type in df.dtypes.items()]
    num_v_i = []  # numerical_variables
    cat_v_i = []  # categorical_variables
    for i, v_type in enumerate(v_i[:-1]):
        cat_v_i.append(i) if v_type == np.object else num_v_i.append(i)

    df, X, y = handle_missing_data(df, num_v_i, cat_v_i)

    X, y, num_v_i = encode_categorical_data(df, X, y, v_i, num_v_i, cat_v_i)

    # Dataset Splitting (into Training & Test sets)
    #   random_state - the RNG's (random number generator's) seed, so we'll get the same split on different runs
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

    if perform_feature_scaling:
        X_train, X_test, y_train, y_test = scale_features(X_train, X_test, y_train, y_test, num_v_i, v_i)

    return X_train, X_test, y_train, y_test


def preprocess_csv_data(path, test_size=0.2, perform_feature_scaling=False):
    # Importing the Dataset
    #   missing values are saved as nan
    return preprocess_dataframe(pd.read_csv(path), test_size, perform_feature_scaling)


def sklearn_to_df(sklearn_dataset):
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df['target'] = pd.Series(sklearn_dataset.target)
    return df


if __name__ == '__main__':
    preprocess_csv_data(path='../datasets/per_field/sl/clss/Customers_Data_1.csv', perform_feature_scaling=True)
