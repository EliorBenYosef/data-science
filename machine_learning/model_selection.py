"""
Model Selection techniques

Dealing with the bias variance tradeoff when building a ML model, and evaluating its performance.
"""

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LassoCV
import sklearn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsClassifier


###############################

"""
k-Fold Cross Validation
    k - n_folds

Used for model selection (finding the most appropriate ML model for the task)
"""


def k_fold_cross_validation(model, X, y, k=10, print_results=False):  # TODO: X_train, y_train
    """
    k-Fold Cross Validation
    """
    scores = cross_val_score(estimator=model, X=X, y=y, cv=k, n_jobs=-1)
    scores_mean, scores_std = scores.mean(), scores.std()
    if print_results:
        print(f"Scores' Mean: {scores_mean * 100:.2f} %")
        print(f"Scores' STD: {scores_std * 100:.2f} %")
    return scores_mean, scores_std


###############################

"""
Grid Search

Used for hyperparameter tuning (choosing the optimal values for the hyperparameters - 
    the parameters that are not learned)
"""


def grid_search_manual_unidimensional(X_train, y_train, X_test, y_test):
    k_range = range(1, 20)

    best_score = 0
    best_param = None

    scores = []
    knn = KNeighborsClassifier()
    for k in k_range:
        knn.n_neighbors = k
        knn.fit(X_train, y_train)
        score = knn.score(X_test, y_test)
        scores.append(score)

        if score > best_score:
            best_score = score
            best_param = k

    print(f'Best Score: {best_score}')
    print(f'Best k: {k}')

    # Plot results
    plt.figure()
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.scatter(k_range, scores)
    plt.xticks([0, 5, 10, 15, 20])
    plt.show()


def grid_search_manual_multidimensional(X_train, y_train, X_test, y_test):
    """
    A simple Grid Search implementation from scratch
    """
    C_values = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]
    gamma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
    # C_values = np.logspace(-2, 2, 9)
    # gamma_values = np.logspace(-2, 2, 9)

    best_score = 0
    best_params = {'C': None, 'gamma': None}

    svc = SVC()
    for C in C_values:
        for gamma in gamma_values:
            svc.C = C
            svc.gamma = gamma
            svc.fit(X_train, y_train)
            score = svc.score(X_test, y_test)

            if score > best_score:
                best_score = score
                best_params['C'] = C
                best_params['gamma'] = gamma

    print(f'Best Score: {best_score}')
    print(f'Best Params: {best_params}')


###############################

"""
Grid Search + k-Fold Cross Validation
"""


def grid_search_cross_validation_svm_unidimensional(svm, X, y, k=10):
    """
    An example of k-Fold Cross Validation on a Grid Search of SVM's C hyperparameter
    """
    C_values = np.logspace(-10, 0, 10)

    means, stds = [], []  # models_scores_means, models_scores_stds
    for C in C_values:
        svm.C = C
        scores_mean, scores_std = k_fold_cross_validation(svm, X, y, k)
        means.append(scores_mean)
        stds.append(scores_std)
    means, stds = np.array(means), np.array(stds)
    stes = stds / np.sqrt(k)

    plot_log(C_values, means, stes, 'C')


def grid_search_cross_validation(model, scoring, parameters, X, y, k=10):
    """
    A more general implementation, which utilizes sklearn's built-in GridSearchCV

    scoring - f1_score, precision, balanced_accuracy
    classification: accuracy, balanced_accuracy, neg_log_loss, roc_auc, average_precision,
        f1, f1_macro, f1_micro, f1_samples, f1_weighted,
        precision, precision_macro, precision_micro, precision_samples, precision_weighted,
        recall, recall_macro, recall_micro, recall_samples, recall_weighted
    regression: r2, explained_variance, neg_mean_absolute_error, neg_mean_squared_error, neg_mean_squared_log_error,
        neg_median_absolute_error
    """
    gscv = GridSearchCV(estimator=model, param_grid=parameters, scoring=scoring, cv=k, n_jobs=-1)  # , refit=False
    gscv.fit(X, y)
    print(f'Best Score ({scoring}): {gscv.best_score_ * 100:.2f} %')
    print(f'Best Parameters: {gscv.best_params_}')

    if len(parameters) == 1 and len(parameters[0]) == 1:
        means = gscv.cv_results_['mean_test_score']
        stds = gscv.cv_results_['std_test_score']
        stes = stds / np.sqrt(k)

        for param_name, param_values in parameters[0].items():
            plot_log(param_values, means, stes, param_name)


def plot_log(param_values, means, ste, param_name):
    """
    STD (Standard Deviation) vs STE (Standard Error):
    https://datascienceplus.com/standard-deviation-vs-standard-error/
    """
    plt.figure(figsize=(8, 6))

    plt.semilogx(param_values, means)
    plt.semilogx(param_values, means + ste, 'b--')  # + error line
    plt.semilogx(param_values, means - ste, 'b--')  # - error line

    # alpha=0.2 controls the translucency of the fill color
    plt.fill_between(param_values, means + ste, means - ste, alpha=0.2)

    max_score = np.max(means)
    max_param = param_values[np.argmax(means)]
    print(f'param={max_param:.5f}, score={max_score:.5f}')
    plt.scatter(max_param, max_score, color='r')
    plt.axhline(max_score, linestyle='--', color='r')
    plt.axvline(max_param, linestyle='--', color='r')

    plt.xlabel(f'Hyperparameter {param_name} value')
    plt.ylabel('Cross Validation Score (+/- STE)')
    plt.xlim([param_values[0], param_values[-1]])

    # locs, _ = plt.yticks()
    # plt.yticks(locs, list(map(lambda x: "%g" % x, locs)))

    # plt.ylim(0, 1.1)

    plt.show()


###############################

def trust_alpha(model_cv, param_name, X, y, k):
    """
    How much can you trust the selection of alpha?
    The answer is: not very much (see how we obtain different alphas for different subsets of the data,
        and moreover, see how the scores for these alphas differ quite substantially).

    We use external cross-validation to see how much the automatically obtained alphas differ across different cross-validation folds.
    """
    print("Hyperparameters maximising the generalization score on different data subsets:")
    for k, (train, test) in enumerate(KFold(k).split(X, y)):
        model_cv.fit(X[train], y[train])
        score = model_cv.score(X[test], y[test])
        print(f'[fold {k}] {param_name}: {model_cv.alpha_:.5f}, score: {score:.5f}')
    plt.show()


###############################


def example_kfcv_and_gscv_01():
    df = pd.read_csv('../datasets/per_field/sl/clss/Social_Network_Ads.csv')
    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    # sc = StandardScaler()
    # X_train, X_test = sc.fit_transform(X_train), sc.transform(X_test)

    classifier = SVC(kernel='rbf', random_state=0)

    k_fold_cross_validation(classifier, X, y, print_results=True)

    parameters = [{'C': [0.25, 0.5, 0.75, 1],
                   'kernel': ['linear']},
                  {'C': [0.25, 0.5, 0.75, 1],
                   'kernel': ['rbf', 'poly', 'sigmoid'],
                   'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
    grid_search_cross_validation(classifier, 'accuracy', parameters, X, y)


def example_kfcv_and_gscv_02():
    diabetes = datasets.load_diabetes()
    # X, y = dataset.data, dataset.target
    X, y = diabetes.data[:150], diabetes.target[:150]

    regressor = Lasso(random_state=0)

    k = 10

    k_fold_cross_validation(regressor, X, y, k, print_results=True)

    alphas = np.logspace(-4, -0.5, 30)
    parameters = [{'alpha': alphas}]
    grid_search_cross_validation(regressor, 'r2', parameters, X, y, k=k)

    regressor_cv = LassoCV(alphas=alphas, cv=k, random_state=0)
    trust_alpha(regressor_cv, 'alpha', X, y, k=3)


def example_gscv_svm():
    dataset = datasets.load_digits()
    X, y = dataset.data, dataset.target

    classifier = SVC(kernel='linear', random_state=0)

    grid_search_cross_validation_svm_unidimensional(classifier, X, y)


def example_gs_manual():
    dataset = datasets.load_iris()
    X, y = dataset.data, dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    grid_search_manual_unidimensional(X_train, y_train, X_test, y_test)
    grid_search_manual_multidimensional(X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    example_kfcv_and_gscv_01()
    example_kfcv_and_gscv_02()
    example_gscv_svm()
    example_gs_manual()
