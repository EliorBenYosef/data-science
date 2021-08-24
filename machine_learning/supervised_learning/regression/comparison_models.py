import numpy as np
from data_tools.data import RegressionDataSets
from models_regression import RegressionModels
from utils import Visualizer


def univariate_models(dataset, is_linear=True, is_nonlinear=True):
    # for performing HighRes Non-Linear Regression -> smoother curve:
    X_range = np.arange(min(dataset.X), max(dataset.X) + 0.1, 0.1)[:, np.newaxis]

    regression_models = RegressionModels(dataset.X_train, dataset.y_train, dataset.X_test, dataset.y_test, X_range)
    if is_linear:
        regression_models.all_linear()
    if is_nonlinear:
        regression_models.all_nonlinear()

    visualizer = Visualizer(dataset.X_test, dataset.y_test, dataset.x_labels[0], None, dataset.y_label)
    if is_linear:
        visualizer.visualize_results_2D(regression_models.predictions)
    if is_nonlinear:
        visualizer.visualize_results_2D(regression_models.predictions_fine, X_range)


def bivariate_models(dataset):
    regression_models = RegressionModels(dataset.X_train, dataset.y_train, dataset.X_test, dataset.y_test)
    regression_models.all_linear()

    visualizer = Visualizer(dataset.X_test, dataset.y_test, *dataset.x_labels, dataset.y_label)
    visualizer.visualize_results_3D(regression_models.predictions, 1)

    # TODO: finalize non linear models:


def multivariate_models(dataset):
    regression_models = RegressionModels(dataset.X_train, dataset.y_train, dataset.X_test, dataset.y_test)
    regression_models.all()
    regression_models.print_models_r2()
    regression_models.print_results()


##################################

def example_univariate_models():
    dataset = RegressionDataSets()

    # Linear models
    dataset.get_Salary_Data()
    univariate_models(dataset, is_nonlinear=False)
    dataset.get_linreg_simple()
    univariate_models(dataset, is_nonlinear=False)
    dataset.get_diabetes(indices=[2])  # Use only one feature
    univariate_models(dataset, is_nonlinear=False)

    # Nonlinear models
    dataset.get_Position_Salaries()
    univariate_models(dataset, is_linear=False)
    dataset.get_Combined_Cycle_Power_Plant(indices=[1])  # use only one feature
    univariate_models(dataset, is_linear=False)


def example_bivariate_models():
    dataset = RegressionDataSets()

    dataset.get_diabetes(indices=[0, 1])  # Use only two features
    bivariate_models(dataset)


def example_multivariate_models():
    dataset = RegressionDataSets()

    dataset.get_Combined_Cycle_Power_Plant()
    multivariate_models(dataset)
    dataset.get_Startups()
    multivariate_models(dataset)


if __name__ == '__main__':
    example_univariate_models()
    example_bivariate_models()
    example_multivariate_models()
