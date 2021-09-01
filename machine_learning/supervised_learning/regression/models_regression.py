import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoCV
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error


class RegressionModels:

    def __init__(self, X_train, y_train, X_test, y_test, X_range=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_range = X_range

        # self.regressors = {}
        self.predictions = {}
        self.predictions_fine = {}
        self.performances = {}
        # self.x2 = {}

    def evaluate_model_performance(self, y_true, y_pred, model_name):
        """
        Model Performance Evaluation
        """
        # train_score = regressor.score(self.X_train, self.y_train)
        r2 = r2_score(y_true, y_pred)
        explained_variance = explained_variance_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)

        self.performances[model_name] = [r2, explained_variance, mse]

    def print_models_r2(self):
        for model_name, performance in self.performances.items():
            print(f'{model_name} R^2 score: {performance[0]:.5f}')

    def print_models_explained_variance(self):
        for model_name, performance in self.performances.items():
            print(f'{model_name} Explained Variance score: {performance[1]:.2f}')

    def print_models_mse(self):
        for model_name, performance in self.performances.items():
            print(f'{model_name} MSE: {performance[2]:.2f}')

    def print_models_performance(self):
        for model_name, performance in self.performances.items():
            print(f'{model_name} R^2 score: {performance[0]:.5f}')
            print(f'{model_name} Explained Variance score: {performance[1]:.2f}')
            print(f'{model_name} MSE: {performance[2]:.2f}')

    def print_results(self):
        np.set_printoptions(precision=2)
        print('target,', ', '.join([model_name for model_name in self.predictions.keys()]), )
        print(np.concatenate((
            self.y_test[:, np.newaxis],
            np.array([y_pred for y_pred in self.predictions.values()]).T
        ), axis=1), '\n')

    """
    LinearModels
    """

    def lin_reg(self, print_formula=False):
        """
        (Simple / Multiple) Linear Regression (LinReg)
        utilizes Ordinary Least Squares (OLS), and attempts to draw a straight line that will best minimize
        the residual sum of squares (RSS) between the prediction (y_pred) in the ground truth (y).

        OLS Variance:
        Due to the straight line that linear regression uses to follow the data points as well as it can,
        if there are few points in each dimension, noise on the observations will cause great variance.
        Every line's slope can vary quite a bit for each prediction due to the noise induced in the observations.
        """
        model_name = 'LinReg'

        regressor = LinearRegression()
        regressor.fit(self.X_train, self.y_train)

        if print_formula:
            if regressor.coef_.size == 1:
                print(f'y = {regressor.coef_[0]:.2f}x + {regressor.intercept_:.2f}')
            else:
                print('Coefficients:', regressor.coef_)
                print('Intercept:', regressor.intercept_)

        y_pred = regressor.predict(self.X_test)
        self.predictions[model_name] = y_pred

        self.evaluate_model_performance(self.y_test, y_pred, model_name)

    def ridge_reg(self):
        """
        Ridge Regression (RidgeReg) models
        Ridge regression is basically minimizing a penalised version of the least-squared function.
        The penalising `shrinks` the value of the regression coefficients.

        RidgeReg Variance:
        Despite having few data points in each dimension, the slope of the prediction is much more stable
        and the variance in the line itself is greatly reduced (compared to that of the standard LinReg).
        """
        model_name = 'RidgeReg'

        regressor = Ridge(alpha=.1)
        regressor.fit(self.X_train, self.y_train)

        y_pred = regressor.predict(self.X_test)
        self.predictions[model_name] = y_pred

        self.evaluate_model_performance(self.y_test, y_pred, model_name)

    def lasso_reg(self):
        """
        Lasso Regression

        Note that LassoCV sets its alpha parameter automatically from the data by internal cross-validation
            (it performs cross-validation on the training data it receives).
        """
        pass
        # TODO: complete

    def lin_svr(self, print_formula=True):
        """
        Linear Support Vector Regression (SVR linear)
            similar to SVR with kernel='linear'

        Linear SVMs use Cross-Entropy loss:
            Binary cross-entropy / log loss --> logistic classifier
            Categorical cross-entropy / softmax loss --> softmax classifier

        very similar to linear regression in its learning approach,
            but the cost and gradient functions are formulated differently.
        it uses:
            continuous output (as in LinReg) + sigmoid (or “logit”) activation function
        """
        model_name = 'SVR linear'

        # Feature Scaling
        sc_X, sc_y = StandardScaler(), StandardScaler()
        X_train_sc = sc_X.fit_transform(self.X_train.astype(float))
        y_train_sc = sc_y.fit_transform(self.y_train.astype(float)[:, np.newaxis])

        regressor = LinearSVR()
        regressor.fit(X_train_sc, y_train_sc.ravel())

        if print_formula:
            print(f'y = {regressor.coef_[0]:.5f}x + {regressor.intercept_[0]:.6f}')

        y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(self.X_test.astype(float))))
        self.predictions[model_name] = y_pred

        self.evaluate_model_performance(self.y_test, y_pred, model_name)

    """
    NonLinearModels
    """

    def poly_reg(self, pol_deg=2):
        """
        Polynomial Regression (PolyReg)
            a special case of Multiple Linear Regression,
            but instead of having different features (x1, x2, ..., xn),
            we have a polynomial (powered) single feature (x1, x1^2, x1^3, ..., x1^n), where n is the max power.
        """
        model_name = f'PolyReg ({pol_deg})'

        poly_features = PolynomialFeatures(degree=pol_deg)
        X_poly = poly_features.fit_transform(
            self.X_train)  # creates a feature matrix which is composed of the powered feature

        regressor = LinearRegression()
        regressor.fit(X_poly, self.y_train)

        y_pred = regressor.predict(poly_features.transform(self.X_test))
        self.predictions[model_name] = y_pred

        if self.X_range is not None:
            y_pred_fine = regressor.predict(poly_features.transform(self.X_range))
            self.predictions_fine[model_name] = y_pred_fine

        self.evaluate_model_performance(self.y_test, y_pred, model_name)

    def kernel_svr(self, kernel='rbf', pol_deg=3):
        """
        Support Vector Regression (SVR) with nonlinear kernels (polynomial, gaussian, gaussian rbf, laplace rbf,
        hyperbolic tangent, sigmoid, ...)

        here, there's an implicit equation of the dependent variable with respect to the features
        (as opposed to Linear Regression, in which there's an explicit equation).
        here, there aren't coefficients multiplying each of the features so applying feature scaling is needed.

        Note: outliers are not caught well by the SVR model.

        https://data-flair.training/blogs/svm-kernel-functions/
        """
        model_name = f'SVR {kernel}'
        if kernel == 'poly':
            model_name += f' ({pol_deg})'

        # Feature Scaling
        sc_X, sc_y = StandardScaler(), StandardScaler()
        X_train_sc = sc_X.fit_transform(self.X_train.astype(float))
        y_train_sc = sc_y.fit_transform(self.y_train.astype(float)[:, np.newaxis])

        regressor = SVR(kernel=kernel, degree=pol_deg)
        regressor.fit(X_train_sc, y_train_sc.ravel())  # y should be 1D here

        y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(self.X_test.astype(float))))
        self.predictions[model_name] = y_pred

        if self.X_range is not None:
            y_pred_fine = sc_y.inverse_transform(regressor.predict(sc_X.transform(self.X_range)))
            self.predictions_fine[model_name] = y_pred_fine

        self.evaluate_model_performance(self.y_test, y_pred, model_name)

    def dtr(self):
        """
        Decision Tree Regression (DTR), AKA Regression Tree.
        Non-continuous model
        this model isn't the best model to use on a single-feature dataset, it's more adapted to high-dimensional
        datasets (with many features).
        """
        model_name = 'Decision Tree'

        regressor = DecisionTreeRegressor(random_state=0)
        regressor.fit(self.X_train, self.y_train)

        y_pred = regressor.predict(self.X_test)
        self.predictions[model_name] = y_pred

        if self.X_range is not None:
            y_pred_fine = regressor.predict(self.X_range)
            self.predictions_fine[model_name] = y_pred_fine

        self.evaluate_model_performance(self.y_test, y_pred, model_name)

    def rfr(self, n_trees=10):
        """
        Random Forest Regression (RFR)
        criterion - the function to measure the quality of the splits

        :param n_trees: n_estimators - number of trees. it's usually recommended to start with 10 trees.
        """
        model_name = f'Random Forest ({n_trees})'

        regressor = RandomForestRegressor(n_estimators=n_trees, random_state=0)
        regressor.fit(self.X_train, self.y_train)

        y_pred = regressor.predict(self.X_test)
        self.predictions[model_name] = y_pred

        if self.X_range is not None:
            y_pred_fine = regressor.predict(self.X_range)
            self.predictions_fine[model_name] = y_pred_fine

        self.evaluate_model_performance(self.y_test, y_pred, model_name)

    def xgb(self):
        """
        XG Boost (XGB).
        https://towardsdatascience.com/catboost-vs-light-gbm-vs-xgboost-5f93620723db
        """
        model_name = 'XG Boost'

        regressor = XGBRegressor()
        regressor.fit(self.X_train, self.y_train)

        y_pred = regressor.predict(self.X_test)
        self.predictions[model_name] = y_pred

        if self.X_range is not None:
            y_pred_fine = regressor.predict(self.X_range)
            self.predictions_fine[model_name] = y_pred_fine

        self.evaluate_model_performance(self.y_test, y_pred, model_name)

    def lgbm(self):
        """
        Light GBM (Light GBM)
        a gradient-based model that uses tree-based learning algorithms.
        https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html
        https://towardsdatascience.com/catboost-vs-light-gbm-vs-xgboost-5f93620723db
        """
        model_name = 'Light GBM'

        classifier = LGBMRegressor()
        classifier.fit(self.X_train, self.y_train)

        y_pred = classifier.predict(self.X_test)

        self.evaluate_model_performance(self.y_test, y_pred, model_name)

    def cb(self):
        """
        Cat Boost (CatB)
        https://catboost.ai/
        https://towardsdatascience.com/catboost-vs-light-gbm-vs-xgboost-5f93620723db
        A gradient-based model that uses gradient boosting algorithm over decision trees models.

        Great quality without parameter tuning (self-tuning?) - provides great results with its default parameters
        Categorical features support - automatically handles categorical data. Allows using non-numeric factors,
            instead of having to pre-process your data or spend time and effort turning it to numbers.
        Fast and scalable GPU version - has a fast (training & tuning) gradient-boosting implementation for GPU.
            for large datasets, theres' a multi-card configuration.
        Improved accuracy - a novel gradient-boosting scheme which reduce overfitting when constructing the models.
        Fast prediction - the 'model applier' applies the trained model quickly and efficiently
            even to latency-critical tasks.
        """
        model_name = 'Cat Boost'

        regressor = CatBoostRegressor()
        regressor.fit(self.X_train, self.y_train)

        y_pred = regressor.predict(self.X_test)
        self.predictions[model_name] = y_pred

        if self.X_range is not None:
            y_pred_fine = regressor.predict(self.X_range)
            self.predictions_fine[model_name] = y_pred_fine

        self.evaluate_model_performance(self.y_test, y_pred, model_name)

    """
    AllModels
    """

    def all_linear(self):
        self.lin_reg()
        self.ridge_reg()
        self.lin_svr()

    def all_nonlinear(self):
        self.poly_reg()
        self.poly_reg(pol_deg=3)
        self.poly_reg(pol_deg=4)
        self.kernel_svr(kernel='poly', pol_deg=2)
        self.kernel_svr(kernel='poly')
        self.kernel_svr(kernel='poly', pol_deg=4)
        self.kernel_svr()
        self.dtr()
        self.rfr()
        self.xgb()
        self.lgbm()
        self.cb()

    def all(self):
        self.all_linear()
        self.all_nonlinear()
