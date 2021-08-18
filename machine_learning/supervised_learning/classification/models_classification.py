"""
Then from a business point of view, you would rather use:
- LogReg / Naive Bayes - when you want to rank your predictions by their probability.
    For example if you want to rank your customers from the highest probability that they buy a certain product,
    to the lowest probability. Eventually that allows you to target your marketing campaigns.
    And of course for this type of business problem, you should use Logistic Regression if your problem is linear,
    and Naive Bayes if your problem is non linear.
- SVM - when you want to predict to which segment your customers belong to.
    Segments can be any kind of segments, for example some market segments you identified earlier with clustering.
- Decision Tree - when you want to have clear interpretation of your model results.
- Random Forest - when you are just looking for high performance with less need for interpretation.

TODO:
add other classification models.
Good ones for NLP include: CART, C5.0, Maximum Entropy.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from nimbusml.ensemble import LightGBMClassifier
# from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


class ClassificationModels:

    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.classifiers = {}
        self.performances = {}
        self.x2 = {}

    def evaluate_model_performance(self, y_true, y_pred, model_name):
        """
        Model Performance Evaluation
        """
        # train_accuracy = classifier.score(self.X_train, self.y_train)
        accuracy = accuracy_score(y_true, y_pred)
        c_matrix = confusion_matrix(y_true, y_pred)
        clss_report = classification_report(y_true, y_pred)

        self.performances[model_name] = [accuracy, c_matrix, clss_report]

    def print_models_accuracy(self):
        for model_name, performance in self.performances.items():
            print(f'{model_name} Accuracy score: {performance[0]:.2f}')

    def print_models_c_matrix(self):
        for model_name, performance in self.performances.items():
            print(f'{model_name} Confusion Matrix: \n{performance[1]}')

    def print_models_clss_report(self):
        for model_name, performance in self.performances.items():
            print(f'{model_name} Classification report: \n{performance[2]}')

    def print_models_performance(self, dimen_reducer_name=None):
        for model_name, performance in self.performances.items():
            if dimen_reducer_name is not None:
                model_name = dimen_reducer_name + ' + ' + model_name
            print(f'{model_name} Accuracy score: {performance[0]:.2f}')
            print(f'{model_name} Confusion Matrix: \n{performance[1]}')
            # print(f'{model_name} Classification report: \n{performance[2]}')

    """
    LinearModels
    """

    def add_decision_boundary_lin_func(self, lin_classifier, sc_x1, sc_x2, model_name):
        """
        Calculates the decision boundary's function (the separating line)
        :param lin_classifier: the linear classifier model
        :param sc_x1: StandardScaler of X1
        :param sc_x2: StandardScaler of X2
        :return:
        """
        w0 = lin_classifier.intercept_[0]
        w1, w2 = np.squeeze(lin_classifier.coef_)
        b = -w0 / w2
        m = -w1 / w2
        self.x2[model_name] = lambda x1: sc_x2.inverse_transform(m * sc_x1.transform(x1) + b)

    def log_reg(self, max_iter=100, sc_x1=None, sc_x2=None):
        """
        Logistic Regression (LogReg)

        multi_class='multinomial' if is_digits else 'auto'
        """
        model_name = 'LogReg'

        classifier = LogisticRegression(solver='lbfgs', max_iter=max_iter, multi_class='auto', random_state=0)
        classifier.fit(self.X_train, self.y_train)

        # p_pred = classifier.predict_proba(self.X_test)
        # log_p_pred = classifier.predict_log_proba(self.X_test)
        y_pred = classifier.predict(self.X_test)

        self.evaluate_model_performance(self.y_test, y_pred, model_name)

        if sc_x1 is not None and sc_x2 is not None:
            self.add_decision_boundary_lin_func(classifier, sc_x1, sc_x2, model_name)

        self.classifiers[model_name] = classifier

    def lin_svc(self, C=1.0, loss='squared_hinge', max_iter=1000, sc_x1=None, sc_x2=None):
        """
        Linear Support Vector Classification (SVC linear)
        Linear SVMs use Hinge loss.
        https://www.johnwittenauer.net/machine-learning-exercises-in-python-part-6/

        LinearSVC() & SVC(kernel='linear') yield slightly different decision boundaries.
        This can be a consequence of the following differences:
            SVC(kernel='linear') - minimizes the regular hinge loss.
                uses the One-vs-One multi-class reduction.
            LinearSVC - minimizes the squared hinge loss
                uses the One-vs-All (AKA One-vs-Rest) multi-class reduction.
        Both linear models have linear decision boundaries (intersecting hyperplanes)

        Linear SVMs use Cross-Entropy loss:
            Binary cross-entropy / log loss --> logistic classifier
            Categorical cross-entropy / softmax loss --> softmax classifier

        very similar to linear regression in its learning approach,
            but the cost and gradient functions are formulated differently.
        it uses:
            continuous output (as in LinReg) + sigmoid (or “logit”) activation function
        """
        model_name = 'SVC linear'

        # classifier = SVC(C=C, kernel='linear', random_state=0)
        classifier = LinearSVC(C=C, loss=loss, max_iter=max_iter, random_state=0)
        classifier.fit(self.X_train, self.y_train)

        y_pred = classifier.predict(self.X_test)

        self.evaluate_model_performance(self.y_test, y_pred, model_name)

        if sc_x1 is not None and sc_x2 is not None:
            self.add_decision_boundary_lin_func(classifier, sc_x1, sc_x2, model_name)

        self.classifiers[model_name] = classifier

    """
    NonLinearModels
    """

    # TODO: include more Nearest Neighbors algorithms here.

    def knn(self, n_neighbors=5, weights='uniform'):
        """
        K-Nearest Neighbors (KNN)
        https://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html
        """
        model_name = f'KNN ({n_neighbors} {weights})'

        # metric='minkowski', p=2 -> Euclidean Distance:
        classifier = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric='minkowski', p=2)
        classifier.fit(self.X_train, self.y_train)

        y_pred = classifier.predict(self.X_test)

        self.evaluate_model_performance(self.y_test, y_pred, model_name)

        self.classifiers[model_name] = classifier

    def kernel_svc(self, kernel='rbf', pol_deg=3, C=1.0, probability=False):
        """
        Kernel Support Vector Classification (SVC)
        https://www.johnwittenauer.net/machine-learning-exercises-in-python-part-6/

        the non-linear kernel models (polynomial or Gaussian RBF) have more flexible non-linear decision boundaries
            with shapes that depend on the kind of kernel and its parameters.

        C - inverse regularization strength
        Smaller C -> Stronger regularization (less overfitting) -> Larger margin - includes more/all the observations, allowing the margins to be calculated using
            all the data in the area.
        Larger C -> Weaker regularization (more overfitting) -> Smaller margin - we do not have that much faith in our data's distribution,
            and will only consider points close to line of separation.

        gamma='auto'\'scale' to account better for unscaled features.
        RBF - gamma: inverse of size of radial kernel.
        """
        model_name = f'SVC {kernel}'
        if kernel == 'poly':
            model_name += f' ({pol_deg})'

        classifier = SVC(C=C, kernel=kernel, degree=pol_deg, probability=probability, random_state=0)
        # classifier = SVC(gamma='auto', kernel='poly', degree=3)
        classifier.fit(self.X_train, self.y_train)

        y_pred = classifier.predict(self.X_test)

        self.evaluate_model_performance(self.y_test, y_pred, model_name)

        self.classifiers[model_name] = classifier

    def naive_bayes(self):
        """
        (Gaussian) Naive Bayes (GNB)
        """
        model_name = 'Naive Bayes'

        classifier = GaussianNB()
        classifier.fit(self.X_train, self.y_train)

        y_pred = classifier.predict(self.X_test)

        self.evaluate_model_performance(self.y_test, y_pred, model_name)

        self.classifiers[model_name] = classifier

    def dtc(self):
        """
        Decision Tree Classification (DTC), AKA Classification Tree

        criterion - the function to measure the quality of the splits
        """
        model_name = 'Decision Tree'

        classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
        classifier.fit(self.X_train, self.y_train)

        y_pred = classifier.predict(self.X_test)

        self.evaluate_model_performance(self.y_test, y_pred, model_name)

        self.classifiers[model_name] = classifier

    def rfc(self):
        """
        Random Forest Classification (RFC)

        criterion - the function to measure the quality of the splits
        """
        model_name = 'Random Forest'

        classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0, n_jobs=-1)  # 100
        classifier.fit(self.X_train, self.y_train)

        y_pred = classifier.predict(self.X_test)

        self.evaluate_model_performance(self.y_test, y_pred, model_name)

        self.classifiers[model_name] = classifier

    # def xgb(self):
    #     """
    #     XG Boost (XGB).
    #     """
    #     model_name = 'XG Boost'
    #
    #     classifier = XGBClassifier()
    #     classifier.fit(self.X_train, self.y_train)
    #
    #     y_pred = classifier.predict(self.X_test)
    #
    #     self.evaluate_model_performance(self.y_test, y_pred, model_name)
    #
    #     self.classifiers[model_name] = classifier

    # def lgbm(self):
    #     """
    #     Light GBM (Light GBM)
    #     A gradient-based model that uses gradient boosting algorithm over decision trees models.
    #     """
    #     model_name = 'Light GBM'
    #
    #     classifier = LightGBMClassifier()
    #     classifier.fit(self.X_train, self.y_train)
    #
    #     y_pred = classifier.predict(self.X_test)
    #
    #     self.evaluate_model_performance(self.y_test, y_pred, model_name)
    #
    #     self.classifiers[model_name] = classifier

    # def cb(self):
    #     """
    #     Cat Boost (CatB)
    #     https://catboost.ai/
    #     A gradient-based model that uses gradient boosting algorithm over decision trees models.
    #
    #     Great quality without parameter tuning (self-tuning?) - provides great results with its default parameters
    #     Categorical features support - automatically handles categorical data. Allows using non-numeric factors,
    #         instead of having to pre-process your data or spend time and effort turning it to numbers.
    #     Fast and scalable GPU version - has a fast (training & tuning) gradient-boosting implementation for GPU.
    #         for large datasets, theres' a multi-card configuration.
    #     Improved accuracy - a novel gradient-boosting scheme which reduce overfitting when constructing the models.
    #     Fast prediction - the 'model applier' applies the trained model quickly and efficiently
    #         even to latency-critical tasks.
    #     """
    #     model_name = 'Cat Boost'
    #
    #     classifier = CatBoostClassifier()
    #     classifier.fit(self.X_train, self.y_train)
    #
    #     y_pred = classifier.predict(self.X_test)
    #
    #     self.evaluate_model_performance(self.y_test, y_pred, model_name)
    #
    #     self.classifiers[model_name] = classifier

    def lvq(self):
        """
        Learning Vector Quantization (LVQ).
        https://mrnuggelz.github.io/sklearn-lvq/
        """
        pass
        # TODO: implement

    """
    AllModels
    """

    def all_linear(self, sc_x1=None, sc_x2=None):
        self.log_reg(sc_x1=sc_x1, sc_x2=sc_x2)
        self.lin_svc(sc_x1=sc_x1, sc_x2=sc_x2)

    def all_nonlinear(self):
        self.knn()
        self.knn(n_neighbors=15)
        self.knn(n_neighbors=15, weights='distance')
        self.kernel_svc()  # probability=True
        self.kernel_svc(kernel='poly')
        self.naive_bayes()
        self.dtc()
        self.rfc()
        self.lvq()

    def all(self):
        self.all_linear()
        self.all_nonlinear()
