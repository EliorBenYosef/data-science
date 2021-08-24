"""
write your own linear classification model.
"""

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import random


def lin_clss_random(X, y):
    """
    here, we:
    1. create the line formula (random-ish)
    2. count the number of point from class A and from class B above the line and beneath the line
    """
    # Create the line formula (random-ish)
    a = - 1 + random.random() * 2
    x_min, x_max = np.min(X[:,0]), np.max(X[:,0])
    x = np.linspace(x_min, x_max, 100)
    y_min, y_max = np.min(X[:,1]), np.max(X[:,1])
    b = y_min + random.random() * (y_max - y_min)
    if a < 0:
        b += 3
    elif a > 0:
        b =- 3
    f = a * x + b

    # Count the number of point from class A and from class B above the line and beneath the line.
    A_correct, A_incorrect, B_correct, B_incorrect = 0,0,0,0
    indexes_above_the_line, indexes_below_the_line = [],[]
    correct_classification_score_01,correct_classification_score_02 = 0,0
    for index in range(len(y)):
        point_x, point_y, correct_class = X[index, 0], X[index, 1], y[index]
        if point_y - a * point_x - b > 0:  # the point is above the line
            if correct_class == 0:
                correct_classification_score_01 += 1
            else:
                correct_classification_score_02 += 1
        elif point_y - a * point_x - b < 0:  # the point is below the line
            if correct_class == 1:
                correct_classification_score_01 += 1
            else:
                correct_classification_score_02 += 1

    # give the classifier a score
    classifier_correct_classifications = np.max((correct_classification_score_01, correct_classification_score_02))
    classifier_success_rate = classifier_correct_classifications/len(y)

    # Draw the scatter plot & line in the 2D space
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(x, f, 'r')
    for color, i, target_name in zip(['navy', 'turquoise'], [0, 1], iris.target_names[:2]):
        ax.scatter(X[y == i, 0], X[y == i, 1], color=color, alpha=.8, lw=2, label=target_name)
    ax.legend(loc='best', shadow=False, scatterpoints=1)
    ax.set_xlabel('sepal length')  # 4.3-7.9
    ax.set_ylabel('sepal width')  # 2.0-4.4
    plt.suptitle('My Classifier')
    plt.title('line formula: y = %.2fx %s%.2f\nclassifier correct classifications: %d\nclassifier success rate: %.2f'
              % (a, '+' if not b<0 else '', b, classifier_correct_classifications, classifier_success_rate))
    # plt.ylim(top=30, bottom=-5)
    # plt.xlim(left=0, right=25)
    plt.show()


def lin_clss_grid_search(X, y):
    """
    here, we optimize the model with grid search (by testing various possibilities).
    """
    # optimize model with grid search (by testing various possibilities)
    best_a, best_b, best_classifier_correct_classifications, best_classifier_success_rate = 0, 0, 0, 0

    x_min, x_max = np.min(X[:, 0]), np.max(X[:, 0])
    x = np.linspace(x_min, x_max, 100)
    y_min, y_max = np.min(X[:, 1]), np.max(X[:, 1])

    for i in range(1000):
        a = - 10 + random.random() * 20
        b = - 10 + random.random() * 30

        # Count the number of point from class A and from class B above the line and beneath the line.
        correct_classification_score_01, correct_classification_score_02 = 0, 0
        for index in range(len(y)):
            point_x, point_y, correct_class = X[index, 0], X[index, 1], y[index]
            if point_y - a * point_x - b > 0:  # the point is above the line
                if correct_class == 0:
                    correct_classification_score_01 += 1
                else:
                    correct_classification_score_02 += 1
            elif point_y - a * point_x - b < 0:  # the point is below the line
                if correct_class == 1:
                    correct_classification_score_01 += 1
                else:
                    correct_classification_score_02 += 1

        # give the classifier a score
        classifier_correct_classifications = np.max((correct_classification_score_01, correct_classification_score_02))
        classifier_success_rate = classifier_correct_classifications / len(y)

        # save the best score yet
        if classifier_correct_classifications > best_classifier_correct_classifications:
            best_classifier_correct_classifications = classifier_correct_classifications
            best_classifier_success_rate = classifier_success_rate
            best_a = a
            best_b = b

    # Draw the scatter plot & best line in the 2D space
    fig, ax = plt.subplots(figsize=(12, 8))
    f = best_a * x + best_b
    ax.plot(x, f, 'r')
    for color, i, target_name in zip(['navy', 'turquoise'], [0, 1], iris.target_names[:2]):
        ax.scatter(X[y == i, 0], X[y == i, 1], color=color, alpha=.8, lw=2, label=target_name)
    ax.legend(loc='best', shadow=False, scatterpoints=1)
    ax.set_xlabel('sepal length')  # 4.3-7.9
    ax.set_ylabel('sepal width')  # 2.0-4.4
    plt.suptitle('My Classifier')
    plt.title('line formula: y = %.2fx %s%.2f\nclassifier correct classifications: %d\nclassifier success rate: %.2f'
              % (best_a, '+' if not best_b < 0 else '', best_b, best_classifier_correct_classifications,
                 best_classifier_success_rate))
    plt.show()


if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Select a subset (first two) of its features and classes.
    #   in order to have a 2 dimensions - 2 classes database.
    X = X[y != 2, :2]
    y = y[y != 2]

    lin_clss_random(X, y)
    lin_clss_grid_search(X, y)

