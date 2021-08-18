import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


class Visualizer:

    def __init__(self, X_test, y_test, x1_label, x2_label, y_label, clss_labels, sc=None):
        self.X_test = X_test
        self.y_test = y_test
        self.labels = np.unique(self.y_test)
        self.n_clss = len(self.labels)
        self.sc = sc

        self.x1_label = x1_label
        self.x2_label = x2_label
        self.y_label = y_label
        self.clss_labels = clss_labels

        self.X1_test, self.X2_test = X_test[:, 0], X_test[:, 1]
        X1_test_min, X1_test_max = self.X1_test.min(), self.X1_test.max()
        X2_test_min, X2_test_max = self.X2_test.min(), self.X2_test.max()

        x1_margin = round((X1_test_max - X1_test_min) / 20, ndigits=5)
        x2_margin = round((X2_test_max - X2_test_min) / 20, ndigits=5)
        x1_step = round((X1_test_max - X1_test_min) / 1000, ndigits=5)
        x2_step = round((X2_test_max - X2_test_min) / 1000, ndigits=5)
        self.x1lim_min, self.x1lim_max = X1_test_min - x1_margin, X1_test_max + x1_margin
        self.x2lim_min, self.x2lim_max = X2_test_min - x2_margin, X2_test_max + x2_margin

        self.X1_test_range = np.array([self.x1lim_min, self.x1lim_max], dtype=np.float)[:, np.newaxis]

        self.X1, self.X2 = np.meshgrid(
            np.arange(start=self.x1lim_min, stop=self.x1lim_max, step=x1_step),
            np.arange(start=self.x2lim_min, stop=self.x2lim_max, step=x2_step)
        )
        # self.X1, self.X2 = np.mgrid[
        #     self.x1lim_min:self.x1lim_max:x1_step,
        #     self.x2lim_min:self.x2lim_max:x2_step
        # ]

        # self.X1X2_pts = np.array([self.X1.ravel(), self.X2.ravel()]).T
        self.X1X2_pts = np.c_[self.X1.ravel(), self.X2.ravel()]

        # ('tab:blue', 'tab:orange')
        # ['navy', 'turquoise', 'darkorange']
        # self.colors_light = ('#FFAAAA', '#AAFFAA', '#AAAAFF')
        # self.colors_bold = ('#FF0000', '#00FF00', '#0000FF')
        self.colors_light = ('#AAAAFF', '#FFAAAA', '#AAFFAA',
                             '#AAFFFF', '#FFAAFF', '#FFFFAA')
        self.colors_bold = ('#0000FF', '#FF0000', '#00FF00',
                            '#00FFFF', '#FF00FF', '#FFFF00')

    def vis_sing_lin_2D(self, x2_f, classifier, model_name):
        fig, ax = plt.subplots(figsize=(10, 6))

        x1_1D = np.squeeze(self.X1_test_range)
        x2_1D = np.squeeze(x2_f(self.X1_test_range))
        plt.fill_between(x1_1D, x2_1D, self.x2lim_min, color='tab:blue', alpha=0.2)
        plt.fill_between(x1_1D, x2_1D, self.x2lim_max, color='tab:orange', alpha=0.2)

        if model_name == 'SVC linear':
            # plot the margin lines - the parallels to the separating hyperplane that pass through the support vectors
            #   (margin away from hyperplane in direction perpendicular to hyperplane).
            #   This is sqrt(1+m^2) away vertically in 2D.
            w1, w2 = np.squeeze(classifier.coef_)
            m = -w1 / w2

            X2 = x2_f(self.X1_test_range)
            margin = 1 / np.sqrt(np.sum(classifier.coef_ ** 2))
            lower_margin = X2 - np.sqrt(1 + m ** 2) * margin
            upper_margin = X2 + np.sqrt(1 + m ** 2) * margin
            # plot the line, the points, and the nearest vectors to the plane
            plt.plot(self.X1_test_range, X2, 'k-')
            plt.plot(self.X1_test_range, lower_margin, 'k--')
            plt.plot(self.X1_test_range, upper_margin, 'k--')

        for i, clss in enumerate(np.unique(self.y_test)):
            ax.scatter(*self.X_test[self.y_test == clss].T,
                       cmap=['tab:blue', 'tab:orange'][i],
                       label=self.clss_labels[clss])

        # ax.legend(loc=2)
        ax.legend(loc='best', shadow=False, scatterpoints=1)
        ax.set_xlim(self.x1lim_min, self.x1lim_max)
        ax.set_ylim(self.x2lim_min, self.x2lim_max)
        ax.set_xlabel(self.x1_label)
        ax.set_ylabel(self.x2_label)
        ax.set_title(f'Predicted "{self.y_label}" label')
        plt.show()

    def vis_mult_lin_2D(self, x2_dict):
        fig, ax = plt.subplots(figsize=(10, 6))

        for i, (model_name, x2_f) in enumerate(x2_dict.items()):
            ax.plot(self.X1_test_range, x2_f(self.X1_test_range), self.colors_bold[i],
                    lw=1, ls='--', label=model_name)  # 'g'

        for i, clss in enumerate(np.unique(self.y_test)):
            ax.scatter(*self.X_test[self.y_test == clss].T,
                       cmap=['tab:blue', 'tab:orange'][i],
                       label=self.clss_labels[clss])

        # ax.legend(loc=2)
        ax.legend(loc='best', shadow=False, scatterpoints=1)
        ax.set_xlim(self.x1lim_min, self.x1lim_max)
        ax.set_ylim(self.x2lim_min, self.x2lim_max)
        ax.set_xlabel(self.x1_label)
        ax.set_ylabel(self.x2_label)
        ax.set_title(f'Predicted "{self.y_label}" label')
        plt.show()

    def visualize_results_2D(self, classifier, model_name, probability=False):
        """
        Visualizes the results of a single classification model
        Plots the decision boundary, by assigning a color to each point in the mesh
        """
        if self.sc is not None:
            clss_pred = classifier.predict(self.sc.transform(self.X1X2_pts)).reshape(self.X1.shape)
        else:
            clss_pred = classifier.predict(self.X1X2_pts).reshape(self.X1.shape)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the decision boundary by assigning a color in the color map to each mesh point:
        ax.contourf(self.X1, self.X2, clss_pred, cmap=ListedColormap(self.colors_light[:self.n_clss]))
        # ax.pcolormesh(self.X1, self.X2, clss_pred, cmap=ListedColormap(self.colors_light[:self.n_clss]),
        #               shading='auto')

        if self.n_clss == 2 and model_name[:3] == 'SVC':
            self.plot_svm(classifier, ax, model_name, probability)
        else:
            # for i, clss in enumerate(self.labels):
            #     ax.scatter(*self.X_test[self.y_test == clss].T,
            #                color=self.colors_bold[:self.n_clss][i],
            #                label=self.clss_labels[clss],
            #                edgecolor='k', s=20)

            ax.scatter(self.X_test[:, 0], self.X_test[:, 1],
                       c=self.y_test, cmap=ListedColormap(self.colors_bold[:self.n_clss]),
                       zorder=10, edgecolor='k', s=20)  # s - plot_symbol_size

        patches = [Patch(color=c, label=l) for c, l in zip(self.colors_bold[:self.n_clss], self.clss_labels)]
        # ax.legend(handles=patches, loc=2)
        plt.legend(handles=patches, loc='best', shadow=False, scatterpoints=1)

        ax.set_xlim(self.x1lim_min, self.x1lim_max)
        ax.set_ylim(self.x2lim_min, self.x2lim_max)
        ax.set_xlabel(self.x1_label)
        ax.set_ylabel(self.x2_label)
        ax.set_title(f'{model_name} - Predicted "{self.y_label}" label')
        # plt.title("4-Class classification (k = %i, weights = '%s')" % (n_neighbors, weights))

    def plot_svm(self, classifier, ax, model_name, probability):
        """
        Plotting SVM's decision boundaries & margins
        """
        # in case of multiclass - reshape(*self.X1.shape, self.n_clss) ?

        # Distance of the samples X to the separating hyperplane:
        if self.sc is not None:
            d_fun_pred = classifier.decision_function(self.sc.transform(self.X1X2_pts)).reshape(self.X1.shape)
        else:
            d_fun_pred = classifier.decision_function(self.X1X2_pts).reshape(self.X1.shape)

        # ax.pcolormesh(self.X1, self.X2, d_fun_pred > 0, cmap=ListedColormap(self.colors_light[:self.n_clss]))
        ax.contour(self.X1, self.X2, d_fun_pred,
                   colors=['k', 'k', 'k'],
                   linestyles=['--', '-', '--'],
                   levels=[-1, 0, 1])

        # if model_name[4:] == 'linear':
        #     d_fun_X_test = classifier.decision_function(sc.transform(self.X_test))
        #     support_vector_indices = np.where(np.abs(d_fun_X_test) <= 1 + 1e-15)[0]
        #     support_vectors = self.X_test[support_vector_indices]
        # else:
        #     support_vectors = sc.inverse_transform(classifier.support_vectors_)
        # ax.scatter(support_vectors[:, 0], support_vectors[:, 1],
        #            s=80, facecolors='none',
        #            zorder=10, edgecolors='k')

        # if probability and self.n_clss == 2:
        #     # Express the predicted probability of having a negative class label, by shading the points based on it:
        #     svm_probability = classifier.predict_proba(sc.transform(self.X_test))[:, 0]
        #     ax.scatter(self.X1_test, self.X2_test, s=30,
        #                c=svm_probability, cmap='Reds')
        # else:

        # Express the predictions' confidence level (a function of the point's distance from the hyperplane):
        if self.sc is not None:
            svm_confidence = classifier.decision_function(self.sc.transform(self.X_test.astype(float)))
        else:
            svm_confidence = classifier.decision_function(self.X_test)

        ax.scatter(self.X1_test, self.X2_test, s=50,
                   c=svm_confidence, cmap='seismic')

    @staticmethod
    def show_results():
        plt.show()
