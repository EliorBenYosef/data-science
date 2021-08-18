import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D


class Visualizer:

    def __init__(self, X_test, y_test, x1_label='X_1', x2_label='X_2', y_label='Y'):
        self.X_test = X_test
        self.y_test = y_test

        self.x1_label = x1_label
        self.x2_label = x2_label
        self.y_label = y_label

        # self.X1_test, self.X2_test = X_test[:, 0], X_test[:, 1]
        # X1_test_min, X1_test_max = self.X1_test.min(), self.X1_test.max()
        # X2_test_min, X2_test_max = self.X2_test.min(), self.X2_test.max()
        #
        # x1_margin = round((X1_test_max - X1_test_min) / 20, ndigits=5)
        # x2_margin = round((X2_test_max - X2_test_min) / 20, ndigits=5)
        # x1_step = round((X1_test_max - X1_test_min) / 1000, ndigits=5)
        # x2_step = round((X2_test_max - X2_test_min) / 1000, ndigits=5)
        # self.x1lim_min, self.x1lim_max = X1_test_min - x1_margin, X1_test_max + x1_margin
        # self.x2lim_min, self.x2lim_max = X2_test_min - x2_margin, X2_test_max + x2_margin

        self.colors_light = ('#AAAAFF', '#FFAAAA', '#AAFFAA',
                             '#AAFFFF', '#FFAAFF', '#FFFFAA')
        self.colors_bold = ('#0000FF', '#FF0000', '#00FF00',
                            '#00FFFF', '#FF00FF', '#FFFF00')

        self.colors = ('green', 'yellow', 'red', 'orange', 'teal', 'purple', 'grey')
        # self.colors = ('yellow', 'orange', 'red', 'teal', 'green', 'purple', 'grey')

    def visualize_results_2D(self, predictions, X_range=None):
        """
        Visualizes the results of a multiple regression models
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(self.X_test, self.y_test, label='Data')
        # ax.scatter(df.Population, df.Profit, label='Data')
        for i, (model_name, y_pred) in enumerate(predictions.items()):
            if X_range is not None:
                ax.plot(X_range, y_pred, color=self.colors[i], label=model_name)
            else:
                ax.plot(self.X_test, y_pred, self.colors[i], label=model_name)
        # ax.legend(loc=2)
        ax.legend(loc='best', shadow=False, scatterpoints=1)
        ax.set_xlabel(self.x1_label)
        ax.set_ylabel(self.y_label)
        ax.set_title(f'Predicted {self.y_label} vs. {self.x1_label}')
        plt.show()

    def visualize_results_3D(self, predictions, fig_num, elev=20, azim=-150):
        fig = plt.figure(fig_num, figsize=(8, 6))
        plt.clf()

        ax = Axes3D(fig, elev=elev, azim=azim, auto_add_to_figure=False)
        fig.add_axes(ax)
        ax.scatter(self.X_test[:, 0], self.X_test[:, 1], self.y_test, c='k', marker='+')

        for i, (model_name, z_pred) in enumerate(predictions.items()):
            ax.plot_trisurf(self.X_test[:, 0], self.X_test[:, 1], z_pred,
                            color=self.colors[i], label=model_name, linewidth=0, antialiased=False)

        ax.set_xlabel(self.x1_label)
        ax.set_ylabel(self.x2_label)
        ax.set_zlabel(self.y_label)
        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        ax.set_title(f'Predicted {self.y_label} vs. {self.x1_label} & {self.x2_label}')
        plt.show()

    # def visualize_results_2D(self, classifier, sc, model_name, is_svm=False, is_linsvc=False, probability=False):
    #     """
    #     Visualizes the results of a single classification model
    #     Plots the decision boundary, by assigning a color to each point in the mesh
    #     """
    #     clss_pred = classifier.predict(sc.transform(self.X1X2_pts)).reshape(self.X1.shape)
    #
    #     fig, ax = plt.subplots(figsize=(10, 6))
    #
    #     # Plot the decision boundary by assigning a color in the color map to each mesh point:
    #     ax.contourf(self.X1, self.X2, clss_pred, cmap=ListedColormap(self.colors_light[:self.n_clss]))
    #     # ax.pcolormesh(self.X1, self.X2, clss_pred, cmap=ListedColormap(self.colors_light[:self.n_clss]),
    #     #               shading='auto')
    #
    #     if is_svm:
    #         self.plot_svm(classifier, sc, ax, is_linsvc, probability)
    #     else:
    #         # for i, clss in enumerate(self.labels):
    #         #     ax.scatter(*self.X_test[self.y_test == clss].T,
    #         #                color=self.colors_bold[:self.n_clss][i],
    #         #                label=self.clss_labels[clss],
    #         #                edgecolor='k', s=20)
    #
    #         ax.scatter(self.X_test[:, 0], self.X_test[:, 1],
    #                    c=self.y_test, cmap=ListedColormap(self.colors_bold[:self.n_clss]),
    #                    zorder=10, edgecolor='k', s=20)  # s - plot_symbol_size
    #
    #     patches = [Patch(color=c, label=l) for c, l in zip(self.colors_bold[:self.n_clss], self.clss_labels)]
    #     ax.legend(handles=patches, loc=2)
    #
    #     ax.set_xlim(self.x1lim_min, self.x1lim_max)
    #     ax.set_ylim(self.x2lim_min, self.x2lim_max)
    #     ax.set_xlabel(self.x1_label)
    #     ax.set_ylabel(self.x2_label)
    #     ax.set_title(f'{model_name} - Predicted "{self.y_label}" label')
    #     # plt.title("4-Class classification (k = %i, weights = '%s')" % (n_neighbors, weights))
    #
    # def plot_svm(self, classifier, sc, ax, is_linsvc, probability):
    #     """
    #     Plotting SVM's decision boundaries & margins
    #     """
    #     # Distance of the samples X to the separating hyperplane:
    #     if self.n_clss == 2:
    #         d_fun_pred = classifier.decision_function(sc.transform(self.X1X2_pts)).reshape(self.X1.shape)
    #     else:
    #         d_fun_pred = classifier.decision_function(sc.transform(self.X1X2_pts)).reshape(*self.X1.shape, self.n_clss)
    #     # ax.pcolormesh(self.X1, self.X2, d_fun_pred > 0, cmap=ListedColormap(self.colors_light[:self.n_clss]))
    #     ax.contour(self.X1, self.X2, d_fun_pred,
    #                colors=['k', 'k', 'k'],
    #                linestyles=['--', '-', '--'],
    #                levels=[-1, 0, 1])
    #
    #     # if is_linsvc:
    #     #     d_fun_X_test = classifier.decision_function(sc.transform(self.X_test))
    #     #     support_vector_indices = np.where(np.abs(d_fun_X_test) <= 1 + 1e-15)[0]
    #     #     support_vectors = self.X_test[support_vector_indices]
    #     # else:
    #     #     support_vectors = sc.inverse_transform(classifier.support_vectors_)
    #     # ax.scatter(support_vectors[:, 0], support_vectors[:, 1],
    #     #            s=80, facecolors='none',
    #     #            zorder=10, edgecolors='k')
    #
    #     # if probability and self.n_clss == 2:
    #     #     # Express the predicted probability of having a negative class label, by shading the points based on it:
    #     #     svm_probability = classifier.predict_proba(sc.transform(self.X_test))[:, 0]
    #     #     ax.scatter(self.X1_test, self.X2_test, s=30,
    #     #                c=svm_probability, cmap='Reds')
    #     # else:
    #
    #     # Express the predictions' confidence level (a function of the point's distance from the hyperplane):
    #     svm_confidence = classifier.decision_function(sc.transform(self.X_test))
    #     ax.scatter(self.X1_test, self.X2_test, s=50,
    #                c=svm_confidence, cmap='seismic')

    @staticmethod
    def show_results():
        plt.show()
