from data_tools.data import ClassificationDataSets
from models_classification import ClassificationModels
from utils import Visualizer


def bivariate_binary_classification():
    indices = [0, 1]  # Use only two features

    dataset = ClassificationDataSets()
    # dataset.get_logreg_simple()
    dataset.get_Social_Network_Ads()

    classification_models = ClassificationModels(dataset.X_train_sc, dataset.y_train, dataset.X_test_sc, dataset.y_test)

    visualizer = Visualizer(dataset.X_test, dataset.y_test,
                            dataset.x_labels[indices[0]], dataset.x_labels[indices[1]], dataset.y_label,
                            dataset.clss_labels, dataset.transformers[-1])

    ######################

    classification_models.all_linear(sc_x1=dataset.sc_x1, sc_x2=dataset.sc_x2)
    classification_models.print_models_accuracy()

    visualizer.vis_mult_lin_2D(classification_models.x2)
    for model_name, classifier in classification_models.classifiers.items():
        visualizer.vis_sing_lin_2D(classification_models.x2[model_name], classifier, model_name)

    ######################

    classification_models.all_nonlinear()
    classification_models.print_models_accuracy()

    for model_name, classifier in classification_models.classifiers.items():
        visualizer.visualize_results_2D(classifier, model_name)
    visualizer.show_results()


def bivariate_multiclass_classification():
    indices = [0, 1]  # Use only two features

    dataset = ClassificationDataSets()
    dataset.get_iris(indices)

    classification_models = ClassificationModels(dataset.X_train_sc, dataset.y_train, dataset.X_test_sc, dataset.y_test)
    classification_models.all()
    classification_models.print_models_accuracy()

    visualizer = Visualizer(dataset.X_test, dataset.y_test,
                            dataset.x_labels[0], dataset.x_labels[1], dataset.y_label,
                            dataset.clss_labels, dataset.transformers[-1])
    for model_name, classifier in classification_models.classifiers.items():
        visualizer.visualize_results_2D(classifier, model_name)
    visualizer.show_results()


def trivariate_binary_classification():
    pass
    # TODO: finalize


# # taken from linear models:
# def trivariate_binary_classification():
#     df = ???
#     indices = [0, 1, 2]  # Use only three features
#     X, y = df.iloc[:, indices].values, df.iloc[:, -1].values
#     x1_label, x2_label, x3_label = df.columns.values[indices[0]], df.columns.values[indices[1]], df.columns.values[indices[2]]
#     y_label = df.columns.values[-1]
#
#     # dataset = datasets.load_diabetes()
#     # X, y = dataset.data[:, np.newaxis, index], dataset.target
#     # x1_label, x2_label, x3_label = dataset.feature_names[indices[0]], dataset.feature_names[indices[1]], dataset.feature_names[indices[2]]
#     # y_label = 'Diabetes progression after one year'
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
#
#     X1_test, X2_test, X3_test = X_test[:, 0], X_test[:, 1], X_test[:, 2]
#     X1_test_range = np.array([min(X1_test), max(X1_test)], dtype=np.float)[:, np.newaxis]
#     X2_test_range = np.array([min(X2_test), max(X2_test)], dtype=np.float)[:, np.newaxis]
#     X3_test_range = np.array([min(X3_test), max(X3_test)], dtype=np.float)[:, np.newaxis]
#
#     # Feature Scaling
#     sc, sc_x1, sc_x2 = StandardScaler(), StandardScaler(), StandardScaler()
#     X_train_sc = sc.fit_transform(X_train.astype(float))
#     sc_x1.fit(X_train[:, np.newaxis, 0].astype(float))
#     sc_x2.fit(X_train[:, np.newaxis, 1].astype(float))
#     X_test_sc = sc.transform(X_test.astype(float))
#
#     classification_models = ClassificationModels(X_train_sc, y_train, X_test_sc, y_test)
#
#     x3_logreg = classification_models.log_reg(sc_x1=sc_x1, sc_x2=sc_x2)
#     x3_linsvc = classification_models.lin_svc(sc_x1=sc_x1, sc_x2=sc_x2)
#
#     def plot_3d_fig(fig_num, elev, azim):
#         fig = plt.figure(fig_num, figsize=(8, 6))
#         plt.clf()
#
#         ax = Axes3D(fig, elev=elev, azim=azim, auto_add_to_figure=False)
#         fig.add_axes(ax)
#         ax.scatter(X_test[:, 0], X_test[:, 1], y_test, c='k', marker='+')
#
#         ax.plot_trisurf(X_test[:, 0], X_test[:, 1], z_pred_linreg, color='green', linewidth=0, antialiased=False)
#         ax.plot_trisurf(X_test[:, 0], X_test[:, 1], z_pred_ridgereg, color='yellow', linewidth=0, antialiased=False)
#         ax.plot_trisurf(X_test[:, 0], X_test[:, 1], z_pred_linsvr, color='red', linewidth=0, antialiased=False)
#
#         ax.set_xlabel('X_1')
#         ax.set_ylabel('X_2')
#         ax.set_zlabel('Y')
#         ax.w_xaxis.set_ticklabels([])
#         ax.w_yaxis.set_ticklabels([])
#         ax.w_zaxis.set_ticklabels([])
#
#
#     # TODO: finalize
#     # Results Visualization
#     fig, ax = plt.subplots(figsize=(10, 6))
#
#     ax.plot(X1_test_range, x2_logreg(X1_test_range), 'k', lw=1, ls='--', label='LogReg')  # 'g'
#     # ax.plot(X_test, y_pred_ridgereg, color='yellow', label='RidgeReg')
#     ax.plot(X1_test_range, x2_linsvc(X1_test_range), 'r', lw=1, ls='--', label='LinSVC')  # color='red'
#
#     for i, clss in enumerate(np.unique(y_test)):
#         ax.scatter(*X_test[y_test == clss].T, cmap=ListedColormap(('tab:blue', 'tab:orange'))(i), label=clss)
#
#     ax.legend(loc=2)
#     plt.xlim(X1_test_range[0], X1_test_range[1])
#     plt.ylim(X2_test_range[0], X2_test_range[1])  # zlim X3_test_range
#     ax.set_xlabel(x1_label)
#     ax.set_ylabel(x2_label)  # x3_label
#     ax.set_title(f'Predicted "{y_label}" label')
#     plt.show()
#
#
#
#
#
#     plot_3d_fig(1, elev=20, azim=-150)
#     plt.show()


def trivariate_multiclass_classification():
    pass
    # TODO: finalize


def multivariate_multiclass_classification():
    dataset = ClassificationDataSets()
    dataset.get_Breast_Cancer_Wisconsin()

    classification_models = ClassificationModels(dataset.X_train_sc, dataset.y_train, dataset.X_test_sc, dataset.y_test)
    classification_models.all()
    classification_models.print_models_accuracy()


if __name__ == '__main__':
    bivariate_binary_classification()
    bivariate_multiclass_classification()

    # trivariate_binary_classification()
    # trivariate_multiclass_classification()

    multivariate_multiclass_classification()
