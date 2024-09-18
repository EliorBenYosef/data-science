import tensorflow as tf
from tensorflow.python.keras.datasets import fashion_mnist
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.python.keras.losses import binary_crossentropy, categorical_crossentropy, sparse_categorical_crossentropy,\
    mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, mean_squared_logarithmic_error,\
    hinge, categorical_hinge, squared_hinge,\
    logcosh, kullback_leibler_divergence, poisson, cosine_proximity
from tensorflow.python.keras.activations import relu, softmax, sigmoid

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow.python.keras.models import Model


batch_size = 1024
epochs = 5

# optimizer = 'adam'  # 'sgd' 'adagrad' 'adadelta' 'rmsprop'

# loss: cross-entropy (Softmax), hinge (SVM)
loss = 'sparse_categorical_crossentropy'  # 'binary_crossentropy' 'categorical_crossentropy' ...

# activation='relu' # 'relu', 'softmax', 'sigmoid'; tf.nn.relu, tf.nn.softmax, tf.nn.sigmoid;

seed = 7
np.random.seed(seed)

##############################################

# Import the dataset, and split to Test & Train sets

######################

# when there's an auto split between Test & Train sets
(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()  # X - images, Y - labels
# where the index is not the class name (in contrary to mnist where the index is the digit and the actual class name)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']  # class names are not included with the dataset
classes_num = 10

######################

# # if there's no auto split between Test & Train sets
# X_train, X_test, Y_train, Y_test = train_test_split(X, y, random_state=0)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

##############################################

# Displaying images
# done to verify that the data is in the correct format and we're ready to build and train the network.
# the pixel values fall in the range of 0 to 255

# Displaying a single image
X_test_single = X_train[0]
plt.figure()
plt.imshow(X_test_single)
plt.colorbar()
plt.grid(False)
plt.show()

# Display multiple images & class name below
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i])  # cmap=plt.cm.binary (after 'to_categorical')
    plt.xlabel(class_names[Y_train[i]])
plt.show()

##############################################

# If we want specific classes (not all)
#   np.isin(Y_train, [0,6])
#   np.ravel(np.logical_or(Y_train == 0, Y_train == 6))

train_mask = np.isin(Y_train, [0,6])   # train_filter
test_mask = np.isin(Y_test, [0,6])     # test_filter

# when there are two classes (don't need to be consecutive):
X_train, Y_train = X_train[train_mask], np.array(Y_train[train_mask] == 6)
X_test, Y_test = X_test[test_mask], np.array(Y_test[test_mask] == 6)

# when the classes are consecutive - 0,1,2,... (can be more than two):
# X_train, Y_train = X_train[train_mask], Y_train[train_mask]
# X_test, Y_test = X_test[test_mask], Y_test[test_mask]

classes_num = 2

##############################################

# If we want to use less images to Train & Test

train_images = 500
X_train = X_train[:train_images]
Y_train = Y_train[:train_images]
test_images = 500
X_test = X_test[:test_images]
Y_test = Y_test[:test_images]

##############################################

# Data Pre-Processing

######################

# Normalization:
#   scaling the data pixel values to a range of 0 to 1 before feeding to the neural network model.
#   It's important that the training set and the testing set are pre-processed in the same way

X_train = X_train / 255.0
X_test = X_test / 255.0

######################

# Reshaping:
#   reshaping the array from 3D to 4D (compatible with keras API)
#   important when the first layer isn't Flatten (it can be Conv).

def reshape_X(X):
    try:
        channels = X.shape[3]
    except IndexError:
        channels = 1

    if K.image_data_format() == 'channels_last':
        return X.reshape(*X.shape[:3], channels)
    else:  # channels_first
        return X.reshape(X.shape[0], channels, *X.shape[1:3])


X_train = reshape_X(X_train)
X_test = reshape_X(X_test)

######################

# 'sparse_categorical_crossentropy' expect integer targets (class vectors).
if loss == 'binary_crossentropy' or loss == 'categorical_crossentropy':
    # expect binary (class) matrices targets (1s and 0s) of shape (samples, classes).
    #   If your targets are integer classes, you can convert them to the expected format using 'to_categorical':
    Y_train = to_categorical(Y_train, classes_num)
    # Y_train = to_categorical(np.ravel(Y_train), classes_num)  # practically the same
    Y_test = to_categorical(Y_test, classes_num)
    # Y_test = to_categorical(np.ravel(Y_test), classes_num)    # practically the same

##############################################

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

##############################################

# NN model - Building & Training

def construct_cnn_model(optimizer):
    model = Sequential([
        # first 'input_shape' is required so that `.summary` works.
        Conv2D(8, kernel_size=(3, 3), activation='relu', input_shape=X_train.shape[1:]),  # [1:3] ? [1:4] ?
        Conv2D(8, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(8, activation='relu'),
        Dropout(0.5),
        Dense(classes_num, activation='softmax')
    ])
    # print(model.summary())
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model


def train_model(model, callbacks_list=None):
    return model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        # validation_split=0.25,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_list,
        # shuffle=True,

        # verbose specifies how do you want to 'see' the training progress for each epoch.
        #   verbose=0 will show you nothing (silent)
        #   verbose=1 (default) will show you an animated progress bar.
        #   verbose=2 will just mention the number of epoch.
    )


# history = model.fit(X[train], y[train], epochs=200, batch_size=5, verbose=0)


def construct_and_train_cnn_model_using_optimizer(optimizer, callbacks_list=None):
    model = construct_cnn_model(optimizer)
    trained_model = train_model(model, callbacks_list)
    return model, trained_model


model, trained_model = construct_and_train_cnn_model_using_optimizer('adam')

##############################################

# model's params number
print('model total params:', model.count_params(), '\n')
print("model layers' params:", '\n', [layer.count_params() for layer in model.layers], '\n')

# model's weights and biases
print('model weights', '\n', model.weights, '\n')
print("model layers' weights and biases:", '\n', [layer.get_weights() for layer in model.layers], '\n')
for layer in model.layers:
    weights, biases = layer.get_weights()
    print("Layer's weights", '\n', weights, '\n')
    print("Layer's biases", '\n', biases, '\n')
model.save_weights('cnn_weights.h5')  # always save your weights after training or during training
# model.load_weights('cnn_weights.h5')

##############################################

# NN model - Testing

# Evaluating accuracy
test_loss, test_acc = model.evaluate(X_test, Y_test)
print('Test accuracy:', test_acc)

# Making predictions about a test set
predictions = model.predict(X_test)
i = 0
pred = np.argmax(predictions[i])  # np.argmax() gives us which label has the highest confidence value.
truth = Y_test[i]
print("model's prediction on index %d: %s, ground truth: %s" % (i, pred, truth))

# Making a prediction about a single image
i = 0
X_test_single = (np.expand_dims(X_test[i], 0))  # Add the image to a batch where it's the only member.
predictions_single = model.predict(X_test_single)
pred = np.argmax(predictions_single[0])
truth = Y_test[i]
print("model's prediction on image %d: %s, ground truth: %s" % (i, pred, truth))

#######################################################

# Plotting the confusion matrix


def plot_confusion_matrix(predictions, Y_test):
    Y_pred_classes = np.argmax(predictions, axis=1)  # Convert predictions classes to one hot vectors
    Y_true = np.argmax(Y_test, axis=1)  # Convert validation observations to one hot vectors
    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)  # compute the confusion matrix

    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx, annot=True, fmt="d")  # xticklabels=2, yticklabels=False
    plt.axes().xaxis.set_ticks_position('top')
    plt.yticks(rotation=0)
    plt.show()


plot_confusion_matrix(predictions, Y_test)

##############################################

# Plotting the predictions
# Graph predictions to look at the full set of the classes (or channels)


# Plots image with: predicted label + percent (true label)
def plot_image(i, predictions_array, true_label, img):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    predicted_label = np.argmax(predictions_array)

    plt.imshow(img, cmap=plt.cm.binary)
    color = 'blue' if predicted_label == true_label else 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label, present_names=False):
    plt.grid(False)
    if present_names:
        plt.xticks(range(classes_num), class_names, rotation=45)
    else:
        plt.xticks([])
    plt.yticks([])
    plt.ylim([0, 1])

    predictions_array, true_label = predictions_array[i], true_label[i]
    predicted_label = np.argmax(predictions_array)

    plot = plt.bar(range(classes_num), predictions_array, color="#777777")
    plot[predicted_label].set_color('red')
    plot[true_label].set_color('blue')


def show_subplots_image_and_value_array(num_rows=1, num_cols=1):
    plt.figure(figsize=(6 * num_cols, 3 * num_rows))
    for i in range(num_rows * num_cols):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions, Y_test, X_test.squeeze())
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions, Y_test)

    plt.show()


# single image - image + predictions
show_subplots_image_and_value_array()

# single image - predictions (with class names)
plot_value_array(0, predictions_single, Y_test, present_names=True)
plt.show()

# multiple images - images + predictions
show_subplots_image_and_value_array(5, 3)

##############################################


def plot_model_accuracy(trained_model, ticks=0):
    epoch_count = [i + 1 for i in trained_model.epoch]
    fig = plt.figure()
    plt.plot(epoch_count, trained_model.history['acc'], 'r--', label='Training')
    plt.plot(epoch_count, trained_model.history['val_acc'], 'b-', label='Validation')
    plt.legend(loc=0)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xlim([epoch_count[0], epoch_count[-1]])
    tick_space = 1 if ticks == 0 else epoch_count[-1] // ticks
    plt.xticks([i for i in epoch_count if i % tick_space == 0])
    plt.grid(True)
    plt.title("Model Accuracy")
    plt.show()
    fig.savefig('model_accuracy.jpg')
    plt.close(fig)


def plot_model_loss(trained_model, ticks=0):
    epoch_count = [i + 1 for i in trained_model.epoch]
    fig = plt.figure()
    plt.plot(epoch_count, trained_model.history['loss'], 'r--', label='Training')
    plt.plot(epoch_count, trained_model.history['val_loss'], 'b-', label='Validation')
    plt.legend(loc=0)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xlim([epoch_count[0], epoch_count[-1]])
    tick_space = 1 if ticks == 0 else epoch_count[-1] // ticks
    plt.xticks([i for i in epoch_count if i % tick_space == 0])
    plt.grid(True)
    plt.title("Model Loss")
    plt.show()
    fig.savefig('model_loss.jpg')
    plt.close(fig)


# visualizing the learning rate schedule (loss_history.lr)
#   we can also visualize the loss history (loss_history.losses)
def plot_learning_rate(loss_history):
    epoch_count = [i + 1 for i in trained_model.epoch]
    fig = plt.figure()
    plt.plot(epoch_count, loss_history.lr, label='Learning Rate')
    plt.legend(loc=0)
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.xlim([1, max(epoch_count)])
    plt.grid(True)
    plt.title("Learning rate")
    plt.show()
    fig.savefig('model_learning_rate.jpg')
    plt.close(fig)


def plot_models_accuracy(trained_models):
    fig = plt.figure(figsize=(16, 10))

    max_epochs = 0
    for name, trained_model in trained_models:
        epoch_count = [i + 1 for i in trained_model.epoch]
        epochs = max(epoch_count)
        if epochs > max_epochs:
            max_epochs = epochs
        val = plt.plot(epoch_count, trained_model.history['val_acc'], label=name.title() + ' Validation')
        plt.plot(epoch_count, trained_model.history['acc'], label=name.title() + ' Training',
                 linestyle='--', color=val[0].get_color())

    plt.legend(loc=0)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xlim([1, max_epochs])
    plt.xticks(range(1, max_epochs + 1))
    plt.grid(True)
    plt.title("Model Accuracy")
    plt.show()
    fig.savefig('models_accuracy.jpg')
    plt.close(fig)


# Required: metrics=['accuracy', loss]
def plot_models_loss(trained_models):
    fig = plt.figure(figsize=(16, 10))

    max_epochs = 0
    for name, trained_model in trained_models:
        epoch_count = [i + 1 for i in trained_model.epoch]
        epochs = max(epoch_count)
        if epochs > max_epochs:
            max_epochs = epochs
        val = plt.plot(epoch_count, trained_model.history['val_' + loss], label=name.title() + ' Validation')
        plt.plot(epoch_count, trained_model.history[loss], label=name.title() + ' Training',
                 linestyle='--', color=val[0].get_color())

    plt.legend(loc=0)
    plt.xlabel('Epoch')
    plt.ylabel(loss.replace('_', ' ').title())
    # plt.ylabel('Loss')
    plt.xlim([1, max_epochs])
    plt.xticks(range(1, max_epochs + 1))
    plt.grid(True)
    plt.title("Model Loss")
    plt.show()
    fig.savefig('models_loss.jpg')
    plt.close(fig)


plot_model_accuracy(trained_model)
plot_model_loss(trained_model)

#######################################################

# Plotting activations maps of the trained model
#   Visualizing every channel in every intermediate activation.


def display_activations(layers_names, layers_activations, layers_filters, enhanced=True):  # images_per_row

    for layer_name, layer_activation, cols in zip(layers_names, layers_activations, layers_filters):  # Displays the feature maps
        features_num = layer_activation.shape[-1]  # features map shape: (H, W, features_num).

        rows = features_num // cols
        if features_num % cols != 0:
            rows += 1

        if enhanced:

            display_grid = np.zeros((rows * layer_activation.shape[0], cols * layer_activation.shape[1]))

            for feature in range(features_num):
                row = feature // cols
                col = feature % cols

                channel_image = layer_activation[:, :, feature]

                # Post-processes the feature to make it visually palatable
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype(np.uint8)

                # Displays the grid:
                display_grid[row * layer_activation.shape[0]: (row + 1) * layer_activation.shape[0],
                             col * layer_activation.shape[1]: (col + 1) * layer_activation.shape[1]] = channel_image

            # scale = 1. / layer_activation.shape[0]
            # plt.figure(figsize=(scale * display_grid.shape[1],
            #                     scale * display_grid.shape[0]))
            fig = plt.figure(figsize=(12, 4))

            plt.imshow(display_grid, cmap='viridis')  # , aspect='auto'

            plt.grid(False)
            plt.xticks([])
            plt.yticks([])

            plt.title(layer_name)

            plt.show()
            fig.savefig('trained-model-image-01-act-enhanced.jpg')
            plt.close(fig)

        else:

            # fig, ax = plt.subplots(rows, cols, figsize=(rows * 2.5, cols * 1.5), squeeze=False)
            fig, ax = plt.subplots(rows, cols, figsize=(12, 4), squeeze=False)
            fig.suptitle(layer_name)  # fontsize=16

            for feature in range(features_num):
                row = feature // cols
                col = feature % cols
                ax[row][col].imshow(layer_activation[:, :, feature], cmap='viridis')

                ax[row][col].grid(False)
                ax[row][col].set_xticks([])
                ax[row][col].set_yticks([])

            plt.show()
            fig.savefig('trained-model-image-01-act-regular.jpg')
            plt.close(fig)


img_tensor = X_test[0]  # selected_image
fig = plt.figure()
plt.imshow(img_tensor[:,:,0])  # show original image
plt.show()
fig.savefig('trained-model-image-01.jpg')
plt.close(fig)

# present onlt conv2d activations
conv2d_layers = [layer for layer in model.layers if 'conv2d' in layer.name]
layers_names = [layer.name for layer in conv2d_layers]
layers_outputs = [layer.output for layer in conv2d_layers]
activation_model = Model(inputs=model.input, outputs=layers_outputs)
layers_activations = activation_model.predict(img_tensor.reshape(1, *img_tensor.shape))
layers_filters = [layer.filters for layer in conv2d_layers]

display_activations(layers_names, layers_activations, layers_filters, enhanced=False)
display_activations(layers_names, layers_activations, layers_filters)

#######################################################

summary_writer = tf.summary.FileWriter("tmp/tensorboard_env_name")

starting_ep, n_episodes, score = 0, 100, 0
for i in range(starting_ep, n_episodes):
    score += 1
    # Export results for Tensorboard (at the end of each episode)
    tfSummary_score = tf.Summary(value=[tf.Summary.Value(tag='score', simple_value=score)])  # Scalar Value Tensorflow Summary
    summary_writer.add_summary(tfSummary_score, global_step=i)
    summary_writer.flush()
