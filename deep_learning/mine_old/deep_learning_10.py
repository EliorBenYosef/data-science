# https://towardsdatascience.com/multi-label-classification-and-class-activation-map-on-fashion-mnist-1454f09f5925
# https://github.com/gabrielpierobon/cnnshapes/blob/master/README.md
# https://towardsdatascience.com/visualizing-intermediate-activation-in-convolutional-neural-networks-with-keras-260b36d60d0

# Construct the minimal network that classify well

# The minimal number of parameters that is enough to achieve above 95% accuracy is
# 2 x 2 kernel - 21,895
# 5 x 5 kernel - 17,368
# 10 x 10 kernel - 11,143


from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Conv2D, Flatten, Dense
import numpy as np
import matplotlib.pyplot as plt


############################

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
classes_num = 10


# Normalization:
X_train = X_train / 255.0
X_test = X_test / 255.0


# Reshaping:
try:
    channels = X_train.shape[3]
except IndexError:
    channels = 1

if K.image_data_format() == 'channels_last':
    X_train = X_train.reshape(*X_train.shape[:3], channels)
    X_test = X_test.reshape(*X_test.shape[:3], channels)
else:  # channels_first
    X_train = X_train.reshape(X_train.shape[0], channels, *X_train.shape[1:3])
    X_test = X_test.reshape(X_test.shape[0], channels, *X_test.shape[1:3])


# Constructing the network
model = Sequential([
    Conv2D(3, kernel_size=(10, 10), activation='relu', input_shape=X_train.shape[1:]),
    Flatten(input_shape=X_train.shape[1:3]),
    Dense(classes_num, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

trained_model = model.fit(X_train, Y_train,
                          validation_data=(X_test, Y_test),
                          epochs=5)

predictions = model.predict(X_test)

#######################################################

print('model total params:', model.count_params(), '\n')
print("model layers' params:", '\n', [layer.count_params() for layer in model.layers], '\n')

#######################################################

# Plotting Accuracy


def plot_model_accuracy(trained_model):
    epoch_count = [i + 1 for i in trained_model.epoch]
    fig = plt.figure()
    plt.plot(epoch_count, trained_model.history['acc'], 'r--', label='Training')
    plt.plot(epoch_count, trained_model.history['val_acc'], 'b-', label='Validation')
    plt.legend(loc=0)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xlim([1, max(epoch_count)])
    plt.xticks(epoch_count)
    plt.grid(True)
    plt.title("Model Accuracy")
    plt.show()
    fig.savefig('DL10_model_accuracy.jpg')
    plt.close(fig)


plot_model_accuracy(trained_model)

#######################################################

# Plotting Loss


def plot_model_loss(trained_model):
    epoch_count = [i + 1 for i in trained_model.epoch]
    fig = plt.figure()
    plt.plot(epoch_count, trained_model.history['loss'], 'r--', label='Training')
    plt.plot(epoch_count, trained_model.history['val_loss'], 'b-', label='Validation')
    plt.legend(loc=0)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xlim([1, max(epoch_count)])
    plt.xticks(epoch_count)
    plt.grid(True)
    plt.title("Model Loss")
    plt.show()
    fig.savefig('DL10_model_loss.jpg')
    plt.close(fig)


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
            fig.savefig('DL10_activations_' + layer_name + '_enhanced.jpg')
            plt.close(fig)

        else:

            # fig, ax = plt.subplots(rows, cols, figsize=(rows * 2.5, cols * 1.5), squeeze=False)
            fig, ax = plt.subplots(rows, cols, figsize=(12, 4), squeeze=False)
            fig.suptitle(layer_name)  # fontsize=16
            # print(features_num)
            # print(rows, cols)

            for feature in range(features_num):
                row = feature // cols
                col = feature % cols
                # print(feature, row, col)
                # print(layer_activation[:, :, feature][0][0][0])
                ax[row][col].imshow(layer_activation[:, :, feature], cmap='viridis')

                ax[row][col].grid(False)
                ax[row][col].set_xticks([])
                ax[row][col].set_yticks([])

            plt.show()
            fig.savefig('DL10_activations_' + layer_name + '_regular.jpg')
            plt.close(fig)


img_tensor = X_test[0]  # selected_image
fig = plt.figure()
plt.imshow(img_tensor[:,:,0])  # show original image
plt.show()
fig.savefig('DL10_image.jpg')
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
