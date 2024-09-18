# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
# https://flyyufelix.github.io/2016/10/08/fine-tuning-in-keras-part2.html
# https://flyyufelix.github.io/2016/10/03/fine-tuning-in-keras-part1.html
# https://www.kaggle.com/dromosys/dogs-vs-cats-keras/

# https://towardsdatascience.com/visualizing-intermediate-activation-in-convolutional-neural-networks-with-keras-260b36d60d0

# https://medium.com/@vijayabhaskar96/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720

# effect of synthetic db size on the mnist fine-tuned model:
#   5K - model's top accuracy: 94%
#   10K - model's top accuracy: 95.8%

# when training the same model from scratch on mnist, the results are much better.

# In order to achieve same score, did you save training time ? were you able to use less db ?


import cv2
import numpy as np
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation
import matplotlib.pyplot as plt
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.models import Model


wanted_generated_db_size = 10000
epochs_shapes = 100
batch_size_shapes = 2000

epochs_mnist = 100
batch_size_mnist = 2000

######################


def reshape_X(X):
    try:
        channels = X.shape[3]
    except IndexError:
        channels = 1

    if K.image_data_format() == 'channels_last':
        return X.reshape(*X.shape[:3], channels)
    else:  # channels_first
        return X.reshape(X.shape[0], channels, *X.shape[1:3])


def plot_model_accuracy(trained_model, title, ticks=0):
    epoch_count = [i + 1 for i in trained_model.epoch]
    fig = plt.figure()
    plt.plot(epoch_count, trained_model.history['acc'], 'r--', label='Training')
    plt.plot(epoch_count, trained_model.history['val_acc'], 'b-', label='Validation')
    plt.legend(loc=0)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xlim([1, max(epoch_count)])
    tick_space = 1 if ticks == 0 else epoch_count[-1] // ticks
    plt.xticks([i for i in epoch_count if i % tick_space == 0])
    plt.grid(True)
    plt.title("Model Accuracy")
    plt.show()
    fig.savefig('results/DL22_model_accuracy_' + title + '.jpg')
    plt.close(fig)


def plot_model_loss(trained_model, title, ticks=0):
    epoch_count = [i + 1 for i in trained_model.epoch]
    fig = plt.figure()
    plt.plot(epoch_count, trained_model.history['loss'], 'r--', label='Training')
    plt.plot(epoch_count, trained_model.history['val_loss'], 'b-', label='Validation')
    plt.legend(loc=0)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xlim([1, max(epoch_count)])
    tick_space = 1 if ticks == 0 else epoch_count[-1] // ticks
    plt.xticks([i for i in epoch_count if i % tick_space == 0])
    plt.grid(True)
    plt.title("Model Loss")
    plt.show()
    fig.savefig('results/DL22_model_loss_' + title + '.jpg')
    plt.close(fig)


######################

# creating a synthetic DB of random squares and circles:

circle = np.expand_dims(cv2.imread('../../datasets/per_field/cv/circle_28.jpg', cv2.IMREAD_GRAYSCALE), 2)
square = np.expand_dims(cv2.imread('../../datasets/per_field/cv/square_28.jpg', cv2.IMREAD_GRAYSCALE), 2)

X = np.stack((circle, square)) / 255.0  # Normalization
Y = np.array([0,1], dtype=np.int32)

X = reshape_X(X)

##############################################

# Data augmentation:

datagen = ImageDataGenerator(
    width_shift_range=0.2, height_shift_range=0.2,
    shear_range=0.2, zoom_range=0.2, rotation_range=90,  # rescale, brightness_range, black to grayscale
    horizontal_flip=True, vertical_flip=True  # validation_split=0.2
)

# datagen.fit(X)  # ???

X_generated = np.empty((0, 28, 28, 1), dtype=np.float32)
Y_generated = np.empty((0,), dtype=np.int32)
for x_batch, y_batch in datagen.flow(X, Y):
    X_generated = np.append(X_generated, x_batch, axis=0)
    print(X_generated.shape)
    Y_generated = np.append(Y_generated, y_batch, axis=0)
    if len(Y_generated) >= wanted_generated_db_size:
        break

# generator = datagen.flow(X, Y, batch_size, shuffle=True, seed=0)

##########################

def generate_images_and_save(X, wanted_generated_db_size):
    i = 1

    # flow() generates batches of randomly transformed images and saves the results to the directory
    for batch in datagen.flow(X, batch_size=2, shuffle=True, seed=0,
                              save_to_dir='deep_learning_22_preview',
                              save_prefix='image', save_format='jpg'):
        i += len(X)
        if i >= wanted_generated_db_size:
            break  # otherwise the generator would loop indefinitely


# generate_images_and_save(X, wanted_generated_db_size)

##############################################

nodes = 16  # 128  # 4
kernel_size = (3, 3)
pool_size=(2, 2)
model_shapes = Sequential([
    Conv2D(nodes, kernel_size=kernel_size, activation='relu', input_shape=X_generated.shape[1:]),
    # MaxPooling2D(pool_size=pool_size),  # creates an ERROR
    Conv2D(nodes, kernel_size=kernel_size, activation='relu'),
    # MaxPooling2D(pool_size=pool_size),  # creates an ERROR
    Conv2D(nodes, kernel_size=kernel_size, activation='relu'),
    # MaxPooling2D(pool_size=pool_size),  # creates an ERROR
    Conv2D(nodes, kernel_size=kernel_size, activation='relu'),
    # MaxPooling2D(pool_size=pool_size),
    Flatten(),  #input_shape=X_generated.shape[1:3]
    Dense(2, activation='softmax')
])

model_shapes.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

trained_model_shapes = model_shapes.fit(
    X_generated, Y_generated, validation_split=0.2,
    epochs=epochs_shapes, batch_size=batch_size_shapes
)

# # using the generators to train our model.
# # Each epoch takes 20-30s on GPU and 300-400s on CPU.
# # So it's definitely viable to run this model on CPU if you aren't in a hurry.
# trained_model_shapes = model_shapes.fit_generator(
#     generator,  # train_generator
#     steps_per_epoch=500,  # train_dataset_size // batch_size, train_generator.samples // batch_size
#     epochs=epochs,
#     validation_data=generator,  # validation_generator
#     validation_steps=500  # test_dataset_size // batch_size, validation_generator.samples // batch_size
# )
#
# # model.save_weights('cnn_weights.h5')  # always save your weights after training or during training
# # model.load_weights('cnn_weights.h5')
#
# weights_trained_model_shapes = model_shapes.weights

plot_model_accuracy(trained_model_shapes, '01_shapes', 5)
plot_model_loss(trained_model_shapes, '01_shapes', 5)

##############################################

# Fine-tuning

# model_shapes.summary()

# Adjust the trained model to suit MNIST (replace last layer).
# Truncate the original softmax layer and replace it with our own:
# model_mnist.layers[-1].outbound_nodes = []
# model_mnist.add(Dense(10, activation='softmax'))
x = Dense(10)(model_shapes.layers[-2].output)
o = Activation('softmax', name='loss')(x)
model_mnist = Model(model_shapes.inputs, [o])

# model_mnist.summary()

# only the last layer should be trainable:
for layer in model_mnist.layers[:len(model_mnist.layers) - 2]:
    layer.trainable = False

##############################################

# Training the model on MNIST

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = reshape_X(X_train) / 255.0
X_test = reshape_X(X_test) / 255.0

model_mnist.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

trained_model_mnist = model_mnist.fit(
    X_train, Y_train, validation_data=(X_test, Y_test),
    epochs=epochs_mnist, batch_size=batch_size_mnist
)

plot_model_accuracy(trained_model_mnist, '02_mnist', 5)
plot_model_loss(trained_model_mnist, '02_mnist', 5)

##############################################

# MNIST comparison model

model_mnist_scratch = Sequential([
    Conv2D(nodes, kernel_size=kernel_size, activation='relu', input_shape=X_generated.shape[1:]),
    # MaxPooling2D(pool_size=pool_size),  # creates an ERROR
    Conv2D(nodes, kernel_size=kernel_size, activation='relu'),
    # MaxPooling2D(pool_size=pool_size),  # creates an ERROR
    Conv2D(nodes, kernel_size=kernel_size, activation='relu'),
    # MaxPooling2D(pool_size=pool_size),  # creates an ERROR
    Conv2D(nodes, kernel_size=kernel_size, activation='relu'),
    # MaxPooling2D(pool_size=pool_size),
    Flatten(),  #input_shape=X_generated.shape[1:3]
    Dense(10, activation='softmax')
])

model_mnist_scratch.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

trained_model_mnist_scratch = model_mnist_scratch.fit(
    X_train, Y_train, validation_data=(X_test, Y_test),
    epochs=epochs_mnist, batch_size=batch_size_mnist
)

plot_model_accuracy(trained_model_mnist_scratch, '03_mnist_scratch', 5)
plot_model_loss(trained_model_mnist_scratch, '03_mnist_scratch', 5)
