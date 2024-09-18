# https://medium.com/coinmonks/the-mathematics-of-neural-network-60a112dd3e05

from tensorflow.python.keras.datasets import fashion_mnist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, Flatten, Dense
import numpy as np
import matplotlib.pyplot as plt


(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
classes_num = 10


# Use only 1000 images to train:
X_train = X_train[:1000]
Y_train = Y_train[:1000]
# Use only 1000 images to test:
X_test = X_test[:1000]
Y_test = Y_test[:1000]


# Normalization:
X_train = X_train / 255.0
X_test = X_test / 255.0


# Reshaping the array from 3D to 4D (compatible with keras API)
X_train = X_train.reshape(*X_train.shape, 1)
X_test = X_test.reshape(*X_test.shape, 1)


# Constructing the network
model = Sequential([
    Conv2D(8, kernel_size=(3, 3), activation='relu'),
    Flatten(input_shape=X_train.shape[1:3]),
    Dense(classes_num, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training the model to always return a constant prediction.
model.fit(X_train, np.full(Y_train.shape, 1), epochs=5)

predictions = model.predict(X_test)

#######################################################

print('model weights', '\n', model.weights, '\n')
weights_0, biases_0 = model.layers[0].get_weights()
print('Conv weights', '\n', weights_0, '\n')
print('Conv biases', '\n', biases_0, '\n')
weights_2, biases_2 = model.layers[2].get_weights()
print('FC weights', '\n', weights_2, '\n')
print('FC biases', '\n', biases_2, '\n')

#######################################################

# Plotting


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(classes_num), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# plotting several images with their predictions.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions, Y_test, X_test.squeeze())
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions, Y_test)

plt.show()
