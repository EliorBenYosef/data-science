import tensorflow as tf
from tensorflow.python.keras.datasets import fashion_mnist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Flatten, Dense
import numpy as np
import matplotlib.pyplot as plt


epochs = 5
classes_num = 10

##############################################

(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Normalization
X_train = X_train / 255.0
X_test = X_test / 255.0

##############################################

# NN model - Building

model = Sequential([
    Flatten(input_shape=X_train.shape[1:3]),  # a layer that reformats the data from 2D to 1D (crucial before the first FC)
    Dense(128, activation=tf.nn.relu),  # 1st layer: FC (128 neurons) +
    Dense(classes_num, activation='softmax')  # 2nd layer: 10-node softmax layer - returns an array of 10 probability scores that sum to 1
])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

##############################################

# NN model - Training

model.fit(X_train, Y_train, epochs=epochs)

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
