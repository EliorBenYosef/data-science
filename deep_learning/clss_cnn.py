"""
Convolutional Neural Network for a Binary Classification task
Computer Vision task:

metrics (tensorflow.keras.metrics):
accuracy, acc
    binary_accuracy
    categorical_accuracy
sparse_categorical_accuracy
top_k_categorical_accuracy (requires you specify a k parameter)
sparse_top_k_categorical_accuracy (requires you specify a k parameter)
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

####################################

# Part 1 - Image Preprocessing
#   https://keras.io/api/preprocessing/image/

target_size = (64, 64)
idg_train = ImageDataGenerator(rescale=1./255,  # feature scaling
                               # validation_split=0.2
                               # image augmentation (applying transformations to avoid overfitting):
                               # width_shift_range=0.2, height_shift_range=0.2,  # shift horizontally & vertically
                               shear_range=0.2, zoom_range=0.2, #rotation_range=90,
                               # rescale=, brightness_range=, black to grayscale=,
                               horizontal_flip=True)#, vertical_flip=True)
                               # zca_whitening=True - highlights the outline of each digit.
training_set = idg_train.flow_from_directory(
    '../datasets/per_field/cv/training_set', target_size, batch_size=32, class_mode='binary')  # class_mode='categorical'

idg_test = ImageDataGenerator(rescale=1./255)  # feature scaling
test_set = idg_test.flow_from_directory(
    '../datasets/per_field/cv/test_set', target_size, batch_size=32, class_mode='binary')  # class_mode='categorical'

####################################

# Part 2 - CNN

# Building:
#   Layers: (Convolution, Pooling) x #, Flattening, Full Connection, Output Layer
classifier = Sequential()
classifier.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[*target_size, 3]))
classifier.add(MaxPool2D(pool_size=2, strides=2))  # padding='valid'/'same'
classifier.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
classifier.add(MaxPool2D(pool_size=2, strides=2))  # padding='valid'/'same'
classifier.add(Flatten())
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid', name='output'))
# classifier.add(Dense(units=n_clss, activation='softmax', name='output'))

# Compiling
metrics=['accuracy']
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)
# classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)

# Training & evaluating the model
history = classifier.fit(x=training_set, validation_data=test_set, epochs=25)

# plot metrics
for metric in metrics:
    plt.plot(history.history[metric])
plt.show()

# Making predictions
y_pred = classifier.predict(test_set)
y_pred = (y_pred > 0.5).astype(np.int)

# evaluating the model
accuracy = accuracy_score(test_set.labels, y_pred)
c_matrix = confusion_matrix(test_set.labels, y_pred)
clss_report = classification_report(test_set.labels, y_pred)
print(f'Accuracy score: {accuracy:.2f}')
print(f'Confusion Matrix: \n{c_matrix}')
print(f'classification report: \n{clss_report}')

####################################

# Part 3 - Making a single prediction


classes = {}
for clss_name, clss_index in training_set.class_indices.items():
    classes[clss_index] = clss_name


def predict_single(path, img_label):
    """
    Predicting the result of a single observation:
    """
    img = load_img(path, target_size=target_size)  # PIL format
    img = img_to_array(img)  # ndarray format
    img = np.expand_dims(img, axis=0)  # batch format
    result = classifier.predict(img)
    print(img_label, 'prediction:', classes[result[0][0]])


predict_single('../datasets/per_field/cv/clss/dog.jpg', 'dog')
predict_single('../datasets/per_field/cv/clss/cat.jpg', 'cat')
