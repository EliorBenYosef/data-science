# Test accuracy: Fit - 0.9694
# Test accuracy: Overfit - 0.8731
# Test accuracy: L1 - 0.8751
# Test accuracy: L2 - 0.8642
# Test accuracy: Dropout - 0.8769
# Test accuracy: L1, L2 - 0.8649
# Test accuracy: L1, Dropout - 0.8703
# Test accuracy: L2, Dropout - 0.8754
# Test accuracy: L1, L2, Dropout - 0.8684


from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.python.keras.regularizers import l1, l2
from tensorflow.python.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np


epochs = 50
loss = 'sparse_categorical_crossentropy'  # 'binary_crossentropy'

seed = 0
np.random.seed(seed)

##########################################


def plot_accuracies(trained_models, title):
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
    # fig.savefig('results/DL15_testing_models_accuracy' + title + '.jpg')
    plt.close(fig)


def plot_losses(trained_models, title):
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
    # fig.savefig('results/DL15_testing_models_loss' + title + '.jpg')
    plt.close(fig)


##########################################

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

############################

# Model (that achieve mnist score of above 95%)

train_images = 10000
X_train = X_train[:train_images]
Y_train = Y_train[:train_images]

model = Sequential([
    Conv2D(16, kernel_size=(2, 2), activation='relu', input_shape=X_train.shape[1:]),
    Conv2D(16, kernel_size=(2, 2), activation='relu'),
    Conv2D(16, kernel_size=(2, 2), activation='relu'),
    Conv2D(16, kernel_size=(2, 2), activation='relu'),
    Flatten(input_shape=X_train.shape[1:3]),
    Dense(classes_num, activation='softmax')
])

model.compile(optimizer='adam', loss=loss, metrics=['accuracy', loss])

trained_model = model.fit(X_train, Y_train,
                          validation_data=(X_test, Y_test),
                          batch_size=2000,
                          epochs=epochs)

############################

def compile_and_fit(model):
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy', loss])
    return model.fit(X_train_overfit, Y_train_overfit,
                     validation_data=(X_test, Y_test),
                     batch_size=500,
                     epochs=epochs)


def compile_and_fit_early_stop(model):
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy', loss])
    return model.fit(X_train_overfit, Y_train_overfit,
                     validation_data=(X_test, Y_test),
                     callbacks=[EarlyStopping(monitor='val_acc', patience=5)],
                     epochs=epochs)

############################

# Overfit Model (more complex + smaller testing dataset)

train_images = 500
X_train_overfit = X_train[:train_images]
Y_train_overfit = Y_train[:train_images]

model = Sequential([
    Conv2D(512, kernel_size=(3, 3), activation='relu', input_shape=X_train.shape[1:]),
    Conv2D(512, kernel_size=(3, 3), activation='relu'),
    Conv2D(512, kernel_size=(3, 3), activation='relu'),
    Conv2D(512, kernel_size=(3, 3), activation='relu'),
    Flatten(input_shape=X_train.shape[1:3]),
    Dense(classes_num, activation='softmax')
])
trained_model_overfit = compile_and_fit(model)

############################

# Reg models

l1_lambda_var = 0.0001  # 1
l2_lambda_var = 0.0001  # 3,4

model = Sequential([
    Conv2D(512, kernel_size=(3, 3), kernel_regularizer=l1(l1_lambda_var), activation='relu', input_shape=X_train.shape[1:]),
    Conv2D(512, kernel_size=(3, 3), activation='relu'),
    Conv2D(512, kernel_size=(3, 3), activation='relu'),
    Conv2D(512, kernel_size=(3, 3), activation='relu'),
    Flatten(input_shape=X_train.shape[1:3]),
    Dense(classes_num, activation='softmax')
])
trained_model_overfit_reg01 = compile_and_fit(model)

model = Sequential([
    Conv2D(512, kernel_size=(3, 3), kernel_regularizer=l2(l2_lambda_var), activation='relu', input_shape=X_train.shape[1:]),
    Conv2D(512, kernel_size=(3, 3), kernel_regularizer=l2(l2_lambda_var), activation='relu'),
    Conv2D(512, kernel_size=(3, 3), kernel_regularizer=l2(l2_lambda_var), activation='relu'),
    Conv2D(512, kernel_size=(3, 3), kernel_regularizer=l2(l2_lambda_var), activation='relu'),
    Flatten(input_shape=X_train.shape[1:3]),
    Dense(classes_num, activation='softmax')
])
trained_model_overfit_reg02 = compile_and_fit(model)

dropout_rate = 0.05
model = Sequential([
    Conv2D(512, kernel_size=(3, 3), activation='relu', input_shape=X_train.shape[1:]),
    Dropout(dropout_rate),
    Conv2D(512, kernel_size=(3, 3), activation='relu'),
    Dropout(dropout_rate),
    Conv2D(512, kernel_size=(3, 3), activation='relu'),
    Dropout(dropout_rate),
    Conv2D(512, kernel_size=(3, 3), activation='relu'),
    Dropout(dropout_rate),
    Flatten(input_shape=X_train.shape[1:3]),
    Dense(classes_num, activation='softmax')
])
trained_model_overfit_reg03 = compile_and_fit(model)

model = Sequential([
    Conv2D(512, kernel_size=(3, 3), kernel_regularizer=l1(l1_lambda_var), activation='relu', input_shape=X_train.shape[1:]),
    Conv2D(512, kernel_size=(3, 3), kernel_regularizer=l2(l2_lambda_var), activation='relu'),
    Conv2D(512, kernel_size=(3, 3), activation='relu'),
    Conv2D(512, kernel_size=(3, 3), activation='relu'),
    Flatten(input_shape=X_train.shape[1:3]),
    Dense(classes_num, activation='softmax')
])
trained_model_overfit_reg04 = compile_and_fit(model)

model = Sequential([
    Conv2D(512, kernel_size=(3, 3), kernel_regularizer=l1(l1_lambda_var), activation='relu', input_shape=X_train.shape[1:]),
    Conv2D(512, kernel_size=(3, 3), activation='relu'),
    Conv2D(512, kernel_size=(3, 3), activation='relu'),
    Dropout(dropout_rate),
    Conv2D(512, kernel_size=(3, 3), activation='relu'),
    Flatten(input_shape=X_train.shape[1:3]),
    Dense(classes_num, activation='softmax')
])
trained_model_overfit_reg05 = compile_and_fit(model)

model = Sequential([
    Conv2D(512, kernel_size=(3, 3), kernel_regularizer=l2(l2_lambda_var), activation='relu', input_shape=X_train.shape[1:]),
    Conv2D(512, kernel_size=(3, 3), activation='relu'),
    Conv2D(512, kernel_size=(3, 3), activation='relu'),
    Dropout(dropout_rate),
    Conv2D(512, kernel_size=(3, 3), activation='relu'),
    Flatten(input_shape=X_train.shape[1:3]),
    Dense(classes_num, activation='softmax')
])
trained_model_overfit_reg06 = compile_and_fit(model)

model = Sequential([
    Conv2D(512, kernel_size=(3, 3), kernel_regularizer=l1(l1_lambda_var), activation='relu', input_shape=X_train.shape[1:]),
    Conv2D(512, kernel_size=(3, 3), kernel_regularizer=l2(l2_lambda_var), activation='relu'),
    Conv2D(512, kernel_size=(3, 3), activation='relu'),
    Dropout(dropout_rate),
    Conv2D(512, kernel_size=(3, 3), activation='relu'),
    Flatten(input_shape=X_train.shape[1:3]),
    Dense(classes_num, activation='softmax')
])
trained_model_overfit_reg07 = compile_and_fit(model)

############################

trained_models = [
    ('Fit', trained_model),
    ('Overfit', trained_model_overfit),
    ('L1', trained_model_overfit_reg01),
    ('L2', trained_model_overfit_reg02),
    ('Dropout', trained_model_overfit_reg03),
    ('L1, L2', trained_model_overfit_reg04),
    ('L1, Dropout', trained_model_overfit_reg05),
    ('L2, Dropout', trained_model_overfit_reg06),
    ('L1, L2, Dropout', trained_model_overfit_reg07)
]

plot_accuracies(trained_models, 'reg')
plot_losses(trained_models, 'reg')
