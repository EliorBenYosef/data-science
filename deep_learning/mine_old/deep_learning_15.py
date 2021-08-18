from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.python.keras.regularizers import l1, l2
from tensorflow.python.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


epochs = 50
loss = 'sparse_categorical_crossentropy'  # 'binary_crossentropy'

##########################################


def plot_accuracies(trained_models):
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
    fig.savefig('DL15_models_accuracy.jpg')
    plt.close(fig)


def plot_losses(trained_models):
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
    fig.savefig('DL15_models_loss.jpg')
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

model.compile(optimizer='adam', loss=loss, metrics=['accuracy', loss])

trained_model_overfit = model.fit(X_train_overfit, Y_train_overfit,
                                  validation_data=(X_test, Y_test),
                                  batch_size=500,
                                  epochs=epochs)

############################

# Regularized Overfit Model

# lambda_var = 0.001  # 0.01? 0.0001?
# dropout_rate = 0.5
#
# model = Sequential([
#     Conv2D(512, kernel_size=(3, 3), activation='relu', input_shape=X_train.shape[1:]),
#     # BatchNormalization(axis=1),
#     Conv2D(512, kernel_size=(3, 3), activation='relu'),
#     Dropout(dropout_rate),
#     Conv2D(512, kernel_size=(3, 3), activation='relu'),  #  kernel_regularizer=l1(lambda_var),
#     Conv2D(512, kernel_size=(3, 3), activation='relu'),  #  kernel_regularizer=l2(lambda_var),
#     Flatten(input_shape=X_train.shape[1:3]),
#     Dense(classes_num, activation='softmax')
# ])
#
# model.compile(optimizer='adam', loss=loss, metrics=['accuracy', loss])
#
# trained_model_overfit_reg = model.fit(X_train_overfit, Y_train_overfit,
#                                       validation_data=(X_test, Y_test),
#                                       callbacks=[EarlyStopping(monitor='val_acc', patience=5)],
#                                       epochs=epochs)

############################

trained_models = [
    ('Fit', trained_model),
    ('Overfit', trained_model_overfit)
    # ,
    # ('Regularized', trained_model_overfit_reg)
]

plot_accuracies(trained_models)
plot_losses(trained_models)
