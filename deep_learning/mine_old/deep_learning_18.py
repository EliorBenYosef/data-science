# Playing with:
#   Optimization \ learning rate methods: adagrad, adadelta, sgd...
#   Learning rate: is the initial value (lr) and its degredation (decay) - here I didn't do it...

# Adadelta-Batch worked best: 0.978 accuracy

from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, Flatten, Dense
import matplotlib.pyplot as plt


# Base hyper-parameters
loss = 'sparse_categorical_crossentropy'
epochs_base = 5

# Changed hyper-parameters
batch_size = 1024
epochs = 20
# learning_rate = 0.1  # 1.0, 0.1, 0.01, 0.001


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
    fig.savefig('results/DL18_' + title + '_models_accuracy.jpg')
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
    fig.savefig('results/DL18_' + title + '_models_loss.jpg')
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

model = Sequential([
    Conv2D(8, kernel_size=(2, 2), activation='relu', input_shape=X_train.shape[1:]),
    Flatten(input_shape=X_train.shape[1:3]),
    Dense(classes_num, activation='softmax')
])

############################

model.compile(optimizer='sgd', loss=loss, metrics=['accuracy', loss])
trained_model_sgd = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs_base)
trained_model_sgd_batch_size = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs_base,
                                         batch_size=batch_size)
trained_model_sgd_epochs = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs)
trained_model_sgd_epochs_and_batch = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs,
                                         batch_size=batch_size)


model.compile(optimizer='adagrad', loss=loss, metrics=['accuracy', loss])
trained_model_adagrad = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs_base)
trained_model_adagrad_batch_size = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs_base,
                                             batch_size=batch_size)
trained_model_adagrad_epochs = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs)
trained_model_adagrad_epochs_and_batch = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs,
                                             batch_size=batch_size)


model.compile(optimizer='adadelta', loss=loss, metrics=['accuracy', loss])
trained_model_adadelta = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs_base)
trained_model_adadelta_batch_size = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs_base,
                                              batch_size=batch_size)
trained_model_adadelta_epochs = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs)
trained_model_adadelta_epochs_and_batch = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs,
                                              batch_size=batch_size)


model.compile(optimizer='rmsprop', loss=loss, metrics=['accuracy', loss])
trained_model_rms = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs_base)
trained_model_rms_batch_size = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs_base,
                                         batch_size=batch_size)
trained_model_rms_epochs = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs)
trained_model_rms_epochs_and_batch = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs,
                                         batch_size=batch_size)


model.compile(optimizer='adam', loss=loss, metrics=['accuracy', loss])
trained_model_adam = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs_base)
trained_model_adam_batch_size = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs_base,
                                          batch_size=batch_size)
trained_model_adam_epochs = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs)
trained_model_adam_epochs_and_batch = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs,
                                          batch_size=batch_size)


############################

trained_models_adam = [('ADAM', trained_model_adam),
                       ('ADAM Batch Size', trained_model_adam_batch_size),
                       ('ADAM Epochs', trained_model_adam_epochs),
                       ('ADAM Epochs & Batch Size', trained_model_adam_epochs_and_batch)]

trained_models_sgd = [('SGD', trained_model_sgd),
                      ('SGD Batch Size', trained_model_sgd_batch_size),
                      ('SGD Epochs', trained_model_sgd_epochs),
                      ('SGD Epochs & Batch Size', trained_model_sgd_epochs_and_batch)]

trained_models_adagrad = [('ASGARD', trained_model_adagrad),
                          ('ASGARD Batch Size', trained_model_adagrad_batch_size),
                          ('ASGARD Epochs', trained_model_adagrad_epochs),
                          ('ASGARD Epochs & Batch Size', trained_model_adagrad_epochs_and_batch)]

trained_models_adadelta = [('ADADELTA', trained_model_adadelta),
                           ('ADADELTA Batch Size', trained_model_adadelta_batch_size),
                           ('ADADELTA Epochs', trained_model_adadelta_epochs),
                           ('ADADELTA Epochs & Batch Size', trained_model_adadelta_epochs_and_batch)]

trained_models_rms = [('RMSPROP', trained_model_rms),
                      ('RMSPROP Batch Size', trained_model_rms_batch_size),
                      ('RMSPROP Epochs', trained_model_rms_epochs),
                      ('RMSPROP Epochs & Batch Size', trained_model_rms_epochs_and_batch)]

############################

plot_accuracies(trained_models_adam, 'adam')
plot_accuracies(trained_models_sgd, 'sgd')
plot_accuracies(trained_models_adagrad, 'adagrad')
plot_accuracies(trained_models_adadelta, 'adadelta')
plot_accuracies(trained_models_rms, 'rms')

plot_losses(trained_models_adam, 'adam')
plot_losses(trained_models_sgd, 'sgd')
plot_losses(trained_models_adagrad, 'adagrad')
plot_losses(trained_models_adadelta, 'adadelta')
plot_losses(trained_models_rms, 'rms')
