# https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
# https://github.com/sukilau/Ziff-deep-learning/blob/master/3-CIFAR10-lrate/CIFAR10-lrate.ipynb

# Learning Rate Schedules & Adaptive Learning Rate Methods (Adaptive Gradient Descent Algorithms) for Deep Learning

# When training deep NNs, it is often useful to reduce learning rate as the training progresses.
# This can be done by using pre-defined learning rate schedules or adaptive learning rate methods.

# Here we'll train a CNN on CIFAR-10 using differing learning rate schedules and adaptive learning rate methods
#   to compare their model performances.


import numpy as np
import math

import matplotlib.pyplot as plt

from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras.optimizers import SGD, Adagrad, Adadelta, RMSprop, Adam
from tensorflow.python.keras.callbacks import LearningRateScheduler, Callback


# A CNN model is constructed to train on CIFAR-10


batch_size = 64
epochs = 100
epoch_count = range(1, epochs + 1)
loss = 'categorical_crossentropy'

############################

# Load CIFAR-10 data (shuffled and split between train and test sets):
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'Ankle truck']

############################

# Only look at cats [=3] and dogs [=5]
train_picks = np.ravel(np.logical_or(Y_train == 3, Y_train == 5))
test_picks = np.ravel(np.logical_or(Y_test == 3, Y_test == 5))

X_train, Y_train = X_train[train_picks], np.array(Y_train[train_picks] == 5, dtype=int)
X_test, Y_test = X_test[test_picks], np.array(Y_test[test_picks] == 5, dtype=int)

classes_num = 2

############################

# Normalization:
X_train = X_train / 255.0
X_test = X_test / 255.0

############################

# Reshaping:
if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 3, *X_train.shape[1:3])
    X_test = X_test.reshape(X_test.shape[0], 3, *X_test.shape[1:3])
else:  # channels_last
    X_train = X_train.reshape(*X_train.shape[:3], 3)
    X_test = X_test.reshape(*X_test.shape[:3], 3)

######################

if loss == 'categorical_crossentropy':
    Y_train = to_categorical(Y_train, classes_num)
    Y_test = to_categorical(Y_test, classes_num)

############################


def construct_cnn_model(optimizer):
    model = Sequential([
        # first 'input_shape' is required so that `.summary` works.
        Conv2D(4, kernel_size=(3, 3), activation='relu', input_shape=X_train.shape[1:]),
        Conv2D(8, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(16, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    print(model.summary())
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model


def train_model(model, callbacks_list=None):
    if callbacks_list is None:
        return model.fit(X_train, Y_train,
                         validation_data=(X_test, Y_test),
                         epochs=epochs,
                         batch_size=batch_size,
                         verbose=2)
    else:
        return model.fit(X_train, Y_train,
                         validation_data=(X_test, Y_test),
                         epochs=epochs,
                         batch_size=batch_size,
                         callbacks=callbacks_list,
                         verbose=2)


def construct_and_train_model_using_optimizer(optimizer, callbacks_list=None):
    model = construct_cnn_model(optimizer)
    trained_model = train_model(model, callbacks_list)
    return trained_model


def construct_and_train_model_using_sgd_optimizer(learning_rate=0.0, decay_rate=0.0, momentum=0.0, callbacks_list=None):
    # return construct_and_train_model_using_optimizer('sgd')
    return construct_and_train_model_using_optimizer(
        SGD(lr=learning_rate, decay=decay_rate, momentum=momentum, nesterov=False), callbacks_list)


############################


def plot_model_accuracy(i, trained_model):
    train_acc = trained_model.history['acc']
    val_acc = trained_model.history['val_acc']

    fig = plt.figure()
    plt.plot(epoch_count, train_acc, 'r--', label='Training')
    plt.plot(epoch_count, val_acc, 'b-', label='Validation')
    plt.legend(loc=0)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xlim([1,epochs])
    plt.xticks(epoch_count)
    plt.grid(True)
    plt.title("Model Accuracy")
    plt.show()
    fig.savefig('img/'+str(i)+'-accuracy.jpg')
    plt.close(fig)


def plot_model_loss(i, trained_model):
    train_acc = trained_model.history['loss']
    val_acc = trained_model.history['val_loss']

    fig = plt.figure()
    plt.plot(epoch_count, train_acc, 'r--', label='Training')
    plt.plot(epoch_count, val_acc, 'b-', label='Validation')
    plt.legend(loc=0)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xlim([1,epochs])
    plt.xticks(epoch_count)
    plt.grid(True)
    plt.title("Model Loss")
    plt.show()
    fig.savefig('img/'+str(i)+'-loss.jpg')
    plt.close(fig)


# visualizing the learning rate schedule (loss_history.lr)
#   we can also visualize the loss history (loss_history.losses)
def plot_learning_rate(i, loss_history):
    learning_rate = loss_history.lr

    fig = plt.figure()
    plt.plot(epoch_count, learning_rate, label='Learning Rate')
    plt.legend(loc=0)
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.xlim([1,epochs+1])
    plt.grid(True)
    plt.title("Learning rate")
    plt.show()
    fig.savefig('img/'+str(i)+'-learning-rate.jpg')
    plt.close(fig)


#######################################################

# Constant Learning Rate
# this is the default learning rate schedule in SGD optimizer in Keras.
#   Momentum and decay rate are both set to zero by default.
# It is tricky to choose the right learning rate.
#   By experimenting with range of learning rates in our example,
#   lr=0.1 shows a relative good performance to start with.
# This can serve as a baseline for us to experiment with different learning rate strategies.

trained_model_01 = construct_and_train_model_using_sgd_optimizer(learning_rate=0.1)
plot_model_accuracy(1, trained_model_01)

#######################################################

# Learning Rate (decay?) Schedules
# seek to adjust the learning rate during training by reducing the learning rate according to a pre-defined schedule.
# For illustrative purpose, I construct a CNN trained on CIFAR-10,
#   using stochastic gradient descent (SGD) optimization algorithm
#   with different learning rate schedules to compare the performances.

# The challenge of using learning rate schedules is that their hyperparameters have to be defined in advance
#   and they depend heavily on the type of model and problem.
# Another problem is that the same learning rate is applied to all parameter updates.
#   If we have sparse data, we may want to update the parameters in different extent instead.

############################

# Time-based decay
# mathematical form: lr = lr0/(1+kt), where lr, k are hyperparameters and t is the iteration number.

learning_rate = 0.1
trained_model_02 = construct_and_train_model_using_sgd_optimizer(learning_rate=learning_rate,
                                                                 decay_rate=learning_rate / epochs,
                                                                 momentum=0.5)  # 0.8?
plot_model_accuracy(2, trained_model_02)

############################

# Step decay
# drops the learning rate by a factor every few epochs.
# mathematical form: lr = lr0 * drop^floor(epoch / epochs_drop)
# A typical way is to to drop the learning rate by half every 10 epochs.


# Creating a custom callback by extending the base class keras.callbacks.Callback
#   to record loss history (self.losses) and learning rate (schedule?) (self.lr) during the training procedure.
# As a digression, a callback is a set of functions to be applied at given stages of the training procedure.
#   We can use callbacks to get a view on internal states and statistics of the model during training.
class LossHistory_StepDecay(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append(step_decay(len(self.losses)))
        print('lr:', step_decay(len(self.losses)))


def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((epoch) / epochs_drop))
    # lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


# learning schedule callback
# LearningRateScheduler callback will return the updated learning rates (according to the decay function arg)
#   for use in SGD optimizer.
#   any custom decay schedule can be implemented in Keras using this approach,
#       The only difference is to define a different custom decay function.
# We can use LearningRateScheduler to create custom learning rate schedules which is specific to our data problem.
loss_history = LossHistory_StepDecay()
lrate = LearningRateScheduler(step_decay)

# we can pass a callback list consisting of LearningRateScheduler callback and our custom callback to fit the model.
trained_model_03 = construct_and_train_model_using_sgd_optimizer(momentum=0.5, callbacks_list=[loss_history, lrate])
plot_model_accuracy(3, trained_model_03)
plot_learning_rate(3, loss_history)

############################

# Exponential decay
# mathematical form: lr = lr0 * e^(−kt), where lr, k are hyperparameters and t is the iteration number (epoch).


class LossHistory_ExpDecay(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append(exp_decay(len(self.losses)))
        print('lr:', exp_decay(len(self.losses)))


def exp_decay(epoch):
    initial_lrate = 0.1
    k = 0.1
    lrate = initial_lrate * np.exp(-k * epoch)
    return lrate


loss_history = LossHistory_ExpDecay()
lrate = LearningRateScheduler(exp_decay)

trained_model_04 = construct_and_train_model_using_sgd_optimizer(momentum=0.8, callbacks_list=[loss_history, lrate])
plot_model_accuracy(4, trained_model_04)
plot_learning_rate(4, loss_history)

#######################################################

# Adaptive Learning Rate Methods
#   demonstrate better performance than Learning Rate Schedules.
#   require much less effort in hyper-parameter settings.

# In Keras, we can implement these adaptive learning algorithms easily using corresponding optimizers.
#   It's usually recommended to leave their hyperparameters at their default values (except lr sometimes).

epsilon = 1e-08

# trained_model_05 = construct_and_train_model_using_optimizer('adagrad')
trained_model_05 = construct_and_train_model_using_optimizer(Adagrad(lr=0.01, epsilon=epsilon, decay=0.0))
# trained_model_06 = construct_and_train_model_using_optimizer('adadelta')
trained_model_06 = construct_and_train_model_using_optimizer(Adadelta(lr=1.0, rho=0.95, epsilon=epsilon, decay=0.0))
# trained_model_07 = construct_and_train_model_using_optimizer('rmsprop')
trained_model_07 = construct_and_train_model_using_optimizer(RMSprop(lr=0.001, rho=0.9, epsilon=epsilon, decay=0.0))
# trained_model_08 = construct_and_train_model_using_optimizer('adam')
trained_model_08 = construct_and_train_model_using_optimizer(Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=epsilon, decay=0.0))

#######################################################

# Compare model accuracy
# Model performance of using different learning rate schedules or adaptive gradient descent algorithms are compared

fig = plt.figure(figsize=(12,8))
plt.plot(range(epochs), trained_model_01.history['val_acc'], label='Constant lr')
plt.plot(range(epochs), trained_model_02.history['val_acc'], label='Time-based')
plt.plot(range(epochs), trained_model_03.history['val_acc'], label='Step decay')
plt.plot(range(epochs), trained_model_04.history['val_acc'], label='Exponential decay')
plt.plot(range(epochs), trained_model_05.history['val_acc'], label='Adagrad')
plt.plot(range(epochs), trained_model_06.history['val_acc'], label='Adadelta')
plt.plot(range(epochs), trained_model_07.history['val_acc'], label='RMSprop')
plt.plot(range(epochs), trained_model_08.history['val_acc'], label='Adam')
plt.legend(loc=0)
plt.xlabel('epochs')
plt.xlim([0,epochs])
plt.ylabel('accuracy om validation set')
plt.grid(True)
plt.title("Comparing Model Accuracy")
plt.show()
fig.savefig('img/compare-accuracy.jpg')
plt.close(fig)
