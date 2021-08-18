# 15. Take any model that achieve mnist score of above 95% and
# play with the db size and with the architecture in order to be in overfit. 
# Add regularizations to solve it. (L2 regularization, Dropout, Data augmentation, Early stopping, Batch normalization)

import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, Dropout, BatchNormalization
from keras.regularizers import l2
from keras.optimizers import SGD, adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras import callbacks
from keras.utils import to_categorical

# Changed hyper-parameters
of_batch_size = 5
of_epochs = 30
nof_batch_size = 50
nof_epochs = 3

classes_num = 10

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalization:
X_train = X_train / 255.0
X_test = X_test / 255.0
# Reshaping:
X_train = X_train.reshape(*X_train.shape[:3], 1)
X_test = X_test.reshape(*X_test.shape[:3], 1)
Y_train = to_categorical(y_train, num_classes=10, dtype='float32')
Y_test = to_categorical(y_test, num_classes=10, dtype='float32')

print("X shape:", X_train.shape, "Y shape:", Y_train.shape)

# example of a picture from the database and its label
x = np.reshape(X_train, [-1, 28, 28])
idx = np.random.randint(len(X_train))
plt.title(np.argmax(Y_train[idx]))
plt.imshow(x[idx])
plt.show()

# Splitting data
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=1)

####################################################################################################


def build_net(model_name, X=X_train, classes_num=classes_num, L2_regularization=0., Drop=1., Early_stopping=0.):
    model = Sequential()
    model.add(Conv2D(4, kernel_size=(2, 2), activation='relu', input_shape=list(X.shape[1:])))
    if model_name == 'NOF':
        model.add(BatchNormalization())
    model.add(Flatten(input_shape=list(X.shape[1:])))
    model.add(Dropout(Drop))
    model.add(Dense(classes_num, kernel_regularizer=l2(L2_regularization), activation='softmax'))
    model.compile(optimizer=adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    es = callbacks.EarlyStopping(monitor='val_loss', min_delta=Early_stopping, patience=6, verbose=0, mode='auto',
                                     restore_best_weights=True)  # restore model weights from the epoch with the best value
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=0.0005)
    print(model.summary())
    return model, reduce_lr, es


def plot_loss(model):
    loss = model.history['loss']
    val_loss = model.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, color='red', label='Training loss')
    plt.plot(epochs, val_loss, color='green', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def plot_acc(model):
    loss = model.history['acc']
    val_loss = model.history['val_acc']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, color='red', label='Training accuracy')
    plt.plot(epochs, val_loss, color='green', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()


#####################################################################################################

# Over-Fitting:
ovf_model = build_net(model_name='OF')

overfit_history = ovf_model.fit(
    X_train, Y_train, validation_data=(X_val, Y_val),
    batch_size=of_batch_size, epochs=of_epochs, verbose=1
)

plot_loss(overfit_history)
plot_acc(overfit_history)

##########################################################################################################

# Regularized Over-Fitting:
nof_model = build_net(model_name='NOF', L2_regularization=0.01, Drop=0.25, Early_stopping=0.01)

not_overfit_history = nof_model[0].fit(
    X_train, Y_train, validation_data=(X_val, Y_val),
    batch_size=nof_batch_size, epochs=nof_epochs, verbose=1,
    callbacks=[nof_model[1], nof_model[2]]
)

plot_loss(not_overfit_history)
plot_acc(not_overfit_history)
