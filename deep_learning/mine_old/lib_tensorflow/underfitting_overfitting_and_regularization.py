# https://keras.io/regularizers/

from tensorflow.python.keras.datasets import imdb
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, BatchNormalization
from tensorflow.python.keras.regularizers import l1, l2
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt


# Multi-hot encoding the sentences.
# Multi-hot encoding our lists means turning them into vectors of 0s and 1s.
#   Concretely, this would mean for instance turning the sequence [3, 5] into a 10,000-dimensional vector
#   that would be all-zeros except for indices 3 and 5, which would be ones.
# This model will quickly overfit to the training set.
#   It will be used to demonstrate when overfitting occurs, and how to fight it.
def multi_hot_sequences(sequences, dimension):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, word_indices in enumerate(sequences):
        results[i, word_indices] = 1.0  # set specific indices of results[i] to 1s
    return results


batch_size = 512
epochs = 20

loss = 'binary_crossentropy'

#####################################################

NUM_WORDS = 10000

# Download the IMDB dataset
(X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words=NUM_WORDS)

X_train = multi_hot_sequences(X_train, dimension=NUM_WORDS)
X_test = multi_hot_sequences(X_test, dimension=NUM_WORDS)

# # Visualizing one of the resulting multi-hot vectors.
# #   The word indices are sorted by frequency, so it is expected that there are more 1-values near index zero.
# plt.plot(train_data[0])
# plt.show()

#####################################################

REG_TECH_L1 = 1
REG_TECH_L2 = 2
REG_TECH_DROPOUT = 3
REG_TECH_BATCH_NORM = 4

lambda_var = 0.001  # 0.01? 0.0001?
dropout_rate = 0.5
early_stopping = False


# create a simple model using only Dense layers
def construct_cnn_model(nodes, optimizer, reg_tech=0):
    # first 'input_shape' is required so that `.summary` works.

    if reg_tech == REG_TECH_L1:
        model = Sequential([
            Dense(nodes, kernel_regularizer=l1(lambda_var), activation='relu', input_shape=(NUM_WORDS,)),
            Dense(nodes, kernel_regularizer=l1(lambda_var), activation='relu'),
            Dense(1, activation='sigmoid')
        ])
    elif reg_tech == REG_TECH_L2:
        model = Sequential([
            Dense(nodes, kernel_regularizer=l2(lambda_var), activation='relu', input_shape=(NUM_WORDS,)),
            Dense(nodes, kernel_regularizer=l2(lambda_var), activation='relu'),
            Dense(1, activation='sigmoid')
        ])
    elif reg_tech == REG_TECH_DROPOUT:
        # the Dropout layer gets applied to the output of layer right before.
        model = Sequential([
            Dense(nodes, activation='relu', input_shape=(NUM_WORDS,)),
            Dropout(dropout_rate),
            Dense(nodes, activation='relu'),
            Dropout(dropout_rate),
            Dense(1, activation='sigmoid')
        ])
    elif reg_tech == REG_TECH_BATCH_NORM:
        # the Batch Normalization layer gets applied to the activation output of layer right before.
        #   axis – specifies the axis for the data that should be normalized (typically the features axis)
        #   beta_initializer & gamma_initializer - the initializers for the arbitrarily set parameters, defaults: 0 & 1.
        model = Sequential([
            Dense(nodes, activation='relu', input_shape=(NUM_WORDS,)),
            BatchNormalization(axis=1),
            Dense(nodes, activation='relu'),
            BatchNormalization(axis=1),
            Dense(1, activation='sigmoid')
        ])
    else:
        model = Sequential([
            Dense(nodes, activation='relu', input_shape=(NUM_WORDS,)),
            Dense(nodes, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

    print(model.summary())

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy', loss])

    return model


def train_model(model):
    if not early_stopping:
        trained_model = model.fit(X_train, Y_train,
                                  validation_data=(X_test, Y_test),
                                  epochs=epochs,
                                  batch_size=batch_size,
                                  verbose=2)
    else:
        trained_model = model.fit(X_train, Y_train,
                                  validation_data=(X_test, Y_test),
                                  epochs=epochs,
                                  batch_size=batch_size,
                                  callbacks=[EarlyStopping(monitor='val_acc', patience=2)],
                                  verbose=2)

    return trained_model


def construct_and_train_cnn_model(nodes, reg_tech=0):
    model = construct_cnn_model(nodes, 'adam', reg_tech)
    trained_model = train_model(model)
    return trained_model


baseline_trained_model = construct_and_train_cnn_model(16)   # Create a baseline model

# Different capacity models
smaller_trained_model = construct_and_train_cnn_model(4)   # Create a smaller model - less hidden units
bigger_trained_model = construct_and_train_cnn_model(512)  # Create a bigger\larger model - will overfit quickly

# Regularization techniques
l1_trained_model = construct_and_train_cnn_model(16, reg_tech=REG_TECH_L1)  # Add L1 weight regularization penalty
l2_trained_model = construct_and_train_cnn_model(16, reg_tech=REG_TECH_L2)  # Add L2 weight regularization penalty
dpt_trained_model = construct_and_train_cnn_model(16, reg_tech=REG_TECH_DROPOUT)  # Add dropout
batchnorm_trained_model = construct_and_train_cnn_model(16, reg_tech=REG_TECH_BATCH_NORM)  # Add batch normalization

#####################################################


def plot_models_accuracy(trained_models, name):
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
    fig.savefig('unovre_models_accuracy' + name + '.jpg')
    plt.close(fig)


# Required: metrics=['accuracy', loss]
def plot_models_loss(trained_models, name):
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
    fig.savefig('unovre_models_loss' + name + '.jpg')
    plt.close(fig)


trained_models_size = [('baseline', baseline_trained_model),
                       ('smaller', smaller_trained_model),
                       ('bigger', bigger_trained_model)]
plot_models_accuracy(trained_models_size, 'size')
plot_models_loss(trained_models_size, 'size')

trained_models_regularization = [('baseline', baseline_trained_model),
                                 ('l1', l1_trained_model),
                                 ('l2', l2_trained_model),
                                 ('dropout', dpt_trained_model),
                                 ('batch norm', batchnorm_trained_model)]
plot_models_accuracy(trained_models_regularization, 'reg')
plot_models_loss(trained_models_regularization, 'reg')


#####################################################

# Data Augmentation
# https://keras.io/preprocessing/image/
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
# https://machinelearningmastery.com/image-augmentation-deep-learning-keras/
# https://www.analyticsvidhya.com/blog/2018/04/fundamentals-deep-learning-regularization-techniques/
# https://mxnet.incubator.apache.org/versions/master/tutorials/python/types_of_data_augmentation.html
# https://towardsdatascience.com/synthetic-data-generation-a-must-have-skill-for-new-data-scientists-915896c0c1ae
# https://webcache.googleusercontent.com/search?q=cache:obcEA-5OnNQJ:https://towardsdatascience.com/synthetic-data-generation-a-must-have-skill-for-new-data-scientists-915896c0c1ae+&cd=1&hl=en&ct=clnk&gl=il

##########################

# creating an ImageDataGenerator

# zca_whitening=True - highlights the outline of each digit.
datagen = ImageDataGenerator(
    width_shift_range=0.2, height_shift_range=0.2,
    shear_range=0.2, zoom_range=0.2, rotation_range=90,  # rescale, brightness_range, black to grayscale
    horizontal_flip=True, vertical_flip=True  # validation_split=0.2
)

##########################

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
# fit the training data in order to augment. fit parameters from data. ???
datagen.fit(X_train)

##########################

# 1 - creating and saving images locally

def generate_images_and_save(X, wanted_generated_db_size):
    i = 0

    # flow() generates batches of randomly transformed images and saves the results to the directory
    for batch in datagen.flow(X,  # batch_size=2, shuffle=True, seed=0,
                              save_to_dir='deep_learning_22_preview',
                              save_prefix='image', save_format='jpg'):
        i += len(X)
        if i >= wanted_generated_db_size:
            break  # otherwise the generator would loop indefinitely


generate_images_and_save(X_train, 10)

##########################

# 2 - creating and saving images in an array

wanted_generated_db_size = 10000

X_generated = np.empty((0, 28, 28, 1), dtype=np.float32)
Y_generated = np.empty((0,), dtype=np.int32)
for x_batch, y_batch in datagen.flow(X_train, Y_train):
    X_generated = np.append(X_generated, x_batch, axis=0)
    print(X_generated.shape)
    Y_generated = np.append(Y_generated, y_batch, axis=0)
    if len(Y_generated) >= wanted_generated_db_size:
        break

# create & compile model first

trained_model_shapes = model.fit(
    X_generated, Y_generated, validation_split=0.2,
    epochs=epochs, batch_size=batch_size
)

##########################

# 3 - creating a generator (ImageDataGenerator flow)

batch_size = 1024

# a generator that will read images from an array
generator = datagen.flow(X_train, Y_train, batch_size, shuffle=True, seed=0)

# a generator that will read images found in subfolers of 'data/train',
#   and indefinitely generate batches of augmented image data:
generator = datagen.flow_from_directory(
        'data/train',             # the target directory
        target_size=(150, 150),   # all images will be resized to 150x150 (W,H)
        batch_size=batch_size,
        class_mode='binary',      # if we use binary_crossentropy loss, we need binary labels
        subset='training')        # training \ validation

# create & compile model first

# using the generators to train our model.
# fit_generator - the number of samples processed for each epoch is batch_size * steps_per_epochs.
# Each epoch takes 20-30s on GPU and 300-400s on CPU.
# So it's definitely viable to run this model on CPU if you aren't in a hurry.
trained_model_shapes = model.fit_generator(
    generator,  # train_generator
    steps_per_epoch=50,  # train_dataset_size // batch_size, train_generator.samples // batch_size
    epochs=5,  # 10
    validation_data=generator,  # validation_generator
    validation_steps=50  # test_dataset_size // batch_size, validation_generator.samples // batch_size
)
