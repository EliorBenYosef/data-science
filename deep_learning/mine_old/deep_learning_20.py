# https://keras.io/losses/

# Test performances with several loss functions:
# best - binary_crossentropy, categorical_crossentropy, sparse_categorical_crossentropy, hinge, logcosh
# intermittent - mean_squared_logarithmic_error, categorical_hinge, kullback_leibler_divergence
# worst - mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, squared_hinge, poisson, cosine_proximity


from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, Flatten, Dense
from tensorflow.python.keras.losses import categorical_hinge
import matplotlib.pyplot as plt


# Base hyper-parameters
epochs = 5
optimizer = 'adam'  # 'sgd' 'adagrad' 'adadelta' 'rmsprop'


##########################################


def compile_and_train_model(loss):
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', loss])
    if loss == 'binary_crossentropy' or loss == 'categorical_crossentropy':
        return model.fit(X_train, Y_train_cc, validation_data=(X_test, Y_test_cc), epochs=epochs)
    else:
        return model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs)


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
    fig.savefig('results/DL20_models_accuracy.jpg')
    plt.close(fig)


def plot_losses(trained_models):
    fig = plt.figure(figsize=(16, 10))

    max_epochs = 0
    for name, trained_model in trained_models:
        epoch_count = [i + 1 for i in trained_model.epoch]
        epochs = max(epoch_count)
        if epochs > max_epochs:
            max_epochs = epochs
        val = plt.plot(epoch_count, trained_model.history['val_loss'], label=name.title() + ' Validation')
        plt.plot(epoch_count, trained_model.history['loss'], label=name.title() + ' Training',
                 linestyle='--', color=val[0].get_color())

    plt.legend(loc=0)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xlim([1, max_epochs])
    plt.xticks(range(1, max_epochs + 1))
    plt.grid(True)
    plt.title("Model Loss")
    plt.show()
    fig.savefig('results/DL20_models_loss.jpg')
    plt.close(fig)


##########################################

# use MNIST

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

# 'categorical_crossentropy' expects binary (class) matrices targets (1s and 0s) of shape (samples, classes).
Y_train_cc = to_categorical(Y_train, classes_num)
Y_test_cc = to_categorical(Y_test, classes_num)

############################

# Construct a 5 layers network

model = Sequential([
    Conv2D(4, kernel_size=(3, 3), activation='relu', input_shape=X_train.shape[1:]),
    Conv2D(4, kernel_size=(3, 3), activation='relu'),
    Conv2D(4, kernel_size=(3, 3), activation='relu'),
    Conv2D(4, kernel_size=(3, 3), activation='relu'),
    Flatten(input_shape=X_train.shape[1:3]),
    Dense(classes_num, activation='softmax')
])

############################

# cross-entropy = Softmax
trained_model_bc = compile_and_train_model('binary_crossentropy')
trained_model_cc = compile_and_train_model('categorical_crossentropy')
trained_model_scc = compile_and_train_model('sparse_categorical_crossentropy')

trained_model_mae = compile_and_train_model('mean_absolute_error')
trained_model_mape = compile_and_train_model('mean_absolute_percentage_error')
trained_model_mse = compile_and_train_model('mean_squared_error')
trained_model_msle = compile_and_train_model('mean_squared_logarithmic_error')

# hinge = SVM
trained_model_h = compile_and_train_model('hinge')
trained_model_ch = compile_and_train_model(categorical_hinge)  # 'categorical_hinge' causes: Unknown metric function:categorical_hinge
trained_model_sh = compile_and_train_model('squared_hinge')

trained_model_l = compile_and_train_model('logcosh')
trained_model_kld = compile_and_train_model('kullback_leibler_divergence')
trained_model_p = compile_and_train_model('poisson')
trained_model_cp = compile_and_train_model('cosine_proximity')

trained_models = [
    ('binary_crossentropy', trained_model_bc),
    ('categorical_crossentropy', trained_model_cc),
    ('sparse_categorical_crossentropy', trained_model_scc),

    ('mean_absolute_error', trained_model_mae),
    ('mean_absolute_percentage_error', trained_model_mape),
    ('mean_squared_error', trained_model_mse),
    ('mean_squared_logarithmic_error', trained_model_msle),

    ('hinge', trained_model_h),
    ('categorical_hinge', trained_model_ch),
    ('squared_hinge', trained_model_sh),

    ('logcosh', trained_model_l),
    ('kullback_leibler_divergence', trained_model_kld),
    ('poisson', trained_model_p),
    ('cosine_proximity', trained_model_cp)
]

plot_accuracies(trained_models)
plot_losses(trained_models)

############################

trained_models_best = [
    ('binary_crossentropy', trained_model_bc),
    ('categorical_crossentropy', trained_model_cc),
    ('sparse_categorical_crossentropy', trained_model_scc),
    ('hinge', trained_model_h),
    ('logcosh', trained_model_l)
]
trained_models_intermittent = [
    ('mean_squared_logarithmic_error', trained_model_msle),
    ('categorical_hinge', trained_model_ch),
    ('kullback_leibler_divergence', trained_model_kld)
]
trained_models_worst = [
    ('mean_absolute_error', trained_model_mae),
    ('mean_absolute_percentage_error', trained_model_mape),
    ('mean_squared_error', trained_model_mse),
    ('squared_hinge', trained_model_sh),
    ('poisson', trained_model_p),
    ('cosine_proximity', trained_model_cp)
]
