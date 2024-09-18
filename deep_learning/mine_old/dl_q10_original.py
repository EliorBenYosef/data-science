# -*- coding: utf-8 -*-
"""Q10_DL_keras_final.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Va7sIVNCC_Pnl3ir-5Hje3TBF8bmlhMu
"""

# Use MNIST and construct the minimal network that classify well! 
# Show accuracy and loss graphs and all activation maps of the trained model. 
# What is the minimal number of parameters that is enough to achieve above 95% accuracy ?
# Answer: Total params: 63,710

from keras.layers import Input,InputLayer, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout
from keras.models import Sequential,Model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint,LearningRateScheduler
import keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

mnist = keras.datasets.mnist
(X_train, Y_train),(X_test, Y_test) = mnist.load_data()

X_train, X_test = X_train / 255.0, X_test / 255.0

X_train = np.reshape(X_train ,[-1, 28, 28, 1])
X_test = np.reshape(X_test ,[-1, 28, 28, 1])

print(X_train.shape)

Y_train = to_categorical(Y_train, num_classes = 10)
print(Y_train.shape)

X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size = 0.2, random_state=42)

inputShape=(28,28,1)
input = Input(inputShape)

x = Conv2D(64,(3,3),strides = (1,1),name='layer_conv1',padding='same')(input)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2,2),name='maxPool1')(x)

# x = Conv2D(64,(3,3),strides = (1,1),name='layer_conv2',padding='same')(x)
# x = BatchNormalization()(x)
# x = Activation('relu')(x)
# x = MaxPooling2D((2,2),name='maxPool2')(x)

x = Conv2D(32,(3,3),strides = (1,1),name='conv3',padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2,2),name='maxPool3')(x)


x = Flatten()(x)
# x = Dense(64,activation = 'relu',name='fc0')(x)
# x = Dropout(0.25)(x)
x = Dense(28,activation = 'relu',name='fc1')(x)
x = Dropout(0.25)(x)
x = Dense(10,activation = 'softmax',name='fc2')(x)

model = Model(inputs = input,outputs = x,name='Predict')

model.summary()

# datagen_train = ImageDataGenerator(
#     width_shift_range=0.2,  # randomly shift images horizontally 
#     height_shift_range=0.2,# randomly shift images vertically 

#     horizontal_flip=True) # randomly flip images horizontally

# # fit augmented image generator on data
# datagen_train.fit(X_train)

# compile the model
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

history=model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid),
                          epochs=5,verbose=1) #callbacks=callbacks_list

# history - model.fit_generator(datagen_train.flow(X_train, Y_train, batch_size=16), validation_data=(X_valid, Y_valid),
#                           epochs=5,steps_per_epoch=X_train.shape[0], verbose=1) #,callbacks=[checkpointer,lrate]

# plotting training and validation loss

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, color='red', label='Training loss')
plt.plot(epochs, val_loss, color='green', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# plotting training and validation accuracy

acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, color='red', label='Training acc')
plt.plot(epochs, val_acc, color='green', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

print("on valid data")
pred1=model.evaluate(X_valid,Y_valid)
print("accuaracy", str(pred1[1]*100))
print("Total loss",str(pred1[0]*100))

layer_outputs = [layer.output for layer in model.layers[1:]]
activation_model = Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(X_train[100].reshape(1,28,28,1))

def display_activation(activations, col_size, row_size, act_index): 
    activation = activations[act_index]
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
    for row in range(0,row_size):
        for col in range(0,col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
            activation_index += 1

plt.imshow(X_train[100][:,:,0]);

# Desplaying above image after layer 2
display_activation(activations, 8, 8, 1)

# Desplaying above image after layer 4
display_activation(activations, 8, 8, 3)

# confusion matrix

Y_prediction = model.predict(X_valid)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_prediction,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_valid,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)

plt.figure(figsize=(10,8))
sns.heatmap(confusion_mtx, annot=True, fmt="d");