# Use MNIST and construct the minimal network that classify well!
# Show accuracy and loss graphs and all activation maps of the trained model. 
# What is the minimal number of parameters that is enough to achieve above 95% accuracy ?
# Answer: Total params: 63,710


from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


epochs = 5
epoch_count = range(1, epochs + 1)
classes_num = 10

#######################################################

(X_train, Y_train),(X_test, Y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

Y_train = to_categorical(Y_train, classes_num)
Y_test = to_categorical(Y_test, classes_num)

print(X_train.shape)
print(Y_train.shape)

#######################################################

inputShape=(28,28,1)
input = Input(inputShape)

x = Conv2D(64,(3,3),strides=(1,1),name='layer_conv1',padding='same')(input)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2,2),name='maxPool1')(x)

# x = Conv2D(64,(3,3),strides=(1,1),name='layer_conv2',padding='same')(x)
# x = BatchNormalization()(x)
# x = Activation('relu')(x)
# x = MaxPooling2D((2,2),name='maxPool2')(x)

x = Conv2D(32,(3,3),strides=(1,1),name='conv3',padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2,2),name='maxPool3')(x)

x = Flatten()(x)
# x = Dense(64,activation='relu',name='fc0')(x)
# x = Dropout(0.25)(x)
x = Dense(28,activation='relu',name='fc1')(x)
x = Dropout(0.25)(x)
x = Dense(classes_num,activation='softmax',name='fc2')(x)

model = Model(inputs=input,outputs=x,name='Predict')
model.summary()

# datagen_train = ImageDataGenerator(
#     width_shift_range=0.2,  # randomly shift images horizontally 
#     height_shift_range=0.2,# randomly shift images vertically 

#     horizontal_flip=True) # randomly flip images horizontally

# # fit augmented image generator on data
# datagen_train.fit(X_train)

# compile the model
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    validation_data=(X_test, Y_test),
                    epochs=5,
                    verbose=1)  # callbacks=callbacks_list

# history = model.fit_generator(datagen_train.flow(X_train, Y_train, batch_size=16), validation_data=(X_valid, Y_valid),
#                           epochs=5,steps_per_epoch=X_train.shape[0], verbose=1) #,callbacks=[checkpointer,lrate]

#######################################################

print("on valid data")
pred1 = model.evaluate(X_test, Y_test)
print("accuaracy", str(pred1[1]*100))
print("Total loss",str(pred1[0]*100))

#######################################################

# confusion matrix

Y_prediction = model.predict(X_test)
# Convert predictions classes to one hot vectors
Y_pred_classes = np.argmax(Y_prediction, axis=1)
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_test, axis=1)
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, annot=True, fmt="d")
plt.show()

#######################################################

# plotting loss

loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(epoch_count, loss, color='red', label='Training loss')
plt.plot(epoch_count, val_loss, color='green', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#######################################################

# plotting accuracy

acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epoch_count, acc, color='red', label='Training acc')
plt.plot(epoch_count, val_acc, color='green', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#######################################################

# plotting activations


def display_activation(activations, col_size, row_size, act_index):
    activation = activations[act_index]
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
    for row in range(0,row_size):
        for col in range(0,col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
            activation_index += 1
    plt.show()

selected_image = X_test[0]
plt.imshow(selected_image[:,:,0])
plt.show()

layer_outputs = [layer.output for layer in model.layers[1:]]
activation_model = Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(selected_image.reshape(1,28,28,1))
display_activation(activations, 8, 8, 1)  # Displaying above image after layer 2
display_activation(activations, 8, 8, 3)  # Displaying above image after layer 4

