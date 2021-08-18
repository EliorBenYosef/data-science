# 50. Chest X-Ray Image
# https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia/home

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
# from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation


kaggle = True

# create generator
datagen = ImageDataGenerator(rescale=1./255)

# load and iterate datasets:
# prepare an iterators for each dataset:
# it \ generator
parent_dir = '../input/chest_xray/chest_xray/' if kaggle else 'C:/DBs/chest_xray/'
batch_size = 8
class_mode = 'binary'
target_size=(150, 150)  # all images will be resized to 150x150 (W,H)
train_it = datagen.flow_from_directory(parent_dir + 'train/', class_mode=class_mode, batch_size=batch_size)
val_it = datagen.flow_from_directory(parent_dir + 'val/', class_mode=class_mode, batch_size=batch_size)
test_it = datagen.flow_from_directory(parent_dir + 'test/', class_mode=class_mode, batch_size=batch_size)

# confirm the iterator works
batchX, batchy = train_it.next()
print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))

nodes = 8
kernel_size = (3, 3)
pool_size=(2, 2)
model = Sequential([
    Conv2D(nodes, kernel_size=kernel_size, activation='relu', input_shape=(256,256,3)),  #
    Conv2D(nodes, kernel_size=kernel_size, activation='relu'),
    # Conv2D(nodes, kernel_size=kernel_size, activation='relu'),
    # Conv2D(nodes, kernel_size=kernel_size, activation='relu'),
    # MaxPooling2D(pool_size=pool_size),
    Flatten(),  #input_shape=X.shape[1:3]
    Dense(2, activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # binary_crossentropy

# model.fit(batchX, batchy, epochs=5, verbose=1)

# model.fit_generator(train_it, steps_per_epoch=16, validation_data=val_it, validation_steps=8)
model.fit_generator(train_it, steps_per_epoch=train_it.samples // batch_size,
                    validation_data=val_it, validation_steps=val_it.samples // batch_size,
                    epochs=5)
loss = model.evaluate_generator(test_it, steps=24)
yhat = model.predict_generator(test_it, steps=24)
print('loss', loss)
print('yhat', yhat)
