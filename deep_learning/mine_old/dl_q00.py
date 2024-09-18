import tensorflow as tf
from tensorflow.python.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

train_mask = np.isin(Y_train, [0])
test_mask = np.isin(Y_test, [0])
X_train, Y_train = X_train[train_mask], Y_train[train_mask]
X_test, Y_test = X_test[test_mask], Y_test[test_mask]

X_train = X_train[:1000]
Y_train = Y_train[:1000]
X_test = X_test[:1000]
Y_test = Y_test[:1000]

X_train = X_train / 255.0
X_test = X_test / 255.0


Y_train_oh = pd.get_dummies(Y_train)
Y_test_oh = pd.get_dummies(Y_test)

# Reshaping the array to 4-dims so that it can work with the Keras API
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)


X = tf.placeholder(tf.float32, shape=[None, X_train.shape[1], X_train.shape[2], 1], name='X')
Y = tf.placeholder(tf.float32, shape=[None, 1], name='Y')
print('X shape:', X.shape)
print('Y shape:', Y.shape)

# building the CNN model
#input_layer = tf.reshape(X_train, [-1, 28, 28, 1])
input_layer = tf.cast(X_train, tf.float64)

# Convolutional Layer #1
conv1 = tf.layers.conv2d(
  inputs=input_layer,
  filters=32,
  kernel_size=[5, 5],
  padding="same",
  activation='relu')

# Pooling Layer #1
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

# fc layer #1
fc1 = tf.reshape(pool1, [-1, 14*14*32])

# Logits Layer
logits = tf.layers.dense(inputs=fc1, units=10)

# Calculate Loss (for both TRAIN and EVAL modes)
loss = tf.losses.sparse_softmax_cross_entropy(labels=Y_train.astype('int32'), logits=logits)

# Configure the Training Op (for TRAIN mode)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
init = tf.global_variables_initializer()

print(X_train.shape)
print(Y_train_oh.shape)
print(X.shape)
print(Y.shape)

# Training
cost_list = []
acc_list = []
epochs = 2
batch = 100
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(0, epochs):
        for i in range(0, len(X_train), batch): 
            y_pred = tf.nn.softmax(logits[i:i+batch], name="y_pred")
            y_pred_cls = tf.argmax(y_pred, dimension=1)
            # get the probability of the class of each ground truth
            y_true_cls = tf.argmax(Y, dimension=1)
            # prediction Accuracy:
            correct_prediction = tf.equal(y_pred_cls, y_true_cls)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
            sess.run(optimizer, feed_dict={X: X_train[i:i+batch], Y: Y_train_oh[i:i+batch]})
            cost = sess.run(loss, feed_dict={X: X_train[i:i+batch], Y: Y_train_oh[i:i+batch]})
            cost_list.append(cost)
            acc = sess.run(accuracy, feed_dict={X: X_train[i:i+batch], Y: Y_train_oh[i:i+batch]})
            acc_list.append(acc)
        print("Epoch:", '%03d' % (epoch + 1), "Accuracy=", "{:.6f}".format(acc) ,"Loss=", "{:.6f}".format(cost))
    # making the prediction
    pred = sess.run(y_pred_cls, feed_dict={X: X_test[0:1], Y: Y_test_oh[0:1]})
    print(pred)

plt.plot(cost_list)
plt.show()

plt.plot(acc_list)
plt.show()
