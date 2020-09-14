import tensorflow as tf
import numpy as np

# train Parameters
seq_length = 4
data_dim = 42
hidden_dim = 20
output_dim = 5
learning_rate = 0.001
iterations = 1500

x_data = np.transpose(np.loadtxt('rnn_train.txt', unpack = True))
x_data = np.reshape(x_data, (-1,seq_length,42))
y_data = np.transpose(np.loadtxt('rnn_label_tensorflow.txt', unpack=True))

trainX = x_data[0]
trainY = y_data[0]
trainX = np.reshape(trainX, (1,4,42))

trainY = np.reshape(trainY, (1,5))

# input place holders
X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, output_dim])

# build a LSTM network
cell = tf.contrib.rnn.BasicLSTMCell(
    num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
Y_pred = tf.contrib.layers.fully_connected(
    outputs[:, -1], output_dim, activation_fn=None)  # We use the last cell's output

saver = tf.train.Saver()

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    saver.restore(sess, "./tmp/model.ckpt")
    #
    # correct_prediction = tf.equal(tf.argmax(Y_pred ,1) ,tf.argmax(Y ,1))
    #
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # train_accuracy = sess.run(accuracy, feed_dict={X: trainX, Y: trainY})
    # print("training accuracy %g" % ( train_accuracy))
    # print(np.dtype(train_accuracy))
    result = sess.run(Y_pred, feed_dict = {X: trainX})
    print(result)