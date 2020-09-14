import tensorflow as tf
import numpy as np
import matplotlib
import os

tf.set_random_seed(777)  # reproducibility

if "DISPLAY" not in os.environ:
    # remove Travis CI Error
    matplotlib.use('Agg')

import matplotlib.pyplot as plt


def MinMaxScaler(data):
    ''' Min Max Normalization
    Parameters
    ----------
    data : numpy.ndarray
        input data to be normalized
        shape: [Batch size, dimension]
    Returns
    ----------
    data : numpy.ndarry
        normalized data
        shape: [Batch size, dimension]
    References
    ----------
    .. [1] http://sebastianraschka.com/Articles/2014_about_feature_scaling.html
    '''
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)


# train Parameters
seq_length = 4
data_dim = 42
hidden_dim = 20
output_dim = 6
learning_rate = 0.001
iterations = 2000

x_data = np.transpose(np.loadtxt('rnn_train.txt', unpack = True))
x_data = np.reshape(x_data, (-1,seq_length,data_dim))
y_data = np.transpose(np.loadtxt('rnn_label_tensorflow.txt', unpack=True))

trainX = x_data
trainY = y_data
testX = x_data
testY = y_data
# input place holders
X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, output_dim])

# build a LSTM network
cell = tf.contrib.rnn.BasicLSTMCell(
    num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
Y_pred = tf.contrib.layers.fully_connected(
    outputs[:, -1], output_dim, activation_fn=None)  # We use the last cell's output

# cost/loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y_pred , labels=Y))
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

saver = tf.train.Saver()

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # saver.save(sess, 'my-model', global_step=iterations-1, write_meta_graph=False)

    # Training step
    for i in range(iterations):
        _, step_loss = sess.run([train, loss], feed_dict={ X: trainX, Y: trainY})
        print("[step: {}] loss: {}".format(i, step_loss))

    save_path = saver.save(sess, "./tmp/model.ckpt")
    print("Model saved in path: %s" % save_path)

    # Plot predictions

    correct_prediction = tf.equal(tf.argmax(Y_pred ,1) ,tf.argmax(Y ,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    train_accuracy = sess.run(accuracy, feed_dict={X: trainX, Y: trainY})
    print("step %d, training accuracy %g" % (i, train_accuracy))