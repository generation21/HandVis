# Lab 12 RNN
import tensorflow as tf
import numpy as np

# Teach hello: hihell -> ihello
x_data = np.transpose(np.loadtxt('rnn_train.txt', unpack = True))
x_data = np.reshape(x_data, (-1,4,42))
y_data = np.transpose(np.loadtxt('rnn_label_tensorflow.txt', unpack=True))
print(np.shape(y_data))

timesteps = seq_length = 4
data_dim = 42
output_dim = 5
hidden_dim = 40
# Open,High,Low,Close,Volume

# xy = np.loadtxt('data-02-stock_daily.csv', delimiter=',')
# xy = xy[::-1]  # reverse order (chronically ordered)
# xy = MinMaxScaler(xy)
# x = xy
# y = xy[:, [-1]]  # Close as label
x = x_data
y = y_data

#
# dataX = []
#
# dataY = []
#
# for i in range(0, len(y) - seq_length):
#
#    _x = x[i:i + seq_length]
#
#    _y = y[i + seq_length]  # Next close price
#
#    # print(_x, "->", _y)
#
#    dataX.append(_x)
#
#    dataY.append(_y)



# 데이터를 MinMaxScaler를 합니다. 7일씩 잘라서 x의 값과 예측하려는 y의 값을 리스트에 넣습니다.

trainX = x_data
trainY = y_data
testX = x_data
testY = y_data
# split to train and testing
#
# train_size = int(len(dataY)[0] * 0.7)
#
# test_size = len(dataY) - train_size
#
# trainX, testX = np.array(dataX[0:train_size]),  np.array(dataX[train_size:len(dataX)])
#
# trainY, testY = np.array(dataY[0:train_size]), np.array(dataY[train_size:len(dataY)])



# input placeholders

X = tf.placeholder(tf.float32, [None, seq_length, data_dim])

Y = tf.placeholder(tf.float32, [None, 5])



# 70%를 training data로 사용하고 나머지를 test로 사용합니다.


#
# # input placeholders
#
# X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
#
# Y = tf.placeholder(tf.float32, [None, 1])
#


cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True)

outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

Y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], output_dim, activation_fn=None)

   # We use the last cell's output



# cost/loss

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y_pred , labels=Y))
# loss = tf.reduce_mean(-Y * tf.log(tf.clip_by_value(Y_pred, 1e-10, 10.0)) - ( 1 -Y ) *tf.log(tf.clip_by_value( 1 -Y_pred,1e-10, 10.0)))
# optimizer1)
# train = optimizer.minimize(loss)
train = tf.train.AdamOptimizer(0.001).minimize(loss)
# optimizer = tf.train.AdamOptimizer(0.0


# Cell을 만들고 이전에 배웠듯이 FC layer를 추가해 줍니다. output에서 마지막 데이터만 사용할 것이기 때문에 outputs[ : , -1] 을 사용합니다. loss에서 위의 모델의 output이 sequence data가 아니기 때문에 하나의 linear loss이기때문에 mean square error를 사용합니다. 그다음 이전처럼 optimizer를 정의하고 train을 정의합니다.

sess = tf.Session()
sess.run(tf.global_variables_initializer())
batch_size = 520
for i in range(2000):
    for start, end in zip(range(0, len(trainX), batch_size), range(batch_size, len(trainX) + 1, batch_size)):
        # sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
        _, l = sess.run([train, loss],  feed_dict={X: trainX[start:end], Y: trainY[start:end]})
    print(i, l)
    # if i % 10 == 0:
    #     train_accuracy = sess.run(accuracy, feed_dict={X: x_data, Y: y_data})
    #     print("step %d, training accuracy %g" % (i, train_accuracy))
# Accruacy computation
correct_prediction = tf.equal(tf.argmax(Y_pred ,1) ,tf.argmax(Y ,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
train_accuracy = sess.run(accuracy, feed_dict={X: x_data, Y: y_data})
print("step %d, training accuracy %g" % (i, train_accuracy))
testPredict = sess.run(Y_pred, feed_dict={X: testX})




