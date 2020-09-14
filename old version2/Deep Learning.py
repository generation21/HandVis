
# Lab 10 MNIST and Deep learning
import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt

# def MinMaxScaler(data):
#    numerator = data - np.min(data, 0)
#    denominator = np.max(data, 0) - np.min(data,0)
#    return numerator / (denominator + 1e-7)

xy = np.loadtxt("train_verstion.txt", unpack = True)
print(xy.shape)
x_data = np.transpose(xy)
# x_data = x_data.flatten()
# x_data = np.array([x_data])

y_data = np.loadtxt("label.txt")
# y_data = y_data[0]
print(x_data.shape)
print(y_data.shape)
# for i in range(99 ,xy[0].size ,99):
#     ans_x = np.transpose(xy[0:13 ,i: 99 +i])
#     ans_x = ans_x.flatten()
#     ans_x = np.array([ans_x])
#     x_data = np.concatenate((x_data, ans_x), 0)
# y_data = np.array([y_data])
# for i in range(99, xy[0].size, 99):
#     ans_y = np.transpose(xy[13: ,i])
#     ans_y = np.array([ans_y])
#     y_data = np.concatenate((y_data ,ans_y), 0)



x_train = tf.placeholder(tf.float32)
y_train = tf.placeholder(tf.float32)

DropoutRate = tf.placeholder(tf.float32)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.02)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.02, shape = shape)
    return tf.Variable(initial)

with tf.name_scope("layer1") as scope:
    W1 = weight_variable([42, 42])
    b1 = bias_variable([42])
    tf.summary.histogram("Weight", W1)
    tf.summary.histogram("bias", b1)

    _h1 = tf.nn.relu(tf.matmul(x_train ,W1 )+ b1)
    h1 = tf.nn.dropout(_h1, DropoutRate)
    tf.summary.histogram("Layer", h1)

with tf.name_scope("layer2") as scope:
    W2 = weight_variable([42, 21])
    b2 = bias_variable([21])
    tf.summary.histogram("Weight", W2)
    tf.summary.histogram("bias", b2)

    _h2 = tf.nn.relu(tf.matmul(h1 ,W2) + b2)
    h2 = tf.nn.dropout(_h2, DropoutRate)
    tf.summary.histogram("Layer", h2)

with tf.name_scope("layer3") as scope:
   W3 = weight_variable([21, 21])
   b3 = bias_variable([21])
   tf.summary.histogram("Weight3", W3)
   tf.summary.histogram("biases3", b3)

   _h3 = tf.nn.relu(tf.matmul(h2, W3) + b3)
   h3 = tf.nn.dropout(_h3, DropoutRate)
   tf.summary.histogram("Layer", h3)


with tf.name_scope("layer4") as scope:
    W4 = weight_variable([21, 5])
    b4 = bias_variable([5])
    tf.summary.histogram("Weight3", W4)
    tf.summary.histogram("biases3" ,b4)

    hypothesis = tf.matmul(h3, W4) + b4
    tf.summary.histogram("hypothesis", hypothesis)
y_ans = tf.nn.softmax(hypothesis)
# cost/ loss function
with tf.name_scope("cost") as scope:
    cross_entropy = tf.reduce_mean(-y_train * tf.log(tf.clip_by_value(y_ans, 1e-10, 10.0)) - (1 - y_train) * tf.log(tf.clip_by_value(1 - y_ans, 1e-10, 10.0)))
    # cross_entropy = tf.reduce_mean(-y_data * tf.log(tf.clip_by_value(y_ans, 1e-10, 10.0)) - ( 1 -y_data ) *tf.log(tf.clip_by_value( 1 -y_ans,1e-10, 10.0)))
    # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_ans , labels=y_train))
    cost_summ = tf.summary.scalar("cost", cross_entropy)
with tf.name_scope("train") as scope:
    train_step = optimizer = tf.train.AdamOptimizer(0.0003).minimize(cross_entropy)

# Accruacy computation
correct_prediction = tf.equal(tf.argmax(y_ans ,1) ,tf.argmax(y_train ,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar("accuracy", accuracy)
# train_x_batch, train_y_batch = tf.train.batch([np.transpose(xy[0:13]) ,np.transpose(xy[13:])], batch_size = 100)

# Launch graph
with tf.Session() as sess:
    # tensorboard --logdir=/logs/tf_logs
    # merged_summary = tf.summary.merge_all()
    # writer = tf.summary.FileWriter("/logs/tf_logs0")
    # writer.add_graph(sess.graph)

    sess.run(tf.global_variables_initializer())

    for i in range(2000):


        # summary, _ = sess.run([merged_summary, train_step], feed_dict = {x_train : x_data, y_train : y_data, DropoutRate : 0.7})
        # writer.add_summary(summary ,i)
        sess.run(train_step, feed_dict = {x_train : x_data, y_train : y_data, DropoutRate : 0.7})
        if i % 10 == 0:
            train_accuracy = sess.run(accuracy, feed_dict = {
                x_train : x_data, y_train : y_data, DropoutRate : 1.0})

            print("step %d, training accuracy %g" %(i, train_accuracy))

    np.savetxt("W1.txt", sess.run(W1))
    np.savetxt("b1.txt", sess.run(b1))
    np.savetxt("W2.txt", sess.run(W2))
    np.savetxt("b2.txt", sess.run(b2))
    np.savetxt("W3.txt", sess.run(W3))
    np.savetxt("b3.txt", sess.run(b3))
    np.savetxt("W4.txt", sess.run(W4))
    np.savetxt("b4.txt", sess.run(b4))
    #
    # test = np.loadtxt("test.txt", unpack=True)
    #
    # x_test = np.transpose(test[0:126])
    #
    # y_test = np.transpose(test[126:])
    #
    # print("test accuracy %g" % sess.run(accuracy, feed_dict={x_train: x_test, y_train: y_test, DropoutRate: 1.0}))

