import numpy as np
from sklearn import datasets, linear_model
from returns_data import read_goog_sp500_data

xData, yData = read_goog_sp500_data()

googModel = linear_model.LinearRegression()

googModel.fit(xData.reshape(-1,1), yData.reshape(-1,1))

print (googModel.coef_)
print (googModel.intercept_)

#####################################################
#Using TensorFlow#
#####################################################
import tensorflow as tf

W = tf.Variable(tf.zeros([1,1]))
b = tf.Variable(tf.zeros([1]))

x = tf.placeholder(tf.float32, [None,1])

Wx = tf.matmul(x, W)

y = Wx + b

W_hist = tf.summary.histogram("weights", W)
b_hist = tf.summary.histogram("biases", b)
y_hist = tf.summary.histogram("y", y)

y_ = tf.placeholder(tf.float32, [None, 1])

cost = tf.reduce_mean(tf.square(y_-y))
cost_hist = tf.summary.histogram("cost", cost)

train_step_ftrl = tf.train.FtrlOptimizer(1).minimize(cost)

dataset_size = len(xData)

def trainWithMultiplePointPerEpoch(steps, train_step, batch_size):
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter('./lrlog', sess.graph)

        for i in range(steps):

            if dataset_size == batch_size:
                batch_start_idx = 0
            elif dataset_size < batch_size:
                raise ValueError("dataset_size: %d, must be greater than batch_size: %d" % (dataset_size, batch_size))
            else:
                batch_start_idx = (i * batch_size) % (dataset_size)

            batch_end_idx = batch_start_idx + batch_size

            batch_xs = xData[batch_start_idx : batch_end_idx]
            batch_ys = yData[batch_start_idx : batch_end_idx]

            feed = {x: batch_xs.reshape(-1,1), y_: batch_ys.reshape(-1,1)}

            sess.run(train_step, feed_dict=feed)

            result = sess.run(merged_summary, feed_dict=feed)
            writer.add_summary(result, i)

            if(i+1) % 500 == 0:
                print("After %d iterations:" % i)

                print("W: %f" % sess.run(W))
                print("b: %f" % sess.run(b))

                print("cost: %f" % sess.run(cost, feed_dict=feed))
        writer.close()

trainWithMultiplePointPerEpoch(5000, train_step_ftrl, 10)
