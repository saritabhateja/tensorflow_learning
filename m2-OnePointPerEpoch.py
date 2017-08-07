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

train_step_constant = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

def trainWithOnePointPerEpoch(steps, train_step):
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter('./lrlog', sess.graph)

        for i in range(steps):

            xs = np.array([[xData[i % len(xData)]]])
            ys = np.array([[yData[i % len(yData)]]])

            feed = {x: xs, y_: ys}

            sess.run(train_step, feed_dict=feed)

            result = sess.run(merged_summary, feed_dict=feed)
            writer.add_summary(result, i)

            if(i+1)%1000 == 0:
                print("After %d iterations:" % i)

                print("W: %f" % sess.run(W))
                print("b: %f" % sess.run(b))

                print("cost: %f" % sess.run(cost, feed_dict=feed))
        writer.close()

trainWithOnePointPerEpoch(10000, train_step_constant)
