import pandas as  pd
import numpy as np
import statsmodels.api as sm

from returns_data_logistic_regression import read_goog_sp500_logistic_data

xData, yData = read_goog_sp500_logistic_data()

logit = sm.Logit(yData, xData)

result = logit.fit()

predictions = (result.predict(xData) > 0.5)

num_accurate_predictions = (list(yData == predictions)).count(True)

pctAccuracy = float(num_accurate_predictions)/float(len(predictions))

print("Accuracy: ", pctAccuracy)

#################################################################
### Logistic Regression using tensorflow#########################
#################################################################

import tensorflow as tf

W = tf.Variable(tf.ones([1,2]), name="W")
b = tf.Variable(tf.zeros([2]), name="b")

x = tf.placeholder(tf.float32, [None,1], name="x")
y_ = tf.placeholder(tf.float32, [None,2], name="y_")

y = tf.matmul(x,W) + b

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

all_xs = np.expand_dims(xData[:,0], axis=1)
#One hot representation
all_ys = np.array([([1,0] if yEl == True else [0,1]) for yEl in yData])

dataset_size = len(all_xs)

def trainWithMultiplePointsPerEpoch(steps, train_step, batch_size):
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for i in range(steps):
            if dataset_size == batch_size:
                batch_start_idx = 0
            elif dataset_size < batch_size:
                raise ValueError("dataset_size: %d, must be greater than batch_size: %d" % (dataset_size, batch_size))
            else:
                batch_start_idx = (i * batch_size) % (dataset_size)

            batch_end_idx = batch_start_idx + batch_size

            batch_xs = all_xs[batch_start_idx : batch_end_idx]
            batch_ys = all_ys[batch_start_idx : batch_end_idx]
            feed = {x: batch_xs, y_: batch_ys}
            sess.run(train_step, feed_dict=feed)

            if(i+1)%1000 == 0:
                print("After %d iterations:" % i)
                print(sess.run(W))
                print(sess.run(b))
                print("cross entropy: %f" % sess.run(cross_entropy, feed_dict=feed))
        correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Accuracy: %f" % sess.run(accuracy, feed_dict={x: all_xs, y_: all_ys}))

trainWithMultiplePointsPerEpoch(20000, train_step, dataset_size)
