from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import matplotlib.pyplot as plt
import time
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def main(_):
  # Import MNIST data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  W1 = tf.Variable(tf.truncated_normal([784, 5],stddev=0.1))
  b1 = tf.Variable(tf.constant(0.1, shape=[5]))
  z = tf.nn.sigmoid(tf.matmul(x, W1) + b1)

  W2 = tf.Variable(tf.truncated_normal([5, 784],stddev=0.1))
  b2 = tf.Variable(tf.constant(0.1, shape=[784]))
  y = tf.nn.sigmoid(tf.matmul(z, W2) + b2)


  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 784])


  Mean_Square_Error=tf.losses.mean_squared_error(y_,y)
  train_step = tf.train.AdamOptimizer(0.05).minimize(Mean_Square_Error)
  # train_step = tf.train.GradientDescentOptimizer(.1).minimize(Mean_Square_Error)

  sess = tf.InteractiveSession()

  tf.global_variables_initializer().run()
  # Train
  for i in range(100000):
    batch_xs, batch_ys = mnist.train.next_batch(1000)

    if i % 1000 == 0:
      X = np.reshape(batch_xs[1, :], [28, 28])
      Y = sess.run(y, feed_dict={x: batch_xs})
      Y = np.reshape(Y[1, :], [28, 28])
      plt.ion()
      plt.subplot(211)
      plt.imshow(X)
      plt.subplot(212)
      plt.imshow(Y)
      plt.pause(.00001)
      sess.run(train_step, feed_dict={x: batch_xs, y_: batch_xs})
      print('Train MSE', sess.run(Mean_Square_Error, feed_dict={x: batch_xs, y_: batch_xs}))
      print('Test MSE', sess.run(Mean_Square_Error, feed_dict={x: mnist.test.images,y_: mnist.test.images}))
      print('iteration=',i)

  # Test trained model
  # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  # print(sess.run(accuracy, feed_dict={x: mnist.test.images,
  #                                     y_: mnist.test.labels}))
  print('Test Mean Square Error',sess.run(Mean_Square_Error, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.images}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)