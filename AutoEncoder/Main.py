from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import matplotlib.pyplot as plt
import time
import numpy as np
import scipy.io as sio

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def main(_):
  # Import MNIST data
  
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  # Set defualt MNIST samples to show in training
  X0=mnist.train._images[7:8,:]
  X1=mnist.train._images[4:5,:]
  X2=mnist.train._images[13:14,:]
  X3=mnist.train._images[1:2,:]
  X4=mnist.train._images[2:3,:]
  X5=mnist.train._images[28:29,:]
  X6=mnist.train._images[3:4,:]
  X7=mnist.train._images[14:15,:]
  X8=mnist.train._images[5:6,:]
  X9=mnist.train._images[8:9,:]
  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  W1 = tf.Variable(tf.truncated_normal([784, 30],stddev=0.1))
  b1 = tf.Variable(tf.truncated_normal([1, 30], stddev=0.1))
  # b1 = tf.Variable(tf.constant(.1,shape=[30]))
  z = tf.nn.sigmoid(tf.matmul(x, W1) + b1)

  W2 = tf.Variable(tf.truncated_normal([30, 784],stddev=0.1))
  b2 = tf.Variable(tf.truncated_normal([1, 784], stddev=0.1))
  y = tf.nn.sigmoid(tf.matmul(z, W2) + b2)


  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 784])


  Mean_Square_Error=tf.losses.mean_squared_error(y_,y)
  train_step = tf.train.AdamOptimizer(0.01).minimize(Mean_Square_Error)
  # train_step = tf.train.GradientDescentOptimizer(.01).minimize(Mean_Square_Error)
  init=tf.initialize_all_variables()
  sess = tf.Session()
  sess.run(init)

  writer = tf.summary.FileWriter('./tmp')
  summary_accuracy = tf.summary.scalar("accuracy", Mean_Square_Error)

  for i in range(20000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_xs})
    # summary (comment these lines and uncomment those in below "if", if you see a drop in speed
    Summary = sess.run(summary_accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.images})
    writer.add_summary(Summary, i)

    if i % 1000 == 0:
      # show original and constructed image
      x0 = np.reshape(X0, [28, 28])
      y0 = sess.run(y, feed_dict={x: X0})
      y0 = np.reshape(y0, [28, 28])

      x1 = np.reshape(X1, [28, 28])
      y1 = sess.run(y, feed_dict={x: X1})
      y1 = np.reshape(y1, [28, 28])

      x2 = np.reshape(X2, [28, 28])
      y2 = sess.run(y, feed_dict={x: X2})
      y2 = np.reshape(y2, [28, 28])

      x3 = np.reshape(X3, [28, 28])
      y3 = sess.run(y, feed_dict={x: X3})
      y3 = np.reshape(y3, [28, 28])

      x4 = np.reshape(X4, [28, 28])
      y4 = sess.run(y, feed_dict={x: X4})
      y4 = np.reshape(y4, [28, 28])

      x5 = np.reshape(X5, [28, 28])
      y5 = sess.run(y, feed_dict={x: X5})
      y5 = np.reshape(y5, [28, 28])

      x6 = np.reshape(X6, [28, 28])
      y6 = sess.run(y, feed_dict={x: X6})
      y6 = np.reshape(y6, [28, 28])

      x7 = np.reshape(X7, [28, 28])
      y7 = sess.run(y, feed_dict={x: X7})
      y7 = np.reshape(y7, [28, 28])

      x8 = np.reshape(X8, [28, 28])
      y8 = sess.run(y, feed_dict={x: X8})
      y8 = np.reshape(y8, [28, 28])

      x9 = np.reshape(X9, [28, 28])
      y9 = sess.run(y, feed_dict={x: X9})
      y9 = np.reshape(y9, [28, 28])

      plt.figure(1)
      plt.ion()
      plt.subplot(2,10,1)
      plt.imshow(x0)
      plt.subplot(2,10,11)
      plt.imshow(y0)

      plt.subplot(2,10,2)
      plt.imshow(x1)
      plt.subplot(2,10,12)
      plt.imshow(y1)

      plt.subplot(2,10,3)
      plt.imshow(x2)
      plt.subplot(2,10,13)
      plt.imshow(y2)

      plt.subplot(2,10,4)
      plt.imshow(x3)
      plt.subplot(2,10,14)
      plt.imshow(y3)

      plt.subplot(2,10,5)
      plt.imshow(x4)
      plt.subplot(2,10,15)
      plt.imshow(y4)

      plt.subplot(2,10,6)
      plt.imshow(x5)
      plt.subplot(2,10,16)
      plt.imshow(y5)

      plt.subplot(2,10,7)
      plt.imshow(x6)
      plt.subplot(2,10,17)
      plt.imshow(y6)

      plt.subplot(2,10,8)
      plt.imshow(x7)
      plt.subplot(2,10,18)
      plt.imshow(y7)

      plt.subplot(2,10,9)
      plt.imshow(x8)
      plt.subplot(2,10,19)
      plt.imshow(y8)

      plt.subplot(2,10,10)
      plt.imshow(x9)
      plt.subplot(2,10,20)
      plt.imshow(y9)

      plt.pause(.01)

      W = (sess.run(W1))
      plt.figure(2)
      plt.ion()
      for ii in range(30):

        plt.subplot(6,5,ii+1)
        plt.imshow(np.transpose(np.reshape(W[:,ii],[28, 28])))


      plt.pause(.01)


      # xtrain=sess.run(z, feed_dict={x: mnist.train.images,y_: mnist.train.images})
      # xtrainlabel=mnist.train.labels
      # xtest=sess.run(z, feed_dict={x: mnist.test.images,y_: mnist.test.images})
      # xtestlabel=mnist.test.labels
      # sio.savemat('train.mat', {'train':xtrain})
      # sio.savemat('trainlabel.mat', {'trainlabel': xtrainlabel})
      # sio.savemat('test.mat', {'test': xtest})
      # sio.savemat('testlabel.mat', {'testlabel': xtestlabel})


      # print('Train MSE', sess.run(Mean_Square_Error, feed_dict={x: batch_xs, y_: batch_xs}))
      print('Test MSE', sess.run(Mean_Square_Error, feed_dict={x: mnist.test.images,y_: mnist.test.images}))
      # Summary = sess.run(summary_accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.images})
      # writer.add_summary(Summary, i)
      print('iteration=',i)

  # Test trained model
  # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  # print(sess.run(accuracy, feed_dict={x: mnist.test.images,
  #                                     y_: mnist.test.labels}))
  # summary = sess.run(summary_accuracy, feed_dict={x: mnist.test.images,y_: mnist.test.images})
  # print('Test Mean Square Error',acc)
  # writer.add_summary(summary, i)



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)