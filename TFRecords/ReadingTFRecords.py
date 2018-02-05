import tensorflow as tf
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import time

# This code read images from the disk


TrainingSize=24630
TestSize=7716

batch_size=1
total_samples=100000



def _parse_function(example_proto):
  features = {"height": tf.FixedLenFeature((), tf.int64),
              "width": tf.FixedLenFeature((), tf.int64),
              "label1_single": tf.FixedLenFeature((), tf.string),
              "label2_multi": tf.FixedLenFeature((), tf.string),
              "label3_multi": tf.FixedLenFeature((), tf.string),
              "filenames": tf.FixedLenFeature((), tf.string),
              "image_raw": tf.FixedLenFeature((), tf.string),

              }
  parsed_features = tf.parse_single_example(example_proto, features)

  h = tf.cast(parsed_features["height"], tf.int64)
  w = tf.cast(parsed_features["width"], tf.int64)

  im = tf.decode_raw(parsed_features['image_raw'], tf.uint8)

  label1 = parsed_features['label1_single']
  label2 = parsed_features['label2_multi']
  label3 = parsed_features['label3_multi']
  FileName = parsed_features['filenames']


  return im,h,w,label1,label2,label3,FileName
  # return image,h,w



TFfilename = ["Okutama_Train.tfrecords"]
dataset = tf.contrib.data.TFRecordDataset(TFfilename)
dataset = dataset.map(_parse_function)
dataset = dataset.shuffle(buffer_size=total_samples)
batched = dataset.batch(batch_size)
iterator = batched.make_one_shot_iterator()
image,height,width,label1_single,label2_multi,label3_multi,filenames = iterator.get_next()



sess=tf.Session()
for i in range(TrainingSize*100):
    try:
        im,h,w,files,label1,label2,label3=sess.run([image,height,width,filenames,label1_single,label2_multi,label3_multi])
        label1 = label1.astype('unicode').tolist()
        label1 = label1[0]
        label2 = label2.astype('unicode').tolist()
        label2 = label2[0]
        label3 = label3.astype('unicode').tolist()
        label3 = label3[0]
        files = files.astype('unicode').tolist()
        files = files[0]

        I=np.reshape(im,[h[0],w[0],3])

        print(files)
        print(label1)
        print(label2)
        print(label3)
        print(I.shape)
        print(I)
        print(h)
        print(w)

        print(i)


    except tf.errors.OutOfRangeError:
        print('Shuffling')
        dataset = dataset.shuffle(buffer_size=total_samples)
        batched_dataset = dataset.batch(batch_size)
        iterator = batched.make_one_shot_iterator()
        image, height, width, label1_single, label2_multi, label3_multi, filenames = iterator.get_next()
