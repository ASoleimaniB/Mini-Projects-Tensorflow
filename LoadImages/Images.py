import tensorflow as tf
from os import listdir
from os.path import isfile, join

batch_size=6


mypath='/home/amir/Desktop/Okutama-Action/Patches/IMG'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
files=list()
for f in listdir(mypath):
  if isfile(join(mypath,f)):
    files.append(join(mypath,f))


def _parse_function(filename):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string)
  image_resized = tf.image.resize_images(image_decoded, [100, 60])
  return image_resized

filenames = files

dataset = tf.contrib.data.Dataset.from_tensor_slices((filenames))
dataset = dataset.map(_parse_function)
dataset = dataset.shuffle(buffer_size=len(files))
batched_dataset = dataset.batch(batch_size)
iterator = batched_dataset.make_one_shot_iterator()
x = iterator.get_next()


sess=tf.Session()
for i in range(20000000):
    try:
      A = sess.run(x)
      print(A.shape)
    except tf.errors.OutOfRangeError:
      print('Shuffling')
      dataset = dataset.shuffle(buffer_size=len(files))
      batched_dataset = dataset.batch(batch_size)
      iterator = batched_dataset.make_one_shot_iterator()
      x = iterator.get_next()