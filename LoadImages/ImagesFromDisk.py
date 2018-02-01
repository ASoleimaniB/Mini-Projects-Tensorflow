import tensorflow as tf
from os import listdir
from os.path import isfile, join

# This code read images from the disk

batch_size=1


mypath='/home/amir/Desktop/Okutama-Action/Patches/IMG'
files=list()
files_test=list()
for f in listdir(mypath):
  if isfile(join(mypath,f)):
    if f[0:5]!='1.1.7' and f[0:6]!='1.1.10' and f[0:6]!='1.1.11' and f[0:5]!='1.2.9' and f[0:6]!='1.2.11' and f[0:5]!='2.1.7' and f[0:6]!='2.1.10' and f[0:5]!='2.2.9' and f[0:6]!='2.2.11':
      files.append(join(mypath,f))
    else:
      files_test.append(join(mypath, f))


# mypath_txt='/home/amir/Desktop/Okutama-Action/Patches/txt'
# files_txt=list()
# files_txt_test=list()
# for f in listdir(mypath_txt):
#   if isfile(join(mypath_txt,f)):
#     if f[0:5]!='1.1.7' and f[0:6]!='1.1.10' and f[0:6]!='1.1.11' and f[0:5]!='1.2.9' and f[0:6]!='1.2.11' and f[0:5]!='2.1.7' and f[0:6]!='2.1.10' and f[0:5]!='2.2.9' and f[0:6]!='2.2.11':
#       files_txt.append(join(mypath_txt,f))
#     else:
#       files_txt_test.append(join(mypath, f))



def _parse_function(filename):
  image_string = tf.read_file(filename['files'])
  # txt_string = tf.read_file(filename['txt'])
  # txt_name=join(mypath_txt, filenames[0][len(mypath_txt) + 1:-3] + 'txt')
  # label=tf.contrib.data.TextLineDataset(filename['txt'])
  # label= tf.contrib.data.TextLineDataset(filename['txt'])
  image_decoded = tf.image.decode_jpeg(image_string)
  image_resized = tf.image.resize_images(image_decoded, [100, 60])
  return image_resized



dataset = tf.contrib.data.Dataset.from_tensor_slices(({'files':files,'txt':files_txt}))
dataset = dataset.map(_parse_function)
dataset = dataset.shuffle(buffer_size=len(files))
batched_dataset = dataset.batch(batch_size)
iterator = batched_dataset.make_one_shot_iterator()
x = iterator.get_next()


sess=tf.Session()
for i in range(1):
    try:
      A = sess.run(x)
      print(A)
    except tf.errors.OutOfRangeError:
      print('Shuffling')
      dataset = dataset.shuffle(buffer_size=len(files))
      batched_dataset = dataset.batch(batch_size)
      iterator = batched_dataset.make_one_shot_iterator()
      x = iterator.get_next()