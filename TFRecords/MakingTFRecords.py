import tensorflow as tf
import numpy as np
from os import listdir
from PIL import Image as PImage
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

# First Read Images and txt files from Disk


def loadImagesAndTxts(path,path_txt):
    # return array of images
    imagesList = listdir(path)
    imagesList = sorted(imagesList)
    loadedImages = []
    Firsttxt = []
    Secondtxt = []
    Thirdtxt = []
    filename= []
    for image in imagesList:
        # if image[0:5]!='1.1.7' and image[0:6]!='1.1.10' and image[0:6]!='1.1.11' and image[0:5]!='1.2.9' and image[0:6]!='1.2.11' and image[0:5]!='2.1.7' and image[0:6]!='2.1.10' and image[0:5]!='2.2.9' and image[0:6]!='2.2.11':
        if image[0:5] == '1.1.7' or image[0:6] == '1.1.10' or image[0:6] == '1.1.11' or image[0:5] == '1.2.9' or image[0:6] == '1.2.11' or image[0:5] == '2.1.7' or image[0:6] == '2.1.10' or image[0:5] == '2.2.9' or image[0:6] == '2.2.11':
            img = PImage.open(path + image)
            txt_name = path_txt + image[0:-4] + '.txt'
            loadedImages.append(img)
            # Text reading
            Text=open(txt_name,"r")
            Text=Text.read()
            firstdash=Text.find('-')
            secondtdash = Text.find('-',firstdash+1)
            Firsttxt.append(Text[0:firstdash])   #single action
            Secondtxt.append(Text[firstdash+1:secondtdash]) #multi-action
            Thirdtxt.append(Text[secondtdash+1:])   #multi-action
            # File Name
            filename.append(image)
    return loadedImages, Firsttxt, Secondtxt, Thirdtxt, filename


def Labling(txt):
    # return array of images
    if txt=='Hand Shaking':
        t='Hand'
    elif txt=='Hugging':
        t = 'Hugg'
    elif txt == 'Reading':
        t = 'Read'
    elif txt == 'Drinking':
        t = 'Drin'
    elif txt == 'Pushing/Pulling':
        t = 'Push'
    elif txt == 'Carrying':
        t = 'Carr'
    elif txt == 'Calling':
        t = 'Call'
    elif txt == 'Running':
        t = 'Runn'
    elif txt == 'Walking':
        t = 'Walk'
    elif txt == 'Lying':
        t = 'Lyin'
    elif txt == 'Sitting':
        t = 'Sitt'
    elif txt == 'Standing':
        t = 'Stan'
    else:
        t = 'dont'
    return t

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))




path = "/home/amir/Desktop/Okutama-Action/Patches/IMG/"
path_txt = "/home/amir/Desktop/Okutama-Action/Patches/txt/"

imgs,txt1,txt2,txt3,filename = loadImagesAndTxts(path,path_txt)


TFfilename ='Okutama_Test.tfrecords'
print('Writing', filename)
writer = tf.python_io.TFRecordWriter(TFfilename)
# counter=0
# for img,t1,t2,t3,f in zip(imgs,txt1,txt2,txt3,filename):
#     counter = counter+1
#     image = np.array(img, dtype=np.uint8)
#     print(f)
#     print(Labling(t1))
#     print(Labling(t2))
#     print(Labling(t3))
#
#
#     ## Making TF Records
#
#     images = image
#     labels1 = Labling(t1)
#     labels2 = Labling(t2)
#     labels3 = Labling(t3)
#
#     # num_examples = data_set.num_examples
#
#     rows = images.shape[0]
#     cols = images.shape[1]
#     depth = images.shape[2]
#
#
#     # for index in range(num_examples):
#     image_raw = images.tostring()
#     example = tf.train.Example(features=tf.train.Features(feature={
#         'height': _int64_feature(rows),
#         'width': _int64_feature(cols),
#         # 'depth': _int64_feature(depth),
#         'label1_single': _bytes_feature(labels1.encode('utf-8')),
#         'label2_multi': _bytes_feature(labels2.encode('utf-8')),
#         'label3_multi': _bytes_feature(labels3.encode('utf-8')),
#         'filenames': _bytes_feature(f.encode('utf-8')),
#         'image_raw': _bytes_feature(image_raw)}))
#     writer.write(example.SerializeToString())
# writer.close()


with tf.python_io.TFRecordWriter(TFfilename) as writer:
    for index in range(len(imgs)):
        image = np.array(imgs[index], dtype=np.uint8)
        ## Making TF Records

        images = image
        labels1 = Labling(txt1[index])
        labels2 = Labling(txt2[index])
        labels3 = Labling(txt3[index])

        # num_examples = data_set.num_examples

        rows = images.shape[0]
        cols = images.shape[1]
        depth = images.shape[2]

        print(filename[index])
        print(labels1)
        print(labels2)
        print(labels3)

        image_raw = images.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'label1_single': _bytes_feature(labels1.encode('utf-8')),
            'label2_multi': _bytes_feature(labels2.encode('utf-8')),
            'label3_multi': _bytes_feature(labels3.encode('utf-8')),
            'filenames': _bytes_feature(filename[index].encode('utf-8')),
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())

print('Number:')
print(len(imgs))