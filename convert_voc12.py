import os
import sys
import numpy as np
import tensorflow as tf

from config import *
from PIL import Image

def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float64_feature(value):
  """Wrapper for inserting float64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer, segmentation, height, width):
  image_format = 'JPEG'
  segmentation_format = 'PNG'
  example = tf.train.Example(features=tf.train.Features(feature={
    'image/height': _int64_feature(height),
    'image/width': _int64_feature(width),
    'image/filename': _bytes_feature(tf.compat.as_bytes(os.path.basename(filename))),
    'image/format': _bytes_feature(tf.compat.as_bytes(image_format)),
    'image/encoded': _bytes_feature(tf.compat.as_bytes(image_buffer)),
    'image/segmentation/format': _bytes_feature(tf.compat.as_bytes(segmentation_format)),
    'image/segmentation/encoded': _bytes_feature(tf.compat.as_bytes(segmentation)),
    }))
  return example

def _convert_to_tfrecord(record_dir):
  """Loads image files and writes files to a TFRecord.
  """
  label_placeholder = tf.placeholder(dtype=tf.uint8)
  encoded_label = tf.image.encode_png(tf.expand_dims(label_placeholder, 2))
  with tf.Session('') as sess:
    record_filename = os.path.join(record_dir, '{}.tfrecord'.format(args.data_name))
    with tf.python_io.TFRecordWriter(record_filename) as tfrecord_writer:
      with open('./libs/datasets/VOC12/train.txt', 'r') as f:
        # 14720 images for train and val
        count = 1
        for line in f:
          line = line.strip()
          img, gt = line.split()
          img_path = args.data_dir + img
          gt_path = args.data_dir + gt
          with tf.gfile.FastGFile(img_path, 'rb') as ff:
              image_data = ff.read()
          segmentation = np.array(Image.open(gt_path))
          h,w = segmentation.shape[0], segmentation.shape[1]
          label_string = sess.run(encoded_label,
            feed_dict={label_placeholder:segmentation})
          example = _convert_to_example(img_path, image_data, label_string, 
            h, w)
          tfrecord_writer.write(example.SerializeToString())
          sys.stdout.write('Write {} {}/14270\n'.format(img_path, count))
          sys.stdout.flush()
          count += 1

  sys.stdout.write('\n')
  sys.stdout.flush()


if __name__ == '__main__':
  record_dir = os.path.join(args.data_dir, 'records')

  if not tf.gfile.Exists(record_dir):
    tf.gfile.MakeDirs(record_dir)

  # process the training, validation data:
  _convert_to_tfrecord(record_dir)

  print('\nFinished converting the VOC12 dataset!')