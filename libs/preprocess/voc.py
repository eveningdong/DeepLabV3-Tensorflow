import tensorflow as tf

from config import *
from libs.preprocess import utils as preprocess_utils

def preprocess_image(image, gt_mask, is_training=False):
    if is_training:
        return _preprocess_for_training(image, gt_mask)
    else:
        return _preprocess_for_test(image, gt_mask)


def _preprocess_for_training(image, gt_mask):
    
    ih, iw = tf.shape(image)[0], tf.shape(image)[1]

    ## random flipping
    coin = tf.to_float(tf.random_uniform([1]))[0]
    image, gt_mask =\
            tf.cond(tf.greater_equal(coin, 0.5), 
                    lambda: (preprocess_utils.flip_image(image),
                            preprocess_utils.flip_image(gt_mask)),
                    lambda: (image, gt_mask))
    
    scale = tf.random_uniform(shape=[1], minval=0.5, maxval=2)[0]
    image, gt_mask = preprocess_utils.rescale(image, gt_mask, ih, iw, scale)
    
    image, gt_mask = preprocess_utils.random_crop_and_pad_image_and_labels(image, gt_mask, args.input_size, args.input_size)

    # rgb to bgr
    image = tf.reverse(image, axis=[-1])
    image -= IMG_MEAN

    return image, gt_mask

def _preprocess_for_test(image, gt_mask):
    image = tf.to_float(image)
    image = tf.reverse(image, axis=[-1])
    image -= IMG_MEAN
    image = tf.expand_dims(image, 0)
    gt_mask = tf.expand_dims(gt_mask, 0)
    return image, gt_mask