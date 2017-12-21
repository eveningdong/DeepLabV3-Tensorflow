import sys
import argparse
import numpy as np
import tensorflow as tf
import time

from config import *
from datetime import datetime
from libs.datasets.dataset_factory import read_data
from libs.datasets.VOC12 import decode_labels, inv_preprocess, prepare_label
from libs.nets import deeplabv3

slim = tf.contrib.slim
streaming_mean_iou = tf.contrib.metrics.streaming_mean_iou

def save(saver, sess, logdir, step):
   '''Save weights.
   
   Args:
     saver: TensorFlow Saver object.
     sess: TensorFlow session.
     logdir: path to the snapshots directory.
     step: current training step.
   '''
   model_name = 'model.ckpt'
   checkpoint_path = os.path.join(logdir, model_name)
    
   if not os.path.exists(logdir):
      os.makedirs(logdir)
   saver.save(sess, checkpoint_path, global_step=step)
   print('The checkpoint has been created.')

def load(saver, sess, ckpt_dir):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow Saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    if args.ckpt == 0:
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        ckpt_path = ckpt.model_checkpoint_path
    else:
        ckpt_path = ckpt_dir+'/model.ckpt-%i' % args.ckpt
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def main():
    """Create the model and start the training."""
    tf.set_random_seed(args.random_seed)
    
    # Create queue coordinator.
    coord = tf.train.Coordinator()

    image_batch, label_batch = read_data(is_training=args.is_training)

    # Create network.
    net, end_points = deeplabv3(image_batch,
                                num_classes=args.num_classes,
                                depth=args.num_layers,
                                is_training=False)

    # For a small batch size, it is better to keep 
    # the statistics of the BN layers (running means and variances)
    # frozen, and to not update the values provided by the pre-trained model. 
    # If is_training=True, the statistics will be updated during the training.
    # Note that is_training=False still updates BN parameters gamma (scale) and beta (offset)
    # if they are presented in var_list of the optimiser definition.

    # Which variables to load. Running means and variances are not trainable,
    # thus all_variables() should be restored.
    restore_var = [v for v in tf.global_variables() if 'fc' not in v.name or not args.not_restore_last]
    
    # Predictions.
    raw_output = end_points['resnet{}/logits'.format(args.num_layers)]
    # Predictions: ignoring all predictions with labels greater or equal than n_classes
    nh, nw = tf.shape(image_batch)[1], tf.shape(image_batch)[2]
    seg_logits = tf.image.resize_bilinear(raw_output, [nh, nw])
    seg_pred = tf.argmax(seg_logits, axis=3)
    seg_pred = tf.expand_dims(seg_pred, 3)
    seg_pred = tf.reshape(seg_pred, [-1,])

    seg_gt = tf.cast(label_batch, tf.int32)
    seg_gt = tf.reshape(seg_gt, [-1,])
    mask = seg_gt <= args.num_classes - 1

    seg_pred = tf.boolean_mask(seg_pred, mask)
    seg_gt = tf.boolean_mask(seg_gt, mask)

    mean_iou, update_mean_iou = streaming_mean_iou(seg_pred, seg_gt, num_classes=args.num_classes)
    
    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    # Load variables if the checkpoint is provided.
    if args.ckpt > 0 or args.restore_from is not None:
        loader = tf.train.Saver(var_list=restore_var)
        load(loader, sess, args.snapshot_dir)
    
    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    
    tf.get_default_graph().finalize()
    
    for step in range(1449):
        start_time = time.time()
        mean_iou_float, _ = sess.run([mean_iou, update_mean_iou])
        duration = time.time() - start_time
        sys.stdout.write('step {:d}, mean_iou: {:.6f}({:.3f} sec/step)\n'.format(step, mean_iou_float, duration))
        sys.stdout.flush()

        if coord.should_stop():
            coord.request_stop()
            coord.join(threads)
    
if __name__ == '__main__':
    main()
