"""Training script for the DeepLab-ResNet network on the PASCAL VOC dataset
   for semantic image segmentation.

This script trains the model using augmented PASCAL VOC,
which contains approximately 10000 images for training and 1500 images for validation.
"""
import os
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
    h = args.input_size
    w = args.input_size
    input_size = (h, w)
    
    tf.set_random_seed(args.random_seed)
    
    # Create queue coordinator.
    coord = tf.train.Coordinator()

    image_batch, label_batch = read_data()
    
    # Create network.
    net, end_points = deeplabv3(image_batch,
                                num_classes=args.num_classes,
                                layer_depth=args.num_layers,
                                )
    # For a small batch size, it is better to keep 
    # the statistics of the BN layers (running means and variances)
    # frozen, and to not update the values provided by the pre-trained model. 
    # If is_training=True, the statistics will be updated during the training.
    # Note that is_training=False still updates BN parameters gamma (scale) and beta (offset)
    # if they are presented in var_list of the optimiser definition.

    # Predictions.
    raw_output = end_points['resnet{}/logits'.format(args.num_layers)]
    # Which variables to load. Running means and variances are not trainable,
    # thus all_variables() should be restored.
    restore_var = [v for v in tf.global_variables() if 'fc' not in v.name or not args.not_restore_last]
    all_trainable = [v for v in tf.trainable_variables()]
    conv_trainable = [v for v in all_trainable if 'fc' not in v.name] # lr * 1.0
    
    # Predictions: ignoring all predictions with labels greater or equal than n_classes
    label_proc = prepare_label(label_batch, tf.shape(raw_output)[1:3], args.num_classes, one_hot=False)
    mask = label_proc <= args.num_classes
    seg_logits = tf.boolean_mask(raw_output, mask)
    seg_gt = tf.boolean_mask(label_proc, mask)
    seg_gt = tf.cast(seg_gt, tf.int32)          
                                                  
    # Pixel-wise softmax loss.
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=seg_logits, labels=seg_gt)
    seg_loss = tf.reduce_mean(loss)
    tf.summary.scalar('loss/seg', seg_loss)
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_loss = tf.add_n(reg_losses)
    tf.summary.scalar('loss/reg', reg_loss)
    total_loss = seg_loss + reg_loss
    tf.summary.scalar('loss/tot', total_loss)
    
    seg_pred = tf.argmax(seg_logits, axis=1)
    mean_iou, update_mean_iou = streaming_mean_iou(seg_pred, seg_gt, 
        args.num_classes)    
    tf.summary.scalar('accuracy/mean_iou', mean_iou)
    
    summary_op = tf.summary.merge_all()

    global_step = tf.train.get_or_create_global_step()

    # Define loss and optimisation parameters.
    base_lr = tf.constant(args.learning_rate)
    step_ph = tf.placeholder(dtype=tf.float32, shape=())
    learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_ph / args.num_steps), args.power))
    # learning_rate = base_lr
    tf.summary.scalar('hyperparameters/learning_rate', learning_rate)
    
    opt = tf.train.MomentumOptimizer(learning_rate, args.momentum)

    grads_conv = tf.gradients(total_loss, conv_trainable)
    # train_op = opt.apply_gradients(zip(grads_conv, conv_trainable))
    train_op = slim.learning.create_train_op(
        total_loss, opt,
        global_step=global_step,
        variables_to_train=conv_trainable,
        summarize_gradients=True)
    
    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=20)
    
    # Load variables if the checkpoint is provided.
    if args.ckpt > 0 or args.restore_from is not None:
        loader = tf.train.Saver(var_list=restore_var)
        load(loader, sess, args.snapshot_dir)
    
    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    
    tf.get_default_graph().finalize()
    summary_writer = tf.summary.FileWriter(args.snapshot_dir,
                                           sess.graph)
    
    # Iterate over training steps.
    for step in range(args.ckpt, args.num_steps):
    # for step in range(1):
        start_time = time.time()
        feed_dict = { step_ph : step }
        if step % args.save_pred_every == 0 and step > args.ckpt:
            tot_loss_float, seg_loss_float, reg_loss_float, images, labels, summary, mean_iou_float, _, _, lr_float = sess.run([total_loss,
              seg_loss, reg_loss, image_batch, label_batch, summary_op,
              mean_iou, update_mean_iou, train_op, learning_rate], feed_dict=feed_dict)
            summary_writer.add_summary(summary, step)
            save(saver, sess, args.snapshot_dir, step)
            sys.stdout.write('step {:d}, tot_loss = {:.6f}, seg_loss = {:.6f}, reg_loss = {:.6f}, mean_iou: {:.6f}, lr: {:.6f}({:.3f} sec/step)\n'.format(step, tot_loss_float, seg_loss_float, reg_loss_float, mean_iou_float, lr_float, duration))
            sys.stdout.flush()
        else:
            tot_loss_float, seg_loss_float, reg_loss_float, mean_iou_float, _, _, lr_float = sess.run([total_loss, seg_loss, reg_loss, mean_iou, update_mean_iou, train_op, learning_rate], feed_dict=feed_dict)
        duration = time.time() - start_time
        sys.stdout.write('step {:d}, tot_loss = {:.6f}, seg_loss = {:.6f}, reg_loss = {:.6f}, mean_iou: {:.6f}, lr: {:.6f}({:.3f} sec/step)\n'.format(step, tot_loss_float, seg_loss_float, reg_loss_float, mean_iou_float, lr_float, duration))
        sys.stdout.flush()

        if coord.should_stop():
            coord.request_stop()
            coord.join(threads)
    
if __name__ == '__main__':
    main()
