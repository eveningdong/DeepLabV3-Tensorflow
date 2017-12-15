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
from tensorflow.python.client import device_lib

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

def get_num_available_gpus():
  num_gpus = args.num_gpus
  local_device_protos = device_lib.list_local_devices()
  num_available_gpus = len([x.name for x in local_device_protos if x.device_type == 'GPU'])
    
  if num_gpus > num_available_gpus:
    return num_available_gpus

  return num_gpus

def average_gradients(gpu_grads):
  """Calculate the average gradient for each shared variable across all GPUs.
  Note that this function provides a synchronization point across all GPUs.
  Args:
    gpu_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each GPU.
  Returns:
    List of pairs of (gradient, variable) where the gradient has been averaged
    across all GPUs.
  """
  average_grads = []
  for grad_and_vars in zip(*gpu_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the GPU.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across GPUs. So .. we will just return the first GPU's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

def main():
  """Create the model and start the training."""
  h = args.input_size
  w = args.input_size
  input_size = (h, w)

  # Check available GPUs  
  num_gpus = get_num_available_gpus()

  tf.set_random_seed(args.random_seed)
    
  # Create queue coordinator.
  coord = tf.train.Coordinator()

  image_batch, label_batch = read_data(batch_size=num_gpus * args.batch_size)
  split_image_batch = tf.split(image_batch, num_gpus, 0)
  split_label_batch = tf.split(label_batch, num_gpus, 0)

  global_step = tf.train.get_or_create_global_step()
  # Define optimization parameters.
  base_lr = tf.constant(args.learning_rate)
  step_ph = tf.placeholder(dtype=tf.float32, shape=())
  learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_ph / args.num_steps), args.power))
  # learning_rate = base_lr
  tf.summary.scalar('hyperparameters/learning_rate', learning_rate)
  opt = tf.train.MomentumOptimizer(learning_rate, args.momentum)

  grads = []
  seg_losses = [] 
  reg_losses = []
  tot_losses = []
  seg_gts = []
  seg_preds = []

  with tf.variable_scope(tf.get_variable_scope()):
    for i in range(num_gpus):
      with tf.device('/gpu:{}'.format(i)):
        with tf.name_scope('gpu{}'.format(i)) as scope:
          # Create network.
          print('Device: {}'.format(scope))
          net, end_points = deeplabv3(split_image_batch[i],
                                      num_classes=args.num_classes,
                                      layer_depth=args.num_layers,
                                      is_training=args.is_training)
          # For a small batch size, it is better to keep 
          # the statistics of the BN layers (running means and variances)
          # frozen, and to not update the values provided by the pre-trained model. 
          # If is_training=True, the statistics will be updated during the training.
          # Note that is_training=False still updates BN parameters gamma (scale) and beta (offset)
          # if they are presented in var_list of the optimiser definition.
          # Predictions.
          raw_output = end_points['gpu{}/resnet{}/logits'.format(i, args.num_layers)]

          # Predictions: ignoring all predictions with labels greater or equal than n_classes
          label_proc = prepare_label(split_label_batch[i], tf.shape(raw_output)[1:3], args.num_classes, one_hot=False)
          mask = label_proc <= args.num_classes
          seg_logits = tf.boolean_mask(raw_output, mask)
          seg_gt = tf.boolean_mask(label_proc, mask)
          seg_gt = tf.cast(seg_gt, tf.int32)
          seg_gts.append(seg_gt)

          # Pixel-wise softmax loss.
          loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=seg_logits, labels=seg_gt)
          seg_loss = tf.reduce_mean(loss)
          # tf.summary.scalar('loss/seg', seg_loss)
          seg_losses.append(seg_loss)

          reg_losses = [args.weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]
          reg_loss = tf.add_n(reg_losses)
          # tf.summary.scalar('loss/reg', reg_loss)
          reg_losses.append(reg_loss)

          total_loss = seg_loss + reg_loss
          # tf.summary.scalar('loss/tot', total_loss)
          tot_losses.append(total_loss)

          seg_pred = tf.argmax(seg_logits, axis=1)
          seg_preds.append(seg_pred)
          
          # Reuse variables for the next GPU.
          tf.get_variable_scope().reuse_variables()

          grad = opt.compute_gradients(total_loss, var_list=[v for v in tf.trainable_variables() if 'beta' not in v.name and 'gamma' not in v.name])
          grads.append(grad)

  # We must calculate the mean of each gradient. Note that this is the
  # synchronization point across all GPUs.
  grads = average_gradients(grads)
  train_op = opt.apply_gradients(grads, global_step=global_step)

  seg_gts = tf.concat(seg_gts, 0)
  seg_preds = tf.concat(seg_preds, 0)
  mean_iou, update_mean_iou = streaming_mean_iou(seg_preds, seg_gts, 
      args.num_classes)    
  tf.summary.scalar('accuracy/mean_iou', mean_iou)

  avg_seg_loss = tf.reduce_mean(seg_losses)
  tf.summary.scalar('loss/seg_loss', avg_seg_loss)

  avg_reg_loss = tf.reduce_mean(reg_losses)
  tf.summary.scalar('loss/reg_loss', avg_reg_loss)

  avg_tot_loss = tf.reduce_mean(tot_losses)
  tf.summary.scalar('loss/tot_loss', avg_tot_loss)
    
  # Which variables to load. Running means and variances are not trainable,
  # thus all_variables() should be restored.
  restore_var = [v for v in tf.global_variables() if 'fc' not in v.name or not args.not_restore_last]

  summary_op = tf.summary.merge_all()
  
  ## Set up tf session and initialize variables.
  # This will resolve the problem if it couldn't place an operation on the 
  # GPU. Since some operations have only CPU implementation.
  config = tf.ConfigProto(allow_soft_placement=True)
  # config = tf.ConfigProto()
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
      tot_loss_float, seg_loss_float, reg_loss_float, summary, mean_iou_float, _, _, lr_float = sess.run([avg_tot_loss, avg_seg_loss, avg_reg_loss, summary_op, mean_iou, update_mean_iou, train_op, learning_rate], feed_dict=feed_dict)
      summary_writer.add_summary(summary, step)
      save(saver, sess, args.snapshot_dir, step)
      sys.stdout.write('step {:d}, tot_loss = {:.6f}, seg_loss = {:.6f}, reg_loss = {:.6f}, mean_iou: {:.6f}, lr: {:.6f}({:.3f} sec/step)\n'.format(step, tot_loss_float, seg_loss_float, reg_loss_float, mean_iou_float, lr_float, duration))
      sys.stdout.flush()
    else:
      tot_loss_float, seg_loss_float, reg_loss_float, mean_iou_float, _, _, lr_float = sess.run([avg_tot_loss, avg_seg_loss, avg_reg_loss, mean_iou, update_mean_iou, train_op, learning_rate], feed_dict=feed_dict)
      duration = time.time() - start_time
      sys.stdout.write('step {:d}, tot_loss = {:.6f}, seg_loss = {:.6f}, reg_loss = {:.6f}, mean_iou: {:.6f}, lr: {:.6f}({:.3f} sec/step)\n'.format(step, tot_loss_float, seg_loss_float, reg_loss_float, mean_iou_float, lr_float, duration))
      sys.stdout.flush()

    if coord.should_stop():
      coord.request_stop()
      coord.join(threads)
    
if __name__ == '__main__':
  main()
