"""Training script for the DeepLab-ResNet network on the PASCAL VOC dataset
   for semantic image segmentation.

This script trains the model using augmented PASCAL VOC,
which contains approximately 10000 images for training and 1500 images for 
validation.
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
        if args.imagenet is not None:
            ckpt_path = os.path.join(args.imagenet, 'resnet_v1_{}.ckpt'.format(args.num_layers).format(args.num_layers))
        else:
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

    image_batch, label_batch = read_data(is_training=True, split_name='train')
    
    # Create network.
    net, end_points = deeplabv3(image_batch,
                                num_classes=args.num_classes,
                                depth=args.num_layers,
                                is_training=True,
                                )
    # For a small batch size, it is better to keep 
    # the statistics of the BN layers (running means and variances)
    # frozen, and to not update the values provided by the pre-trained model. 
    # If is_training=True, the statistics will be updated during the training.
    # Note that is_training=False still updates BN parameters gamma (scale) 
    # and beta (offset)
    # if they are presented in var_list of the optimizer definition.

    # Predictions.
    raw_output = end_points['resnet_v1_{}/logits'.format(args.num_layers)]
    # Which variables to load. Running means and variances are not trainable,
    # thus all_variables() should be restored.
    if args.imagenet is not None:
        restore_var = [v for v in tf.global_variables() if 
          ('aspp' not in v.name) and 
          ('img_pool' not in v.name) and 
          ('fusion' not in v.name) and
          ('block5' not in v.name) and
          ('block6' not in v.name) and
          ('block7' not in v.name) and
          ('logits' not in v.name)]
    else:
        restore_var = [v for v in tf.global_variables()]
    if args.freeze_bn:
        all_trainable = [v for v in tf.trainable_variables() if 'beta' not in 
            v.name and 'gamma' not in v.name]
    else:
        all_trainable = [v for v in tf.trainable_variables()]
    conv_trainable = [v for v in all_trainable] 
    
    # Upsample the logits instead of donwsample the ground truth
    raw_output_up = tf.image.resize_bilinear(raw_output, [h, w])

    # Predictions: ignoring all predictions with labels greater or equal than 
    # n_classes
    label_proc = tf.squeeze(label_batch)
    mask = label_proc <= args.num_classes
    seg_logits = tf.boolean_mask(raw_output_up, mask)
    seg_gt = tf.boolean_mask(label_proc, mask)
    seg_gt = tf.cast(seg_gt, tf.int32)          
                                                  
    # Pixel-wise softmax loss.
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=seg_logits,
        labels=seg_gt)
    seg_loss = tf.reduce_mean(loss)
    seg_loss_sum = tf.summary.scalar('loss/seg', seg_loss)
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_loss = tf.add_n(reg_losses)
    reg_loss_sum = tf.summary.scalar('loss/reg', reg_loss)
    tot_loss = seg_loss + reg_loss
    tot_loss_sum = tf.summary.scalar('loss/tot', tot_loss)

    seg_pred = tf.argmax(seg_logits, axis=1)
    train_mean_iou, train_update_mean_iou = streaming_mean_iou(seg_pred, 
        seg_gt, args.num_classes, name="train_iou")  
    train_iou_sum = tf.summary.scalar('accuracy/train_mean_iou', 
        train_mean_iou)
    train_initializer = tf.variables_initializer(var_list=tf.get_collection(
        tf.GraphKeys.LOCAL_VARIABLES, scope="train_iou"))

    # Define loss and optimisation parameters.
    base_lr = tf.constant(args.learning_rate)
    step_ph = tf.placeholder(dtype=tf.float32, shape=())
    learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_ph / args.num_steps), args.power))
    learning_rate = base_lr
    lr_sum = tf.summary.scalar('params/learning_rate', learning_rate)

    train_sum_op = tf.summary.merge([seg_loss_sum, reg_loss_sum, 
        tot_loss_sum, train_iou_sum, lr_sum])

    image_batch_val, label_batch_val = read_data(is_training=False, split_name='val')
    _, end_points_val = deeplabv3(image_batch_val,
                                  num_classes=args.num_classes,
                                  depth=args.num_layers,
                                  reuse=True,
                                  is_training=False,
                                  )
    raw_output_val = end_points_val['resnet_v1_{}/logits'.format(args.num_layers)]
    nh, nw = tf.shape(image_batch_val)[1], tf.shape(image_batch_val)[2]

    seg_logits_val = tf.image.resize_bilinear(raw_output_val, [nh, nw])
    seg_pred_val = tf.argmax(seg_logits_val, axis=3)
    seg_pred_val = tf.expand_dims(seg_pred_val, 3)
    seg_pred_val = tf.reshape(seg_pred_val, [-1,])

    seg_gt_val = tf.cast(label_batch_val, tf.int32)
    seg_gt_val = tf.reshape(seg_gt_val, [-1,])
    mask_val = seg_gt_val <= args.num_classes - 1

    seg_pred_val = tf.boolean_mask(seg_pred_val, mask_val)
    seg_gt_val = tf.boolean_mask(seg_gt_val, mask_val)

    val_mean_iou, val_update_mean_iou = streaming_mean_iou(seg_pred_val, 
        seg_gt_val, num_classes=args.num_classes, name="val_iou")        
    val_iou_sum = tf.summary.scalar('accuracy/val_mean_iou', val_mean_iou)
    val_initializer = tf.variables_initializer(var_list=tf.get_collection(
        tf.GraphKeys.LOCAL_VARIABLES, scope="val_iou"))
    test_sum_op = tf.summary.merge([val_iou_sum])

    global_step = tf.train.get_or_create_global_step()
    
    opt = tf.train.MomentumOptimizer(learning_rate, args.momentum)

    grads_conv = tf.gradients(tot_loss, conv_trainable)
    # train_op = opt.apply_gradients(zip(grads_conv, conv_trainable))
    train_op = slim.learning.create_train_op(
        tot_loss, opt,
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
    if args.ckpt > 0 or args.restore_from is not None or args.imagenet is not None:
        loader = tf.train.Saver(var_list=restore_var)
        load(loader, sess, args.snapshot_dir)
    
    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    
    # tf.get_default_graph().finalize()
    summary_writer = tf.summary.FileWriter(args.snapshot_dir,
                                           sess.graph)
    
    # Iterate over training steps.
    for step in range(args.ckpt, args.num_steps):
        start_time = time.time()
        feed_dict = { step_ph : step }
        tot_loss_float, seg_loss_float, reg_loss_float, _, lr_float, _,train_summary = sess.run([tot_loss, seg_loss, reg_loss, train_op,
            learning_rate, train_update_mean_iou, train_sum_op], 
            feed_dict=feed_dict)
        train_mean_iou_float = sess.run(train_mean_iou)
        duration = time.time() - start_time
        sys.stdout.write('step {:d}, tot_loss = {:.6f}, seg_loss = {:.6f}, ' \
            'reg_loss = {:.6f}, mean_iou = {:.6f}, lr: {:.6f}({:.3f}' \
            'sec/step)\n'.format(step, tot_loss_float, seg_loss_float,
             reg_loss_float, train_mean_iou_float, lr_float, duration)
            )
        sys.stdout.flush()

        if step % args.save_pred_every == 0 and step > args.ckpt:
            sess.run(val_initializer)
            for val_step in range(NUM_VAL-1):
                _, test_summary = sess.run([val_update_mean_iou, test_sum_op],
                feed_dict=feed_dict)
            
            summary_writer.add_summary(train_summary, step)
            summary_writer.add_summary(test_summary, step)
            val_mean_iou_float= sess.run(val_mean_iou)

            save(saver, sess, args.snapshot_dir, step)
            sys.stdout.write('step {:d}, train_mean_iou: {:.6f}, ' \
                'val_mean_iou: {:.6f}\n'.format(step, train_mean_iou_float, 
                val_mean_iou_float))
            sys.stdout.flush()
            sess.run(train_initializer)

        if coord.should_stop():
            coord.request_stop()
            coord.join(threads)
    
if __name__ == '__main__':
    main()
