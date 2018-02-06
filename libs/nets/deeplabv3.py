import tensorflow as tf

from config import *
from libs.nets import resnet_utils
from libs.nets.resnet_v1 import bottleneck, resnet_arg_scope

slim = tf.contrib.slim

'''Commented out due to suspicious implementation. 
@slim.add_arg_scope
def bottleneck_hdc(inputs,
               depth,
               depth_bottleneck,
               stride,
               rate=1,
               multi_grid=(1,2,4),
               outputs_collections=None,
               scope=None,
               use_bounded_activations=False):
  """Hybrid Dilated Convolution Bottleneck.
  Multi_Grid = (1,2,4)
  See Understanding Convolution for Semantic Segmentation.
  When putting together two consecutive ResNet blocks that use this unit, one
  should use stride = 2 in the last unit of the first block.
  Args:
    inputs: A tensor of size [batch, height, width, channels].
    depth: The depth of the ResNet unit output.
    depth_bottleneck: The depth of the bottleneck layers.
    stride: The ResNet unit's stride. Determines the amount of downsampling of
      the units output compared to its input.
    rate: An integer, rate for atrous convolution.
    multi_grid: multi_grid sturcture.
    outputs_collections: Collection to add the ResNet unit output.
    scope: Optional variable_scope.
    use_bounded_activations: Whether or not to use bounded activations. Bounded
      activations better lend themselves to quantized inference.
  Returns:
    The ResNet unit's output.
  """
  with tf.variable_scope(scope, 'bottleneck_v1', [inputs]) as sc:
    depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    if depth == depth_in:
      shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
    else:
      shortcut = slim.conv2d(
          inputs,
          depth, [1, 1],
          stride=stride,
          activation_fn=tf.nn.relu6 if use_bounded_activations else None,
          scope='shortcut')

    residual = slim.conv2d(inputs, depth_bottleneck, [1, 1], stride=1, 
      rate=rate*multi_grid[0], scope='conv1')
    residual = resnet_utils.conv2d_same(residual, depth_bottleneck, 3, stride,
      rate=rate*multi_grid[1], scope='conv2')
    residual = slim.conv2d(residual, depth, [1, 1], stride=1, 
      rate=rate*multi_grid[2], activation_fn=None, scope='conv3')

    if use_bounded_activations:
      # Use clip_by_value to simulate bandpass activation.
      residual = tf.clip_by_value(residual, -6.0, 6.0)
      output = tf.nn.relu6(shortcut + residual)
    else:
      output = tf.nn.relu(shortcut + residual)

    return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.name,
                                            output)
'''

def deeplabv3(inputs,
              num_classes,
              depth=50,
              aspp=True,
              reuse=None,
              is_training=True):
  """DeepLabV3
  Args:
    inputs: A tensor of size [batch, height, width, channels].
    depth: The number of layers of the ResNet.
    aspp: Whether to use ASPP module, if True, will use 4 blocks with 
      multi_grid=(1,2,4), if False, will use 7 blocks with multi_grid=(1,2,1).
    reuse: Whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
  Returns:
    net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
    end_points: A dictionary from components of the network to the 
      corresponding activation.
  """
  if aspp:
    multi_grid = (1,2,4)
  else:
    multi_grid = (1,2,1)
  scope ='resnet_v1_{}'.format(depth)
  with tf.variable_scope(scope, [inputs], reuse=reuse) as sc:
    end_points_collection = sc.name + '_end_points'
    with slim.arg_scope(resnet_arg_scope(weight_decay=args.weight_decay, 
      batch_norm_decay=args.bn_weight_decay)):
      with slim.arg_scope([slim.conv2d, bottleneck],
                          outputs_collections=end_points_collection):
        with slim.arg_scope([slim.batch_norm], is_training=is_training):
          net = inputs
          net = resnet_utils.conv2d_same(net, 64, 7, stride=2, scope='conv1')
          net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')

          with tf.variable_scope('block1', [net]) as sc:
            base_depth = 64
            for i in range(2):
              with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                net = bottleneck(net, depth=base_depth * 4, 
                  depth_bottleneck=base_depth, stride=1)
            with tf.variable_scope('unit_3', values=[net]):
              net = bottleneck(net, depth=base_depth * 4, 
                depth_bottleneck=base_depth, stride=2)
            net = slim.utils.collect_named_outputs(end_points_collection, 
              sc.name, net)

          with tf.variable_scope('block2', [net]) as sc:
            base_depth = 128
            for i in range(3):
              with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                net = bottleneck(net, depth=base_depth * 4, 
                  depth_bottleneck=base_depth, stride=1)
            with tf.variable_scope('unit_4', values=[net]):
              net = bottleneck(net, depth=base_depth * 4, 
                depth_bottleneck=base_depth, stride=2)
            net = slim.utils.collect_named_outputs(end_points_collection, 
              sc.name, net)

          with tf.variable_scope('block3', [net]) as sc:
            base_depth = 256

            num_units = 6
            if depth == 101:
              num_units = 23
            elif depth == 152:
              num_units = 36

            for i in range(num_units):
              with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                net = bottleneck(net, depth=base_depth * 4, 
                  depth_bottleneck=base_depth, stride=1)
            net = slim.utils.collect_named_outputs(end_points_collection, 
              sc.name, net)

          with tf.variable_scope('block4', [net]) as sc:
            base_depth = 512

            for i in range(3):
              with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                net = bottleneck(net, depth=base_depth * 4, 
                  depth_bottleneck=base_depth, stride=1, rate=2*multi_grid[i])
            net = slim.utils.collect_named_outputs(end_points_collection, 
              sc.name, net)

          if aspp:
            with tf.variable_scope('aspp', [net]) as sc:
              aspp_list = []
              branch_1 = slim.conv2d(net, 256, [1,1], stride=1, 
                scope='1x1conv')
              branch_1 = slim.utils.collect_named_outputs(
                end_points_collection, sc.name, branch_1)
              aspp_list.append(branch_1)

              for i in range(3):
                branch_2 = slim.conv2d(net, 256, [3,3], stride=1, rate=6*(i+1), scope='rate{}'.format(6*(i+1)))
                branch_2 = slim.utils.collect_named_outputs(end_points_collection, sc.name, branch_2)
                aspp_list.append(branch_2)

              # aspp = tf.add_n(aspp_list)
              # aspp = slim.utils.collect_named_outputs(end_points_collection, sc.name, aspp)

            with tf.variable_scope('img_pool', [net]) as sc:
              """Image Pooling
              See ParseNet: Looking Wider to See Better
              """
              pooled = tf.reduce_mean(net, [1, 2], name='avg_pool', 
                keep_dims=True)
              pooled = slim.utils.collect_named_outputs(end_points_collection, 
                sc.name, pooled)

              pooled = slim.conv2d(pooled, 256, [1,1], stride=1, scope='1x1conv')
              pooled = slim.utils.collect_named_outputs(end_points_collection, 
                sc.name, pooled)

              pooled = tf.image.resize_bilinear(pooled, tf.shape(net)[1:3])
              pooled = slim.utils.collect_named_outputs(end_points_collection, 
                sc.name, pooled)

            with tf.variable_scope('fusion', [aspp_list, pooled]) as sc:
              aspp_list.append(pooled)
              net = tf.concat(aspp_list, 3)
              net = slim.utils.collect_named_outputs(end_points_collection, 
                sc.name, net)

              net = slim.conv2d(net, 256, [1,1], stride=1, scope='1x1conv')
              net = slim.utils.collect_named_outputs(end_points_collection, 
                sc.name, net)
          else:
            with tf.variable_scope('block5', [net]) as sc:
              base_depth = 512

              for i in range(3):
                with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                  net = bottleneck(net, depth=base_depth * 4, 
                    depth_bottleneck=base_depth, stride=1, rate=4*multi_grid[i])
              net = slim.utils.collect_named_outputs(end_points_collection, 
                sc.name, net)

            with tf.variable_scope('block6', [net]) as sc:
              base_depth = 512

              for i in range(3):
                with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                  net = bottleneck(net, depth=base_depth * 4, 
                    depth_bottleneck=base_depth, stride=1, rate=8*multi_grid[i])
              net = slim.utils.collect_named_outputs(end_points_collection, 
                sc.name, net)

            with tf.variable_scope('block7', [net]) as sc:
              base_depth = 512

              for i in range(3):
                with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                  net = bottleneck(net, depth=base_depth * 4, 
                    depth_bottleneck=base_depth, stride=1, rate=16*multi_grid[i])
              net = slim.utils.collect_named_outputs(end_points_collection, 
                sc.name, net)
          
          with tf.variable_scope('logits',[net]) as sc:
            net = slim.conv2d(net, num_classes, [1,1], stride=1, 
              activation_fn=None, normalizer_fn=None)
            net = slim.utils.collect_named_outputs(end_points_collection, 
            sc.name, net)

          end_points = slim.utils.convert_collection_to_dict(
              end_points_collection)

          return net, end_points

if __name__ == "__main__":
  x = tf.placeholder(tf.float32, [None, 512, 512, 3])

  net, end_points = deeplabv3(x, 21)
  for i in end_points:
    print(i, end_points[i])
