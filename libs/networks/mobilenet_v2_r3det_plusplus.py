# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division
import tensorflow.contrib.slim as slim
import tensorflow as tf

from libs.networks.mobilenet import mobilenet_v2
from libs.networks.mobilenet.mobilenet import training_scope
from libs.networks.mobilenet.mobilenet_v2 import op
from libs.networks.mobilenet.mobilenet_v2 import ops
from libs.networks.resnet import fusion_two_layer
from libs.configs import cfgs
expand_input = ops.expand_input_by_factor

V2_BASE_DEF = dict(
    defaults={
        # Note: these parameters of batch norm affect the architecture
        # that's why they are here and not in training_scope.
        (slim.batch_norm,): {'center': True, 'scale': True},
        (slim.conv2d, slim.fully_connected, slim.separable_conv2d): {
            'normalizer_fn': slim.batch_norm, 'activation_fn': tf.nn.relu6
        },
        (ops.expanded_conv,): {
            'expansion_size': expand_input(6),
            'split_expansion': 1,
            'normalizer_fn': slim.batch_norm,
            'residual': True
        },
        (slim.conv2d, slim.separable_conv2d): {'padding': 'SAME'}
    },
    spec=[
        op(slim.conv2d, stride=2, num_outputs=32, kernel_size=[3, 3]),
        op(ops.expanded_conv,
           expansion_size=expand_input(1, divisible_by=1),
           num_outputs=16, scope='expanded_conv'),
        op(ops.expanded_conv, stride=2, num_outputs=24, scope='expanded_conv_1'),
        op(ops.expanded_conv, stride=1, num_outputs=24, scope='expanded_conv_2'),
        op(ops.expanded_conv, stride=2, num_outputs=32, scope='expanded_conv_3'),
        op(ops.expanded_conv, stride=1, num_outputs=32, scope='expanded_conv_4'),
        op(ops.expanded_conv, stride=1, num_outputs=32, scope='expanded_conv_5'),
        op(ops.expanded_conv, stride=2, num_outputs=64, scope='expanded_conv_6'),
        op(ops.expanded_conv, stride=1, num_outputs=64, scope='expanded_conv_7'),
        op(ops.expanded_conv, stride=1, num_outputs=64, scope='expanded_conv_8'),
        op(ops.expanded_conv, stride=1, num_outputs=64, scope='expanded_conv_9'),
        op(ops.expanded_conv, stride=1, num_outputs=96, scope='expanded_conv_10'),
        op(ops.expanded_conv, stride=1, num_outputs=96, scope='expanded_conv_11'),
        op(ops.expanded_conv, stride=1, num_outputs=96, scope='expanded_conv_12'),
        op(ops.expanded_conv, stride=2, num_outputs=160, scope='expanded_conv_13'),
        op(ops.expanded_conv, stride=1, num_outputs=160, scope='expanded_conv_14'),
        op(ops.expanded_conv, stride=1, num_outputs=160, scope='expanded_conv_15'),
        op(ops.expanded_conv, stride=1, num_outputs=320, scope='expanded_conv_16'),
        op(slim.conv2d, stride=1, kernel_size=[1, 1], num_outputs=1280, scope='Conv_1')
    ],
)


V2_HEAD_DEF = dict(
    defaults={
        # Note: these parameters of batch norm affect the architecture
        # that's why they are here and not in training_scope.
        (slim.batch_norm,): {'center': True, 'scale': True},
        (slim.conv2d, slim.fully_connected, slim.separable_conv2d): {
            'normalizer_fn': slim.batch_norm, 'activation_fn': tf.nn.relu6
        },
        (ops.expanded_conv,): {
            'expansion_size': expand_input(6),
            'split_expansion': 1,
            'normalizer_fn': slim.batch_norm,
            'residual': True
        },
        (slim.conv2d, slim.separable_conv2d): {'padding': 'SAME'}
    },
    spec=[
        op(ops.expanded_conv, stride=2, num_outputs=160, scope='expanded_conv_13'),
        op(ops.expanded_conv, stride=1, num_outputs=160, scope='expanded_conv_14'),
        op(ops.expanded_conv, stride=1, num_outputs=160, scope='expanded_conv_15'),
        op(ops.expanded_conv, stride=1, num_outputs=320, scope='expanded_conv_16'),
        op(slim.conv2d, stride=1, kernel_size=[1, 1], num_outputs=1280, scope='Conv_1')
    ],
)

def mobilenetv2_scope(is_training=True,
                      trainable=True,
                      weight_decay=0.00004,
                      stddev=0.09,
                      dropout_keep_prob=0.8,
                      bn_decay=0.997):
  """Defines Mobilenet training scope.
  In default. We do not use BN
  ReWrite the scope.
  """
  batch_norm_params = {
      'is_training': False,
      'trainable': False,
      'decay': bn_decay,
  }
  with slim.arg_scope(training_scope(is_training=is_training, weight_decay=weight_decay)):
      with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.separable_conv2d],
                          trainable=trainable):
          with slim.arg_scope([slim.batch_norm], **batch_norm_params) as sc:
              return sc


def mobilenetv2_base(img_batch, is_training=True):

    with slim.arg_scope(mobilenetv2_scope(is_training=is_training, trainable=True)):
        feature_to_crop, endpoints = mobilenet_v2.mobilenet_base(input_tensor=img_batch,
                                                                 num_classes=None,
                                                                 is_training=False,
                                                                 depth_multiplier=1.0,
                                                                 scope='MobilenetV2',
                                                                 conv_defs=V2_BASE_DEF,
                                                                 finegrain_classification_mode=False)

        feature_dict = {"C3": endpoints['layer_5'],
                        'C4': endpoints['layer_8'],
                        'C5': endpoints['layer_15']}
        pyramid_dict = {}
        with tf.variable_scope('build_pyramid'):
            with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(cfgs.WEIGHT_DECAY),
                                activation_fn=None, normalizer_fn=None):

                P5 = slim.conv2d(feature_dict['C5'],
                                 num_outputs=256,
                                 kernel_size=[1, 1],
                                 stride=1, scope='build_P5')

                pyramid_dict['P5'] = P5

                for level in range(4, 2, -1):  # build [P4, P3]

                    pyramid_dict['P%d' % level] = fusion_two_layer(C_i=feature_dict["C%d" % level],
                                                                   P_j=pyramid_dict["P%d" % (level + 1)],
                                                                   scope='build_P%d' % level)
                for level in range(5, 2, -1):
                    pyramid_dict['P%d' % level] = slim.conv2d(pyramid_dict['P%d' % level],
                                                              num_outputs=256, kernel_size=[3, 3], padding="SAME",
                                                              stride=1, scope="fuse_P%d" % level)

                p6 = slim.conv2d(pyramid_dict['P5'] if cfgs.USE_P5 else feature_dict['C5'],
                                 num_outputs=256, kernel_size=[3, 3], padding="SAME",
                                 stride=2, scope='p6_conv')
                pyramid_dict['P6'] = p6

                p7 = tf.nn.relu(p6, name='p6_relu')

                p7 = slim.conv2d(p7,
                                 num_outputs=256, kernel_size=[3, 3], padding="SAME",
                                 stride=2, scope='p7_conv')

                pyramid_dict['P7'] = p7

        print("we are in Pyramid::-======>>>>")
        print(cfgs.LEVEL)
        print("base_anchor_size are: ", cfgs.BASE_ANCHOR_SIZE_LIST)
        print(20 * "__")

        if cfgs.USE_SUPERVISED_MASK:
            mask_list = []
            dot_layer_list = []
            with tf.variable_scope("enrich_semantics"):
                with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(cfgs.WEIGHT_DECAY),
                                    normalizer_fn=None):
                    for i, l_name in enumerate(cfgs.GENERATE_MASK_LIST):
                        G, mask, dot_layer = generate_mask(net=pyramid_dict[l_name],
                                                           num_layer=cfgs.ADDITION_LAYERS[i],
                                                           level_name=l_name)
                        # add_heatmap(G, name="MASK/G_%s" % l_name)
                        # add_heatmap(mask, name="MASK/mask_%s" % l_name)

                        # if cfgs.MASK_ACT_FET:
                        #     pyramid_dict[l_name] = pyramid_dict[l_name] * dot_layer
                        dot_layer_list.append(dot_layer)
                        mask_list.append(mask)

            return pyramid_dict, mask_list, dot_layer_list
        else:
            return pyramid_dict


def mobilenetv2_head(inputs, is_training=True):
    with slim.arg_scope(mobilenetv2_scope(is_training=is_training, trainable=True)):
        net, _ = mobilenet_v2.mobilenet(input_tensor=inputs,
                                        num_classes=None,
                                        is_training=False,
                                        depth_multiplier=1.0,
                                        scope='MobilenetV2',
                                        conv_defs=V2_HEAD_DEF,
                                        finegrain_classification_mode=False)

        net = tf.squeeze(net, [1, 2])

        return net


def generate_mask(net, num_layer, level_name):
    G = enrich_semantics_supervised(net=net,
                                    num_layer=num_layer,
                                    channels=cfgs.FPN_CHANNEL, scope="enrich_%s" % level_name)

    last_dim = 2 if cfgs.BINARY_MASK else cfgs.CLASS_NUM + 1
    mask = slim.conv2d(G, num_outputs=last_dim, kernel_size=[1, 1], stride=1, padding="SAME",
                       activation_fn=None,
                       scope='gmask_%s' % level_name)

    act_fn = tf.nn.sigmoid if cfgs.SIGMOID_ON_DOT else None
    dot_layer = slim.conv2d(G, num_outputs=cfgs.FPN_CHANNEL, kernel_size=[1, 1], stride=1, padding="SAME",
                            activation_fn=act_fn,
                            scope='gdot_%s' % level_name)

    return G, mask, dot_layer


def enrich_semantics_supervised(net, channels, num_layer, scope):
    with tf.variable_scope(scope):
        for _ in range(num_layer-1):
            net = slim.conv2d(net, num_outputs=channels, kernel_size=[3, 3], stride=1, rate=2, padding="SAME")

        net = slim.conv2d(net, num_outputs=channels, kernel_size=[3, 3], stride=1, rate=4, padding="SAME")
        net = slim.conv2d(net, num_outputs=channels, kernel_size=[1, 1], stride=1, padding="SAME")
        return net