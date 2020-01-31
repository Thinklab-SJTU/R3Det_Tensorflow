import tensorflow as tf
import tensorflow.contrib.slim as slim

from libs.configs import cfgs

USE_FUSED_BN = True
BN_EPSILON = 0.001
BN_MOMENTUM = 0.99


def fusion_two_layer(C_i, P_j, scope):
    '''
    i = j+1
    :param C_i: shape is [1, h, w, c]
    :param P_j: shape is [1, h/2, w/2, 256]
    :return:
    P_i
    '''
    with tf.variable_scope(scope):
        level_name = scope.split('_')[1]

        h, w = tf.shape(C_i)[1], tf.shape(C_i)[2]
        upsample_p = tf.image.resize_bilinear(P_j,
                                              size=[h, w],
                                              name='up_sample_'+level_name)

        reduce_dim_c = slim.conv2d(C_i,
                                   num_outputs=256,
                                   kernel_size=[1, 1], stride=1,
                                   scope='reduce_dim_'+level_name)

        add_f = 0.5*upsample_p + 0.5*reduce_dim_c

        # P_i = slim.conv2d(add_f,
        #                   num_outputs=256, kernel_size=[3, 3], stride=1,
        #                   padding='SAME',
        #                   scope='fusion_'+level_name)
        return add_f


def reduced_kernel_size_for_small_input(input_tensor, kernel_size):
    shape = input_tensor.get_shape().as_list()
    if shape[1] is None or shape[2] is None:
        kernel_size_out = kernel_size
    else:
        kernel_size_out = [
          min(shape[1], kernel_size[0]), min(shape[2], kernel_size[1])
        ]
    return kernel_size_out


def relu_separable_bn_block(inputs, filters, name_prefix, is_training, data_format, has_bn=False):
    bn_axis = -1 if data_format == 'channels_last' else 1

    inputs = tf.nn.relu(inputs, name=name_prefix + '_act')
    inputs = tf.layers.separable_conv2d(inputs, filters, (3, 3),
                                        strides=(1, 1), padding='same',
                                        data_format=data_format,
                                        activation=None, use_bias=False,
                                        depthwise_initializer=tf.contrib.layers.xavier_initializer(),
                                        pointwise_initializer=tf.contrib.layers.xavier_initializer(),
                                        bias_initializer=tf.zeros_initializer(),
                                        name=name_prefix, reuse=None)
    # inputs = tf.layers.batch_normalization(inputs, momentum=BN_MOMENTUM, name=name_prefix + '_bn', axis=bn_axis,
    #                                        epsilon=BN_EPSILON, training=has_bn, reuse=None, fused=USE_FUSED_BN)
    return inputs


def XceptionModel(input_image, num_classes, is_training=False, has_bn=False, data_format='channels_last'):
    feature_dict = {}

    bn_axis = -1 if data_format == 'channels_last' else 1
    # Entry Flow
    inputs = tf.layers.conv2d(input_image, 32, (3, 3), use_bias=False, name='block1_conv1', strides=(2, 2),
                              padding='valid', data_format=data_format, activation=None,
                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                              bias_initializer=tf.zeros_initializer())
    # inputs = tf.layers.batch_normalization(inputs, momentum=BN_MOMENTUM, name='block1_conv1_bn', axis=bn_axis,
    #                                        epsilon=BN_EPSILON, training=has_bn, reuse=None, fused=USE_FUSED_BN)
    inputs = tf.nn.relu(inputs, name='block1_conv1_act')

    inputs = tf.layers.conv2d(inputs, 64, (3, 3), use_bias=False, name='block1_conv2', strides=(1, 1),
                              padding='valid', data_format=data_format, activation=None,
                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                              bias_initializer=tf.zeros_initializer())
    # inputs = tf.layers.batch_normalization(inputs, momentum=BN_MOMENTUM, name='block1_conv2_bn', axis=bn_axis,
    #                                        epsilon=BN_EPSILON, training=has_bn, reuse=None, fused=USE_FUSED_BN)
    inputs = tf.nn.relu(inputs, name='block1_conv2_act')

    residual = tf.layers.conv2d(inputs, 128, (1, 1), use_bias=False, name='conv2d_1', strides=(2, 2),
                                padding='same', data_format=data_format, activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.zeros_initializer())
    # residual = tf.layers.batch_normalization(residual, momentum=BN_MOMENTUM, name='batch_normalization_1', axis=bn_axis,
    #                                          epsilon=BN_EPSILON, training=has_bn, reuse=None, fused=USE_FUSED_BN)

    inputs = tf.layers.separable_conv2d(inputs, 128, (3, 3),
                                        strides=(1, 1), padding='same',
                                        data_format=data_format,
                                        activation=None, use_bias=False,
                                        depthwise_initializer=tf.contrib.layers.xavier_initializer(),
                                        pointwise_initializer=tf.contrib.layers.xavier_initializer(),
                                        bias_initializer=tf.zeros_initializer(),
                                        name='block2_sepconv1', reuse=None)
    # inputs = tf.layers.batch_normalization(inputs, momentum=BN_MOMENTUM, name='block2_sepconv1_bn', axis=bn_axis,
    #                                        epsilon=BN_EPSILON, training=has_bn, reuse=None, fused=USE_FUSED_BN)

    inputs = relu_separable_bn_block(inputs, 128, 'block2_sepconv2', is_training, data_format, has_bn)

    inputs = tf.layers.max_pooling2d(inputs, pool_size=(3, 3), strides=(2, 2),
                                     padding='same', data_format=data_format,
                                     name='block2_pool')

    feature_dict['C2'] = inputs

    inputs = tf.add(inputs, residual, name='residual_add_0')
    residual = tf.layers.conv2d(inputs, 256, (1, 1), use_bias=False, name='conv2d_2', strides=(2, 2),
                                padding='same', data_format=data_format, activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.zeros_initializer())
    # residual = tf.layers.batch_normalization(residual, momentum=BN_MOMENTUM, name='batch_normalization_2', axis=bn_axis,
    #                                          epsilon=BN_EPSILON, training=has_bn, reuse=None, fused=USE_FUSED_BN)

    inputs = relu_separable_bn_block(inputs, 256, 'block3_sepconv1', is_training, data_format, has_bn)
    inputs = relu_separable_bn_block(inputs, 256, 'block3_sepconv2', is_training, data_format, has_bn)

    inputs = tf.layers.max_pooling2d(inputs, pool_size=(3, 3), strides=(2, 2),
                                     padding='same', data_format=data_format,
                                     name='block3_pool')
    inputs = tf.add(inputs, residual, name='residual_add_1')

    feature_dict['C3'] = inputs

    residual = tf.layers.conv2d(inputs, 728, (1, 1), use_bias=False, name='conv2d_3', strides=(2, 2),
                                padding='same', data_format=data_format, activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.zeros_initializer())
    # residual = tf.layers.batch_normalization(residual, momentum=BN_MOMENTUM, name='batch_normalization_3', axis=bn_axis,
    #                                          epsilon=BN_EPSILON, training=has_bn, reuse=None, fused=USE_FUSED_BN)

    inputs = relu_separable_bn_block(inputs, 728, 'block4_sepconv1', is_training, data_format, has_bn)
    inputs = relu_separable_bn_block(inputs, 728, 'block4_sepconv2', is_training, data_format, has_bn)

    inputs = tf.layers.max_pooling2d(inputs, pool_size=(3, 3), strides=(2, 2),
                                     padding='same', data_format=data_format,
                                     name='block4_pool')
    inputs = tf.add(inputs, residual, name='residual_add_2')

    feature_dict['C4'] = inputs

    # Middle Flow
    for index in range(8):
        residual = inputs
        prefix = 'block' + str(index + 5)

        inputs = relu_separable_bn_block(inputs, 728, prefix + '_sepconv1', is_training, data_format, has_bn)
        inputs = relu_separable_bn_block(inputs, 728, prefix + '_sepconv2', is_training, data_format, has_bn)
        inputs = relu_separable_bn_block(inputs, 728, prefix + '_sepconv3', is_training, data_format, has_bn)
        inputs = tf.add(inputs, residual, name=prefix + '_residual_add')
    # Exit Flow
    residual = tf.layers.conv2d(inputs, 1024, (1, 1), use_bias=False, name='conv2d_4', strides=(2, 2),
                                padding='same', data_format=data_format, activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.zeros_initializer())
    # residual = tf.layers.batch_normalization(residual, momentum=BN_MOMENTUM, name='batch_normalization_4', axis=bn_axis,
    #                                          epsilon=BN_EPSILON, training=has_bn, reuse=None, fused=USE_FUSED_BN)

    inputs = relu_separable_bn_block(inputs, 728, 'block13_sepconv1', is_training, data_format, has_bn)
    inputs = relu_separable_bn_block(inputs, 1024, 'block13_sepconv2', is_training, data_format, has_bn)

    inputs = tf.layers.max_pooling2d(inputs, pool_size=(3, 3), strides=(2, 2),
                                     padding='same', data_format=data_format,
                                     name='block13_pool')
    inputs = tf.add(inputs, residual, name='residual_add_3')

    feature_dict['C5'] = inputs

    # inputs = tf.layers.separable_conv2d(inputs, 1536, (3, 3),
    #                                     strides=(1, 1), padding='same',
    #                                     data_format=data_format,
    #                                     activation=None, use_bias=False,
    #                                     depthwise_initializer=tf.contrib.layers.xavier_initializer(),
    #                                     pointwise_initializer=tf.contrib.layers.xavier_initializer(),
    #                                     bias_initializer=tf.zeros_initializer(),
    #                                     name='block14_sepconv1', reuse=None)
    # inputs = tf.layers.batch_normalization(inputs, momentum=BN_MOMENTUM, name='block14_sepconv1_bn', axis=bn_axis,
    #                                        epsilon=BN_EPSILON, training=has_bn, reuse=None, fused=USE_FUSED_BN)
    # inputs = tf.nn.relu(inputs, name='block14_sepconv1_act')
    #
    # inputs = tf.layers.separable_conv2d(inputs, 2048, (3, 3),
    #                                     strides=(1, 1), padding='same',
    #                                     data_format=data_format,
    #                                     activation=None, use_bias=False,
    #                                     depthwise_initializer=tf.contrib.layers.xavier_initializer(),
    #                                     pointwise_initializer=tf.contrib.layers.xavier_initializer(),
    #                                     bias_initializer=tf.zeros_initializer(),
    #                                     name='block14_sepconv2', reuse=None)
    # inputs = tf.layers.batch_normalization(inputs, momentum=BN_MOMENTUM, name='block14_sepconv2_bn', axis=bn_axis,
    #                                        epsilon=BN_EPSILON, training=has_bn, reuse=None, fused=USE_FUSED_BN)
    # inputs = tf.nn.relu(inputs, name='block14_sepconv2_act')
    #
    # if data_format == 'channels_first':
    #     channels_last_inputs = tf.transpose(inputs, [0, 2, 3, 1])
    # else:
    #     channels_last_inputs = inputs
    #
    # inputs = tf.layers.average_pooling2d(inputs, pool_size=reduced_kernel_size_for_small_input(channels_last_inputs, [10, 10]), strides = 1, padding='valid', data_format=data_format, name='avg_pool')
    #
    # if data_format == 'channels_first':
    #     inputs = tf.squeeze(inputs, axis=[2, 3])
    # else:
    #     inputs = tf.squeeze(inputs, axis=[1, 2])
    #
    # outputs = tf.layers.dense(inputs, num_classes,
    #                           activation=tf.nn.softmax, use_bias=True,
    #                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
    #                           bias_initializer=tf.zeros_initializer(),
    #                           name='dense', reuse=None)

    return feature_dict


def xception_base(input_image, is_training=True):
    feature_dict = XceptionModel(input_image, 1000, is_training=is_training, has_bn=False, data_format='channels_last')

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

    # for level in range(7, 1, -1):
    #     add_heatmap(pyramid_dict['P%d' % level], name='Layer%d/P%d_heat' % (level, level))

    return pyramid_dict

