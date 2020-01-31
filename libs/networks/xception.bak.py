import tensorflow as tf

USE_FUSED_BN = True
BN_EPSILON = 0.001
BN_MOMENTUM = 0.99


def reduced_kernel_size_for_small_input(input_tensor, kernel_size):
    shape = input_tensor.get_shape().as_list()
    if shape[1] is None or shape[2] is None:
        kernel_size_out = kernel_size
    else:
        kernel_size_out = [
          min(shape[1], kernel_size[0]), min(shape[2], kernel_size[1])
        ]
    return kernel_size_out


def relu_separable_bn_block(inputs, filters, name_prefix, is_training, data_format):
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
    inputs = tf.layers.batch_normalization(inputs, momentum=BN_MOMENTUM, name=name_prefix + '_bn', axis=bn_axis,
                                           epsilon=BN_EPSILON, training=is_training, reuse=None, fused=USE_FUSED_BN)
    return inputs


def XceptionModel(input_image, num_classes, is_training = False, data_format='channels_last'):
    bn_axis = -1 if data_format == 'channels_last' else 1
    # Entry Flow
    inputs = tf.layers.conv2d(input_image, 32, (3, 3), use_bias=False, name='block1_conv1', strides=(2, 2),
                              padding='valid', data_format=data_format, activation=None,
                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                              bias_initializer=tf.zeros_initializer())
    inputs = tf.layers.batch_normalization(inputs, momentum=BN_MOMENTUM, name='block1_conv1_bn', axis=bn_axis,
                                           epsilon=BN_EPSILON, training=is_training, reuse=None, fused=USE_FUSED_BN)
    inputs = tf.nn.relu(inputs, name='block1_conv1_act')

    inputs = tf.layers.conv2d(inputs, 64, (3, 3), use_bias=False, name='block1_conv2', strides=(1, 1),
                              padding='valid', data_format=data_format, activation=None,
                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                              bias_initializer=tf.zeros_initializer())
    inputs = tf.layers.batch_normalization(inputs, momentum=BN_MOMENTUM, name='block1_conv2_bn', axis=bn_axis,
                                           epsilon=BN_EPSILON, training=is_training, reuse=None, fused=USE_FUSED_BN)
    inputs = tf.nn.relu(inputs, name='block1_conv2_act')

    residual = tf.layers.conv2d(inputs, 128, (1, 1), use_bias=False, name='conv2d_1', strides=(2, 2),
                                padding='same', data_format=data_format, activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.zeros_initializer())
    residual = tf.layers.batch_normalization(residual, momentum=BN_MOMENTUM, name='batch_normalization_1', axis=bn_axis,
                                             epsilon=BN_EPSILON, training=is_training, reuse=None, fused=USE_FUSED_BN)

    inputs = tf.layers.separable_conv2d(inputs, 128, (3, 3),
                                        strides=(1, 1), padding='same',
                                        data_format=data_format,
                                        activation=None, use_bias=False,
                                        depthwise_initializer=tf.contrib.layers.xavier_initializer(),
                                        pointwise_initializer=tf.contrib.layers.xavier_initializer(),
                                        bias_initializer=tf.zeros_initializer(),
                                        name='block2_sepconv1', reuse=None)
    inputs = tf.layers.batch_normalization(inputs, momentum=BN_MOMENTUM, name='block2_sepconv1_bn', axis=bn_axis,
                                           epsilon=BN_EPSILON, training=is_training, reuse=None, fused=USE_FUSED_BN)

    inputs = relu_separable_bn_block(inputs, 128, 'block2_sepconv2', is_training, data_format)

    inputs = tf.layers.max_pooling2d(inputs, pool_size=(3, 3), strides=(2, 2),
                                     padding='same', data_format=data_format,
                                     name='block2_pool')

    inputs = tf.add(inputs, residual, name='residual_add_0')
    residual = tf.layers.conv2d(inputs, 256, (1, 1), use_bias=False, name='conv2d_2', strides=(2, 2),
                                padding='same', data_format=data_format, activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.zeros_initializer())
    residual = tf.layers.batch_normalization(residual, momentum=BN_MOMENTUM, name='batch_normalization_2', axis=bn_axis,
                                             epsilon=BN_EPSILON, training=is_training, reuse=None, fused=USE_FUSED_BN)

    inputs = relu_separable_bn_block(inputs, 256, 'block3_sepconv1', is_training, data_format)
    inputs = relu_separable_bn_block(inputs, 256, 'block3_sepconv2', is_training, data_format)

    inputs = tf.layers.max_pooling2d(inputs, pool_size=(3, 3), strides=(2, 2),
                                     padding='same', data_format=data_format,
                                     name='block3_pool')
    inputs = tf.add(inputs, residual, name='residual_add_1')

    residual = tf.layers.conv2d(inputs, 728, (1, 1), use_bias=False, name='conv2d_3', strides=(2, 2),
                                padding='same', data_format=data_format, activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.zeros_initializer())
    residual = tf.layers.batch_normalization(residual, momentum=BN_MOMENTUM, name='batch_normalization_3', axis=bn_axis,
                                             epsilon=BN_EPSILON, training=is_training, reuse=None, fused=USE_FUSED_BN)

    inputs = relu_separable_bn_block(inputs, 728, 'block4_sepconv1', is_training, data_format)
    inputs = relu_separable_bn_block(inputs, 728, 'block4_sepconv2', is_training, data_format)

    inputs = tf.layers.max_pooling2d(inputs, pool_size=(3, 3), strides=(2, 2),
                                     padding='same', data_format=data_format,
                                     name='block4_pool')
    inputs = tf.add(inputs, residual, name='residual_add_2')
    # Middle Flow
    for index in range(8):
        residual = inputs
        prefix = 'block' + str(index + 5)

        inputs = relu_separable_bn_block(inputs, 728, prefix + '_sepconv1', is_training, data_format)
        inputs = relu_separable_bn_block(inputs, 728, prefix + '_sepconv2', is_training, data_format)
        inputs = relu_separable_bn_block(inputs, 728, prefix + '_sepconv3', is_training, data_format)
        inputs = tf.add(inputs, residual, name=prefix + '_residual_add')
    # Exit Flow
    residual = tf.layers.conv2d(inputs, 1024, (1, 1), use_bias=False, name='conv2d_4', strides=(2, 2),
                                padding='same', data_format=data_format, activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.zeros_initializer())
    residual = tf.layers.batch_normalization(residual, momentum=BN_MOMENTUM, name='batch_normalization_4', axis=bn_axis,
                                             epsilon=BN_EPSILON, training=is_training, reuse=None, fused=USE_FUSED_BN)

    inputs = relu_separable_bn_block(inputs, 728, 'block13_sepconv1', is_training, data_format)
    inputs = relu_separable_bn_block(inputs, 1024, 'block13_sepconv2', is_training, data_format)

    inputs = tf.layers.max_pooling2d(inputs, pool_size=(3, 3), strides=(2, 2),
                                     padding='same', data_format=data_format,
                                     name='block13_pool')
    inputs = tf.add(inputs, residual, name='residual_add_3')

    inputs = tf.layers.separable_conv2d(inputs, 1536, (3, 3),
                                        strides=(1, 1), padding='same',
                                        data_format=data_format,
                                        activation=None, use_bias=False,
                                        depthwise_initializer=tf.contrib.layers.xavier_initializer(),
                                        pointwise_initializer=tf.contrib.layers.xavier_initializer(),
                                        bias_initializer=tf.zeros_initializer(),
                                        name='block14_sepconv1', reuse=None)
    inputs = tf.layers.batch_normalization(inputs, momentum=BN_MOMENTUM, name='block14_sepconv1_bn', axis=bn_axis,
                                           epsilon=BN_EPSILON, training=is_training, reuse=None, fused=USE_FUSED_BN)
    inputs = tf.nn.relu(inputs, name='block14_sepconv1_act')

    inputs = tf.layers.separable_conv2d(inputs, 2048, (3, 3),
                                        strides=(1, 1), padding='same',
                                        data_format=data_format,
                                        activation=None, use_bias=False,
                                        depthwise_initializer=tf.contrib.layers.xavier_initializer(),
                                        pointwise_initializer=tf.contrib.layers.xavier_initializer(),
                                        bias_initializer=tf.zeros_initializer(),
                                        name='block14_sepconv2', reuse=None)
    inputs = tf.layers.batch_normalization(inputs, momentum=BN_MOMENTUM, name='block14_sepconv2_bn', axis=bn_axis,
                                           epsilon=BN_EPSILON, training=is_training, reuse=None, fused=USE_FUSED_BN)
    inputs = tf.nn.relu(inputs, name='block14_sepconv2_act')

    if data_format == 'channels_first':
        channels_last_inputs = tf.transpose(inputs, [0, 2, 3, 1])
    else:
        channels_last_inputs = inputs

    inputs = tf.layers.average_pooling2d(inputs, pool_size = reduced_kernel_size_for_small_input(channels_last_inputs, [10, 10]), strides = 1, padding='valid', data_format=data_format, name='avg_pool')

    if data_format == 'channels_first':
        inputs = tf.squeeze(inputs, axis=[2, 3])
    else:
        inputs = tf.squeeze(inputs, axis=[1, 2])

    outputs = tf.layers.dense(inputs, num_classes,
                              activation=tf.nn.softmax, use_bias=True,
                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                              bias_initializer=tf.zeros_initializer(),
                              name='dense', reuse=None)

    return outputs


if __name__ == '__main__':
    '''model test samples
    '''
    import numpy as np
    # from tensorflow.python.keras._impl.keras.applications.imagenet_utils import decode_predictions  # pylint: disable=unused-import
    import scipy
    import tensorflow.contrib.slim as slim

    tf.reset_default_graph()

    input_image = tf.placeholder(tf.float32,  shape = (None, 299, 299, 3), name = 'input_placeholder')
    outputs = XceptionModel(input_image, 1000, is_training = True, data_format='channels_last')

    saver = tf.train.Saver()

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        model_variables = tf.trainable_variables()
        print(model_variables)

        saver.restore(sess, "/data/RetinaNet_TF/data/pretrained_weights/xception_tf_model/xception_model.ckpt")

        image_file = ['test_images/000013.jpg', 'test_images/000018.jpg', 'test_images/000031.jpg', 'test_images/000038.jpg', 'test_images/000045.jpg']
        image_array = []
        for file in image_file:
            np_image = scipy.misc.imread(file, mode='RGB')
            np_image = scipy.misc.imresize(np_image, (299, 299))
            np_image = np.expand_dims(np_image, axis=0).astype(np.float32)
            image_array.append(np_image)
            np_image = np.concatenate(image_array, axis=0)
            np_image /= 127.5
            np_image -= 1.
            #np_image = np.transpose(np_image, (0, 3, 1, 2))
            predict = sess.run(outputs, feed_dict = {input_image : np_image})
            #print(predict)
            print(np.argmax(predict))
            # print('Predicted:', decode_predictions(predict, 1))


