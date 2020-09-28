# -*-coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from libs.networks import resnet, resnet_gluoncv_r3det_plusplus, mobilenet_v2, xception, mobilenet_v2_r3det_plusplus
from libs.box_utils import anchor_utils, generate_anchors, generate_rotate_anchors
from libs.configs import cfgs
from libs.losses import losses
from libs.box_utils import show_box_in_tensor
from libs.detection_oprations.refine_proposal_opr import postprocess_detctions
from libs.detection_oprations.anchor_target_layer_without_boxweight import anchor_target_layer
from libs.detection_oprations.refinebox_target_layer_without_boxweight import refinebox_target_layer
from libs.box_utils import bbox_transform
from libs.box_utils import mask_utils


class DetectionNetwork(object):

    def __init__(self, base_network_name, is_training):

        self.base_network_name = base_network_name
        self.is_training = is_training
        if cfgs.METHOD == 'H':
            self.num_anchors_per_location = len(cfgs.ANCHOR_SCALES) * len(cfgs.ANCHOR_RATIOS)
        else:
            self.num_anchors_per_location = len(cfgs.ANCHOR_SCALES) * len(cfgs.ANCHOR_RATIOS) * len(cfgs.ANCHOR_ANGLES)
        self.method = cfgs.METHOD
        self.losses_dict = {}

    def build_base_network(self, input_img_batch):

        if self.base_network_name.startswith('resnet_v1'):
            return resnet.resnet_base(input_img_batch, scope_name=self.base_network_name, is_training=self.is_training)

        elif self.base_network_name in ['resnet152_v1d', 'resnet101_v1d', 'resnet50_v1d']:

            return resnet_gluoncv_r3det_plusplus.resnet_base(input_img_batch, scope_name=self.base_network_name,
                                                             is_training=self.is_training)

        elif self.base_network_name.startswith('MobilenetV2'):
            return mobilenet_v2_r3det_plusplus.mobilenetv2_base(input_img_batch, is_training=self.is_training)

        elif self.base_network_name.startswith('xception'):
            return xception.xception_base(input_img_batch, is_training=self.is_training)

        else:
            raise ValueError('Sry, we only support resnet, mobilenet_v2 and xception')

    def rpn_cls_net(self, inputs, scope_list, reuse_flag, level):
        rpn_conv2d_3x3 = inputs
        for i in range(4):
            rpn_conv2d_3x3 = slim.conv2d(inputs=rpn_conv2d_3x3,
                                         num_outputs=cfgs.FPN_CHANNEL,
                                         kernel_size=[3, 3],
                                         stride=1,
                                         activation_fn=tf.nn.relu,
                                         weights_initializer=cfgs.SUBNETS_WEIGHTS_INITIALIZER,
                                         biases_initializer=cfgs.SUBNETS_BIAS_INITIALIZER,
                                         scope='{}_{}'.format(scope_list[0], i),
                                         reuse=reuse_flag)

        rpn_box_scores = slim.conv2d(rpn_conv2d_3x3,
                                     num_outputs=cfgs.CLASS_NUM * self.num_anchors_per_location,
                                     kernel_size=[3, 3],
                                     stride=1,
                                     weights_initializer=cfgs.SUBNETS_WEIGHTS_INITIALIZER,
                                     biases_initializer=cfgs.FINAL_CONV_BIAS_INITIALIZER,
                                     scope=scope_list[2],
                                     activation_fn=None,
                                     reuse=reuse_flag)

        rpn_box_scores = tf.reshape(rpn_box_scores, [-1, cfgs.CLASS_NUM],
                                    name='rpn_{}_classification_reshape'.format(level))
        rpn_box_probs = tf.sigmoid(rpn_box_scores, name='rpn_{}_classification_sigmoid'.format(level))

        return rpn_box_scores, rpn_box_probs

    def refine_cls_net(self, inputs, scope_list, reuse_flag, level):
        rpn_conv2d_3x3 = inputs
        for i in range(4):
            rpn_conv2d_3x3 = slim.conv2d(inputs=rpn_conv2d_3x3,
                                         num_outputs=cfgs.FPN_CHANNEL,
                                         kernel_size=[3, 3],
                                         stride=1,
                                         activation_fn=tf.nn.relu,
                                         weights_initializer=cfgs.SUBNETS_WEIGHTS_INITIALIZER,
                                         biases_initializer=cfgs.SUBNETS_BIAS_INITIALIZER,
                                         scope='{}_{}'.format(scope_list[0], i),
                                         reuse=reuse_flag)

        rpn_box_scores = slim.conv2d(rpn_conv2d_3x3,
                                     num_outputs=cfgs.CLASS_NUM,
                                     kernel_size=[3, 3],
                                     stride=1,
                                     weights_initializer=cfgs.SUBNETS_WEIGHTS_INITIALIZER,
                                     biases_initializer=cfgs.FINAL_CONV_BIAS_INITIALIZER,
                                     scope=scope_list[2],
                                     activation_fn=None,
                                     reuse=reuse_flag)

        rpn_box_scores = tf.reshape(rpn_box_scores, [-1, cfgs.CLASS_NUM],
                                    name='refine_{}_classification_reshape'.format(level))
        rpn_box_probs = tf.sigmoid(rpn_box_scores, name='refine_{}_classification_sigmoid'.format(level))

        return rpn_box_scores, rpn_box_probs

    def rpn_reg_net(self, inputs, scope_list, reuse_flag, level):
        rpn_delta_boxes = inputs
        for i in range(4):
            rpn_delta_boxes = slim.conv2d(inputs=rpn_delta_boxes,
                                          num_outputs=cfgs.FPN_CHANNEL,
                                          kernel_size=[3, 3],
                                          weights_initializer=cfgs.SUBNETS_WEIGHTS_INITIALIZER,
                                          biases_initializer=cfgs.SUBNETS_BIAS_INITIALIZER,
                                          stride=1,
                                          activation_fn=tf.nn.relu,
                                          scope='{}_{}'.format(scope_list[1], i),
                                          reuse=reuse_flag)

        rpn_delta_boxes = slim.conv2d(rpn_delta_boxes,
                                      num_outputs=5 * self.num_anchors_per_location,
                                      kernel_size=[3, 3],
                                      stride=1,
                                      weights_initializer=cfgs.SUBNETS_WEIGHTS_INITIALIZER,
                                      biases_initializer=cfgs.SUBNETS_BIAS_INITIALIZER,
                                      scope=scope_list[3],
                                      activation_fn=None,
                                      reuse=reuse_flag)

        rpn_delta_boxes = tf.reshape(rpn_delta_boxes, [-1, 5],
                                     name='rpn_{}_regression_reshape'.format(level))
        return rpn_delta_boxes

    def refine_reg_net(self, inputs, scope_list, reuse_flag, level):
        rpn_delta_boxes = inputs
        for i in range(4):
            rpn_delta_boxes = slim.conv2d(inputs=rpn_delta_boxes,
                                          num_outputs=cfgs.FPN_CHANNEL,
                                          kernel_size=[3, 3],
                                          weights_initializer=cfgs.SUBNETS_WEIGHTS_INITIALIZER,
                                          biases_initializer=cfgs.SUBNETS_BIAS_INITIALIZER,
                                          stride=1,
                                          activation_fn=tf.nn.relu,
                                          scope='{}_{}'.format(scope_list[1], i),
                                          reuse=reuse_flag)

        rpn_delta_boxes = slim.conv2d(rpn_delta_boxes,
                                      num_outputs=5,
                                      kernel_size=[3, 3],
                                      stride=1,
                                      weights_initializer=cfgs.SUBNETS_WEIGHTS_INITIALIZER,
                                      biases_initializer=cfgs.SUBNETS_BIAS_INITIALIZER,
                                      scope=scope_list[3],
                                      activation_fn=None,
                                      reuse=reuse_flag)

        rpn_delta_boxes = tf.reshape(rpn_delta_boxes, [-1, 5],
                                     name='refine_{}_regression_reshape'.format(level))
        return rpn_delta_boxes

    def rpn_net(self, feature_pyramid, name):

        rpn_delta_boxes_list = []
        rpn_scores_list = []
        rpn_probs_list = []
        with tf.variable_scope(name):
            with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(cfgs.WEIGHT_DECAY)):
                for level in cfgs.LEVEL:

                    if cfgs.SHARE_NET:
                        reuse_flag = None if level == 'P3' else True
                        scope_list = ['conv2d_3x3_cls', 'conv2d_3x3_reg', 'rpn_classification', 'rpn_regression']
                    else:
                        reuse_flag = None
                        scope_list = ['conv2d_3x3_cls_' + level, 'conv2d_3x3_reg_' + level,
                                      'rpn_classification_' + level, 'rpn_regression_' + level]

                    rpn_box_scores, rpn_box_probs = self.rpn_cls_net(feature_pyramid[level],
                                                                     scope_list, reuse_flag,
                                                                     level)
                    rpn_delta_boxes = self.rpn_reg_net(feature_pyramid[level], scope_list, reuse_flag, level)

                    rpn_scores_list.append(rpn_box_scores)
                    rpn_probs_list.append(rpn_box_probs)
                    rpn_delta_boxes_list.append(rpn_delta_boxes)

                # rpn_all_delta_boxes = tf.concat(rpn_delta_boxes_list, axis=0)
                # rpn_all_boxes_scores = tf.concat(rpn_scores_list, axis=0)
                # rpn_all_boxes_probs = tf.concat(rpn_probs_list, axis=0)

            return rpn_delta_boxes_list, rpn_scores_list, rpn_probs_list

    def refine_net(self, feature_pyramid, name):

        refine_delta_boxes_list = []
        refine_scores_list = []
        refine_probs_list = []
        with tf.variable_scope(name):
            with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(cfgs.WEIGHT_DECAY)):
                for level in cfgs.LEVEL:

                    if cfgs.SHARE_NET:
                        reuse_flag = None if level == 'P3' else True
                        scope_list = ['conv2d_3x3_cls', 'conv2d_3x3_reg', 'refine_classification', 'refine_regression']
                    else:
                        reuse_flag = None
                        scope_list = ['conv2d_3x3_cls_' + level, 'conv2d_3x3_reg_' + level,
                                      'refine_classification_' + level, 'refine_regression_' + level]

                    refine_box_scores, refine_box_probs = self.refine_cls_net(feature_pyramid[level],
                                                                              scope_list, reuse_flag,
                                                                              level)
                    refine_delta_boxes = self.refine_reg_net(feature_pyramid[level], scope_list, reuse_flag, level)

                    refine_scores_list.append(refine_box_scores)
                    refine_probs_list.append(refine_box_probs)
                    refine_delta_boxes_list.append(refine_delta_boxes)

                # refine_all_delta_boxes = tf.concat(refine_delta_boxes_list, axis=0)
                # refine_all_boxes_scores = tf.concat(refine_scores_list, axis=0)
                # refine_all_boxes_probs = tf.concat(refine_probs_list, axis=0)

            return refine_delta_boxes_list, refine_scores_list, refine_probs_list

    def generate_mask(self, mask_list, img_shape, gtboxes_batch_h, gtboxes_batch_r, feature_pyramid):
        mask_gt_list = []

        for i, l in enumerate(cfgs.LEVEL):
            p_h, p_w = tf.shape(feature_pyramid[l])[1], tf.shape(feature_pyramid[l])[2]
            if cfgs.USE_SUPERVISED_MASK and i < len(mask_list) and self.is_training:
                if cfgs.MASK_TYPE.strip() == 'h':
                    mask = tf.py_func(mask_utils.make_gt_mask,
                                      [p_h, p_w, img_shape[1], img_shape[2], gtboxes_batch_h],
                                      Tout=tf.int32)
                elif cfgs.MASK_TYPE.strip() == 'r':
                    mask = tf.py_func(mask_utils.make_r_gt_mask,
                                      [p_h, p_w, img_shape[1], img_shape[2], gtboxes_batch_r],
                                      Tout=tf.int32)
                if cfgs.BINARY_MASK:
                    mask = tf.where(tf.greater(mask, 0), tf.ones_like(mask), tf.zeros_like(mask))
                mask_gt_list.append(mask)
                mask_utils.vis_mask_tfsmry(mask, name="MASK/%s" % l)
        return mask_gt_list

    def make_anchors(self, feature_pyramid):
        with tf.variable_scope('make_anchors'):
            anchor_list = []
            level_list = cfgs.LEVEL
            with tf.name_scope('make_anchors_all_level'):
                for level, base_anchor_size, stride in zip(level_list, cfgs.BASE_ANCHOR_SIZE_LIST, cfgs.ANCHOR_STRIDE):
                    '''
                    (level, base_anchor_size) tuple:
                    (P3, 32), (P4, 64), (P5, 128), (P6, 256), (P7, 512)
                    '''
                    featuremap_height, featuremap_width = tf.shape(feature_pyramid[level])[1], \
                                                          tf.shape(feature_pyramid[level])[2]

                    featuremap_height = tf.cast(featuremap_height, tf.float32)
                    featuremap_width = tf.cast(featuremap_width, tf.float32)

                    if self.method == 'H':
                        tmp_anchors = tf.py_func(generate_anchors.generate_anchors_pre,
                                                 inp=[featuremap_height, featuremap_width, stride,
                                                      np.array(cfgs.ANCHOR_SCALES) * stride, cfgs.ANCHOR_RATIOS, 4.0],
                                                 Tout=[tf.float32])

                        tmp_anchors = tf.reshape(tmp_anchors, [-1, 4])
                    else:
                        tmp_anchors = generate_rotate_anchors.make_anchors(base_anchor_size=base_anchor_size,
                                                                           anchor_scales=cfgs.ANCHOR_SCALES,
                                                                           anchor_ratios=cfgs.ANCHOR_RATIOS,
                                                                           anchor_angles=cfgs.ANCHOR_ANGLES,
                                                                           featuremap_height=featuremap_height,
                                                                           featuremap_width=featuremap_width,
                                                                           stride=stride)
                        tmp_anchors = tf.reshape(tmp_anchors, [-1, 5])

                    anchor_list.append(tmp_anchors)

                # all_level_anchors = tf.concat(anchor_list, axis=0)
            return anchor_list

    def add_anchor_img_smry(self, img, anchors, labels, method):

        positive_anchor_indices = tf.reshape(tf.where(tf.greater_equal(labels, 1)), [-1])
        # negative_anchor_indices = tf.reshape(tf.where(tf.equal(labels, 0)), [-1])

        positive_anchor = tf.gather(anchors, positive_anchor_indices)
        # negative_anchor = tf.gather(anchors, negative_anchor_indices)

        pos_in_img = show_box_in_tensor.only_draw_boxes(img_batch=img,
                                                        boxes=positive_anchor,
                                                        method=method)
        # neg_in_img = show_box_in_tensor.only_draw_boxes(img_batch=img,
        #                                                 boxes=negative_anchor)

        tf.summary.image('positive_anchor', pos_in_img)
        # tf.summary.image('negative_anchors', neg_in_img)

    def build_whole_detection_network(self, input_img_batch, gtboxes_batch_h, gtboxes_batch_r, gpu_id=0):

        if self.is_training:
            gtboxes_batch_h = tf.reshape(gtboxes_batch_h, [-1, 5])
            gtboxes_batch_h = tf.cast(gtboxes_batch_h, tf.float32)

            gtboxes_batch_r = tf.reshape(gtboxes_batch_r, [-1, 6])
            gtboxes_batch_r = tf.cast(gtboxes_batch_r, tf.float32)

        img_shape = tf.shape(input_img_batch)

        # 1. build base network
        if cfgs.USE_SUPERVISED_MASK:
            feature_pyramid, mask_list, dot_layer_list = self.build_base_network(input_img_batch)
        else:
            feature_pyramid = self.build_base_network(input_img_batch)
            dot_layer_list = None
            mask_list = []

        # 2. build rpn
        # if cfgs.USE_SUPERVISED_MASK:
        #     for i, d in enumerate(dot_layer_list):
        #         feature_pyramid['P{}'.format(i + 3)] *= d
        rpn_box_pred_list, rpn_cls_score_list, rpn_cls_prob_list = self.rpn_net(feature_pyramid, 'rpn_net')

        # 3. generate_anchors and mask
        anchor_list = self.make_anchors(feature_pyramid)

        if cfgs.USE_SUPERVISED_MASK:
            mask_gt_list = self.generate_mask(mask_list, img_shape, gtboxes_batch_h, gtboxes_batch_r, feature_pyramid)

        rpn_box_pred = tf.concat(rpn_box_pred_list, axis=0)
        rpn_cls_score = tf.concat(rpn_cls_score_list, axis=0)
        # rpn_cls_prob = tf.concat(rpn_cls_prob_list, axis=0)
        anchors = tf.concat(anchor_list, axis=0)

        if self.is_training:
            with tf.variable_scope('build_loss'):
                labels, target_delta, anchor_states, target_boxes = tf.py_func(func=anchor_target_layer,
                                                                               inp=[gtboxes_batch_h, gtboxes_batch_r,
                                                                                    anchors],
                                                                               Tout=[tf.float32, tf.float32,
                                                                                     tf.float32, tf.float32])

                if self.method == 'H':
                    self.add_anchor_img_smry(input_img_batch, anchors, anchor_states, 0)
                else:
                    self.add_anchor_img_smry(input_img_batch, anchors, anchor_states, 1)

                cls_loss = losses.focal_loss(labels, rpn_cls_score, anchor_states)
                if cfgs.USE_IOU_FACTOR:
                    reg_loss = losses.iou_smooth_l1_loss_(target_delta, rpn_box_pred, anchor_states, target_boxes, anchors)
                else:
                    reg_loss = losses.smooth_l1_loss(target_delta, rpn_box_pred, anchor_states)

                if cfgs.USE_SUPERVISED_MASK:
                    with tf.variable_scope("supervised_mask_loss"):
                        mask_loss = 0.0
                        for i in range(len(mask_list)):
                            a_mask, a_mask_gt = mask_list[i], mask_gt_list[i]
                            # b, h, w, c = a_mask.shape
                            last_dim = 2 if cfgs.BINARY_MASK else cfgs.CLASS_NUM + 1
                            a_mask = tf.reshape(a_mask, shape=[-1, last_dim])
                            a_mask_gt = tf.reshape(a_mask_gt, shape=[-1])
                            a_mask_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=a_mask,
                                                                                                        labels=a_mask_gt))
                            mask_loss += a_mask_loss
                        self.losses_dict['mask_loss'] = mask_loss * cfgs.SUPERVISED_MASK_LOSS_WEIGHT / float(len(mask_list))

                self.losses_dict['cls_loss'] = cls_loss * cfgs.CLS_WEIGHT
                self.losses_dict['reg_loss'] = reg_loss * cfgs.REG_WEIGHT

        box_pred_list, cls_prob_list, proposal_list = rpn_box_pred_list, rpn_cls_prob_list, anchor_list

        all_box_pred_list, all_cls_prob_list, all_proposal_list = [], [], []

        for i in range(cfgs.NUM_REFINE_STAGE):
            box_pred_list, cls_prob_list, proposal_list = self.refine_stage(input_img_batch,
                                                                            gtboxes_batch_r,
                                                                            box_pred_list,
                                                                            cls_prob_list,
                                                                            proposal_list,
                                                                            feature_pyramid,
                                                                            dot_layer_list,
                                                                            gpu_id,
                                                                            pos_threshold=cfgs.REFINE_IOU_POSITIVE_THRESHOLD[i],
                                                                            neg_threshold=cfgs.REFINE_IOU_NEGATIVE_THRESHOLD[i],
                                                                            stage='' if i == 0 else '_stage{}'.format(i + 2),
                                                                            proposal_filter=True if i == 0 else False)

            if not self.is_training:
                all_box_pred_list.extend(box_pred_list)
                all_cls_prob_list.extend(cls_prob_list)
                all_proposal_list.extend(proposal_list)
            else:
                all_box_pred_list, all_cls_prob_list, all_proposal_list = box_pred_list, cls_prob_list, proposal_list

        with tf.variable_scope('postprocess_detctions'):
            box_pred = tf.concat(all_box_pred_list, axis=0)
            cls_prob = tf.concat(all_cls_prob_list, axis=0)
            proposal = tf.concat(all_proposal_list, axis=0)

            boxes, scores, category = postprocess_detctions(refine_bbox_pred=box_pred,
                                                            refine_cls_prob=cls_prob,
                                                            anchors=proposal,
                                                            is_training=self.is_training)
            boxes = tf.stop_gradient(boxes)
            scores = tf.stop_gradient(scores)
            category = tf.stop_gradient(category)

        if self.is_training:
            return boxes, scores, category, self.losses_dict
        else:
            return boxes, scores, category

    def get_restorer(self):
        checkpoint_path = tf.train.latest_checkpoint(os.path.join(cfgs.TRAINED_CKPT, cfgs.VERSION))

        if checkpoint_path != None:
            if cfgs.RESTORE_FROM_RPN:
                print('___restore from rpn___')
                model_variables = slim.get_model_variables()
                restore_variables = [var for var in model_variables if not var.name.startswith('FastRCNN_Head')] + \
                                    [slim.get_or_create_global_step()]
                for var in restore_variables:
                    print(var.name)
                restorer = tf.train.Saver(restore_variables)
            else:
                restorer = tf.train.Saver()
            print("model restore from :", checkpoint_path)
        else:
            checkpoint_path = cfgs.PRETRAINED_CKPT
            print("model restore from pretrained mode, path is :", checkpoint_path)

            model_variables = slim.get_model_variables()

            # for var in model_variables:
            #     print(var.name)
            # print(20*"__++__++__")

            def name_in_ckpt_rpn(var):
                return var.op.name

            def name_in_ckpt_fastrcnn_head(var):
                '''
                Fast-RCNN/resnet_v1_50/block4 -->resnet_v1_50/block4
                Fast-RCNN/MobilenetV2/** -- > MobilenetV2 **
                :param var:
                :return:
                '''
                return '/'.join(var.op.name.split('/')[1:])

            nameInCkpt_Var_dict = {}
            for var in model_variables:
                if var.name.startswith('Fast-RCNN/'+self.base_network_name):  # +'/block4'
                    var_name_in_ckpt = name_in_ckpt_fastrcnn_head(var)
                    nameInCkpt_Var_dict[var_name_in_ckpt] = var
                else:
                    if var.name.startswith(self.base_network_name):
                        var_name_in_ckpt = name_in_ckpt_rpn(var)
                        nameInCkpt_Var_dict[var_name_in_ckpt] = var
                    else:
                        continue
            restore_variables = nameInCkpt_Var_dict
            for key, item in restore_variables.items():
                print("var_in_graph: ", item.name)
                print("var_in_ckpt: ", key)
                print(20*"___")
            restorer = tf.train.Saver(restore_variables)
            print(20 * "****")
            print("restore from pretrained_weighs in IMAGE_NET")
        return restorer, checkpoint_path

    def get_gradients(self, optimizer, loss):
        '''

        :param optimizer:
        :param loss:
        :return:

        return vars and grads that not be fixed
        '''

        # if cfgs.FIXED_BLOCKS > 0:
        #     trainable_vars = tf.trainable_variables()
        #     # trained_vars = slim.get_trainable_variables()
        #     start_names = [cfgs.NET_NAME + '/block%d'%i for i in range(1, cfgs.FIXED_BLOCKS+1)] + \
        #                   [cfgs.NET_NAME + '/conv1']
        #     start_names = tuple(start_names)
        #     trained_var_list = []
        #     for var in trainable_vars:
        #         if not var.name.startswith(start_names):
        #             trained_var_list.append(var)
        #     # slim.learning.train()
        #     grads = optimizer.compute_gradients(loss, var_list=trained_var_list)
        #     return grads
        # else:
        #     return optimizer.compute_gradients(loss)
        return optimizer.compute_gradients(loss)

    def enlarge_gradients_for_bias(self, gradients):

        final_gradients = []
        with tf.variable_scope("Gradient_Mult") as scope:
            for grad, var in gradients:
                scale = 1.0
                if cfgs.MUTILPY_BIAS_GRADIENT and './biases' in var.name:
                    scale = scale * cfgs.MUTILPY_BIAS_GRADIENT
                if not np.allclose(scale, 1.0):
                    grad = tf.multiply(grad, scale)
                final_gradients.append((grad, var))
        return final_gradients

    def refine_feature_op(self, points, feature_map, name):

        h, w = tf.cast(tf.shape(feature_map)[1], tf.int32), tf.cast(tf.shape(feature_map)[2], tf.int32)

        xmin = tf.maximum(0.0, tf.floor(points[:, 0]))
        xmin = tf.minimum(tf.cast(w - 1, tf.float32), tf.ceil(xmin))

        ymin = tf.maximum(0.0, tf.floor(points[:, 1]))
        ymin = tf.minimum(tf.cast(h - 1, tf.float32), tf.ceil(ymin))

        xmax = tf.minimum(tf.cast(w - 1, tf.float32), tf.ceil(points[:, 0]))
        xmax = tf.maximum(0.0, tf.floor(xmax))

        ymax = tf.minimum(tf.cast(h - 1, tf.float32), tf.ceil(points[:, 1]))
        ymax = tf.maximum(0.0, tf.floor(ymax))

        left_top = tf.cast(tf.transpose(tf.stack([ymin, xmin], axis=0)), tf.int32)
        right_bottom = tf.cast(tf.transpose(tf.stack([ymax, xmax], axis=0)), tf.int32)
        left_bottom = tf.cast(tf.transpose(tf.stack([ymax, xmin], axis=0)), tf.int32)
        right_top = tf.cast(tf.transpose(tf.stack([ymin, xmax], axis=0)), tf.int32)

        feature_1x5 = slim.conv2d(inputs=feature_map,
                                  num_outputs=cfgs.FPN_CHANNEL,
                                  kernel_size=[1, 5],
                                  weights_initializer=cfgs.SUBNETS_WEIGHTS_INITIALIZER,
                                  biases_initializer=cfgs.SUBNETS_BIAS_INITIALIZER,
                                  stride=1,
                                  activation_fn=None,
                                  scope='refine_1x5_{}'.format(name))

        feature5x1 = slim.conv2d(inputs=feature_1x5,
                                 num_outputs=cfgs.FPN_CHANNEL,
                                 kernel_size=[5, 1],
                                 weights_initializer=cfgs.SUBNETS_WEIGHTS_INITIALIZER,
                                 biases_initializer=cfgs.SUBNETS_BIAS_INITIALIZER,
                                 stride=1,
                                 activation_fn=None,
                                 scope='refine_5x1_{}'.format(name))

        feature_1x1 = slim.conv2d(inputs=feature_map,
                                  num_outputs=cfgs.FPN_CHANNEL,
                                  kernel_size=[1, 1],
                                  weights_initializer=cfgs.SUBNETS_WEIGHTS_INITIALIZER,
                                  biases_initializer=cfgs.SUBNETS_BIAS_INITIALIZER,
                                  stride=1,
                                  activation_fn=None,
                                  scope='refine_1x1_{}'.format(name))

        feature = feature5x1 + feature_1x1

        left_top_feature = tf.gather_nd(tf.squeeze(feature), left_top)
        right_bottom_feature = tf.gather_nd(tf.squeeze(feature), right_bottom)
        left_bottom_feature = tf.gather_nd(tf.squeeze(feature), left_bottom)
        right_top_feature = tf.gather_nd(tf.squeeze(feature), right_top)

        refine_feature = right_bottom_feature * tf.tile(
            tf.reshape((tf.abs((points[:, 0] - xmin) * (points[:, 1] - ymin))), [-1, 1]),
            [1, cfgs.FPN_CHANNEL]) \
                         + left_top_feature * tf.tile(
            tf.reshape((tf.abs((xmax - points[:, 0]) * (ymax - points[:, 1]))), [-1, 1]),
            [1, cfgs.FPN_CHANNEL]) \
                         + right_top_feature * tf.tile(
            tf.reshape((tf.abs((points[:, 0] - xmin) * (ymax - points[:, 1]))), [-1, 1]),
            [1, cfgs.FPN_CHANNEL]) \
                         + left_bottom_feature * tf.tile(
            tf.reshape((tf.abs((xmax - points[:, 0]) * (points[:, 1] - ymin))), [-1, 1]),
            [1, cfgs.FPN_CHANNEL])

        refine_feature = tf.reshape(refine_feature, [1, tf.cast(h, tf.int32), tf.cast(w, tf.int32), cfgs.FPN_CHANNEL])

        # refine_feature = tf.reshape(refine_feature, [1, tf.cast(feature_size[1], tf.int32),
        #                                              tf.cast(feature_size[0], tf.int32), 256])

        return refine_feature + feature

    def refine_stage(self, input_img_batch, gtboxes_batch_r, box_pred_list, cls_prob_list, proposal_list,
                     feature_pyramid, dot_layer_list, gpu_id, pos_threshold, neg_threshold,
                     stage, proposal_filter=False):
        with tf.variable_scope('refine_feature_pyramid{}'.format(stage)):
            refine_feature_pyramid = {}
            refine_boxes_list = []

            for box_pred, cls_prob, proposal, stride, level in \
                    zip(box_pred_list, cls_prob_list, proposal_list,
                        cfgs.ANCHOR_STRIDE, cfgs.LEVEL):

                if proposal_filter:
                    box_pred = tf.reshape(box_pred, [-1, self.num_anchors_per_location, 5])
                    proposal = tf.reshape(proposal, [-1, self.num_anchors_per_location, 5 if self.method == 'R' else 4])
                    cls_prob = tf.reshape(cls_prob, [-1, self.num_anchors_per_location, cfgs.CLASS_NUM])

                    cls_max_prob = tf.reduce_max(cls_prob, axis=-1)
                    box_pred_argmax = tf.cast(tf.reshape(tf.argmax(cls_max_prob, axis=-1), [-1, 1]), tf.int32)
                    indices = tf.cast(tf.cumsum(tf.ones_like(box_pred_argmax), axis=0), tf.int32) - tf.constant(1, tf.int32)
                    indices = tf.concat([indices, box_pred_argmax], axis=-1)

                    box_pred = tf.reshape(tf.gather_nd(box_pred, indices), [-1, 5])
                    proposal = tf.reshape(tf.gather_nd(proposal, indices), [-1, 5 if self.method == 'R' else 4])

                    if cfgs.METHOD == 'H':
                        x_c = (proposal[:, 2] + proposal[:, 0]) / 2
                        y_c = (proposal[:, 3] + proposal[:, 1]) / 2
                        h = proposal[:, 2] - proposal[:, 0] + 1
                        w = proposal[:, 3] - proposal[:, 1] + 1
                        theta = -90 * tf.ones_like(x_c)
                        proposal = tf.transpose(tf.stack([x_c, y_c, w, h, theta]))
                else:
                    box_pred = tf.reshape(box_pred, [-1, 5])
                    proposal = tf.reshape(proposal, [-1, 5])

                bboxes = bbox_transform.rbbox_transform_inv(boxes=proposal, deltas=box_pred)
                refine_boxes_list.append(bboxes)
                center_point = bboxes[:, :2] / stride

                refine_feature_pyramid[level] = self.refine_feature_op(points=center_point,
                                                                       feature_map=feature_pyramid[level],
                                                                       name=level)

            if cfgs.USE_SUPERVISED_MASK and stage in ['', '_stage2']:
                for i, d in enumerate(dot_layer_list):
                    refine_feature_pyramid['P{}'.format(i+3)] *= d

            refine_box_pred_list, refine_cls_score_list, refine_cls_prob_list = self.refine_net(refine_feature_pyramid,
                                                                                                'refine_net{}'.format(stage))

            refine_box_pred = tf.concat(refine_box_pred_list, axis=0)
            refine_cls_score = tf.concat(refine_cls_score_list, axis=0)
            # refine_cls_prob = tf.concat(refine_cls_prob_list, axis=0)
            refine_boxes = tf.concat(refine_boxes_list, axis=0)

        if self.is_training:
            with tf.variable_scope('build_refine_loss{}'.format(stage)):
                refine_labels, refine_target_delta, refine_box_states, refine_target_boxes = tf.py_func(
                    func=refinebox_target_layer,
                    inp=[gtboxes_batch_r, refine_boxes, pos_threshold, neg_threshold, gpu_id],
                    Tout=[tf.float32, tf.float32,
                          tf.float32, tf.float32])

                self.add_anchor_img_smry(input_img_batch, refine_boxes, refine_box_states, 1)

                refine_cls_loss = losses.focal_loss(refine_labels, refine_cls_score, refine_box_states)
                if cfgs.USE_IOU_FACTOR:
                    refine_reg_loss = losses.iou_smooth_l1_loss_(refine_target_delta, refine_box_pred,
                                                                 refine_box_states, refine_target_boxes,
                                                                 refine_boxes, is_refine=True)
                else:
                    refine_reg_loss = losses.smooth_l1_loss(refine_target_delta, refine_box_pred, refine_box_states)

                self.losses_dict['refine_cls_loss{}'.format(stage)] = refine_cls_loss * cfgs.CLS_WEIGHT
                self.losses_dict['refine_reg_loss{}'.format(stage)] = refine_reg_loss * cfgs.REG_WEIGHT

        return refine_box_pred_list, refine_cls_prob_list, refine_boxes_list