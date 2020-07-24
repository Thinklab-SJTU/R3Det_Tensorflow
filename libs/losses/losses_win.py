# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import cv2

from libs.box_utils import bbox_transform
from libs.configs import cfgs


def iou_rotate_calculate2(boxes1, boxes2):
    ious = []
    if boxes1.shape[0] != 0:
        area1 = boxes1[:, 2] * boxes1[:, 3]
        area2 = boxes2[:, 2] * boxes2[:, 3]

        for i in range(boxes1.shape[0]):
            temp_ious = []
            r1 = ((boxes1[i][0], boxes1[i][1]), (boxes1[i][2], boxes1[i][3]), boxes1[i][4])
            r2 = ((boxes2[i][0], boxes2[i][1]), (boxes2[i][2], boxes2[i][3]), boxes2[i][4])

            int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
            if int_pts is not None:
                order_pts = cv2.convexHull(int_pts, returnPoints=True)

                int_area = cv2.contourArea(order_pts)

                inter = int_area * 1.0 / (area1[i] + area2[i] - int_area)
                temp_ious.append(inter)
            else:
                temp_ious.append(0.0)
            ious.append(temp_ious)

    return np.array(ious, dtype=np.float32)


def focal_loss(labels, pred, anchor_state, alpha=0.25, gamma=2.0):

    # filter out "ignore" anchors
    indices = tf.reshape(tf.where(tf.not_equal(anchor_state, -1)), [-1, ])
    labels = tf.gather(labels, indices)
    pred = tf.gather(pred, indices)

    # compute the focal loss
    per_entry_cross_ent = (tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels, logits=pred))
    prediction_probabilities = tf.sigmoid(pred)
    p_t = ((labels * prediction_probabilities) +
           ((1 - labels) * (1 - prediction_probabilities)))
    modulating_factor = 1.0
    if gamma:
        modulating_factor = tf.pow(1.0 - p_t, gamma)
    alpha_weight_factor = 1.0
    if alpha is not None:
        alpha_weight_factor = (labels * alpha +
                               (1 - labels) * (1 - alpha))
    focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor *
                                per_entry_cross_ent)

    # compute the normalizer: the number of positive anchors
    normalizer = tf.stop_gradient(tf.where(tf.equal(anchor_state, 1)))
    normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
    normalizer = tf.maximum(1.0, normalizer)

    # normalizer = tf.stop_gradient(tf.cast(tf.equal(anchor_state, 1), tf.float32))
    # normalizer = tf.maximum(tf.reduce_sum(normalizer), 1)

    return tf.reduce_sum(focal_cross_entropy_loss) / normalizer


def _smooth_l1_loss_base(bbox_pred, bbox_targets, sigma=1.0):
    '''

    :param bbox_pred: [-1, 4] in RPN. [-1, cls_num+1, 4] in Fast-rcnn
    :param bbox_targets: shape is same as bbox_pred
    :param sigma:
    :return:
    '''
    sigma_2 = sigma**2

    box_diff = bbox_pred - bbox_targets

    abs_box_diff = tf.abs(box_diff)

    smoothL1_sign = tf.stop_gradient(
        tf.to_float(tf.less(abs_box_diff, 1. / sigma_2)))
    loss_box = tf.pow(box_diff, 2) * (sigma_2 / 2.0) * smoothL1_sign \
               + (abs_box_diff - (0.5 / sigma_2)) * (1.0 - smoothL1_sign)
    return loss_box


def smooth_l1_loss_rcnn(bbox_targets, bbox_pred, anchor_state, sigma=3.0):

    outside_mask = tf.stop_gradient(tf.to_float(tf.greater(anchor_state, 0)))

    bbox_pred = tf.reshape(bbox_pred, [-1, 1, 4])
    bbox_targets = tf.reshape(bbox_targets, [-1, 1, 4])

    value = _smooth_l1_loss_base(bbox_pred,
                                 bbox_targets,
                                 sigma=sigma)
    value = tf.reduce_sum(value, 2)
    value = tf.reshape(value, [-1, 1])

    normalizer = tf.stop_gradient(tf.where(tf.equal(anchor_state, 1)))
    normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
    normalizer = tf.maximum(1.0, normalizer)

    bbox_loss = tf.reduce_sum(
        tf.reduce_sum(value, 1)*outside_mask) / normalizer

    return bbox_loss


def smooth_l1_loss(targets, preds, anchor_state, sigma=3.0, weight=None):
    sigma_squared = sigma ** 2
    indices = tf.reshape(tf.where(tf.equal(anchor_state, 1)), [-1, ])
    preds = tf.gather(preds, indices)
    targets = tf.gather(targets, indices)

    # compute smooth L1 loss
    # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
    #        |x| - 0.5 / sigma / sigma    otherwise
    regression_diff = preds - targets
    regression_diff = tf.abs(regression_diff)

    regression_loss = tf.where(
        tf.less(regression_diff, 1.0 / sigma_squared),
        0.5 * sigma_squared * tf.pow(regression_diff, 2),
        regression_diff - 0.5 / sigma_squared
    )

    if weight is not None:
        regression_loss = tf.reduce_sum(regression_loss, axis=-1)
        weight = tf.gather(weight, indices)
        regression_loss *= weight

    normalizer = tf.stop_gradient(tf.where(tf.equal(anchor_state, 1)))
    normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
    normalizer = tf.maximum(1.0, normalizer)

    # normalizer = tf.stop_gradient(tf.cast(tf.equal(anchor_state, 1), tf.float32))
    # normalizer = tf.maximum(tf.reduce_sum(normalizer), 1)

    return tf.reduce_sum(regression_loss) / normalizer


def smooth_l1_loss_atan(targets, preds, anchor_state, sigma=3.0, weight=None):

    sigma_squared = sigma ** 2
    indices = tf.reshape(tf.where(tf.equal(anchor_state, 1)), [-1, ])
    preds = tf.gather(preds, indices)
    targets = tf.gather(targets, indices)

    # compute smooth L1 loss
    # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
    #        |x| - 0.5 / sigma / sigma    otherwise
    regression_diff = preds - targets
    regression_diff = tf.abs(regression_diff)

    regression_diff = tf.reshape(regression_diff, [-1, 5])
    dx, dy, dw, dh, dtheta = tf.unstack(regression_diff, axis=-1)
    dtheta = tf.atan(dtheta)
    regression_diff = tf.transpose(tf.stack([dx, dy, dw, dh, dtheta]))

    regression_loss = tf.where(
        tf.less(regression_diff, 1.0 / sigma_squared),
        0.5 * sigma_squared * tf.pow(regression_diff, 2),
        regression_diff - 0.5 / sigma_squared
    )

    if weight is not None:
        regression_loss = tf.reduce_sum(regression_loss, axis=-1)
        weight = tf.gather(weight, indices)
        regression_loss *= weight

    normalizer = tf.stop_gradient(tf.where(tf.equal(anchor_state, 1)))
    normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
    normalizer = tf.maximum(1.0, normalizer)

    # normalizer = tf.stop_gradient(tf.cast(tf.equal(anchor_state, 1), tf.float32))
    # normalizer = tf.maximum(tf.reduce_sum(normalizer), 1)

    return tf.reduce_sum(regression_loss) / normalizer


def iou_smooth_l1_loss(targets, preds, anchor_state, target_boxes, anchors, sigma=3.0, is_refine=False):
    if cfgs.METHOD == 'H' and not is_refine:
        x_c = (anchors[:, 2] + anchors[:, 0]) / 2
        y_c = (anchors[:, 3] + anchors[:, 1]) / 2
        h = anchors[:, 2] - anchors[:, 0] + 1
        w = anchors[:, 3] - anchors[:, 1] + 1
        theta = -90 * tf.ones_like(x_c)
        anchors = tf.transpose(tf.stack([x_c, y_c, w, h, theta]))

    sigma_squared = sigma ** 2
    indices = tf.reshape(tf.where(tf.equal(anchor_state, 1)), [-1, ])

    preds = tf.gather(preds, indices)
    targets = tf.gather(targets, indices)
    target_boxes = tf.gather(target_boxes, indices)
    anchors = tf.gather(anchors, indices)

    boxes_pred = bbox_transform.rbbox_transform_inv(boxes=anchors, deltas=preds,
                                                    scale_factors=cfgs.ANCHOR_SCALE_FACTORS)

    # compute smooth L1 loss
    # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
    #        |x| - 0.5 / sigma / sigma    otherwise
    regression_diff = preds - targets
    regression_diff = tf.abs(regression_diff)
    regression_loss = tf.where(
        tf.less(regression_diff, 1.0 / sigma_squared),
        0.5 * sigma_squared * tf.pow(regression_diff, 2),
        regression_diff - 0.5 / sigma_squared
    )

    overlaps = tf.py_func(iou_rotate_calculate2,
                          inp=[tf.reshape(boxes_pred, [-1, 5]), tf.reshape(target_boxes[:, :-1], [-1, 5])],
                          Tout=[tf.float32])

    overlaps = tf.reshape(overlaps, [-1, 1])
    regression_loss = tf.reshape(tf.reduce_sum(regression_loss, axis=1), [-1, 1])
    # -ln(x)
    iou_factor = tf.stop_gradient(-1 * tf.log(overlaps)) / (tf.stop_gradient(regression_loss) + cfgs.EPSILON)
    # iou_factor = tf.Print(iou_factor, [iou_factor], 'iou_factor', summarize=50)

    normalizer = tf.stop_gradient(tf.where(tf.equal(anchor_state, 1)))
    normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
    normalizer = tf.maximum(1.0, normalizer)

    # normalizer = tf.stop_gradient(tf.cast(tf.equal(anchor_state, 1), tf.float32))
    # normalizer = tf.maximum(tf.reduce_sum(normalizer), 1)

    return tf.reduce_sum(regression_loss * iou_factor) / normalizer


def iou_smooth_l1_loss_(targets, preds, anchor_state, target_boxes, anchors, sigma=3.0, alpha=1.0, beta=1.0, is_refine=False):
    if cfgs.METHOD == 'H' and not is_refine:
        x_c = (anchors[:, 2] + anchors[:, 0]) / 2
        y_c = (anchors[:, 3] + anchors[:, 1]) / 2
        h = anchors[:, 2] - anchors[:, 0] + 1
        w = anchors[:, 3] - anchors[:, 1] + 1
        theta = -90 * tf.ones_like(x_c)
        anchors = tf.transpose(tf.stack([x_c, y_c, w, h, theta]))

    sigma_squared = sigma ** 2
    indices = tf.reshape(tf.where(tf.equal(anchor_state, 1)), [-1, ])

    preds = tf.gather(preds, indices)
    targets = tf.gather(targets, indices)
    target_boxes = tf.gather(target_boxes, indices)
    anchors = tf.gather(anchors, indices)

    boxes_pred = bbox_transform.rbbox_transform_inv(boxes=anchors, deltas=preds,
                                                    scale_factors=cfgs.ANCHOR_SCALE_FACTORS)

    # compute smooth L1 loss
    # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
    #        |x| - 0.5 / sigma / sigma    otherwise
    regression_diff = preds - targets
    regression_diff = tf.abs(regression_diff)
    regression_loss = tf.where(
        tf.less(regression_diff, 1.0 / sigma_squared),
        0.5 * sigma_squared * tf.pow(regression_diff, 2),
        regression_diff - 0.5 / sigma_squared
    )

    overlaps = tf.py_func(iou_rotate_calculate2,
                          inp=[tf.reshape(boxes_pred, [-1, 5]), tf.reshape(target_boxes[:, :-1], [-1, 5])],
                          Tout=[tf.float32])

    overlaps = tf.reshape(overlaps, [-1, 1])
    regression_loss = tf.reshape(tf.reduce_sum(regression_loss, axis=1), [-1, 1])
    # 1-exp(1-x)
    iou_factor = tf.stop_gradient(tf.exp(alpha*(1-overlaps)**beta)-1) / (tf.stop_gradient(regression_loss) + cfgs.EPSILON)
    # iou_factor = tf.Print(iou_factor, [iou_factor], 'iou_factor', summarize=50)

    normalizer = tf.stop_gradient(tf.where(tf.equal(anchor_state, 1)))
    normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
    normalizer = tf.maximum(1.0, normalizer)

    # normalizer = tf.stop_gradient(tf.cast(tf.equal(anchor_state, 1), tf.float32))
    # normalizer = tf.maximum(tf.reduce_sum(normalizer), 1)

    return tf.reduce_sum(regression_loss * iou_factor) / normalizer


def angle_focal_loss(labels, pred, anchor_state, alpha=0.25, gamma=2.0):

    # filter out "ignore" anchors
    indices = tf.reshape(tf.where(tf.not_equal(anchor_state, -1)), [-1, ])
    labels = tf.gather(labels, indices)
    pred = tf.gather(pred, indices)

    # compute the focal loss
    per_entry_cross_ent = - labels * tf.log(tf.sigmoid(pred) + cfgs.EPSILON) \
                          - (1 - labels) * tf.log(1 - tf.sigmoid(pred) + cfgs.EPSILON)

    prediction_probabilities = tf.sigmoid(pred)
    p_t = ((labels * prediction_probabilities) +
           ((1 - labels) * (1 - prediction_probabilities)))
    modulating_factor = 1.0
    if gamma:
        modulating_factor = tf.pow(1.0 - p_t, gamma)
    alpha_weight_factor = 1.0
    if alpha is not None:
        alpha_weight_factor = (labels * alpha +
                               (1 - labels) * (1 - alpha))
    focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor *
                                per_entry_cross_ent)

    # compute the normalizer: the number of positive anchors
    normalizer = tf.stop_gradient(tf.where(tf.greater(anchor_state, -2)))
    normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
    normalizer = tf.maximum(1.0, normalizer)

    # normalizer = tf.stop_gradient(tf.cast(tf.equal(anchor_state, 1), tf.float32))
    # normalizer = tf.maximum(tf.reduce_sum(normalizer), 1)

    return tf.reduce_sum(focal_cross_entropy_loss) / normalizer


def scale_iou_smooth_l1_loss(targets, preds, anchor_state, target_boxes, anchors, sigma=3.0,
                             is_refine=False, use_scale_factor=False):
    if cfgs.METHOD == 'H' and not is_refine:
        x_c = (anchors[:, 2] + anchors[:, 0]) / 2
        y_c = (anchors[:, 3] + anchors[:, 1]) / 2
        h = anchors[:, 2] - anchors[:, 0] + 1
        w = anchors[:, 3] - anchors[:, 1] + 1
        theta = -90 * tf.ones_like(x_c)
        anchors = tf.transpose(tf.stack([x_c, y_c, w, h, theta]))

    sigma_squared = sigma ** 2
    indices = tf.reshape(tf.where(tf.equal(anchor_state, 1)), [-1, ])

    preds = tf.gather(preds, indices)
    targets = tf.gather(targets, indices)
    target_boxes = tf.gather(target_boxes, indices)
    anchors = tf.gather(anchors, indices)

    boxes_pred = bbox_transform.rbbox_transform_inv(boxes=anchors, deltas=preds,
                                                    scale_factors=cfgs.ANCHOR_SCALE_FACTORS)

    # compute smooth L1 loss
    # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
    #        |x| - 0.5 / sigma / sigma    otherwise
    regression_diff = preds - targets
    regression_diff = tf.abs(regression_diff)
    regression_loss = tf.where(
        tf.less(regression_diff, 1.0 / sigma_squared),
        0.5 * sigma_squared * tf.pow(regression_diff, 2),
        regression_diff - 0.5 / sigma_squared
    )

    overlaps = tf.py_func(iou_rotate_calculate2,
                          inp=[tf.reshape(boxes_pred, [-1, 5]), tf.reshape(target_boxes[:, :-1], [-1, 5])],
                          Tout=[tf.float32])

    overlaps = tf.reshape(overlaps, [-1, 1])
    regression_loss = tf.reshape(tf.reduce_sum(regression_loss, axis=1), [-1, 1])
    # 1-exp(1-x)
    iou_factor = tf.stop_gradient(tf.exp(1-overlaps)-1) / (tf.stop_gradient(regression_loss) + cfgs.EPSILON)

    if use_scale_factor:
        area = target_boxes[:, 2] * target_boxes[:, 3]
        area = tf.reshape(area, [-1, 1])
        scale_factor = tf.stop_gradient(tf.exp(-1 * area) + 1)
    else:
        scale_factor = 1.0

    normalizer = tf.stop_gradient(tf.where(tf.equal(anchor_state, 1)))
    normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
    normalizer = tf.maximum(1.0, normalizer)

    # normalizer = tf.stop_gradient(tf.cast(tf.equal(anchor_state, 1), tf.float32))
    # normalizer = tf.maximum(tf.reduce_sum(normalizer), 1)

    return tf.reduce_sum(regression_loss * iou_factor * scale_factor) / normalizer


def scale_focal_loss(labels, pred, anchor_state, target_boxes, alpha=0.25, gamma=2.0, use_scale_factor=False):

    # filter out "ignore" anchors
    indices = tf.reshape(tf.where(tf.not_equal(anchor_state, -1)), [-1, ])
    labels = tf.gather(labels, indices)
    pred = tf.gather(pred, indices)
    target_boxes = tf.gather(target_boxes, indices)

    # compute the focal loss
    per_entry_cross_ent = (tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels, logits=pred))
    prediction_probabilities = tf.sigmoid(pred)
    p_t = ((labels * prediction_probabilities) +
           ((1 - labels) * (1 - prediction_probabilities)))
    modulating_factor = 1.0
    if gamma:
        modulating_factor = tf.pow(1.0 - p_t, gamma)
    alpha_weight_factor = 1.0
    if alpha is not None:
        alpha_weight_factor = (labels * alpha +
                               (1 - labels) * (1 - alpha))
    focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor *
                                per_entry_cross_ent)

    if use_scale_factor:
        area = target_boxes[:, 2] * target_boxes[:, 3]
        area = tf.reshape(area, [-1, 1])
        scale_factor = tf.stop_gradient(tf.exp(-1 * area) + 1)
    else:
        scale_factor = 1.0

    # compute the normalizer: the number of positive anchors
    normalizer = tf.stop_gradient(tf.where(tf.equal(anchor_state, 1)))
    normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
    normalizer = tf.maximum(1.0, normalizer)

    # normalizer = tf.stop_gradient(tf.cast(tf.equal(anchor_state, 1), tf.float32))
    # normalizer = tf.maximum(tf.reduce_sum(normalizer), 1)

    return tf.reduce_sum(focal_cross_entropy_loss * scale_factor) / normalizer