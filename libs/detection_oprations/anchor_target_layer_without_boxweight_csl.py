# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from libs.configs import cfgs
import numpy as np
from libs.box_utils.cython_utils.cython_bbox import bbox_overlaps
from libs.box_utils.rbbox_overlaps import rbbx_overlaps
from libs.box_utils.iou_cpu import get_iou_matrix
from libs.box_utils import bbox_transform
from libs.box_utils.coordinate_convert import coordinate_present_convert


def anchor_target_layer(gt_boxes_h, gt_boxes_r, gt_smooth_label, anchors, gpu_id=0):

    anchor_states = np.zeros((anchors.shape[0],))
    labels = np.zeros((anchors.shape[0], cfgs.CLASS_NUM))
    if gt_boxes_r.shape[0]:
        # [N, M]

        if cfgs.METHOD == 'H':
            overlaps = bbox_overlaps(np.ascontiguousarray(anchors, dtype=np.float),
                                     np.ascontiguousarray(gt_boxes_h, dtype=np.float))
        else:
            overlaps = rbbx_overlaps(np.ascontiguousarray(anchors, dtype=np.float32),
                                     np.ascontiguousarray(gt_boxes_r[:, :-1], dtype=np.float32), gpu_id)

            # overlaps = get_iou_matrix(np.ascontiguousarray(anchors, dtype=np.float32),
            #                           np.ascontiguousarray(gt_boxes_r[:, :-1], dtype=np.float32))

        argmax_overlaps_inds = np.argmax(overlaps, axis=1)
        max_overlaps = overlaps[np.arange(overlaps.shape[0]), argmax_overlaps_inds]

        # compute box regression targets
        target_boxes = gt_boxes_r[argmax_overlaps_inds]
        target_smooth_label = gt_smooth_label[argmax_overlaps_inds]

        if cfgs.USE_ANGLE_COND:
            if cfgs.METHOD == 'R':
                delta_theta = np.abs(target_boxes[:, -2] - anchors[:, -1])
                theta_indices = delta_theta < 15
                positive_indices = (max_overlaps >= cfgs.IOU_POSITIVE_THRESHOLD) & theta_indices
            else:
                positive_indices = max_overlaps >= cfgs.IOU_POSITIVE_THRESHOLD

            ignore_indices = (max_overlaps > cfgs.IOU_NEGATIVE_THRESHOLD) & (max_overlaps < cfgs.IOU_POSITIVE_THRESHOLD)

        else:
            positive_indices = max_overlaps >= cfgs.IOU_POSITIVE_THRESHOLD
            ignore_indices = (max_overlaps > cfgs.IOU_NEGATIVE_THRESHOLD) & ~positive_indices

        anchor_states[ignore_indices] = -1
        anchor_states[positive_indices] = 1

        # compute target class labels
        labels[positive_indices, target_boxes[positive_indices, -1].astype(int) - 1] = 1
    else:
        # no annotations? then everything is background
        target_boxes = np.zeros((anchors.shape[0], gt_boxes_r.shape[1]))
        target_smooth_label = np.zeros((anchors.shape[0], gt_smooth_label.shape[1]))

    if cfgs.METHOD == 'H':
        x_c = (anchors[:, 2] + anchors[:, 0]) / 2
        y_c = (anchors[:, 3] + anchors[:, 1]) / 2
        h = anchors[:, 2] - anchors[:, 0] + 1
        w = anchors[:, 3] - anchors[:, 1] + 1
        theta = -90 * np.ones_like(x_c)
        anchors = np.vstack([x_c, y_c, w, h, theta]).transpose()

    if cfgs.ANGLE_RANGE == 180:
        anchors = coordinate_present_convert(anchors, mode=-1)
        target_boxes = coordinate_present_convert(target_boxes, mode=-1)
    target_delta = bbox_transform.rbbox_transform(ex_rois=anchors, gt_rois=target_boxes)

    return np.array(labels, np.float32), np.array(target_delta, np.float32), \
           np.array(anchor_states, np.float32), np.array(target_boxes, np.float32),\
           np.array(target_smooth_label, np.float32)




