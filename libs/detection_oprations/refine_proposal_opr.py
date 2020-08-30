# encoding: utf-8
from libs.configs import cfgs
from libs.box_utils import bbox_transform
from libs.box_utils import nms_rotate
import tensorflow as tf


def filter_detections(boxes, scores, is_training):
    """
    :param boxes: [-1, 4]
    :param scores: [-1, ]
    :param labels: [-1, ]
    :return:
    """
    if is_training:
        indices = tf.reshape(tf.where(tf.greater(scores, cfgs.VIS_SCORE)), [-1, ])
    else:
        indices = tf.reshape(tf.where(tf.greater(scores, cfgs.FILTERED_SCORE)), [-1, ])

    if cfgs.NMS:
        filtered_boxes = tf.gather(boxes, indices)
        filtered_scores = tf.gather(scores, indices)

        # perform NMS

        nms_indices = nms_rotate.nms_rotate(decode_boxes=filtered_boxes,
                                            scores=filtered_scores,
                                            iou_threshold=cfgs.NMS_IOU_THRESHOLD,
                                            max_output_size=100 if is_training else 1000,
                                            use_angle_condition=False,
                                            angle_threshold=15,
                                            use_gpu=False)

        # filter indices based on NMS
        indices = tf.gather(indices, nms_indices)

    # add indices to list of all indices
    return indices


def postprocess_detctions(refine_bbox_pred, refine_cls_prob, anchors, is_training):

    boxes_pred = bbox_transform.rbbox_transform_inv(boxes=anchors, deltas=refine_bbox_pred,
                                                    scale_factors=cfgs.ANCHOR_SCALE_FACTORS)

    return_boxes_pred = []
    return_scores = []
    return_labels = []
    for j in range(0, cfgs.CLASS_NUM):
        indices = filter_detections(boxes_pred, refine_cls_prob[:, j], is_training)
        tmp_boxes_pred = tf.reshape(tf.gather(boxes_pred, indices), [-1, 5])
        tmp_scores = tf.reshape(tf.gather(refine_cls_prob[:, j], indices), [-1, ])

        return_boxes_pred.append(tmp_boxes_pred)
        return_scores.append(tmp_scores)
        return_labels.append(tf.ones_like(tmp_scores)*(j+1))

    return_boxes_pred = tf.concat(return_boxes_pred, axis=0)
    return_scores = tf.concat(return_scores, axis=0)
    return_labels = tf.concat(return_labels, axis=0)

    return return_boxes_pred, return_scores, return_labels
