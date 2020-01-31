# encoding: utf-8
from libs.configs import cfgs
from libs.box_utils import bbox_transform
from libs.box_utils import nms_rotate
import tensorflow as tf

from libs.box_utils.coordinate_convert import coordinate_present_convert, coords_regular


def postprocess_detctions(rpn_bbox_pred, rpn_cls_prob, anchors, is_training):

    return_boxes_pred = []
    return_scores = []
    return_labels = []
    for j in range(0, cfgs.CLASS_NUM):
        scores = rpn_cls_prob[:, j]
        if is_training:
            indices = tf.reshape(tf.where(tf.greater(scores, cfgs.VIS_SCORE)), [-1, ])
        else:
            indices = tf.reshape(tf.where(tf.greater(scores, cfgs.FILTERED_SCORE)), [-1, ])

        anchors_ = tf.gather(anchors, indices)
        rpn_bbox_pred_ = tf.gather(rpn_bbox_pred, indices)
        scores = tf.gather(scores, indices)

        if cfgs.METHOD == 'H':
            x_c = (anchors_[:, 2] + anchors_[:, 0]) / 2
            y_c = (anchors_[:, 3] + anchors_[:, 1]) / 2
            h = anchors_[:, 2] - anchors_[:, 0] + 1
            w = anchors_[:, 3] - anchors_[:, 1] + 1
            theta = -90 * tf.ones_like(x_c)
            anchors_ = tf.transpose(tf.stack([x_c, y_c, w, h, theta]))

        if cfgs.ANGLE_RANGE == 180:
            anchors_ = tf.py_func(coordinate_present_convert,
                                 inp=[anchors_, -1],
                                 Tout=[tf.float32])
            anchors_ = tf.reshape(anchors_, [-1, 5])

        boxes_pred = bbox_transform.rbbox_transform_inv(boxes=anchors_, deltas=rpn_bbox_pred_)

        if cfgs.ANGLE_RANGE == 180:

            _, _, _, _, theta = tf.unstack(boxes_pred, axis=1)
            indx = tf.reshape(tf.where(tf.logical_and(tf.less(theta, 0), tf.greater_equal(theta, -180))), [-1, ])
            boxes_pred = tf.gather(boxes_pred, indx)
            scores = tf.gather(scores, indx)

            boxes_pred = tf.py_func(coordinate_present_convert,
                                    inp=[boxes_pred, 1],
                                    Tout=[tf.float32])
            boxes_pred = tf.reshape(boxes_pred, [-1, 5])

        nms_indices = nms_rotate.nms_rotate(decode_boxes=boxes_pred,
                                            scores=scores,
                                            iou_threshold=cfgs.NMS_IOU_THRESHOLD,
                                            max_output_size=100,
                                            use_angle_condition=False,
                                            angle_threshold=15,
                                            use_gpu=False)

        tmp_boxes_pred = tf.reshape(tf.gather(boxes_pred, nms_indices), [-1, 5])
        tmp_scores = tf.reshape(tf.gather(scores, nms_indices), [-1, ])

        return_boxes_pred.append(tmp_boxes_pred)
        return_scores.append(tmp_scores)
        return_labels.append(tf.ones_like(tmp_scores)*(j+1))

    return_boxes_pred = tf.concat(return_boxes_pred, axis=0)
    return_scores = tf.concat(return_scores, axis=0)
    return_labels = tf.concat(return_labels, axis=0)

    return return_boxes_pred, return_scores, return_labels
