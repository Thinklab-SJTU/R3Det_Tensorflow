# encoding: utf-8
from libs.configs import cfgs
from libs.box_utils import bbox_transform
import tensorflow as tf
import numpy as np

from libs.box_utils.coordinate_convert import coordinate_present_convert, coords_regular


def nms_rotate_cpu(boxes, scores, iou_threshold, max_output_size):

    keep = []

    order = scores.argsort()[::-1]
    num = boxes.shape[0]

    suppressed = np.zeros((num), dtype=np.int)

    for _i in range(num):
        if len(keep) >= max_output_size:
            break

        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        r1 = ((boxes[i, 0], boxes[i, 1]), (boxes[i, 2], boxes[i, 3]), boxes[i, 4])
        area_r1 = boxes[i, 2] * boxes[i, 3]
        for _j in range(_i + 1, num):
            j = order[_j]
            if suppressed[i] == 1:
                continue
            r2 = ((boxes[j, 0], boxes[j, 1]), (boxes[j, 2], boxes[j, 3]), boxes[j, 4])
            area_r2 = boxes[j, 2] * boxes[j, 3]
            inter = 0.0

            try:
                int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]

                if int_pts is not None:
                    order_pts = cv2.convexHull(int_pts, returnPoints=True)

                    int_area = cv2.contourArea(order_pts)

                    inter = int_area * 1.0 / (area_r1 + area_r2 - int_area + cfgs.EPSILON)

            except:
                """
                  cv2.error: /io/opencv/modules/imgproc/src/intersection.cpp:247:
                  error: (-215) intersection.size() <= 8 in function rotatedRectangleIntersection
                """
                # print(r1)
                # print(r2)
                inter = 0.9999

            if inter >= iou_threshold:
                suppressed[j] = 1

    return np.array(keep, np.int64)


def nms_rotate(decode_boxes, scores, iou_threshold, max_output_size):
    keep = tf.py_func(nms_rotate_cpu,
                      inp=[decode_boxes, scores, iou_threshold, max_output_size],
                      Tout=tf.int64)
    return keep


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

        nms_indices = nms_rotate(decode_boxes=boxes_pred,
                                 scores=scores,
                                 iou_threshold=cfgs.NMS_IOU_THRESHOLD,
                                 max_output_size=100 if is_training else 1000)

        tmp_boxes_pred = tf.reshape(tf.gather(boxes_pred, nms_indices), [-1, 5])
        tmp_scores = tf.reshape(tf.gather(scores, nms_indices), [-1, ])

        return_boxes_pred.append(tmp_boxes_pred)
        return_scores.append(tmp_scores)
        return_labels.append(tf.ones_like(tmp_scores)*(j+1))

    return_boxes_pred = tf.concat(return_boxes_pred, axis=0)
    return_scores = tf.concat(return_scores, axis=0)
    return_labels = tf.concat(return_labels, axis=0)

    return return_boxes_pred, return_scores, return_labels
