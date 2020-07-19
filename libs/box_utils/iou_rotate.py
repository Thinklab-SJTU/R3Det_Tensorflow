# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import math
from help_utils.gaussian_wasserstein_distance import get_element1, get_element4
from libs.box_utils.coordinate_convert import *
from libs.box_utils.rbbox_overlaps import rbbx_overlaps
from libs.box_utils.iou_cpu import get_iou_matrix


def iou_rotate_calculate(boxes1, boxes2, use_gpu=True, gpu_id=0):
    '''

    :param boxes_list1:[N, 8] tensor
    :param boxes_list2: [M, 8] tensor
    :return:
    '''

    boxes1 = tf.cast(boxes1, tf.float32)
    boxes2 = tf.cast(boxes2, tf.float32)
    if use_gpu:

        iou_matrix = tf.py_func(rbbx_overlaps,
                                inp=[boxes1, boxes2, gpu_id],
                                Tout=tf.float32)
    else:
        iou_matrix = tf.py_func(get_iou_matrix, inp=[boxes1, boxes2],
                                Tout=tf.float32)

    iou_matrix = tf.reshape(iou_matrix, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])

    return iou_matrix


def iou_rotate_calculate1(boxes1, boxes2, use_gpu=True, gpu_id=0):

    # start = time.time()
    if use_gpu:
        ious = rbbx_overlaps(boxes1, boxes2, gpu_id)
    else:
        area1 = boxes1[:, 2] * boxes1[:, 3]
        area2 = boxes2[:, 2] * boxes2[:, 3]
        ious = []
        for i, box1 in enumerate(boxes1):
            temp_ious = []
            r1 = ((box1[0], box1[1]), (box1[2], box1[3]), box1[4])
            for j, box2 in enumerate(boxes2):
                r2 = ((box2[0], box2[1]), (box2[2], box2[3]), box2[4])

                int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
                if int_pts is not None:
                    order_pts = cv2.convexHull(int_pts, returnPoints=True)

                    int_area = cv2.contourArea(order_pts)

                    inter = int_area * 1.0 / (area1[i] + area2[j] - int_area)
                    temp_ious.append(inter)
                else:
                    temp_ious.append(0.0)
            ious.append(temp_ious)

    # print('{}s'.format(time.time() - start))

    return np.array(ious, dtype=np.float32)


def iou_rotate_calculate2(boxes1, boxes2):
    ious = []
    if boxes1.shape[0] != 0:
        boxes1[:, 2] += 1.0
        boxes1[:, 3] += 1.0
        boxes2[:, 2] += 1.0
        boxes2[:, 3] += 1.0

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

                inter = int_area * 1.0 / (area1[i] + area2[i] - int_area + 1.0)

                temp_ious.append(inter)
            else:
                temp_ious.append(0.0)
            ious.append(temp_ious)

    return np.array(ious, dtype=np.float32)


def diou_rotate_calculate(boxes1, boxes2):

    if boxes1.shape[0] != 0:
        area1 = boxes1[:, 2] * boxes1[:, 3]
        area2 = boxes2[:, 2] * boxes2[:, 3]
        d = (boxes1[:, 0] - boxes2[:, 0]) ** 2 + (boxes1[:, 1] - boxes2[:, 1])

        boxes1_ = forward_convert(boxes1, with_label=False)
        boxes2_ = forward_convert(boxes2, with_label=False)

        xmin = np.minimum(np.min(boxes1_[:, 0::2]), np.min(boxes2_[:, 0::2]))
        xmax = np.maximum(np.max(boxes1_[:, 0::2]), np.max(boxes2_[:, 0::2]))
        ymin = np.minimum(np.min(boxes1_[:, 1::2]), np.min(boxes2_[:, 1::2]))
        ymax = np.maximum(np.max(boxes1_[:, 1::2]), np.max(boxes2_[:, 1::2]))

        c = (xmax - xmin) ** 2 + (ymax - ymin) ** 2
        ious = []
        for i in range(boxes1.shape[0]):
            r1 = ((boxes1[i][0], boxes1[i][1]), (boxes1[i][2], boxes1[i][3]), boxes1[i][4])
            r2 = ((boxes2[i][0], boxes2[i][1]), (boxes2[i][2], boxes2[i][3]), boxes2[i][4])

            int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
            if int_pts is not None:
                order_pts = cv2.convexHull(int_pts, returnPoints=True)

                int_area = cv2.contourArea(order_pts)

                iou = int_area * 1.0 / (area1[i] + area2[i] - int_area)
            else:
                iou = 0.0

            ious.append(iou)
        ious = np.array(ious)

        dious = ious - d / c
    else:
        dious = []

    return np.reshape(np.array(dious, dtype=np.float32), [-1, 1])


def adiou_rotate_calculate(boxes1, boxes2):

    if boxes1.shape[0] != 0:
        area1 = boxes1[:, 2] * boxes1[:, 3]
        area2 = boxes2[:, 2] * boxes2[:, 3]
        d = (boxes1[:, 0] - boxes2[:, 0]) ** 2 + (boxes1[:, 1] - boxes2[:, 1])

        boxes1_ = forward_convert(boxes1, with_label=False)
        boxes2_ = forward_convert(boxes2, with_label=False)

        xmin = np.minimum(np.min(boxes1_[:, 0::2]), np.min(boxes2_[:, 0::2]))
        xmax = np.maximum(np.max(boxes1_[:, 0::2]), np.max(boxes2_[:, 0::2]))
        ymin = np.minimum(np.min(boxes1_[:, 1::2]), np.min(boxes2_[:, 1::2]))
        ymax = np.maximum(np.max(boxes1_[:, 1::2]), np.max(boxes2_[:, 1::2]))

        c = (xmax - xmin) ** 2 + (ymax - ymin) ** 2

        # v = (4 / (math.pi ** 2)) * (np.arctan(boxes1[:, 2]/boxes1[:, 3]) - np.arctan(boxes2[:, 2]/boxes2[:, 3])) ** 2

        ious = []
        for i in range(boxes1.shape[0]):
            r1 = ((boxes1[i][0], boxes1[i][1]), (boxes1[i][2], boxes1[i][3]), boxes1[i][4])
            r2 = ((boxes2[i][0], boxes2[i][1]), (boxes2[i][2], boxes2[i][3]), boxes2[i][4])

            int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
            if int_pts is not None:
                order_pts = cv2.convexHull(int_pts, returnPoints=True)

                int_area = cv2.contourArea(order_pts)

                iou = int_area * 1.0 / (area1[i] + area2[i] - int_area)
            else:
                iou = 0.0

            ious.append(iou)
        ious = np.array(ious)

        # S = 1 - ious
        # alpha = v / (S + v)
        # w_temp = 2 * boxes1[:, 2]
        # ar = (8 / (math.pi ** 2)) * (np.arctan(boxes1[:, 2]/boxes1[:, 3]) - np.arctan(boxes2[:, 2]/boxes2[:, 3])) \
        #      * ((boxes1[:, 2] - w_temp) * boxes1[:, 3])
        # cious = ious - d / c - alpha * ar
        cious = (ious - d / c) * np.abs(np.cos(boxes1[:, 4] - boxes2[:, 4]))
    else:
        cious = []

    return np.reshape(np.array(cious, dtype=np.float32), [-1, 1])


def gaussian_wasserstein_distance_(boxes1, boxes2):
    boxes1 = coordinate_present_convert(boxes1, -1)
    boxes1[:, 4] += 90
    boxes1[:, 4] *= (-np.pi / 180)

    boxes2 = coordinate_present_convert(boxes2, -1)
    boxes2[:, 4] += 90
    boxes2[:, 4] *= (-np.pi / 180)

    dis = (boxes1[:, 0] - boxes2[:, 0])**2 + (boxes1[:, 1] - boxes2[:, 1])**2 + \
          ((boxes1[:, 2] / 2 * np.cos(boxes1[:, 4])**2 + boxes1[:, 3] / 2 * np.sin(boxes1[:, 4])**2) - (boxes2[:, 2] / 2 * np.cos(boxes2[:, 4])**2 + boxes2[:, 3] / 2 * np.sin(boxes2[:, 4])**2))**2 + \
          2*((boxes1[:, 2] / 2 * np.cos(boxes1[:, 4])* np.sin(boxes1[:, 4] - boxes1[:, 3] / 2 * np.cos(boxes1[:, 4]) * np.sin(boxes1[:, 4]))) - (boxes2[:, 2] / 2 * np.cos(boxes2[:, 4]) * np.sin(boxes2[:, 4] - boxes2[:, 3] / 2 * np.cos(boxes2[:, 4]) * np.sin(boxes2[:, 4]))))**2 + \
          ((boxes1[:, 2] / 2 * np.sin(boxes1[:, 4]) ** 2 + boxes1[:, 3] / 2 * np.cos(boxes1[:, 4]) ** 2) - (
          boxes2[:, 2] / 2 * np.sin(boxes2[:, 4]) ** 2 + boxes2[:, 3] / 2 * np.cos(boxes2[:, 4]) ** 2)) ** 2
    return dis


def gaussian_wasserstein_distance(boxes1, boxes2):
    boxes1 = coordinate_present_convert(boxes1, -1)
    boxes1[:, 4] += 90
    boxes1[:, 4] *= (-np.pi / 180)

    boxes2 = coordinate_present_convert(boxes2, -1)
    boxes2[:, 4] += 90
    boxes2[:, 4] *= (-np.pi / 180)

    element1 = get_element1(boxes1[:, 2], boxes1[:, 3], boxes1[:, 4], boxes2[:, 2], boxes2[:, 3], boxes2[:, 4])
    element4 = get_element4(boxes1[:, 2], boxes1[:, 3], boxes1[:, 4], boxes2[:, 2], boxes2[:, 3], boxes2[:, 4])
    dis = (boxes1[:, 0] - boxes2[:, 0])**2 + (boxes1[:, 1] - boxes2[:, 1])**2 + (element1 + element4)
    return dis

if __name__ == '__main__':
    boxes1 = np.array([
                       [50, 50, 10, 70, -45],
                       [50, 50, 10, 70, -45],
                       [50, 50, 10, 70, -45],
                       [50, 50, 10, 70, -45],
                       [50, 50, 10, 70, -45],
                       [50, 50, 10, 70, -45],
                       [50, 50, 10, 70, -45]
                       ], np.float32)

    boxes2 = np.array([
                       [350, 350, 10, 70, -65],
                       [250, 250, 10, 70, -45],
                       [50, 40, 20, 65, -40],
                       [50, 40, 20, 65, -41],
                       [50, 50, 10, 70, -25],
                       [50, 50, 10, 70, -15],
                       [50, 50, 10, 70, -5]], np.float32)

    print(iou_rotate_calculate2(boxes1, boxes2).reshape(-1,))
    print(iou_rotate_calculate2(coordinate_present_convert(boxes1, mode=-1), coordinate_present_convert(boxes2, mode=-1)).reshape(-1,))
    print(diou_rotate_calculate(boxes1, boxes2).reshape(-1,))
    print(gaussian_wasserstein_distance(boxes1, boxes2))
    print(gaussian_wasserstein_distance_(boxes1, boxes2))

    print(np.argsort(iou_rotate_calculate2(boxes1, boxes2).reshape(-1, )*-1))
    print(np.argsort(diou_rotate_calculate(boxes1, boxes2).reshape(-1, )*-1))
    print(np.argsort(np.array(gaussian_wasserstein_distance(boxes1, boxes2))))
    print(np.argsort(np.array(gaussian_wasserstein_distance_(boxes1, boxes2))))




