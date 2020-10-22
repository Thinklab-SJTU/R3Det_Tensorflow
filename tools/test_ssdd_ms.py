# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import tensorflow as tf
import time
import cv2
import pickle
import numpy as np
import argparse
from tqdm import tqdm
sys.path.append("../")

from data.io.image_preprocess import short_side_resize_for_inference_data
from libs.configs import cfgs
from libs.networks import build_whole_network
from libs.val_libs import voc_eval, voc_eval_r
from libs.box_utils import draw_box_in_img
from libs.box_utils.coordinate_convert import forward_convert, backward_convert
from help_utils import tools


def eval_with_plac(img_dir, det_net, num_imgs, image_ext, draw_imgs=False):

    # 1. preprocess img
    img_plac = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])  # is RGB. not BGR
    img_batch = tf.cast(img_plac, tf.float32)

    img_batch = short_side_resize_for_inference_data(img_tensor=img_batch,
                                                     target_shortside_len=cfgs.IMG_SHORT_SIDE_LEN,
                                                     length_limitation=cfgs.IMG_MAX_LENGTH)
    if cfgs.NET_NAME in ['resnet152_v1d', 'resnet101_v1d', 'resnet50_v1d']:
        img_batch = (img_batch / 255 - tf.constant(cfgs.PIXEL_MEAN_)) / tf.constant(cfgs.PIXEL_STD)
    else:
        img_batch = img_batch - tf.constant(cfgs.PIXEL_MEAN)

    img_batch = tf.expand_dims(img_batch, axis=0)

    detection_boxes, detection_scores, detection_category = det_net.build_whole_detection_network(
        input_img_batch=img_batch,
        gtboxes_batch_h=None,
        gtboxes_batch_r=None)

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    restorer, restore_ckpt = det_net.get_restorer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        if not restorer is None:
            restorer.restore(sess, restore_ckpt)
            print('restore model')

        all_boxes_r = []
        imgs = os.listdir(img_dir)
        pbar = tqdm(imgs)
        for a_img_name in pbar:
            a_img_name = a_img_name.split(image_ext)[0]

            raw_img = cv2.imread(os.path.join(img_dir,
                                              a_img_name + image_ext))
            raw_h, raw_w = raw_img.shape[0], raw_img.shape[1]

            resized_img, det_boxes_r_, det_scores_r_, det_category_r_ = \
                sess.run(
                    [img_batch, detection_boxes, detection_scores, detection_category],
                    feed_dict={img_plac: raw_img[:, :, ::-1]}
                )

            if draw_imgs:
                detected_indices = det_scores_r_ >= cfgs.VIS_SCORE
                detected_scores = det_scores_r_[detected_indices]
                detected_boxes = det_boxes_r_[detected_indices]
                detected_categories = det_category_r_[detected_indices]

                det_detections_r = draw_box_in_img.draw_boxes_with_label_and_scores(np.squeeze(resized_img, 0),
                                                                                    boxes=detected_boxes,
                                                                                    labels=detected_categories,
                                                                                    scores=detected_scores,
                                                                                    method=1,
                                                                                    in_graph=True)

                save_dir = os.path.join('test_hrsc', cfgs.VERSION, 'hrsc2016_img_vis')
                tools.mkdir(save_dir)

                cv2.imwrite(save_dir + '/{}.jpg'.format(a_img_name),
                            det_detections_r[:, :, ::-1])

            if det_boxes_r_.shape[0] != 0:
                resized_h, resized_w = resized_img.shape[1], resized_img.shape[2]
                det_boxes_r_ = forward_convert(det_boxes_r_, False)
                det_boxes_r_[:, 0::2] *= (raw_w / resized_w)
                det_boxes_r_[:, 1::2] *= (raw_h / resized_h)
                det_boxes_r_ = backward_convert(det_boxes_r_, False)

            x_c, y_c, w, h, theta = det_boxes_r_[:, 0], det_boxes_r_[:, 1], det_boxes_r_[:, 2], \
                                    det_boxes_r_[:, 3], det_boxes_r_[:, 4]

            boxes_r = np.transpose(np.stack([x_c, y_c, w, h, theta]))
            dets_r = np.hstack((det_category_r_.reshape(-1, 1),
                                det_scores_r_.reshape(-1, 1),
                                boxes_r))
            all_boxes_r.append(dets_r)

            pbar.set_description("Eval image %s" % a_img_name)

        # fw1 = open(cfgs.VERSION + '_detections_r.pkl', 'wb')
        # pickle.dump(all_boxes_r, fw1)
        return all_boxes_r


def eval(num_imgs, img_dir, image_ext, test_annotation_path, draw_imgs):

    retinanet = build_whole_network.DetectionNetwork(base_network_name=cfgs.NET_NAME,
                                                     is_training=False)

    all_boxes_r = eval_with_plac(img_dir=img_dir, det_net=retinanet,
                                 num_imgs=num_imgs, image_ext=image_ext, draw_imgs=draw_imgs)

    # with open(cfgs.VERSION + '_detections_r.pkl', 'rb') as f2:
    #     all_boxes_r = pickle.load(f2)
    #
    #     print(len(all_boxes_r))

    imgs = os.listdir(img_dir)
    real_test_imgname_list = [i.split(image_ext)[0] for i in imgs]

    print(10 * "**")
    print('rotation eval:')
    voc_eval_r.voc_evaluate_detections(all_boxes=all_boxes_r,
                                       test_imgid_list=real_test_imgname_list,
                                       test_annotation_path=test_annotation_path)


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a R2CNN network')
    parser.add_argument('--img_dir', dest='img_dir',
                        help='images path',
                        default='/data/yangxue/dataset/SSDD++/test/JPEGImages', type=str)
    parser.add_argument('--image_ext', dest='image_ext',
                        help='image format',
                        default='.jpg', type=str)
    parser.add_argument('--test_annotation_path', dest='test_annotation_path',
                        help='test annotate path',
                        default='/data/yangxue/dataset/SSDD++/test/Annotations', type=str)
    parser.add_argument('--gpu', dest='gpu',
                        help='gpu index',
                        default='0', type=str)
    parser.add_argument('--draw_imgs', '-s', default=False,
                        action='store_true')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    eval(np.inf, args.img_dir, args.image_ext, args.test_annotation_path, args.draw_imgs)

