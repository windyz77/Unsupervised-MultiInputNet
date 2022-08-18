
"""Fully convolutional model for monocular depth estimation
    by Clement Godard, Oisin Mac Aodha and Gabriel J. Brostow
    http://visual.cs.ucl.ac.uk/pubs/monoDepth/
"""

from __future__ import absolute_import, division, print_function
from collections import namedtuple

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from general_funcs.bilinear_sampler_4 import *

"""monodepth LF model"""
class MultiDepthModel(object):

    def __init__(self, args, mode, image_list, reuse_variables=None):

        ori_disp, refine_disp = self.build_resnet50(image_list, reuse_variables)
        if mode == 'test_fliplrud':
            self.ori_disp = ori_disp
            self.refine_disp = refine_disp
            return

        self.total_loss, loss_dict = self.compute_losses(ori_disp, refine_disp, image_list, reuse_variables, args)
        self.build_summaries(ori_disp, refine_disp, self.total_loss, loss_dict, image_list)
        return

    def build_resnet50(self, image_list, reuse_variables):
        def conv(x, num_out_layers, kernel_size, stride, activation_fn=tf.nn.relu):
            p = np.floor((kernel_size - 1) / 2).astype(np.int32)
            p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
            return slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID', activation_fn=activation_fn)

        def upsample_nn(x, ratio):
            s = tf.shape(x)
            h = s[1]
            w = s[2]
            return tf.image.resize_nearest_neighbor(x, [h * ratio, w * ratio])

        def upconv(x, num_out_layers, kernel_size, scale):
            upsample = upsample_nn(x, scale)
            x = conv(upsample, num_out_layers, kernel_size, 1)
            return x

        def maxpool(x, kernel_size):
            p = np.floor((kernel_size - 1) / 2).astype(np.int32)
            p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
            return slim.max_pool2d(p_x, kernel_size)

        def resconv(x, num_layers, stride):
            do_proj = tf.shape(x)[3] != num_layers or stride == 2
            shortcut = []
            conv1 = conv(x, num_layers, 1, 1)
            conv2 = conv(conv1, num_layers, 3, stride)
            conv3 = conv(conv2, 4 * num_layers, 1, 1, None)
            if do_proj:
                shortcut = conv(x, 4 * num_layers, 1, stride, None)
            else:
                shortcut = x
            return tf.nn.elu(conv3 + shortcut)

        def resblock(x, num_layers, num_blocks):
            out = x
            for i in range(num_blocks - 1):
                out = resconv(out, num_layers, 1)
            out = resconv(out, num_layers, 2)
            return out

        def get_disp(x):
            disp = (conv(x, 1, 3, 1, activation_fn=tf.nn.sigmoid) - 0.5) * 8
            return disp


        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=tf.nn.elu):
            with tf.variable_scope('model', reuse=reuse_variables):
                center_image = image_list[0]
                multi_input = tf.concat([image_list[0],
                                         image_list[1], image_list[6],
                                         image_list[7], image_list[12],
                                         image_list[13], image_list[18],
                                         image_list[19], image_list[24]], axis=3)

                with tf.variable_scope('preconv'):
                    preconv = conv(multi_input, 64, 3, 1)
                    conv1 = conv(preconv, 64, 7, 2)  # H/2  -   64D
                    pool1 = maxpool(conv1, 3)  # H/4  -   64D

                with tf.variable_scope('encoder'):
                    conv2 = resblock(pool1, 64, 3)  # H/8  -  256D
                    conv3 = resblock(conv2, 128, 4)  # H/16 -  512D
                    conv4 = resblock(conv3, 256, 6)  # H/32 - 1024D
                    conv5 = resblock(conv4, 512, 3)  # H/64 - 2048D

                with tf.variable_scope('skips'):
                    skip1 = conv1
                    skip2 = pool1
                    skip3 = conv2
                    skip4 = conv3
                    skip5 = conv4

                # DECODING
                with tf.variable_scope('decoder'):
                    upconv6 = upconv(conv5, 512, 3, 2)  # H/32
                    concat6 = tf.concat([upconv6, skip5], 3)
                    iconv6 = conv(concat6, 512, 3, 1)
                    disp6 = get_disp(iconv6)
                    udisp6 = upsample_nn(disp6, 2)  # H/32

                    upconv5 = upconv(iconv6, 256, 3, 2)  # H/16
                    concat5 = tf.concat([upconv5, skip4, udisp6], 3)
                    iconv5 = conv(concat5, 256, 3, 1)
                    disp5 = get_disp(iconv5)
                    udisp5 = upsample_nn(disp5, 2)  # H/32

                    upconv4 = upconv(iconv5, 128, 3, 2)  # H/8
                    concat4 = tf.concat([upconv4, skip3, udisp5], 3)
                    iconv4 = conv(concat4, 128, 3, 1)
                    disp4 = get_disp(iconv4)
                    udisp4 = upsample_nn(disp4, 2)

                    upconv3 = upconv(iconv4, 64, 3, 2)  # H/4
                    concat3 = tf.concat([upconv3, skip2, udisp4], 3)
                    iconv3 = conv(concat3, 64, 3, 1)
                    disp3 = get_disp(iconv3)
                    udisp3 = upsample_nn(disp3, 2)

                    upconv2 = upconv(iconv3, 32, 3, 2)  # H/2
                    concat2 = tf.concat([upconv2, skip1, udisp3], 3)
                    iconv2 = conv(concat2, 32, 3, 1)
                    disp2 = get_disp(iconv2)
                    udisp2 = upsample_nn(disp2, 2)

                    upconv1 = upconv(iconv2, 16, 3, 2)  # H
                    concat1 = tf.concat([upconv1, udisp2], 3)
                    iconv1 = conv(concat1, 16, 3, 1)
                    ori_disp = get_disp(iconv1)

                with tf.variable_scope('refine'):
                    gray = tf.image.rgb_to_grayscale(center_image)
                    stack_ref = tf.stack([ori_disp, gray], axis=1)
                    stack_ref1 = slim.conv3d(stack_ref, 16, [3, 3, 3], 1)
                    stack_ref2 = slim.conv3d(stack_ref1, 32, [3, 3, 3], 1)
                    stack_ref3 = slim.conv3d(stack_ref2, 16, [3, 3, 3], 1)
                    stack_ref4 = slim.conv3d(stack_ref3, 16, [2, 1, 1], 1, padding='VALID')
                    stack_ref4 = tf.squeeze(stack_ref4)
                    refine_disp = get_disp(stack_ref4)
                    return ori_disp, refine_disp

    def compute_losses(self, ori_disp, refine_disp, image_list, reuse_variables, args):
        def SSIM(x, y):
            C1 = 0.01 ** 2
            C2 = 0.03 ** 2

            mu_x = slim.avg_pool2d(x, 3, 1, 'SAME')
            mu_y = slim.avg_pool2d(y, 3, 1, 'SAME')

            sigma_x = slim.avg_pool2d(x ** 2, 3, 1, 'SAME') - mu_x ** 2
            sigma_y = slim.avg_pool2d(y ** 2, 3, 1, 'SAME') - mu_y ** 2
            sigma_xy = slim.avg_pool2d(x * y, 3, 1, 'SAME') - mu_x * mu_y

            SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
            SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

            SSIM = SSIM_n / SSIM_d

            return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

        # compute image loss: l1 + ssim
        def compute_imageloss(im, im_est, args):
            im, im_est = im[:, 12:500, 12:500, :], im_est[:, 12:500, 12:500, :]
            l1_im = tf.abs(im_est - im)
            ssim_im = SSIM(im_est, im)
            return tf.reduce_mean(args.alpha_image_loss * ssim_im + (1 - args.alpha_image_loss) * l1_im)

        def generate_image_left(img, disp):
            return bilinear_sampler_1d_h(img, -disp)

        def generate_image_right(img, disp):
            return bilinear_sampler_1d_h(img, disp)

        def generate_image_top(img, disp):
            return bilinear_sampler_1d_v(img, -disp)

        def generate_image_bottom(img, disp):
            return bilinear_sampler_1d_v(img, disp)

        def generate_image_topleft(img, disp_x, disp_y):
            return bilinear_sampler_2d(img, -disp_x, -disp_y)

        def generate_image_topright(img, disp_x, disp_y):
            return bilinear_sampler_2d(img, disp_x, -disp_y)

        def generate_image_bottomleft(img, disp_x, disp_y):
            return bilinear_sampler_2d(img, -disp_x, disp_y)

        def generate_image_bottomright(img, disp_x, disp_y):
            return bilinear_sampler_2d(img, disp_x, disp_y)

        def comput_CAD_loss(centerdisp, total_est_image):
            # IMAGE LOSS OF EACH PIXEL: WEIGTHED SUM
            centerdisp = centerdisp[:, 12:500, 12:500, :]
            centerdisp = tf.expand_dims(centerdisp, axis=1)
            total_est_image = total_est_image[:, :, 12:500, 12:500, :]

            # GENERATE L1 DIFFERENCE
            l1_im = tf.abs(total_est_image - centerdisp)  # RGB different
            l1_im = -l1_im
            zu_of_num = 3

            temp = tf.expand_dims(tf.reduce_mean(l1_im[:, 0 * zu_of_num:(0 + 1) * zu_of_num, :, :, :], axis=1), axis=1)
            # temp /= 3.0  # each direcention mean
            color_constraint = tf.expand_dims(
                tf.reduce_mean(l1_im[:, 0 * zu_of_num:(0 + 1) * zu_of_num, :, :, :], axis=4), axis=4)
            with_constraint = temp + 0.1 * tf.reduce_mean(color_constraint)

            for i in range(1, int(int(l1_im.shape[1]) / zu_of_num)):
                temp = tf.expand_dims(tf.reduce_mean(l1_im[:, i * zu_of_num:(i + 1) * zu_of_num, :, :, :], axis=1),
                                      axis=1)
                # temp /= 3.0
                color_constraint = tf.expand_dims(
                    tf.reduce_mean(l1_im[:, i * zu_of_num:(i + 1) * zu_of_num, :, :, :], axis=4), axis=4)
                temp = temp + 0.1 * tf.reduce_mean(color_constraint)
                with_constraint = tf.concat([with_constraint, temp], axis=1)

            adaptive_refocusloss = slim.max_pool3d(with_constraint, kernel_size=[8, 1, 1],
                                                   stride=[1, 1, 1])  # the min direction mean
            adaptive_refocusloss = -adaptive_refocusloss * 8  # (4, 244, 244 ,3)

            return tf.reduce_mean(adaptive_refocusloss)

        def comput_CAE_loss(centerdisp, total_est_image):
            # IMAGE LOSS OF EACH PIXEL: WEIGTHED SUM
            centerdisp = centerdisp[:, 12:500, 12:500, :]
            centerdisp = tf.expand_dims(centerdisp, axis=1)
            total_est_image =  total_est_image[:, :, 12:500, 12:500, :]
            l1_im = tf.abs(centerdisp - total_est_image)  # RGB different (4, 24, 244, 244, 3)

            variance = 0.5
            fine = tf.maximum(0.0, 1.0 - tf.divide(l1_im, variance))
            h1 = tf.divide(tf.reduce_sum(fine, axis=1), 24.0)

            wi = tf.exp(-tf.divide(tf.square(l1_im), 2 * variance))
            wi = tf.divide(tf.reduce_sum(wi, axis=1), 24.0)
            gi = wi * h1
            gi = gi * 0.6 + 0.4

            AE_loss = - gi * tf.log(gi)
            CAE_loss = tf.reduce_mean(AE_loss)

            l1_pie = (1. - fine) * l1_im
            l1_pie_loss = tf.reduce_mean(l1_pie)

            return CAE_loss + l1_pie_loss

        def compute_loss_45degree_orientation(center_image, disp, image_list, args, disp_name):
            # get images
            topleft_im_3 = image_list[1]  # 000
            topleft_im_2 = image_list[2]  # 010
            topleft_im_1 = image_list[3]  # 020
            bottomright_im_1 = image_list[4]  # 040
            bottomright_im_2 = image_list[5]  # 050
            bottomright_im_3 = image_list[6]  # 060

            # GENERATE ESTIMATED CENTER IMAGES
            centerimg_est_from_topleft_1 = generate_image_bottomright(topleft_im_1, disp, disp)
            centerimg_est_from_topleft_2 = generate_image_bottomright(topleft_im_2, disp * 2.,
                                                                           disp * 2.)
            centerimg_est_from_topleft_3 = generate_image_bottomright(topleft_im_3, disp * 3.,
                                                                           disp * 3.)
            centerimg_est_from_bottomright_1 = generate_image_topleft(bottomright_im_1, disp, disp)
            centerimg_est_from_bottomright_2 = generate_image_topleft(bottomright_im_2, disp * 2.,
                                                                           disp * 2.)
            centerimg_est_from_bottomright_3 = generate_image_topleft(bottomright_im_3, disp * 3.,
                                                                           disp * 3.)

            # COMPUTE IMAGE LOSS
            imloss_center_with_topleft_1 = compute_imageloss(center_image, centerimg_est_from_topleft_1, args)  # if disocc not nee, argsd
            imloss_center_with_topleft_2 = compute_imageloss(center_image, centerimg_est_from_topleft_2, args)
            imloss_center_with_topleft_3 = compute_imageloss(center_image, centerimg_est_from_topleft_3, args)

            imloss_center_with_bottomright_1 = compute_imageloss(center_image, centerimg_est_from_bottomright_1, args)  # if disocc not nee, argsd
            imloss_center_with_bottomright_2 = compute_imageloss(center_image, centerimg_est_from_bottomright_2, args)
            imloss_center_with_bottomright_3 = compute_imageloss(center_image, centerimg_est_from_bottomright_3, args)

            cur_refocused_image = (centerimg_est_from_topleft_1 + centerimg_est_from_topleft_2 + centerimg_est_from_topleft_3 +
                                   centerimg_est_from_bottomright_1 + centerimg_est_from_bottomright_2 + centerimg_est_from_bottomright_3)

            if disp_name == 'refine_disp':
                est_image_45 = tf.stack([centerimg_est_from_topleft_1, centerimg_est_from_topleft_2, centerimg_est_from_topleft_3,
                                          centerimg_est_from_bottomright_1, centerimg_est_from_bottomright_2,
                                          centerimg_est_from_bottomright_3], axis=1)
            else:
                est_image_45 = None

            cur_imageloss = (imloss_center_with_topleft_1 + imloss_center_with_topleft_2 + imloss_center_with_topleft_3 +
                    imloss_center_with_bottomright_1 + imloss_center_with_bottomright_2 + imloss_center_with_bottomright_3)
            return cur_imageloss, cur_refocused_image, est_image_45

        def compute_loss_135degree_orientation(center_image, disp, image_list, args, disp_name):
            # get images
            topright_im_3 = image_list[7]  # 006
            topright_im_2 = image_list[8]  # 014
            topright_im_1 = image_list[9]  # 022
            bottomleft_im_1 = image_list[10]  # 038
            bottomleft_im_2 = image_list[11]  # 046
            bottomleft_im_3 = image_list[12]  # 054
            # GENERATE ESTIMATED CENTER IMAGES
            centerimg_est_from_topright_1 = generate_image_bottomleft(topright_im_1, disp, disp)
            centerimg_est_from_topright_2 = generate_image_bottomleft(topright_im_2, disp * 2.,
                                                                           disp * 2.)
            centerimg_est_from_topright_3 = generate_image_bottomleft(topright_im_3, disp * 3.,
                                                                           disp * 3.)
            centerimg_est_from_bottomleft_1 = generate_image_topright(bottomleft_im_1, disp, disp)
            centerimg_est_from_bottomleft_2 = generate_image_topright(bottomleft_im_2, disp * 2.,
                                                                           disp * 2.)
            centerimg_est_from_bottomleft_3 = generate_image_topright(bottomleft_im_3, disp * 3.,
                                                                           disp * 3.)

            # COMPUTE IMAGE LOSS
            imloss_center_with_topright_1 = compute_imageloss(center_image, centerimg_est_from_topright_1, args)  # if disocc not nee, argsd
            imloss_center_with_topright_2 = compute_imageloss(center_image, centerimg_est_from_topright_2, args)
            imloss_center_with_topright_3 = compute_imageloss(center_image, centerimg_est_from_topright_3, args)

            imloss_center_with_bottomleft_1 = compute_imageloss(center_image, centerimg_est_from_bottomleft_1, args)  # if disocc not nee, argsd
            imloss_center_with_bottomleft_2 = compute_imageloss(center_image, centerimg_est_from_bottomleft_2, args)
            imloss_center_with_bottomleft_3 = compute_imageloss(center_image, centerimg_est_from_bottomleft_3, args)

            cur_refocused_image = (centerimg_est_from_topright_1 + centerimg_est_from_topright_2 + centerimg_est_from_topright_3 +
                                   centerimg_est_from_bottomleft_1 + centerimg_est_from_bottomleft_2 + centerimg_est_from_bottomleft_3)

            if disp_name == 'refine_disp':
                est_image_135 = tf.stack([centerimg_est_from_topright_1, centerimg_est_from_topright_2, centerimg_est_from_topright_3,
                                          centerimg_est_from_bottomleft_1, centerimg_est_from_bottomleft_2, centerimg_est_from_bottomleft_3], axis=1)
            else:
                est_image_135 = None

            # TOTAL IMAGE LOSS
            cur_imageloss = (imloss_center_with_topright_1 + imloss_center_with_topright_2 + imloss_center_with_topright_3 +
                    imloss_center_with_bottomleft_1 + imloss_center_with_bottomleft_2 + imloss_center_with_bottomleft_3)
            return cur_imageloss, cur_refocused_image, est_image_135

        # compute loss among horizental subapertures
        def compute_loss_horizental_orientation(center_image, disp, image_list, args, disp_name):
            # get images
            left_im_3 = image_list[13]  # 027
            left_im_2 = image_list[14]  # 028
            left_im_1 = image_list[15]  # 029
            right_im_1 = image_list[16]  # 031
            right_im_2 = image_list[17]  # 032
            right_im_3 = image_list[18]  # 033

            # GENERATE ESTIMATED CENTER IMAGES
            centerimg_est_from_left_1 = generate_image_right(left_im_1, disp)
            centerimg_est_from_left_2 = generate_image_right(left_im_2, disp * 2.)
            centerimg_est_from_left_3 = generate_image_right(left_im_3, disp * 3.)
            centerimg_est_from_right_1 = generate_image_left(right_im_1, disp)
            centerimg_est_from_right_2 = generate_image_left(right_im_2, disp * 2.)
            centerimg_est_from_right_3 = generate_image_left(right_im_3, disp * 3.)

            # COMPUTE IMAGE LOSS
            imloss_center_with_left_1 = compute_imageloss(center_image, centerimg_est_from_left_1, args)
            imloss_center_with_left_2 = compute_imageloss(center_image, centerimg_est_from_left_2, args)
            imloss_center_with_left_3 = compute_imageloss(center_image, centerimg_est_from_left_3, args)

            imloss_center_with_right_1 = compute_imageloss(center_image, centerimg_est_from_right_1, args)
            imloss_center_with_right_2 = compute_imageloss(center_image, centerimg_est_from_right_2, args)
            imloss_center_with_right_3 = compute_imageloss(center_image, centerimg_est_from_right_3, args)

            cur_refocused_image = (centerimg_est_from_left_1 + centerimg_est_from_left_2 + centerimg_est_from_left_3 +
                                  centerimg_est_from_right_1 + centerimg_est_from_right_2 + centerimg_est_from_right_3)

            if disp_name == 'refine_disp':
                est_image_0 = tf.stack([centerimg_est_from_left_1, centerimg_est_from_left_2, centerimg_est_from_left_3,
                                        centerimg_est_from_right_1, centerimg_est_from_right_2, centerimg_est_from_right_3], axis=1)
            else:
                est_image_0 = None

            cur_imageloss = (imloss_center_with_left_1 + imloss_center_with_left_2 + imloss_center_with_left_3 +
                    imloss_center_with_right_1 + imloss_center_with_right_2 + imloss_center_with_right_3)
            return cur_imageloss, cur_refocused_image, est_image_0

        # compute loss among vertical subapertures
        def compute_loss_vertical_orientation(center_image, disp, image_list, args, disp_name):
            # get images
            top_im_3 = image_list[19]  # 003
            top_im_2 = image_list[20]  # 012
            top_im_1 = image_list[21]  # 021
            bottom_im_1 = image_list[22]  # 039
            bottom_im_2 = image_list[23]  # 048
            bottom_im_3 = image_list[24]  # 057
            # GENERATE ESTIMATED CENTER IMAGES
            centerimg_est_from_top_1 = generate_image_bottom(top_im_1, disp)
            centerimg_est_from_top_2 = generate_image_bottom(top_im_2, disp * 2.)
            centerimg_est_from_top_3 = generate_image_bottom(top_im_3, disp * 3.)
            centerimg_est_from_bottom_1 = generate_image_top(bottom_im_1, disp)
            centerimg_est_from_bottom_2 = generate_image_top(bottom_im_2, disp * 2.)
            centerimg_est_from_bottom_3 = generate_image_top(bottom_im_3, disp * 3.)

            #
            # COMPUTE IMAGE LOSS
            imloss_center_with_top_1 = compute_imageloss(center_image, centerimg_est_from_top_1, args)
            imloss_center_with_top_2 = compute_imageloss(center_image, centerimg_est_from_top_2, args)
            imloss_center_with_top_3 = compute_imageloss(center_image, centerimg_est_from_top_3, args)

            imloss_center_with_bottom_1 = compute_imageloss(center_image, centerimg_est_from_bottom_1, args)
            imloss_center_with_bottom_2 = compute_imageloss(center_image, centerimg_est_from_bottom_2, args)
            imloss_center_with_bottom_3 = compute_imageloss(center_image, centerimg_est_from_bottom_3, args)
            cur_refocused_image = (centerimg_est_from_top_1 + centerimg_est_from_top_2 + centerimg_est_from_top_3 +
                                   centerimg_est_from_bottom_1 + centerimg_est_from_bottom_2 + centerimg_est_from_bottom_3)

            if disp_name == 'refine_disp':
                est_image_90 = tf.stack([centerimg_est_from_top_1, centerimg_est_from_top_2, centerimg_est_from_top_3,
                                         centerimg_est_from_bottom_1, centerimg_est_from_bottom_2, centerimg_est_from_bottom_3], axis=1)
            else:
                est_image_90 = None

            # TOTAL IMAGE LOSS
            cur_imageloss = (imloss_center_with_top_1 + imloss_center_with_top_2 + imloss_center_with_top_3 +
                    imloss_center_with_bottom_1 + imloss_center_with_bottom_2 + imloss_center_with_bottom_3)
            return cur_imageloss, cur_refocused_image, est_image_90

        with tf.variable_scope('losses', reuse=reuse_variables):
            # IMAGE LOSS: WEIGTHED SUM
            oridisp_image_loss_45, oridisp_refocused_image_45, _ = compute_loss_45degree_orientation(image_list[0], ori_disp, image_list, args, 'ori_disp')
            oridisp_image_loss_135, oridisp_refocused_image_135, _ = compute_loss_135degree_orientation(image_list[0], ori_disp, image_list, args, 'ori_disp')
            oridisp_image_loss_0, oridisp_refocused_image_0, _ = compute_loss_horizental_orientation(image_list[0], ori_disp, image_list, args, 'ori_disp')
            oridisp_image_loss_90, oridisp_refocused_image_90, _ = compute_loss_vertical_orientation(image_list[0], ori_disp, image_list, args, 'ori_disp')
            oridisp_imageloss = oridisp_image_loss_0 + oridisp_image_loss_90 + oridisp_image_loss_45 + oridisp_image_loss_135

            oridisp_refocusdim = (image_list[0]+ oridisp_refocused_image_45 + oridisp_refocused_image_135 + oridisp_refocused_image_0 + oridisp_refocused_image_90) * 0.04
            oridisp_refocusloss = compute_imageloss(image_list[0], oridisp_refocusdim, args)

            refinedisp_imageloss_45, refinedisp_refocused_image_45, est_image_45 = compute_loss_45degree_orientation(image_list[0], refine_disp, image_list, args, 'refine_disp')
            refinedisp_imageloss_135, refinedisp_refocused_image_135, est_image_135 = compute_loss_135degree_orientation(image_list[0], refine_disp, image_list, args, 'refine_disp')
            refinedisp_imageloss_0, refinedisp_refocused_image_0, est_image_0 = compute_loss_horizental_orientation(image_list[0], refine_disp, image_list, args, 'refine_disp')
            refinedisp_imageloss_90, refinedisp_refocused_image_90, est_image_90 = compute_loss_vertical_orientation(image_list[0], refine_disp, image_list, args, 'refine_disp')
            refinedisp_imageloss = refinedisp_imageloss_45 + refinedisp_imageloss_135 + refinedisp_imageloss_0 + refinedisp_imageloss_90

            refinedisp_refocusdim = (image_list[0]+ refinedisp_refocused_image_45 + refinedisp_refocused_image_135 + refinedisp_refocused_image_0 + refinedisp_refocused_image_90) * 0.04
            refinedisp_refocusloss = compute_imageloss(image_list[0], refinedisp_refocusdim, args)


            total_est_image = tf.concat([est_image_45, est_image_135, est_image_0, est_image_90], axis=1)
            CADloss = comput_CAD_loss(image_list[0], total_est_image)
            CAEloss = comput_CAE_loss(image_list[0], total_est_image)

            total_loss = oridisp_imageloss + refinedisp_imageloss + (oridisp_refocusloss + refinedisp_refocusloss) * 10. + CADloss + CAEloss

            loss_dict = {'oridisp_imageloss':oridisp_imageloss,
                         'oridisp_refocusloss':oridisp_refocusloss,
                         'refinedisp_imageloss':refinedisp_imageloss,
                         'refinedisp_refocusloss':refinedisp_refocusloss,
                         'CADloss':CADloss,
                         'CAEloss':CAEloss}
            return total_loss, loss_dict

    def build_summaries(self, ori_disp, refine_disp, total_loss, loss_dict, image_list):
        # SUMMARIES
        with tf.device('/cpu:0'):
            tf.summary.scalar('total_loss', total_loss)
            tf.summary.scalar('oridisp_imageloss', loss_dict['oridisp_imageloss'])
            tf.summary.scalar('oridisp_refocusloss', loss_dict['oridisp_refocusloss'])
            tf.summary.scalar('refinedisp_imageloss', loss_dict['refinedisp_imageloss'])
            tf.summary.scalar('refinedisp_refocusloss', loss_dict['refinedisp_refocusloss'])
            tf.summary.scalar('CADloss', loss_dict['CADloss'])
            tf.summary.scalar('CAEloss', loss_dict['CAEloss'])

            # show disparity
            tf.summary.image('ori_disp', (ori_disp + 4.) * 32, max_outputs=4)
            tf.summary.image('refine_disp', (refine_disp + 4.) * 32, max_outputs=4)

            tf.summary.image('center_image', image_list[0], max_outputs=4)
            return

