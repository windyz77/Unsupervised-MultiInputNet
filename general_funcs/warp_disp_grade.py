from optical_flow_warp_fwd import *
from optical_flow_warp_old import *
from bilinear_sampler_4 import *
import tensorflow as tf
from file_io import *
import os
import math
import cv2

pfm = read_pfm('/home/jethong/Desktop/MII/disp_maps/boxes.pfm')
# pfm = np.expand_dims(pfm, axis=0)
# pfm = np.expand_dims(pfm, axis=3)

png_file = '/home/jethong/data/full_data/training/boxes/'
pfm_list = ['input_Cam030.png', 'input_Cam031.png', 'input_Cam032.png', 'input_Cam039.png', 'input_Cam041.png',
            'input_Cam048.png', 'input_Cam049.png', 'input_Cam050.png']
# list_func = [warp_from_topleft(img, disp, i)]

# def warp_from_left(img, disp, i):
#     temp = np.zeros([1, 512, 512, 2]).astype(np.float32)
#     temp[:, :, :, 0] = disp
#     return transformerFwd(img, -i * temp, (512, 512))
#
#
# def warp_from_right(img, disp, i):
#     temp = np.zeros([1, 512, 512, 2]).astype(np.float32)
#     temp[:, :, :, 0] = disp
#     return transformerFwd(img, i * temp, (512, 512))
#
#
# def warp_from_top(img, disp, i):
#     temp = np.zeros([1, 512, 512, 2]).astype(np.float32)
#     temp[:, :, :, 1] = disp
#     return transformerFwd(img, -i * temp, (512, 512))
#
#
# def warp_from_bottom(img, disp, i):
#     temp = np.zeros([1, 512, 512, 2]).astype(np.float32)
#     temp[:, :, :, 1] = disp
#     return transformerFwd(img, i * temp, (512, 512))
#
#
# def warp_from_topleft(img, disp, i):
#     temp = np.zeros([1, 512, 512, 2]).astype(np.float32)
#     temp[:, :, :, 0] = disp * (math.sqrt(2) / 2)
#     temp[:, :, :, 1] = disp * (math.sqrt(2) / 2)
#     return transformerFwd(img, -i * temp, (512, 512))
#
#
# def warp_from_bottomright(img, disp, i):
#     temp = np.zeros([1, 512, 512, 2]).astype(np.float32)
#     temp[:, :, :, 0] = disp * (math.sqrt(2) / 2)
#     temp[:, :, :, 1] = disp * (math.sqrt(2) / 2)
#     return transformerFwd(img, i * temp, (512, 512))
#
#
# def warp_from_topright(img, disp, i):
#     temp = np.zeros([1, 512, 512, 2]).astype(np.float32)
#     temp[:, :, :, 0] = disp * (math.sqrt(2) / 2)
#     temp[:, :, :, 1] = -1 * disp * (math.sqrt(2) / 2)
#     return transformerFwd(img, i * temp, (512, 512))
#
#
# def warp_from_bottomleft(img, disp, i):
#     temp = np.zeros([1, 512, 512, 2]).astype(np.float32)
#     temp[:, :, :, 0] = -1 * disp * (math.sqrt(2) / 2)
#     temp[:, :, :, 1] = disp * (math.sqrt(2) / 2)
#     return transformerFwd(img, i * temp, (512, 512))

def warp_from_left(img, disp, i):
    temp = np.zeros([1, 512, 512, 2]).astype(np.float32)
    temp[:, :, :, 0] = disp
    return transformer_old(img, i * temp, (512, 512))


def warp_from_right(img, disp, i):
    temp = np.zeros([1, 512, 512, 2]).astype(np.float32)
    temp[:, :, :, 0] = disp
    return transformer_old(img, -i * temp, (512, 512))


def warp_from_top(img, disp, i):
    temp = np.zeros([1, 512, 512, 2]).astype(np.float32)
    temp[:, :, :, 1] = disp
    return transformer_old(img, i * temp, (512, 512))


def warp_from_bottom(img, disp, i):
    temp = np.zeros([1, 512, 512, 2]).astype(np.float32)
    temp[:, :, :, 1] = disp
    return transformer_old(img, -i * temp, (512, 512))


def warp_from_topleft(img, disp, i):
    temp = np.zeros([1, 512, 512, 2]).astype(np.float32)
    temp[:, :, :, 0] = disp * (math.sqrt(2) / 2)
    temp[:, :, :, 1] = disp * (math.sqrt(2) / 2)
    return transformer_old(img, i * temp, (512, 512))


def warp_from_bottomright(img, disp, i):
    temp = np.zeros([1, 512, 512, 2]).astype(np.float32)
    temp[:, :, :, 0] = disp * (math.sqrt(2) / 2)
    temp[:, :, :, 1] = disp * (math.sqrt(2) / 2)
    return transformer_old(img, -i * temp, (512, 512))


def warp_from_topright(img, disp, i):
    temp = np.zeros([1, 512, 512, 2]).astype(np.float32)
    temp[:, :, :, 0] = -1 * disp * (math.sqrt(2) / 2)
    temp[:, :, :, 1] = disp * (math.sqrt(2) / 2)
    return transformer_old(img, i * temp, (512, 512))


def warp_from_bottomleft(img, disp, i):
    temp = np.zeros([1, 512, 512, 2]).astype(np.float32)
    temp[:, :, :, 0] = disp * (math.sqrt(2) / 2)
    temp[:, :, :, 1] = -1 * disp * (math.sqrt(2) / 2)
    return transformer_old(img, i * temp, (512, 512))

# temp = np.zeros([1, 512, 512, 2]).astype(np.float32)
# temp[:, :, :, 0] = pfm # horizontal
# temp[:, :, :, 1] = pfm    #vertical
# temp[:, :, :, 0] = pfm * (1 / 2)
# temp[:, :, :, 1] = pfm * (1 / 2)  # topleft-----bottomright
# temp[:, :, :, 0] = pfm * (math.sqrt(2) / 2)
# temp[:, :, :, 1] = -1 * pfm * (math.sqrt(2) / 2)  #


# result = transformer_old(png, 3 * temp, (512, 512))  # ----------------left-right----------

# result = transformerFwd(png, -3 * temp, (512, 512))  # ----------------left-right----------
# result = transformerFwd(png, temp, (512, 512)) #----------------right-left----------
# result = transformerFwd(png, -temp, (512, 512)) #----------------top-low----------
# result = transformerFwd(png, 3 * temp, (512, 512)) #----------------low-top----------
# result = transformerFwd(png, -temp, (512, 512))  # ----------------topleft-----bottomright----------
# result = transformerFwd(png, temp, (512, 512))  # ----------------bottomright----topleft---------------
# result = transformerFwd(png, temp, (512, 512))  # ----------------bottomright----topleft---------------
temp_name = ''
for i in range(len(pfm_list)):
    png = cv2.imread(png_file + pfm_list[i])
    png = np.expand_dims(png, axis=0)
    png = tf.cast(png, tf.float32)
    if i == 0:
        result = warp_from_topleft(png, pfm, 1)
        temp_name = 'warp_from_topleft'
    if i == 1:
        result = warp_from_top(png, pfm, 1)
        temp_name = 'warp_from_top'
    if i == 2:
        result = warp_from_topright(png, pfm, 1)
        temp_name = 'warp_from_topright'
    if i == 3:
        result = warp_from_left(png, pfm, 1)
        temp_name = 'warp_from_left'
    if i == 4:
        result = warp_from_right(png, pfm, 1)
        temp_name = 'warp_from_right'
    if i == 5:
        result = warp_from_bottomleft(png, pfm, 1)
        temp_name = 'warp_from_bottomleft'
    if i == 6:
        result = warp_from_bottom(png, pfm, 1)
        temp_name = 'warp_from_bottom'
    if i == 7:
        result = warp_from_bottomright(png, pfm, 1)
        temp_name = 'warp_from_bottomright'
    # result = warp_from_bottomright(png, pfm, 1)

    # warp = tf.squeeze(result)
    # with tf.Session() as sess:
    #     a = sess.run(warp)
    #     a = np.uint8(a)
    #     cv2.imwrite(temp_name + '.png', a)
