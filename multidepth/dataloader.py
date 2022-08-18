#coding=utf-8
from __future__ import absolute_import, division, print_function
import tensorflow as tf
import os

class Dataloader(object):
    def string_length_tf(self, t):
        return tf.py_func(len, [t], [tf.int64])

    def read_image(self, image_path, resize_h, resize_w):
        image = tf.image.decode_png(tf.read_file(image_path))
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize_images(image, [resize_h, resize_w], tf.image.ResizeMethod.AREA)
        return image

    def augment_image_pair_list(self, image_list):
        # randomly shift gamma
        random_gamma = tf.random_uniform([], 0.8, 1.2)

        image_aug_list = [single **random_gamma for single in image_list]

        # randomly shift brightness
        random_brightness = tf.random_uniform([], 0.5, 2.0)
        image_aug_list = [single * random_brightness for single in image_aug_list]

        # randomly shift color
        random_colors = tf.random_uniform([3], 0.8, 1.2)
        white = tf.ones([tf.shape(image_list[0])[0], tf.shape(image_list[0])[1]])
        color_image = tf.stack([white * random_colors[i] for i in range(3)], axis=2)
        image_aug_list = [single * color_image for single in image_aug_list]

        # saturate
        image_aug_list = [tf.clip_by_value(single, 0, 1) for single in image_aug_list]

        return image_aug_list

    def __init__(self, data_path, file_path, args, mode):
        input_queue = tf.train.string_input_producer([file_path], shuffle=False)
        line_reader = tf.TextLineReader()
        _, line = line_reader.read(input_queue)

        line_split = tf.string_split([line]).values

        image_path = [tf.string_join([data_path, '/', line_split[i]]) for i in range(25)]
        image_list = [self.read_image(single_path, args.input_height, args.input_width) for single_path in image_path]

        center_image = image_list[0]
        na_list = image_list[1:7]  # 1,2,3,4,5,6  # left up to right down
        pie_list = image_list[7:13]  # 7,8,9,10,11,12  # right up to left down
        heng_list = image_list[13:19]  # 13,14,15,16,17,18  # left to right
        heng_list_inverse = image_list[18:12:-1]  # 18,17,16,15,14,13  # right to left
        shu_list = image_list[19:25]  # 19,20,21,22,23,24  # up to down

        if mode == 'train':
            flip_leftright = tf.random_uniform([], 0, 1)
            flip_updown = tf.random_uniform([], 0, 1)
        elif mode == 'test_fliplr':
            flip_leftright = tf.random_uniform([], 0.6, 1)
            flip_updown = tf.random_uniform([], 0, 0.4)
        elif mode == 'test_flipud':
            flip_leftright = tf.random_uniform([], 0, 0.4)
            flip_updown = tf.random_uniform([], 0.6, 1)
        elif mode == 'test_fliplrud':
            flip_leftright = tf.random_uniform([], 0.6, 1)
            flip_updown = tf.random_uniform([], 0.6, 1)
        elif mode == 'code_test':
            flip_leftright = tf.random_uniform([], 0.6, 1)
            flip_updown = tf.random_uniform([], 0.6, 1)
        else:
            raise Exception('wrong mode')

        # flip left right
        center_image_fliplr = tf.cond(flip_leftright > 0.5, lambda: tf.image.flip_left_right(center_image), lambda: center_image)
        na_list_fliplr = tf.cond(flip_leftright > 0.5, lambda: [tf.image.flip_left_right(single) for single in pie_list], lambda: na_list)
        pie_list_fliplr = tf.cond(flip_leftright > 0.5, lambda: [tf.image.flip_left_right(single) for single in na_list], lambda: pie_list)
        heng_list_fliplr = tf.cond(flip_leftright > 0.5, lambda: [tf.image.flip_left_right(single) for single in heng_list_inverse], lambda: heng_list)
        shu_list_fliplr = tf.cond(flip_leftright > 0.5, lambda: [tf.image.flip_left_right(single) for single in shu_list], lambda: shu_list)

        # flip up down after fliplr
        na_list_fliplr_inverse = na_list_fliplr[::-1]
        pie_list_fliplr_inverse = pie_list_fliplr[::-1]
        shu_list_fliplr_inverse = shu_list_fliplr[::-1]

        center_image_fliplrud = tf.cond(flip_updown > 0.5, lambda: tf.image.flip_up_down(center_image_fliplr), lambda: center_image_fliplr)
        na_list_fliplrud = tf.cond(flip_updown > 0.5, lambda: [tf.image.flip_up_down(single) for single in pie_list_fliplr_inverse], lambda: na_list_fliplr)
        pie_list_fliplrud = tf.cond(flip_updown > 0.5, lambda: [tf.image.flip_up_down(single) for single in na_list_fliplr_inverse], lambda: pie_list_fliplr)
        heng_list_fliplrud = tf.cond(flip_updown > 0.5, lambda: [tf.image.flip_up_down(single) for single in heng_list_fliplr], lambda: heng_list_fliplr)
        shu_list_fliplrud = tf.cond(flip_updown > 0.5, lambda: [tf.image.flip_up_down(single) for single in shu_list_fliplr_inverse], lambda: shu_list_fliplr)

        # 找25张图，检查左右翻转和上下翻转是否正确。因为尤其实上下翻转的时候撇捺的方向。物体移动的方向是否符合预期。
        # os._exit(0)

        if mode == 'train':
            image_list_fliplrud = [center_image_fliplrud] + na_list_fliplrud + pie_list_fliplrud + heng_list_fliplrud + shu_list_fliplrud

            # data augement
            augment = tf.random_uniform([], 0, 1)
            image_list_fliplrud = tf.cond(augment > 0.5, lambda: self.augment_image_pair_list(image_list_fliplrud), lambda: image_list_fliplrud)
            for single_image in image_list_fliplrud:
                single_image.set_shape([None, None, 3])

            min_after_dequeue = 254
            capacity = min_after_dequeue + 4 * args.batch_size
            self.data_batch = tf.train.shuffle_batch(image_list_fliplrud, args.batch_size, capacity, min_after_dequeue, args.num_threads)
        elif mode == 'test_fliplr':
            image_list_fliplr = [center_image_fliplr] + na_list_fliplr + pie_list_fliplr + heng_list_fliplr + shu_list_fliplr
            self.data_batch = []
            for i in range(25):
                # check shape
                cur = tf.stack([image_list[i], image_list_fliplr[i]], axis=0)
                cur.set_shape([2, None, None, 3])
                self.data_batch.append(cur)
        elif mode == 'test_flipud':
            image_list_fliplrud = [center_image_fliplrud] + na_list_fliplrud + pie_list_fliplrud + heng_list_fliplrud + shu_list_fliplrud
            self.data_batch = []
            for i in range(25):
                # check shape
                cur = tf.stack([image_list[i], image_list_fliplrud[i]], axis=0)
                cur.set_shape([2, None, None, 3])
                self.data_batch.append(cur)
        elif mode == 'test_fliplrud':
            # flip up down only
            na_list_inverse = na_list[::-1]
            pie_list_inverse = pie_list[::-1]
            shu_list_inverse = shu_list[::-1]
            center_image_flipud = tf.cond(flip_updown > 0.5, lambda: tf.image.flip_up_down(center_image), lambda: center_image)
            na_list_flipud = tf.cond(flip_updown > 0.5, lambda: [tf.image.flip_up_down(single) for single in pie_list_inverse], lambda: na_list)
            pie_list_flipud = tf.cond(flip_updown > 0.5, lambda: [tf.image.flip_up_down(single) for single in na_list_inverse], lambda: pie_list)
            heng_list_flipud = tf.cond(flip_updown > 0.5, lambda: [tf.image.flip_up_down(single) for single in heng_list], lambda: heng_list)
            shu_list_flipud = tf.cond(flip_updown > 0.5, lambda: [tf.image.flip_up_down(single) for single in shu_list_inverse], lambda: shu_list)

            image_list_fliplr = [center_image_fliplr] + na_list_fliplr + pie_list_fliplr + heng_list_fliplr + shu_list_fliplr
            image_list_flipud = [center_image_flipud] + na_list_flipud + pie_list_flipud + heng_list_flipud + shu_list_flipud

            self.data_batch = []
            for i in range(25):
                temp = tf.stack([image_list[i], image_list_fliplr[i], image_list_flipud[i]], axis=0)
                temp.set_shape([3, None, None, 3])
                self.data_batch.append(temp)
        elif mode == 'code_test':
            # flip up down only
            na_list_inverse = na_list[::-1]
            pie_list_inverse = pie_list[::-1]
            shu_list_inverse = shu_list[::-1]
            center_image_flipud = tf.cond(flip_updown > 0.5, lambda: tf.image.flip_up_down(center_image), lambda: center_image)
            na_list_flipud = tf.cond(flip_updown > 0.5, lambda: [tf.image.flip_up_down(single) for single in pie_list_inverse], lambda: na_list)
            pie_list_flipud = tf.cond(flip_updown > 0.5, lambda: [tf.image.flip_up_down(single) for single in na_list_inverse], lambda: pie_list)
            heng_list_flipud = tf.cond(flip_updown > 0.5, lambda: [tf.image.flip_up_down(single) for single in heng_list], lambda: heng_list)
            shu_list_flipud = tf.cond(flip_updown > 0.5, lambda: [tf.image.flip_up_down(single) for single in shu_list_inverse], lambda: shu_list)

            image_list_fliplr = [center_image_fliplr] + na_list_fliplr + pie_list_fliplr + heng_list_fliplr + shu_list_fliplr
            image_list_flipud = [center_image_flipud] + na_list_flipud + pie_list_flipud + heng_list_flipud + shu_list_flipud
            image_list_fliplrud = [center_image_fliplrud] + na_list_fliplrud + pie_list_fliplrud + heng_list_fliplrud + shu_list_fliplrud

            self.data_batch = []
            for i in range(25):
                temp = tf.stack([image_list[i], image_list_fliplr[i], image_list_flipud[i], image_list_fliplrud[i]], axis=0)
                temp.set_shape([4, None, None, 3])
                self.data_batch.append(temp)
        return










