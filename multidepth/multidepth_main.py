#coding=utf-8
from __future__ import absolute_import, division, print_function
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf
from multidepth.dataloader import Dataloader
from multidepth.unet_model import MultiDepthModel
import argparse
from tensorflow.python import pywrap_tensorflow
import numpy as np
from general_funcs.evalfunctions7x7 import *


def get_args():
    parser = argparse.ArgumentParser(description='4DLightFiled MultiDepth')

    parser.add_argument('--model_name', type=str, help='model_name', default='MultiDepth')
    parser.add_argument('--data_path', type=str, help='path to data', default='/root/liqiujian/AI/MultiDepth2020/MultiDepth2020/full_data')
    parser.add_argument('--input_height', type=int, help='input height', default=512)
    parser.add_argument('--input_width', type=int, help='input width', default=512)
    parser.add_argument('--batch_size', type=int, help='batch_size', default=2)
    parser.add_argument('--total_epoch', type=int, help='number of epoch', default=300)
    parser.add_argument('--learning_rate', type=float, help='initial learning rate', default=1e-4)
    parser.add_argument('--lr_decay_boundries', type=list, help='learning rate decay boundries of epcoh', default=[100, 150, 180, 200])
    parser.add_argument('--lr_decay_values', type=list, help='learning rate decay values of each boundries', default=[1.0, 1/2, 1/4, 1/8, 1/16])
    parser.add_argument('--alpha_image_loss', type=float, help='weight between SSIM and L1 in image loss', default=0.85)
    parser.add_argument('--num_threads', type=int, help='number of threads to use for data loading', default=8)
    parser.add_argument('--output_path', type=str, help='directory to save log and results', default='./output/check_code_oldcode')
    parser.add_argument('--train_txt_path', type=str, help='filenames of train txt path', default='/root/liqiujian/AI/MultiDepth2020/MultiDepth2020/train_val_txt/4dlf_7x7star_train.txt')
    parser.add_argument('--val_txt_path', type=str, help='filenames of val txt path', default='/root/liqiujian/AI/MultiDepth2020/MultiDepth2020/train_val_txt/4dlf_7x7star_val.txt')
    parser.add_argument('--gt_path', type=str, help='path to gt', default='/root/liqiujian/AI/MultiDepth2020/MultiDepth2020/train_val_txt/4dlf_gt.txt')
    # parser.add_argument('--checkpoint_path', type=str, help='checkpoint path if load', default='./output/check_code1/MultiDepth/model-26664')
    # parser.add_argument('--retrain_epoch', type=int, help='retrain epoch if retrain', default=100)
    parser.add_argument('--checkpoint_path', type=str, help='checkpoint path if load', default='')
    parser.add_argument('--retrain_epoch', type=int, help='retrain epoch if retrain', default=0)
    parser.add_argument('--compute_refocusloss', type=bool, help='whether compute refocus loss', default=True)
    args = parser.parse_args()
    return args

def count_text_lines(txt_path):
    with open(txt_path, 'r') as txt_file:
        lines = txt_file.readlines()
    return len(lines)

def load_param_only(sess, ckpt_path):
    reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
    print('load params from path {}'.format(ckpt_path))

    restore_dict = dict()
    for v in tf.trainable_variables():
        tensor_name = v.name.split(':')[0]
        if reader.has_tensor(tensor_name):
            restore_dict[tensor_name] = v
    saver = tf.train.Saver(restore_dict)
    saver.restore(sess, ckpt_path)
    return

def post_process_disp(disp, mode):
    _, h, w = disp.shape
    l_disp = disp[0, :, :]
    if mode == 'fliplr':
        r_disp = np.fliplr(disp[1, :, :])
    elif mode == 'flipud':
        r_disp = np.flipud(disp[1, :, :])
    else:
        raise Exception('wrong post process disp mode')
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    if mode == 'flipud':
        l = np.rot90(l, -1)
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def get_post_process_disp(disps):
    pp_lr = post_process_disp(disps[0:2, :, :].squeeze(), 'fliplr')
    temp_disp = np.expand_dims(disps[2, :, :].squeeze(), axis=0)
    pp_lr_result = np.concatenate((np.expand_dims(pp_lr, axis=0), temp_disp), axis=0)
    return post_process_disp(pp_lr_result.squeeze(), 'flipud')

def generate_result(disp, scenename, print_str, res_output_path, name, avgscore):
    print_str += '--------------------{}--------------------\n'.format(name)
    error_img, error_score, print_str = get_scores_file_by_name(disp, scenename, print_str)
    write_pfm(disp, os.path.join(res_output_path, scenename + '_{}.pfm'.format(name)))
    save_singledisp(disp, res_output_path, scenename + '{}_'.format(name))
    save_singledisp_error(error_img, res_output_path, scenename + '{}_error_'.format(name))
    avgscore += error_score
    return print_str, avgscore


def eval_all_fliplrud(args, sess, global_step, mode):
    dataloader = Dataloader(args.data_path, args.val_txt_path, args, mode)
    model = MultiDepthModel(args, mode, dataloader.data_batch, reuse_variables=tf.AUTO_REUSE) # data_batch(oriimage, fliplr, flipud)
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    num_test_samples = count_text_lines(args.val_txt_path)

    print('now testing {} files'.format(num_test_samples))
    ori_disps = np.zeros((num_test_samples, args.input_height, args.input_width), dtype=np.float32)
    ori_disps_pp = np.zeros((num_test_samples, args.input_height, args.input_width), dtype=np.float32)
    refine_disps = np.zeros((num_test_samples, args.input_height, args.input_width), dtype=np.float32)
    refine_disps_pp = np.zeros((num_test_samples, args.input_height, args.input_width), dtype=np.float32)

    for step in range(num_test_samples):
        ori_disp, refine_disp = sess.run([model.ori_disp, model.refine_disp])
        ori_disps[step] = ori_disp[0].squeeze()
        refine_disps[step] = refine_disp[0].squeeze()

        if mode == 'test_fliplrud':
            ori_disps_pp[step] = get_post_process_disp(ori_disp)
            refine_disps_pp[step] = get_post_process_disp(refine_disp)
        else:
            raise Exception('wrong eval_all_fliplrud mode')
    print('writing disparities')
    np.save(os.path.join(args.output_path, 'ori_disps.npy'), ori_disps)
    np.save(os.path.join(args.output_path, 'ori_disps_.npy'), ori_disps_pp)
    np.save(os.path.join(args.output_path, 'refine_disps.npy'), refine_disps)
    np.save(os.path.join(args.output_path, 'refine_disps_pp.npy'), refine_disps_pp)

    ori_avgscore, ori_pp_avgscore, refine_avgscore, refine_pp_avgscore= 0.0, 0.0, 0.0, 0.0
    with open(os.path.join(args.output_path, 'result.txt'), 'a+') as result_file:
        with open(args.val_txt_path) as val_txt:
            print_str = '--------------------load checkpoint {}--------------------'.format(global_step) + '\n'
            print_str += 'mode: {}\n'.format(mode)
            for i in range(num_test_samples):
                val_txt_line = val_txt.readline().split('/')
                scenename = val_txt_line[2]
                print_str += 'origin result\n'
                res_output_path = os.path.join(args.output_path, str(global_step))
                if not os.path.exists(res_output_path):
                    os.mkdir(res_output_path)
                # cur_gtpath = os.path.join(args.gt_path, scenename, 'valid_disp_map.npy')
                # if not os.path.exists(cur_gtpath):
                #     raise Exception('wrong path of gt: {}'.format(cur_gtpath))
                # gt_image = np.load(cur_gtpath)

                print_str, refine_avgscore =\
                    generate_result(refine_disps[i, :, :], scenename, print_str, res_output_path, 'refine_disp', refine_avgscore)
                print_str, refine_pp_avgscore =\
                    generate_result(refine_disps_pp[i, :, :], scenename, print_str, res_output_path, 'refine_disp_pp', refine_pp_avgscore)
                print_str, ori_avgscore =\
                    generate_result(ori_disps[i, :, :], scenename, print_str, res_output_path, 'ori_disp', ori_avgscore)
                print_str, ori_pp_avgscore =\
                    generate_result(ori_disps_pp[i, :, :], scenename, print_str, res_output_path, 'ori_disp_pp', ori_pp_avgscore)
            print_str += '--------------------ori_disp avg score {}\n'.format(100 - ori_avgscore / 8)
            print_str += '--------------------ori_disp pp avg score {}\n'.format(100 - ori_pp_avgscore / 8)
            print_str += '--------------------refine disp avg score {}\n'.format(100 - refine_avgscore / 8)
            print_str += '--------------------refine disp pp avg score {}\n'.format(100 - refine_pp_avgscore / 8)
            print_str += '\n'
            print(print_str)
            result_file.write(print_str)
    return

def train(args):
    with tf.Graph().as_default(), tf.device('/gpu:0'):
        global_step = tf.Variable(0, trainable=False)
        num_train_samples = count_text_lines(args.train_txt_path)
        steps_per_epoch = np.ceil(num_train_samples / args.batch_size).astype(np.int32)
        num_total_steps = args.total_epoch * steps_per_epoch

        boundries = [(epoch * steps_per_epoch).astype(np.int32) for epoch in args.lr_decay_boundries]
        values = [args.learning_rate * value for value in args.lr_decay_values]
        learning_rate = tf.train.piecewise_constant(global_step, boundries, values)

        # optimizer
        opt = tf.train.AdamOptimizer(learning_rate)

        # loading_data
        mode = 'train'
        dataloader = Dataloader(args.data_path, args.train_txt_path, args, mode)

        # loss
        with tf.variable_scope(tf.get_variable_scope()):
            with tf.device('/gpu:0'):
                model = MultiDepthModel(args, mode, dataloader.data_batch, reuse_variables=tf.AUTO_REUSE)
                grads = opt.compute_gradients(model.total_loss)

        apply_grad_op = opt.apply_gradients(grads, global_step=global_step)
        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.scalar('total_loss', model.total_loss)
        summary_op = tf.summary.merge_all()

        # session
        config = tf.ConfigProto(allow_soft_placement=True)
        sess = tf.Session(config=config)

        # saver
        summary_writer = tf.summary.FileWriter(os.path.join(args.output_path, args.model_name), sess.graph)
        train_saver = tf.train.Saver(max_to_keep=30)

        # count params
        total_num_params = 0
        for var in tf.trainable_variables():
            total_num_params += np.array(var.get_shape().as_list()).prod()
        print('total num params: {}'.format(total_num_params))

        # init
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)
        # load checkpoint if set
        if args.checkpoint_path != '':
            load_param_only(sess, args.checkpoint_path)
            sess.run(global_step.assign(args.retrain_epoch * steps_per_epoch))

        # go
        for epoch in range(args.retrain_epoch, args.total_epoch):
            print(epoch)
            for step in range(steps_per_epoch):
                _, loss_value, cur_global_step = sess.run([apply_grad_op, model.total_loss, global_step])
                if step and step % 100 == 0:
                     print('epoch{:>6} | step {:>6} | global_step {:>6} | loss: {:.5f}'.format(epoch, step, cur_global_step, loss_value))
                     summary_str = sess.run(summary_op)
                     summary_writer.add_summary(summary_str, global_step=cur_global_step)
                # print('epoch{:>6} | step {:>6} | global_step {:>6} | loss: {:.5f}'.format(epoch, step, cur_global_step,loss_value))
                # summary_str = sess.run(summary_op)
                # summary_writer.add_summary(summary_str, global_step=cur_global_step)
            if epoch % 100 == 0:
            # if epoch == 0:
                train_saver.save(sess, os.path.join(args.output_path, args.model_name, 'model'), global_step=cur_global_step)
                eval_all_fliplrud(args, sess, cur_global_step, 'test_fliplrud')
    return

def test(args):
    mode = 'test_fliplrud' # now only support test_fliplrud mode
    dataloader = Dataloader(args.data_path, args.val_txt_path, args, mode)
    model = MultiDepthModel(args, mode, dataloader.data_batch, reuse_variables=tf.AUTO_REUSE) # data_batch(oriimage, fliplr, flipud)

    # session
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    # saver
    train_saver = tf.train.Saver()

    # init
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    # restore
    if args.checkpoint_path == '':
        raise Exception('please give args.checkpoint_path to load')
    train_saver.restore(sess, args.checkpoint_path)

    num_test_samples = count_text_lines(args.val_txt_path)

    print('now testing {} files'.format(num_test_samples))
    ori_disps = np.zeros((num_test_samples, args.input_height, args.input_width), dtype=np.float32)
    ori_disps_pp = np.zeros((num_test_samples, args.input_height, args.input_width), dtype=np.float32)
    refine_disps = np.zeros((num_test_samples, args.input_height, args.input_width), dtype=np.float32)
    refine_disps_pp = np.zeros((num_test_samples, args.input_height, args.input_width), dtype=np.float32)

    for step in range(num_test_samples):
        ori_disp, refine_disp = sess.run([model.ori_disp, model.refine_disp])
        ori_disps[step] = ori_disp[0].squeeze()
        refine_disps[step] = refine_disp[0].squeeze()

        if mode == 'test_fliplrud':
            ori_disps_pp[step] = get_post_process_disp(ori_disp)
            refine_disps_pp[step] = get_post_process_disp(refine_disp)
        else:
            raise Exception('wrong eval_all_fliplrud mode')
    print('writing disparities')
    np.save(os.path.join(args.output_path, 'ori_disps.npy'), ori_disps)
    np.save(os.path.join(args.output_path, 'ori_disps_.npy'), ori_disps_pp)
    np.save(os.path.join(args.output_path, 'refine_disps.npy'), refine_disps)
    np.save(os.path.join(args.output_path, 'refine_disps_pp.npy'), refine_disps_pp)

    ori_avgscore, ori_pp_avgscore, refine_avgscore, refine_pp_avgscore= 0.0, 0.0, 0.0, 0.0
    with open(os.path.join(args.output_path, 'result.txt'), 'a+') as result_file:
        with open(args.val_txt_path) as val_txt:
            print_str = '--------------------load checkpoint {}--------------------'.format(args.checkpoint_path) + '\n'
            print_str += 'mode: {}\n'.format(mode)
            for i in range(num_test_samples):
                val_txt_line = val_txt.readline().split('/')
                scenename = val_txt_line[2]
                print_str += 'origin result\n'
                res_output_path = os.path.join(args.output_path, 'test_res')
                if not os.path.exists(res_output_path):
                    os.mkdir(res_output_path)
                # cur_gtpath = os.path.join(args.gt_path, scenename, 'valid_disp_map.npy')
                # if not os.path.exists(cur_gtpath):
                #     raise Exception('wrong path of gt: {}'.format(cur_gtpath))
                # gt_image = np.load(cur_gtpath)

                print_str, refine_avgscore = \
                    generate_result(refine_disps[i, :, :], scenename, print_str, res_output_path, 'refine_disp', refine_avgscore)
                print_str, refine_pp_avgscore = \
                    generate_result(refine_disps_pp[i, :, :], scenename, print_str, res_output_path, 'refine_disp_pp', refine_pp_avgscore)
                print_str, ori_avgscore = \
                    generate_result(ori_disps[i, :, :], scenename, print_str, res_output_path, 'ori_disp', ori_avgscore)
                print_str, ori_pp_avgscore = \
                    generate_result(ori_disps_pp[i, :, :], scenename, print_str, res_output_path, 'ori_disp_pp', ori_pp_avgscore)
            print_str += '--------------------ori_disp avg score {}\n'.format(100 - ori_avgscore / 8)
            print_str += '--------------------ori_disp pp avg score {}\n'.format(100 - ori_pp_avgscore / 8)
            print_str += '--------------------refine disp avg score {}\n'.format(100 - refine_avgscore / 8)
            print_str += '--------------------refine disp pp avg score {}\n'.format(100 - refine_pp_avgscore / 8)
            print(print_str)
            result_file.write(print_str)
    return





def main(argv=None):
    args = get_args()
    train_mode = True
    if train_mode:
        train(args)
    else:
        test(args)
    return

if __name__ == '__main__':
    tf.app.run()


