'''
    Single-GPU training.
'''
import argparse
import math
from datetime import datetime
import h5py
import socket
import importlib
import os
import sys
import glob

import numpy as np
import tensorflow as tf
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
print('ROOT_DIR: ', ROOT_DIR)
sys.path.append(BASE_DIR) # model
sys.path.append(ROOT_DIR) # provider
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
import tf_util
import pc_util
import chair_clean_noisy_patch_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='p2p_translation', help='Model name [default: p2p_translation]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 12]')
parser.add_argument('--pool_size', type=int, default=50, help='Generated point cloud pool size [default: 12]')
parser.add_argument('--output_interval', type=int, default=50, help='Statistics output interval [default: 50]')
FLAGS = parser.parse_args()

EPOCH_CNT = 0

OUTPUT_INTERVAL = FLAGS.output_interval
BATCH_SIZE = FLAGS.batch_size
POOL_SIZE = FLAGS.pool_size
NUM_POINT = FLAGS.num_point
GPU_INDEX = FLAGS.gpu

MODEL = importlib.import_module(FLAGS.model) # import network module
#################### back up code for this run ##########################
MODEL_FILE = os.path.join(BASE_DIR, FLAGS.model+'.py')
LOG_DIR = os.path.join('run', 'log_' + FLAGS.model + '_' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
script_name = os.path.basename(__file__)
os.system('cp %s %s' % (script_name, LOG_DIR)) # bkp of train procedure
os.system('cp %s %s' % ('discriminator.py', LOG_DIR)) # bkp of D procedure
os.system('cp %s %s' % ('generator.py', LOG_DIR)) # bkp of G procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

HOSTNAME = socket.gethostname()

# Shapenet official train/test split
assert(NUM_POINT<=2048)
TRAIN_DATASET = chair_clean_noisy_patch_dataset.ChairSeparatedCleanNoisyDataset(data_dir='data/ShapeNet_v2_point_cloud/03001627', 
                                                                                split='train', 
                                                                                batch_size = BATCH_SIZE, 
                                                                                npoints = NUM_POINT, 
                                                                                shuffle=True)

IMG_OUTPUT_DIR = os.path.join(LOG_DIR, 'img')
PC_OUTPUT_DIR = os.path.join(LOG_DIR, 'pcloud')

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def clean_dir(dirname, name_match='*.jpg'):
    if not os.path.exists(dirname):
        return
    files = glob.glob(os.path.join(dirname, name_match))
    for pf in files:
        os.remove(pf)

def vis_point_cloud_preds(point_clouds, postfix_name='generated'):
    '''
    input:
        point_clouds: np array, BxNx3
    '''    
    if not os.path.exists(PC_OUTPUT_DIR):
        os.makedirs(PC_OUTPUT_DIR)
    clean_dir(PC_OUTPUT_DIR, name_match='*'+postfix_name+'*')
    for idx, pc in enumerate(point_clouds):
        pc_name = 'pc%d_%s.ply' % (idx, postfix_name)
        pc_output_filename = os.path.join(PC_OUTPUT_DIR, pc_name)
        pc_util.write_ply(pc, pc_output_filename)

def print_trainable_vars():
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    for tv in trainable_vars:
        print(tv.name)

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            cycle_gan = MODEL.CycleGAN3D(batch_size=1, 
                                         npoint=2048, 
                                         data_channel=3, 
                                         use_lsgan=True, 
                                         norm_method='instance', 
                                         lambda1=10, lambda2=10, 
                                         learning_rate=0.0002, 
                                         beta1=0.5)
            G_loss, D_Y_loss, F_loss, D_X_loss, \
            cycle_loss, D_Y_fake_loss, D_Y_real_loss, D_X_fake_loss, D_X_real_loss, \
            fake_y, fake_x = cycle_gan.model()
            optimizers = cycle_gan.optimize(G_loss, D_Y_loss, F_loss, D_X_loss)

            summary_op = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'summary'))
            saver = tf.train.Saver()

            train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            for var in train_vars:
                print(var.name)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)

        fake_X_pool = MODEL.PointCloudPool('fake_X_pool', POOL_SIZE)
        fake_Y_pool = MODEL.PointCloudPool('fake_Y_pool', POOL_SIZE)

        for i in range(1000000):
            
            # get previously generated images
            input_x_4_infer = TRAIN_DATASET.next_noisy_batch()
            input_y_4_infer = TRAIN_DATASET.next_clean_batch()
            fake_y_val, fake_x_val = sess.run([fake_y, fake_x], 
                                              feed_dict={cycle_gan.x: input_x_4_infer,
                                                         cycle_gan.y: input_y_4_infer})
            # train
            _, G_loss_val, D_Y_loss_val, F_loss_val, D_X_loss_val, cycle_loss_val, D_Y_fake_loss_val, D_Y_real_loss_val, D_X_fake_loss_val, D_X_real_loss_val, summary = (
                sess.run(
                    [optimizers, G_loss, D_Y_loss, F_loss, D_X_loss, cycle_loss, D_Y_fake_loss, D_Y_real_loss, D_X_fake_loss, D_X_real_loss, summary_op],
                    feed_dict={cycle_gan.fake_x_prev: fake_X_pool.query(fake_x_val, input_y_4_infer),
                               cycle_gan.fake_y_prev: fake_Y_pool.query(fake_y_val, input_x_4_infer),
                               cycle_gan.x: TRAIN_DATASET.next_noisy_batch(),
                               cycle_gan.y: TRAIN_DATASET.next_clean_batch()}
                )
            )

            if i % 100 == 0:
                log_string('-----------Step %d:-------------' % i)
                log_string('  G_loss   : {:.6f}'.format(G_loss_val))
                log_string('  D_Y_loss : {:.6f} = (f){:.6f} + (r){:.6f}'.format(D_Y_loss_val, D_Y_fake_loss_val, D_Y_real_loss_val))
                log_string('  F_loss   : {:.6f}'.format(F_loss_val))
                log_string('  D_X_loss : {:.6f} = (f){:.6f} + (r){:.6f}'.format(D_X_loss_val, D_X_fake_loss_val, D_X_real_loss_val))
                log_string('  cyc_loss : {:.6f}'.format(cycle_loss_val))

                train_writer.add_summary(summary, i)
                train_writer.flush()

            # output fake pool for check
            if i % 500 == 0:
                output_pc_dir_cur = os.path.join(PC_OUTPUT_DIR, 'step_'+str(i))
                log_string('Writing fake pools to disk: %s'%(output_pc_dir_cur))
                if not os.path.exists(output_pc_dir_cur):
                    os.makedirs(output_pc_dir_cur)
                fake_X_pool.dump2disk(output_pc_dir_cur)
                fake_Y_pool.dump2disk(output_pc_dir_cur)
                log_string('Done.')



if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
