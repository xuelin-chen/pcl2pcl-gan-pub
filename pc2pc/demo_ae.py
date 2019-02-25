'''
    Single-GPU training.
'''
import numpy as np

import math
from datetime import datetime
import socket
import os
import sys
import pickle
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
print('ROOT_DIR: ', ROOT_DIR)
sys.path.append(BASE_DIR) # model
sys.path.append(ROOT_DIR) # provider
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
import tf_util
import pc_util
from shapenet_pc_dataset import DemoPointCloudDataset
import autoencoder

para_config = {
    'exp_name': 'ae_rotation_test_r=0.25',
    'random_seed': None, # None for totally random

    'point_cloud_dir': '/workspace/pointnet2/pc2pc/data/demo_data/for_rotation_test_r=0.5',

    'ckpt': '/workspace/pointnet2/pc2pc/run_ae/log_ae_chair_np2np_2019-02-15-17-03-41/ckpts/model_1850.ckpt' ,

    ########################## the paras below should be exactly the same with those of training #######################################

    'batch_size': 1, # important NOTE: batch size should be the same with that of competetor, otherwise, the randomness is not fixed!
    'lr': 0.0005,
    'epoch': 2000,
    'output_interval': 10, # unit in batch
    'test_interval': 10, # unit in epoch
    
    'loss': 'emd',

    # encoder
    'latent_code_dim': 128,
    'n_filters': [64,128,128,256],
    'filter_size': 1,
    'stride': 1,
    'encoder_bn': True,

    # decoder
    'point_cloud_shape': [2048, 3],
    'fc_sizes': [256, 256], 
    'decoder_bn': False,

    'activation_fn': tf.nn.relu,    
}

#################### back up code for this run ##########################
LOG_DIR = os.path.join('run_demo', para_config['exp_name'] + '_' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)

script_name = os.path.basename(__file__)
bk_filenames = ['autoencoder.py', 
                 script_name, 
                 'shapenet_pc_dataset.py', 
                 'pointnet_utils/pointnet_encoder_decoder.py']
for bf in bk_filenames:
    os.system('cp %s %s' % (bf, LOG_DIR))
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(para_config)+'\n')

HOSTNAME = socket.gethostname()

DEMO_DATASET = DemoPointCloudDataset(para_config['point_cloud_dir'], npoint=para_config['point_cloud_shape'][0], batch_size=para_config['batch_size'])

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def test():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(0)):
            ae = autoencoder.AutoEncoder(paras=para_config)

            _, reconstr, _ = ae.model()

            saver = tf.train.Saver()

        # print
        log_string('Net layers:')
        log_string(str(ae))

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False

        with tf.Session(config=config) as sess:
            if para_config['ckpt'] is None or not 'ckpt' in para_config:
                print('Error, no check point is provided for test.')
                return
            saver.restore(sess, para_config['ckpt'])

            # test on whole dataset
            DEMO_DATASET.reset()
            all_reconstr_clouds_test = []
            all_input_clouds_test = []
            while DEMO_DATASET.has_next_batch():

                input_batch_test = DEMO_DATASET.next_batch_rotated()
                
                reconstr_val_test = sess.run(reconstr,
                                                                feed_dict={
                                                                            ae.input_pl: input_batch_test, 
                                                                            ae.is_training: False
                                                                        }
                                                                )

                all_reconstr_clouds_test.extend(reconstr_val_test)
                all_input_clouds_test.extend(input_batch_test)
            
            # write out
            pc_util.write_ply_batch(np.asarray(all_reconstr_clouds_test), os.path.join(LOG_DIR, 'pcloud', 'reconstruction'))
            pc_util.write_ply_batch(np.asarray(all_input_clouds_test), os.path.join(LOG_DIR, 'pcloud', 'input'))


             
if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    test()
    LOG_FOUT.close()
