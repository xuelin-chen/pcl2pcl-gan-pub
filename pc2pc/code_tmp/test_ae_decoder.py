'''
    Single-GPU training.
'''
import math
from datetime import datetime
import socket
import os
import sys
import pickle

import numpy as np
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
import shapenet_pc_dataset
import autoencoder
from latent_gan import LatentCodeDataset

para_config = {
    'exp_name': 'test_ae_decoder_emd_chair_2048_fromL2LGAN400',
    'ckpt': '/workspace/pointnet2/pc2pc/run/log_ae_emd_chair_2048_good/ckpts/model_960.ckpt' ,
    'ltcode_pkl_filename': '/workspace/pointnet2/pc2pc/run/log_latent2latent_gan_test_model400/gened_fake_codes.pickle',

    ########################## the paras below should be exactly the same with those of training #######################################

    'batch_size': 50,
    'lr': 0.0005,
    'epoch': 1000,
    'output_interval': 1, # unit in batch
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

latent_code_dataset = LatentCodeDataset(para_config['ltcode_pkl_filename'], batch_size=para_config['batch_size'])

#################### back up code for this run ##########################
LOG_DIR = os.path.join('run', 'log_' + para_config['exp_name'] + '_' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
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

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def test_decoder():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(0)):
            ae = autoencoder.AutoEncoder(paras=para_config)

            reconstr_loss, reconstr, latent_code = ae.model()

            saver = tf.train.Saver()

        # print
        log_string('Net layers:')
        log_string(str(ae))

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        if para_config['ckpt'] is None or not 'ckpt' in para_config:
            print('Error, no check point is provided for test.')
            return
        saver.restore(sess, para_config['ckpt'])

        # test on whole dataset
        all_reconstr_clouds_test = []
        while latent_code_dataset.has_next_batch():
            lcodes = latent_code_dataset.next_batch()
            reconstr_test = sess.run(reconstr, feed_dict={ae.latent_code: lcodes, ae.is_training: False})
        
            all_reconstr_clouds_test.extend(reconstr_test)
        
        # write out ply point clouds
        pc_util.write_ply_batch(np.asarray(all_reconstr_clouds_test), os.path.join(LOG_DIR, 'pcloud_from_latent_code'))

        log_string('Test %d codes done.'%(len(all_reconstr_clouds_test)))
        

if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    test_decoder()
    LOG_FOUT.close()
