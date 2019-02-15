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

para_config = {
    'exp_name': 'ae_chair_2048_denoise_test900',
    'point_cloud_dir': '/workspace/pointnet2/pc2pc/data/ShapeNet_v2_point_cloud/03001627/point_cloud_clean',
    'ckpt': '/workspace/pointnet2/pc2pc/run1/log_ae_chair_2048_denoise_good/ckpts/model_900.ckpt' ,

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

#################### back up code for this run ##########################
LOG_DIR = os.path.join('run1', 'log_' + para_config['exp_name'] + '_' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
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

TRAIN_DATASET = shapenet_pc_dataset.ShapeNetPartPointsDataset(para_config['point_cloud_dir'], batch_size=para_config['batch_size'], npoint=para_config['point_cloud_shape'][0], shuffle=False, split='all')
TEST_DATASET = TRAIN_DATASET

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def test():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(0)):
            ae = autoencoder.AutoEncoderDenoise(paras=para_config)

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
        TEST_DATASET.reset()
        all_reconstr_clouds_test = []
        all_input_clouds_test = []
        all_latent_codes_test = []
        reconstr_loss_mean_test = []
        while TEST_DATASET.has_next_batch():
            #input_batch_test = TEST_DATASET.next_batch_noise_added()
            noisy_batch_test, clean_batch_test = TEST_DATASET.next_batch_noisy_clean_pair()
            latent_code_val_test, reconstr_val_test, reconstr_loss_val_test = sess.run([latent_code, reconstr, reconstr_loss],
                                                            feed_dict={
                                                                        ae.input_pl: noisy_batch_test, 
                                                                                ae.gt: clean_batch_test,
                                                                                ae.is_training: False
                                                                      }
                                                            )

            if para_config['loss'] == 'emd':
                reconstr_loss_val_test = reconstr_loss_val_test / TEST_DATASET.get_npoint()
            all_reconstr_clouds_test.extend(reconstr_val_test)
            all_input_clouds_test.extend(noisy_batch_test)
            all_latent_codes_test.extend(latent_code_val_test)
            reconstr_loss_mean_test.append(reconstr_loss_val_test)
        
        reconstr_loss_mean_test = np.mean(reconstr_loss_mean_test)
        
        # write out
        pc_util.write_ply_batch(np.asarray(all_reconstr_clouds_test), os.path.join(LOG_DIR, 'pcloud', 'reconstruction'))
        pc_util.write_ply_batch(np.asarray(all_input_clouds_test), os.path.join(LOG_DIR, 'pcloud', 'input'))
        latent_pickle_file = open(os.path.join(LOG_DIR, 'latent_codes.pickle'), 'wb')
        pickle.dump(np.asarray(all_latent_codes_test), latent_pickle_file)
        latent_pickle_file.close()

        log_string('--------- test on whole dataset --------')
        log_string('Mean Reconstruction loss: {:.6f}'.format(reconstr_loss_mean_test))
             
if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    test()
    LOG_FOUT.close()
