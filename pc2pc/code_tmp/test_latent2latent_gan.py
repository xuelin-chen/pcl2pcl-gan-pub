'''
    Single-GPU training.
'''
import math
from datetime import datetime
import socket
import os
import sys

import numpy as np
import tensorflow as tf
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
print('ROOT_DIR: ', ROOT_DIR)
sys.path.append(BASE_DIR) # model
sys.path.append(ROOT_DIR) # provider
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
import tf_util
import pc_util
from latent_gan import LatentGAN, LatentCodeDataset

# paras for autoencoder on all chairs
para_config = {
    'exp_name': 'latent2latent_gan_test_model400',
    'A_ltcode_pkl_filename': '/workspace/pointnet2/pc2pc/run/log_ae_emd_chair_2048_noisy_test_good/latent_codes.pickle',
    'ckpt': '/workspace/pointnet2/pc2pc/run/log_debug_latent2latent_gan_2019-01-23-14-37-50/ckpts/model_400.ckpt',

    'batch_size': 50,
    'lr': 0.0001,
    'beta1': 0.5,
    'epoch': 2001,
    'k': 1, # train k times for D each loop when training
    'kk': 1, # train k times for G each loop when training
    'output_interval': 10, # unit in epoch
    'shuffle_lcode_dataset': True,

    'noise_mu': 0.0,
    'noise_sigma': 0.2,
    'noise_dim': 128,
    'latent_dim': 128,

    # G paras
    'g_fc_sizes': [128],
    'g_activation_fn': tf.nn.relu,
    'g_bn': False,

    #D paras
    'd_fc_sizes': [256, 512],
    'd_activation_fn': tf.nn.leaky_relu,
    'd_bn': False,
}

A_latent_code_dataset = LatentCodeDataset(para_config['A_ltcode_pkl_filename'],
                                             batch_size=para_config['batch_size'],
                                             shuffle=para_config['shuffle_lcode_dataset'])

#################### dir, back up code and etc for this run ##########################
LOG_DIR = os.path.join('run', 'log_' + para_config['exp_name'] + '_' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
if not os.path.exists(os.path.join(LOG_DIR, 'fake_codes')): os.mkdir(os.path.join(LOG_DIR, 'fake_codes'))

script_name = os.path.basename(__file__)
bk_filenames = ['latent_gan.py', 
                 script_name,  
                 'latent_generator_discriminator.py']
for bf in bk_filenames:
    os.system('cp %s %s' % (bf, LOG_DIR))
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(para_config)+'\n')

HOSTNAME = socket.gethostname()
#############################################################################

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def print_trainable_vars():
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    for tv in trainable_vars:
        print(tv.name)

def test():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(0)):
            latent_gan = LatentGAN(para_config)
            print_trainable_vars()
            _, _, _, _, fake_code = latent_gan.model()

            saver = tf.train.Saver()

        # print
        log_string('Net layers:')
        log_string(str(latent_gan))

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

        all_gened_fake_codes = []
        while A_latent_code_dataset.has_next_batch():
            noise_cur = A_latent_code_dataset.next_batch()
            fake_codes_val = sess.run(fake_code, 
                                      feed_dict={
                                                    latent_gan.noise_pl: noise_cur, 
                                                    latent_gan.is_training: False
                                                })
            all_gened_fake_codes.extend(fake_codes_val)

        with open(os.path.join(LOG_DIR, 'gened_fake_codes.pickle'), 'wb') as pf:
            pickle.dump(np.asarray(all_gened_fake_codes), pf)
        
        log_string('Test %d codes done.'%(len(all_gened_fake_codes)))
            
if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    test()
    LOG_FOUT.close()
