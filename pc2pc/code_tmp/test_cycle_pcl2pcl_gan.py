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
from latent_gan import CyclePCL2PCLGAN
import shapenet_pc_dataset

# paras for autoencoder
para_config_gan = {
    'exp_name': 'cycle_pcl2pcl_gan_PN_test',
    'point_cloud_dir': '/workspace/pointnet2/pc2pc/data/ShapeNet_v2_point_cloud/03001627/point_cloud_clean',
    'noisy_ae_ckpt': '/workspace/pointnet2/pc2pc/run2/log_ae_chair_2k_noisy-partial_2019-01-30-19-21-35/ckpts/model_1800.ckpt',
    'clean_ae_ckpt': '/workspace/pointnet2/pc2pc/run/log_ae_emd_chair_2048_good/ckpts/model_960.ckpt',
    'cycle_pcl2pcl_gan_ckpt': '/workspace/pointnet2/pc2pc/run2/log_cycle_pcl2pcl_gan_PN_2019-01-31-13-49-28/ckpts/model_300.ckpt',

    'batch_size': 50,
    'lr': 0.0001,
    'beta1': 0.5,
    'epoch': 3001,
    'k': 1, # train k times for D each loop when training
    'kk': 1, # train k times for G each loop when training
    'output_interval': 1, # unit in epoch
    'save_interval': 10, # unit in epoch
    'shuffle_dataset': True,
    'loss': 'emd',

    'latent_dim': 128,
    'point_cloud_shape': [2048, 3],

    # G paras
    'g_fc_sizes': [128],
    'g_activation_fn': tf.nn.relu,
    'g_bn': False,

    #D paras
    'd_fc_sizes': [256, 512],
    'd_activation_fn': tf.nn.leaky_relu,
    'd_bn': False,
}

para_config_ae = {
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


NOISY_TEST_DATASET = shapenet_pc_dataset.ShapeNetPartPointsDataset(para_config_gan['point_cloud_dir'], batch_size=para_config_gan['batch_size'], npoint=para_config_gan['point_cloud_shape'][0], shuffle=False, split='all')

#################### dirs, code backup and etc for this run ##########################
LOG_DIR = os.path.join('run2', 'log_' + para_config_gan['exp_name'] + '_' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)

script_name = os.path.basename(__file__)
bk_filenames = ['latent_gan.py', 
                 script_name,  
                 'latent_generator_discriminator.py']
for bf in bk_filenames:
    os.system('cp %s %s' % (bf, LOG_DIR))
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(para_config_gan)+'\n')
LOG_FOUT.write(str(para_config_ae)+'\n')

HOSTNAME = socket.gethostname()
##########################################################################

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def print_trainable_vars():
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    for tv in trainable_vars:
        print(tv.name)

def get_restore_dict(stored_vars, current_vars):
    res = {}
    for v in current_vars:
        v_name_stored = v.name[6:-2]
        res[v_name_stored] = v

    return res

def test():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(0)):
            gan = CyclePCL2PCLGAN(para_config_gan, para_config_ae)
            print_trainable_vars()
            _, _, _, _, _, _, _, _, _, _, fake_clean_cloud, eval_loss = gan.model()

            saver = tf.train.Saver(max_to_keep=None)

        # print
        log_string('Net layers:')
        log_string(str(gan))

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        
        with tf.Session(config=config) as sess:

            if para_config_gan['cycle_pcl2pcl_gan_ckpt'] is None or not 'cycle_pcl2pcl_gan_ckpt' in para_config_gan:
                print('Error, no check point is provided for test.')
                return
            saver.restore(sess, para_config_gan['cycle_pcl2pcl_gan_ckpt'])

            all_inputs = []
            all_recons = []
            all_eval_losses = []
            while NOISY_TEST_DATASET.has_next_batch():
                noise_cur, clean_cur = NOISY_TEST_DATASET.next_batch_noise_added_with_partial(with_gt=True)
                feed_dict={
                            gan.input_noisy_cloud: noise_cur,
                            gan.gt: clean_cur,
                            gan.is_training: False,
                            }
                fake_clean_cloud_val, eval_loss_val = sess.run([fake_clean_cloud, eval_loss], feed_dict=feed_dict)

                all_inputs.extend(noise_cur)
                all_recons.extend(fake_clean_cloud_val)
                all_eval_losses.append(eval_loss_val)

            NOISY_TEST_DATASET.reset()

            pc_util.write_ply_batch(np.asarray(all_inputs), os.path.join(LOG_DIR, 'pcloud', 'input_noisys'))
            pc_util.write_ply_batch(np.asarray(all_recons), os.path.join(LOG_DIR, 'pcloud', 'fake_cleans'))
            eval_loss_mean = np.mean(all_eval_losses)
            print('Eval loss on all data: %f'%(np.mean(all_eval_losses)))
            return eval_loss_mean
            


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    '''
    model_index = [100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000]
    ckpt_str_template = '/workspace/pointnet2/pc2pc/run2/log_cycle_pcl2pcl_gan_PN_2019-01-31-13-49-28/ckpts/model_%s.ckpt'
    eval_losses = {}
    for midx in model_index:
        para_config_gan['cycle_pcl2pcl_gan_ckpt'] = ckpt_str_template%(str(midx))
        print('Evaluating for %s'%(para_config_gan['cycle_pcl2pcl_gan_ckpt']))

        eval_loss = test()
        eval_losses[midx] = eval_loss
    
    print(eval_losses)
    '''

    test()
    
    LOG_FOUT.close()
