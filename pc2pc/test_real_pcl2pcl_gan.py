'''
    Single-GPU training.
'''
import numpy as np

import math
from datetime import datetime
import socket
import os
import sys

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
from latent_gan import PCL2PCLGAN
import shapenet_pc_dataset
import config

cat_name = 'MP_table'
ckpt_idx = '630'

# paras for autoencoder
para_config_gan = {
    'exp_name': '%s_real_pcl2pcl_m%s'%(cat_name, ckpt_idx),
    'random_seed': None, # None for totally random

    'batch_size': 1, # important NOTE: batch size should be the same with that of competetor, otherwise, the randomness is not fixed!
    'lr': 0.0001,
    'beta1': 0.5,
    'epoch': 3001,
    'k': 1, # train k times for D each loop when training
    'kk': 1, # train k times for G each loop when training
    'output_interval': 1, # unit in epoch
    'save_interval': 10, # unit in epoch

    'loss': 'hausdorff',
    'lambda': 1.0, # parameter on back-reconstruction loss
    #'eval_loss': 'emd',
    'eval_loss': 'hausdorff',

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

if cat_name == 'scannet_chair':
    para_config_gan['real_point_cloud_dir'] = config.real_scannet_chair_aligned_data_dir

    para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pointnet2/pc2pc/run_real/run_scannet_chair/pcl2pcl/log_scannet_chair_real_pcl2pcl_gan_2019-03-16-20-38-43/ckpts/model_%s.ckpt'%(ckpt_idx)

elif cat_name == 'scannet_table':
    para_config_gan['real_point_cloud_dir'] = config.real_scannet_table_aligned_data_dir

    para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pointnet2/pc2pc/run_real/run_scannet_table/pcl2pcl/log_scannet_table_real_pcl2pcl_gan_2019-03-16-21-55-56/ckpts/model_%s.ckpt'%(ckpt_idx)

elif cat_name == 'MP_chair':
    para_config_gan['real_point_cloud_dir'] = config.real_MP_chair_aligned_data_dir

    para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pointnet2/pc2pc/run_real/run_scannet_chair/pcl2pcl/log_scannet_chair_real_pcl2pcl_gan_2019-03-16-20-38-43/ckpts/model_%s.ckpt'%(ckpt_idx)

elif cat_name == 'MP_table':
    para_config_gan['real_point_cloud_dir'] = config.real_MP_table_aligned_data_dir

    para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pointnet2/pc2pc/run_real/run_scannet_table/pcl2pcl/log_scannet_table_real_pcl2pcl_gan_2019-03-16-21-55-56/ckpts/model_%s.ckpt'%(ckpt_idx)

NOISY_TEST_DATASET = shapenet_pc_dataset.RealWorldPointsDataset(para_config_gan['real_point_cloud_dir'], batch_size=para_config_gan['batch_size'], npoint=para_config_gan['point_cloud_shape'][0], shuffle=False, split='test')

#################### dirs, code backup and etc for this run ##########################
LOG_DIR = os.path.join('test_real', 'test_%s'%(cat_name), 'pcl2pcl_test', 'log_test_' + para_config_gan['exp_name'] + '_' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)

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

def test():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(0)):
            latent_gan = PCL2PCLGAN(para_config_gan, para_config_ae)
            print_trainable_vars()
            _, _, _, _, _, _, fake_clean_reconstr, eval_loss = latent_gan.model()
            
            saver = tf.train.Saver(max_to_keep=None)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        
        with tf.Session(config=config) as sess:

            if para_config_gan['pcl2pcl_gan_ckpt'] is None or not 'pcl2pcl_gan_ckpt' in para_config_gan:
                print('Error, no check point is provided for test.')
                return
            saver.restore(sess, para_config_gan['pcl2pcl_gan_ckpt'])

            all_inputs = []
            all_recons = []
            all_eval_losses = []
            while NOISY_TEST_DATASET.has_next_batch():

                noise_cur = NOISY_TEST_DATASET.next_batch()
                clean_cur = noise_cur

                feed_dict={
                            latent_gan.input_noisy_cloud: noise_cur,
                            latent_gan.gt: clean_cur,
                            latent_gan.is_training: False,
                            }
                fake_clean_reconstr_val, eval_losses_val = sess.run([fake_clean_reconstr, eval_loss], feed_dict=feed_dict)

                all_inputs.extend(noise_cur)
                all_recons.extend(fake_clean_reconstr_val)
                all_eval_losses.append(eval_losses_val)

            NOISY_TEST_DATASET.reset()

            pc_util.write_ply_batch(np.asarray(all_inputs), os.path.join(LOG_DIR, 'pcloud', 'input'))
            pc_util.write_ply_batch(np.asarray(all_recons), os.path.join(LOG_DIR, 'pcloud', 'reconstruction'))
            eval_loss_mean = np.mean(all_eval_losses)
            print('(Useless) Eval loss (%s) on all data: %f'%(para_config_gan['eval_loss'], np.mean(all_eval_losses)))
            print(LOG_DIR)
            return eval_loss_mean
           
if __name__ == "__main__":

    '''
    #model_index = [100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000]
    model_index = [100,200,300,400,500,600,700,800,900]
    ckpt_str_template = '/workspace/pointnet2/pc2pc/run_pcl2pcl/log_pcl2pcl_gan_PN_lamda=1_2019-02-12-19-34-37/ckpts/model_%s.ckpt'
    eval_losses = {}
    for midx in model_index:
        para_config_gan['pcl2pcl_gan_ckpt'] = ckpt_str_template%(str(midx))
        print('Evaluating for %s'%(para_config_gan['pcl2pcl_gan_ckpt']))

        eval_loss = test()
        eval_losses[midx] = eval_loss
    
    for k in sorted(eval_losses.keys()):
        print(k, eval_losses[k])
    '''
    test()

    LOG_FOUT.close()
