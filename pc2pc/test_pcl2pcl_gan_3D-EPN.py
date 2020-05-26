'''
    Single-GPU training.
'''
import numpy as np
import argparse
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
import tf_util
import pc_util
from latent_gan import PCL2PCLGAN
import shapenet_pc_dataset
import config

parser = argparse.ArgumentParser()
parser.add_argument('--cat_name', default='chair', help='category name for training')
parser.add_argument('--split_name', default='test', help='split name for inferring')
parser.add_argument('--pcl2pcl_mode', default='sharedAE', help='[sharedAE | separateAE | withoutGAN | withoutRecon | EMD | GT]')
FLAGS = parser.parse_args()

split_name = FLAGS.split_name
pcl2pcl_mode = FLAGS.pcl2pcl_mode
cat_name = FLAGS.cat_name

loss = 'hausdorff'

para_config_gan = {
    'exp_name': '%s_pcl2pcl_gan_3D-EPN'%(cat_name),
    'random_seed': 0,

    'recover_ckpt': None,

    'batch_size': 1,
    'lr': 0.0001,
    'beta1': 0.5,
    'epoch': 3001,
    'k': 1, # train k times for D each loop when training
    'kk': 1, # train k times for G each loop when training
    'output_interval': 1, # unit in epoch
    'save_interval': 10, # unit in epoch

        # parameters on loss
    'l_alpha': 0.25, # weight on gan loss of G
    'l_beta': 0.75, # weight on reconstruction loss of G
    'loss': loss,
    'eval_loss': loss,

    # parameters on networks
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

    'LOG_DIR': '',
}
# paras for autoencoder
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

if FLAGS.pcl2pcl_mode is not None:
    if FLAGS.pcl2pcl_mode == 'withoutGAN':
        para_config_gan['l_alpha'] = 0.
        para_config_gan['l_beta'] = 1.
    elif FLAGS.pcl2pcl_mode == 'withoutRecon':
        para_config_gan['l_alpha'] = 1.
        para_config_gan['l_beta'] = 0.
    elif FLAGS.pcl2pcl_mode == 'EMD':
        para_config_gan['l_alpha'] = 0.25
        para_config_gan['l_beta'] = 0.75
    elif FLAGS.pcl2pcl_mode == 'sharedAE' or FLAGS.pcl2pcl_mode == 'separateAE':
        para_config_gan['l_alpha'] = 0.25
        para_config_gan['l_beta'] = 0.75
    else:
        raise NotImplementedError('Pcl2pcl mode %s not implemented!'%(FLAGS.pcl2pcl_mode))
else:
    para_config_gan['l_alpha'] = 0.25
    para_config_gan['l_beta'] = 0.75

if cat_name == 'chair':
    para_config_gan['3D-EPN_test_point_cloud_dir'] = config.EPN_chair_point_cloud_dir

    if pcl2pcl_mode == 'sharedAE':
        model_idx = 1000
        para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pcl2pcl-gan/pc2pc/run/run_3D-EPN_pcl2pcl/run_chair/log_chair_pcl2pcl_gan_3D-EPN_hausdorff_sharedAE_2019-07-26-13-18-41/ckpts/model_%s.ckpt'%(model_idx)
    elif pcl2pcl_mode == 'separateAE':
        model_idx = 1000
        para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pcl2pcl-gan/pc2pc/run/run_3D-EPN_pcl2pcl/run_chair/log_chair_pcl2pcl_gan_3D-EPN_hausdorff_separateAE_2019-07-30-11-54-02/ckpts/model_%s.ckpt'%(model_idx)
    elif pcl2pcl_mode == 'withoutGAN':
        model_idx = 1000
        para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pcl2pcl-gan/pc2pc/run/run_3D-EPN_pcl2pcl_withoutGAN/run_chair/log_chair_pcl2pcl_gan_3D-EPN_hausdorff_sharedAE_2019-08-01-09-47-36/ckpts/model_%s.ckpt'%(model_idx)
    elif pcl2pcl_mode == 'withoutRecon':
        model_idx = 1000
        para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pcl2pcl-gan/pc2pc/run/run_3D-EPN_pcl2pcl_withoutRecon/run_chair/log_chair_pcl2pcl_gan_3D-EPN_hausdorff_sharedAE_2019-08-04-16-25-19/ckpts/model_%s.ckpt'%(model_idx)
    elif pcl2pcl_mode == 'EMD':
        model_idx = 1000
        para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pcl2pcl-gan/pc2pc/run/run_3D-EPN_pcl2pcl_EMD/run_chair/log_chair_pcl2pcl_gan_3D-EPN_emd_sharedAE_2019-08-05-08-57-22/ckpts/model_%s.ckpt'%(model_idx)

elif cat_name == 'table':
    para_config_gan['3D-EPN_test_point_cloud_dir'] = config.EPN_table_point_cloud_dir

    if pcl2pcl_mode == 'sharedAE':
        model_idx = 1000
        para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pcl2pcl-gan/pc2pc/run/run_3D-EPN_pcl2pcl/run_table/log_table_pcl2pcl_gan_3D-EPN_hausdorff_sharedAE_2019-07-26-13-22-18/ckpts/model_%s.ckpt'%(model_idx)
    elif pcl2pcl_mode == 'separateAE':
        model_idx = 1000
        para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pcl2pcl-gan/pc2pc/run/run_3D-EPN_pcl2pcl/run_table/log_table_pcl2pcl_gan_3D-EPN_hausdorff_separateAE_2019-07-29-20-37-05/ckpts/model_%s.ckpt'%(model_idx)
    elif pcl2pcl_mode == 'withoutGAN':
        model_idx = 1000
        para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pcl2pcl-gan/pc2pc/run/run_3D-EPN_pcl2pcl_withoutGAN/run_table/log_table_pcl2pcl_gan_3D-EPN_hausdorff_sharedAE_2019-08-01-09-35-09/ckpts/model_%s.ckpt'%(model_idx)
    elif pcl2pcl_mode == 'withoutRecon':
        model_idx = 1000
        para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pcl2pcl-gan/pc2pc/run/run_3D-EPN_pcl2pcl_withoutRecon/run_table/log_table_pcl2pcl_gan_3D-EPN_hausdorff_sharedAE_2019-08-04-16-25-02/ckpts/model_%s.ckpt'%(model_idx)
    elif pcl2pcl_mode == 'EMD':
        model_idx = 1000
        para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pcl2pcl-gan/pc2pc/run/run_3D-EPN_pcl2pcl_EMD/run_table/log_table_pcl2pcl_gan_3D-EPN_emd_sharedAE_2019-08-05-08-29-24/ckpts/model_%s.ckpt'%(model_idx)

elif cat_name == 'plane':
    para_config_gan['3D-EPN_test_point_cloud_dir'] = config.EPN_plane_point_cloud_dir

    if pcl2pcl_mode == 'sharedAE':
        model_idx = 1000
        para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pcl2pcl-gan/pc2pc/run/run_3D-EPN_pcl2pcl/run_plane/log_plane_pcl2pcl_gan_3D-EPN_hausdorff_sharedAE_2019-07-26-13-23-10/ckpts/model_%s.ckpt'%(model_idx)
    elif pcl2pcl_mode == 'separateAE':
        model_idx = 1000
        para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pcl2pcl-gan/pc2pc/run/run_3D-EPN_pcl2pcl/run_plane/log_plane_pcl2pcl_gan_3D-EPN_hausdorff_separateAE_2019-07-30-11-51-26/ckpts/model_%s.ckpt'%(model_idx)
    elif pcl2pcl_mode == 'withoutGAN':
        model_idx = 1000
        para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pcl2pcl-gan/pc2pc/run/run_3D-EPN_pcl2pcl_withoutGAN/run_plane/log_plane_pcl2pcl_gan_3D-EPN_hausdorff_sharedAE_2019-08-02-11-11-58/ckpts/model_%s.ckpt'%(model_idx)
    elif pcl2pcl_mode == 'withoutRecon':
        model_idx = 1000
        para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pcl2pcl-gan/pc2pc/run/run_3D-EPN_pcl2pcl_withoutRecon/run_plane/log_plane_pcl2pcl_gan_3D-EPN_hausdorff_sharedAE_2019-08-04-16-22-03/ckpts/model_%s.ckpt'%(model_idx)
    elif pcl2pcl_mode == 'EMD':
        model_idx = 1000
        para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pcl2pcl-gan/pc2pc/run/run_3D-EPN_pcl2pcl_EMD/run_plane/log_plane_pcl2pcl_gan_3D-EPN_emd_sharedAE_2019-08-05-08-55-27/ckpts/model_%s.ckpt'%(model_idx)

elif cat_name == 'car':
    para_config_gan['3D-EPN_test_point_cloud_dir'] = config.EPN_car_point_cloud_dir

    if pcl2pcl_mode == 'sharedAE':
        model_idx = 1000
        #para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pcl2pcl-gan/pc2pc/run/run_3D-EPN_pcl2pcl/run_car/log_car_pcl2pcl_gan_3D-EPN_hausdorff_sharedAE_2019-07-26-13-26-42/ckpts/model_820.ckpt'
        para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pcl2pcl-gan/pc2pc/run/run_3D-EPN_pcl2pcl/run_car/log_car_pcl2pcl_gan_3D-EPN_hausdorff_sharedAE_2019-07-26-13-26-42/ckpts/model_%s.ckpt'%(model_idx)
    elif pcl2pcl_mode == 'separateAE':
        model_idx = 1000
        para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pcl2pcl-gan/pc2pc/run/run_3D-EPN_pcl2pcl/run_car/log_car_pcl2pcl_gan_3D-EPN_hausdorff_separateAE_2019-07-30-11-54-27/ckpts/model_%s.ckpt'%(model_idx)
    elif pcl2pcl_mode == 'withoutGAN':
        model_idx = 1000
        para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pcl2pcl-gan/pc2pc/run/run_3D-EPN_pcl2pcl_withoutGAN/run_car/log_car_pcl2pcl_gan_3D-EPN_hausdorff_sharedAE_2019-08-01-09-44-32/ckpts/model_%s.ckpt'%(model_idx)
    elif pcl2pcl_mode == 'withoutRecon':
        model_idx = 1000
        para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pcl2pcl-gan/pc2pc/run/run_3D-EPN_pcl2pcl_withoutRecon/run_car/log_car_pcl2pcl_gan_3D-EPN_hausdorff_sharedAE_2019-08-04-16-25-54/ckpts/model_%s.ckpt'%(model_idx)
    elif pcl2pcl_mode == 'EMD':
        model_idx = 1000
        para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pcl2pcl-gan/pc2pc/run/run_3D-EPN_pcl2pcl_EMD/run_car/log_car_pcl2pcl_gan_3D-EPN_emd_sharedAE_2019-08-05-08-58-19/ckpts/model_%s.ckpt'%(model_idx)

elif cat_name == 'lamp':
    para_config_gan['3D-EPN_test_point_cloud_dir'] = config.EPN_lamp_point_cloud_dir

    if pcl2pcl_mode == 'sharedAE':
        model_idx = 1000
        para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pcl2pcl-gan/pc2pc/run/run_3D-EPN_pcl2pcl/run_lamp/log_lamp_pcl2pcl_gan_3D-EPN_hausdorff_sharedAE_2019-07-28-13-55-06/ckpts/model_%s.ckpt'%(model_idx)
    elif pcl2pcl_mode == 'separateAE':
        model_idx = 1000
        para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pcl2pcl-gan/pc2pc/run/run_3D-EPN_pcl2pcl/run_lamp/log_lamp_pcl2pcl_gan_3D-EPN_hausdorff_separateAE_2019-07-29-12-12-48/ckpts/model_%s.ckpt'%(model_idx)
    elif pcl2pcl_mode == 'withoutGAN':
        model_idx = 1000
        para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pcl2pcl-gan/pc2pc/run/run_3D-EPN_pcl2pcl_withoutGAN/run_lamp/log_lamp_pcl2pcl_gan_3D-EPN_hausdorff_sharedAE_2019-08-03-16-29-46/ckpts/model_%s.ckpt'%(model_idx)
    elif pcl2pcl_mode == 'withoutRecon':
        model_idx = 1000
        para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pcl2pcl-gan/pc2pc/run/run_3D-EPN_pcl2pcl_withoutRecon/run_lamp/log_lamp_pcl2pcl_gan_3D-EPN_hausdorff_sharedAE_2019-08-05-08-22-26/ckpts/model_%s.ckpt'%(model_idx)
    elif pcl2pcl_mode == 'EMD':
        model_idx = 1000
        para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pcl2pcl-gan/pc2pc/run/run_3D-EPN_pcl2pcl_EMD/run_lamp/log_lamp_pcl2pcl_gan_3D-EPN_emd_sharedAE_2019-08-05-09-26-50/ckpts/model_%s.ckpt'%(model_idx)

elif cat_name == 'sofa':
    para_config_gan['3D-EPN_test_point_cloud_dir'] = config.EPN_sofa_point_cloud_dir

    if pcl2pcl_mode == 'sharedAE':
        model_idx = 1000
        para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pcl2pcl-gan/pc2pc/run/run_3D-EPN_pcl2pcl/run_sofa/log_sofa_pcl2pcl_gan_3D-EPN_hausdorff_sharedAE_2019-07-27-15-52-20/ckpts/model_%s.ckpt'%(model_idx)
    elif pcl2pcl_mode == 'separateAE':
        model_idx = 1000
        para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pcl2pcl-gan/pc2pc/run/run_3D-EPN_pcl2pcl/run_sofa/log_sofa_pcl2pcl_gan_3D-EPN_hausdorff_separateAE_2019-07-28-18-54-48/ckpts/model_%s.ckpt'%(model_idx)
    elif pcl2pcl_mode == 'withoutGAN':
        model_idx = 1000
        para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pcl2pcl-gan/pc2pc/run/run_3D-EPN_pcl2pcl_withoutGAN/run_sofa/log_sofa_pcl2pcl_gan_3D-EPN_hausdorff_sharedAE_2019-08-03-16-29-09/ckpts/model_%s.ckpt'%(model_idx)
    elif pcl2pcl_mode == 'withoutRecon':
        model_idx = 1000
        para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pcl2pcl-gan/pc2pc/run/run_3D-EPN_pcl2pcl_withoutRecon/run_sofa/log_sofa_pcl2pcl_gan_3D-EPN_hausdorff_sharedAE_2019-08-04-19-13-11/ckpts/model_%s.ckpt'%(model_idx)
    elif pcl2pcl_mode == 'EMD':
        model_idx = 1000
        para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pcl2pcl-gan/pc2pc/run/run_3D-EPN_pcl2pcl_EMD/run_sofa/log_sofa_pcl2pcl_gan_3D-EPN_emd_sharedAE_2019-08-05-09-01-17/ckpts/model_%s.ckpt'%(model_idx)

elif cat_name == 'boat':
    para_config_gan['3D-EPN_test_point_cloud_dir'] = config.EPN_boat_point_cloud_dir

    if pcl2pcl_mode == 'sharedAE':
        model_idx = 1000
        #para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pcl2pcl-gan/pc2pc/run/run_3D-EPN_pcl2pcl/run_boat/log_boat_pcl2pcl_gan_3D-EPN_hausdorff_sharedAE_2019-07-28-13-57-53/ckpts/model_1000.ckpt'
        para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pcl2pcl-gan/pc2pc/run/run_3D-EPN_pcl2pcl/run_boat/log_boat_pcl2pcl_gan_3D-EPN_hausdorff_sharedAE_2019-08-27-21-47-12/ckpts/model_%s.ckpt'%(model_idx)
    elif pcl2pcl_mode == 'separateAE':
        model_idx = 1000
        para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pcl2pcl-gan/pc2pc/run/run_3D-EPN_pcl2pcl/run_boat/log_boat_pcl2pcl_gan_3D-EPN_hausdorff_separateAE_2019-07-29-12-13-37/ckpts/model_%s.ckpt'%(model_idx)
    elif pcl2pcl_mode == 'withoutGAN':
        model_idx = 1000
        para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pcl2pcl-gan/pc2pc/run/run_3D-EPN_pcl2pcl_withoutGAN/run_boat/log_boat_pcl2pcl_gan_3D-EPN_hausdorff_sharedAE_2019-08-03-18-50-30/ckpts/model_%s.ckpt'%(model_idx)
    elif pcl2pcl_mode == 'withoutRecon':
        model_idx = 1000
        para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pcl2pcl-gan/pc2pc/run/run_3D-EPN_pcl2pcl_withoutRecon/run_boat/log_boat_pcl2pcl_gan_3D-EPN_hausdorff_sharedAE_2019-08-05-08-23-08/ckpts/model_%s.ckpt'%(model_idx)
    elif pcl2pcl_mode == 'EMD':
        model_idx = 1000
        para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pcl2pcl-gan/pc2pc/run/run_3D-EPN_pcl2pcl_EMD/run_boat/log_boat_pcl2pcl_gan_3D-EPN_emd_sharedAE_2019-08-05-09-35-20/ckpts/model_%s.ckpt'%(model_idx)

elif cat_name == 'dresser':
    para_config_gan['3D-EPN_test_point_cloud_dir'] = config.EPN_dresser_point_cloud_dir

    if pcl2pcl_mode == 'sharedAE':
        model_idx = 1000
        para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pcl2pcl-gan/pc2pc/run/run_3D-EPN_pcl2pcl/run_dresser/log_dresser_pcl2pcl_gan_3D-EPN_hausdorff_sharedAE_2019-07-28-14-01-35/ckpts/model_%s.ckpt'%(model_idx)
    elif pcl2pcl_mode == 'separateAE':
        model_idx = 1000
        para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pcl2pcl-gan/pc2pc/run/run_3D-EPN_pcl2pcl/run_dresser/log_dresser_pcl2pcl_gan_3D-EPN_hausdorff_separateAE_2019-07-29-12-14-36/ckpts/model_%s.ckpt'%(model_idx)
    elif pcl2pcl_mode == 'withoutGAN':
        model_idx = 1000
        para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pcl2pcl-gan/pc2pc/run/run_3D-EPN_pcl2pcl_withoutGAN/run_dresser/log_dresser_pcl2pcl_gan_3D-EPN_hausdorff_sharedAE_2019-08-03-22-14-44/ckpts/model_%s.ckpt'%(model_idx)
    elif pcl2pcl_mode == 'withoutRecon':
        model_idx = 1000
        para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pcl2pcl-gan/pc2pc/run/run_3D-EPN_pcl2pcl_withoutRecon/run_dresser/log_dresser_pcl2pcl_gan_3D-EPN_hausdorff_sharedAE_2019-08-05-08-23-55/ckpts/model_%s.ckpt'%(model_idx)
    elif pcl2pcl_mode == 'EMD':
        model_idx = 1000
        para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pcl2pcl-gan/pc2pc/run/run_3D-EPN_pcl2pcl_EMD/run_dresser/log_dresser_pcl2pcl_gan_3D-EPN_emd_sharedAE_2019-08-05-09-37-28/ckpts/model_%s.ckpt'%(model_idx)
    elif pcl2pcl_mode == 'GT':
        model_idx = 1000
        para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pcl2pcl-gan/pc2pc/run/run_3D-EPN_pcl2pcl_GT/run_dresser/log_dresser_pcl2pcl_GT_3D-EPN_emd_sharedAE_2019-08-06-18-42-35/ckpts/model_%s.ckpt'%(model_idx)

if not os.path.exists(os.path.dirname(para_config_gan['pcl2pcl_gan_ckpt'])):
    print('pcl2pcl_gan_ckpt not exist! %s'%(os.path.dirname(para_config_gan['pcl2pcl_gan_ckpt'])))
    exit()

NOISY_TEST_DATASET = shapenet_pc_dataset.ShapeNet_3DEPN_PointsDataset(para_config_gan['3D-EPN_test_point_cloud_dir'], batch_size=para_config_gan['batch_size'], npoint=para_config_gan['point_cloud_shape'][0], shuffle=False, split=split_name, preprocess=False)

SCAN_PC_DIR = config.ShapeNet_v1_point_cloud_dir

def prepare4test():
    #################### dirs, code backup and etc for this run ##########################
    model_name = para_config_gan['pcl2pcl_gan_ckpt'].split('/')[-1].split('.')[0]
    #para_config_gan['LOG_DIR'] = os.path.join('results', 'test_3D-EPN_pcl2pcl_'+pcl2pcl_mode, 'test_%s'%(cat_name), split_name + '_' + para_config_gan['exp_name'] + '_' + model_name + '_' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    para_config_gan['LOG_DIR'] = os.path.join('results', 'test_3D-EPN_pcl2pcl_'+pcl2pcl_mode, split_name + '_' + para_config_gan['exp_name'] + '_' + model_name + '_' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    print(para_config_gan['LOG_DIR'])
    if not os.path.exists(para_config_gan['LOG_DIR']): os.makedirs(para_config_gan['LOG_DIR'])

    script_name = os.path.basename(__file__)
    bk_filenames = ['latent_gan.py', 
                    script_name,  
                    'latent_generator_discriminator.py']
    for bf in bk_filenames:
        os.system('cp %s %s' % (bf, para_config_gan['LOG_DIR']))
##########################################################################

def print_trainable_vars():
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    for tv in trainable_vars:
        print(tv.name)

def get_gt_point_clouds(cls_name, model_names, sample_nb=2048):
    cls_id = shapenet_pc_dataset.get_cls_id(cls_name)
    pc_dir = os.path.join(SCAN_PC_DIR, cls_id, 'point_cloud_clean')
    pc_arr = []
    for mn in model_names:
        mn = mn.split('_')[0]
        gt_pc_filename = os.path.join(pc_dir, mn+'_clean.ply')

        if not os.path.exists(gt_pc_filename):
            print('GT points not found: %s'%(gt_pc_filename))
            pc_arr.append(np.zeros((sample_nb, 3)))
            continue

        gt_pc = pc_util.read_ply_xyz(gt_pc_filename)
        if 'v1' in SCAN_PC_DIR:
            # for v1 data, rotate it to align with v2
            gt_pc = pc_util.rotate_point_cloud_by_axis_angle(gt_pc, [0,1,0], 90)
        gt_pc = pc_util.sample_point_cloud(gt_pc, sample_nb)
        pc_arr.append(gt_pc)
    pc_arr = np.array(pc_arr)
    return pc_arr

def test():
    prepare4test()
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(0)):
            latent_gan = PCL2PCLGAN(para_config_gan, para_config_ae)
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
            all_name = []
            all_recons = []
            all_gt = []
            all_eval_losses = []
            while NOISY_TEST_DATASET.has_next_batch():

                noise_cur, name_cur = NOISY_TEST_DATASET.next_batch_with_name()
                clean_cur = noise_cur

                feed_dict={
                            latent_gan.input_noisy_cloud: noise_cur,
                            latent_gan.gt: clean_cur,
                            latent_gan.is_training: False,
                            }
                fake_clean_reconstr_val, eval_losses_val = sess.run([fake_clean_reconstr, eval_loss], feed_dict=feed_dict)

                all_inputs.extend(noise_cur)
                all_name.extend(name_cur)
                all_recons.extend(fake_clean_reconstr_val)
                all_gt.extend(clean_cur)
                all_eval_losses.append(eval_losses_val)

            NOISY_TEST_DATASET.reset()

            all_gt = get_gt_point_clouds(cat_name, all_name)
            pc_util.write_ply_batch_with_name(np.asarray(all_inputs), all_name, os.path.join(para_config_gan['LOG_DIR'], 'pcloud', 'input'))
            pc_util.write_ply_batch_with_name(np.asarray(all_gt), all_name, os.path.join(para_config_gan['LOG_DIR'], 'pcloud', 'gt'))
            pc_util.write_ply_batch_with_name(np.asarray(all_recons), all_name, os.path.join(para_config_gan['LOG_DIR'], 'pcloud', 'reconstruction'))
            eval_loss_mean = np.mean(all_eval_losses)
            print('Eval loss (%s) on all data: %f'%(para_config_gan['eval_loss'], np.mean(all_eval_losses)))
            return eval_loss_mean
           
if __name__ == "__main__":

    model_dir = os.path.dirname(para_config_gan['pcl2pcl_gan_ckpt'])

    if para_config_gan['pcl2pcl_gan_ckpt'][-6] == '?':
        for model_idx in range(500, 1001, 10):
            model_ckpt_filename = os.path.join(model_dir, 'model_%d.ckpt'%(model_idx))
            para_config_gan['pcl2pcl_gan_ckpt'] = model_ckpt_filename
            test()
    else:
        test()

