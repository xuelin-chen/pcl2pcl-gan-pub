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
import tf_util
import pc_util
from latent_gan import PCL2PCLGAN
import shapenet_pc_dataset
import config

cat_name = 'plane'
loss = 'hausdorff'

para_config_gan = {
    'exp_name': '%s_pcl2pcl_gan_3D-EPN_gt-retrieved_from-continue'%(cat_name),
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

    'loss': loss,
    'lambda': 1.0, # parameter on back-reconstruction loss
    'eval_loss': loss,

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

if cat_name == 'chair':
    para_config_gan['3D-EPN_train_point_cloud_dir'] = '/workspace/pointnet2/pc2pc/data/3D-EPN_dataset/shapenet_dim32_sdf_pc/03001627/point_cloud'
    para_config_gan['3D-EPN_test_point_cloud_dir'] = '/workspace/pointnet2/pc2pc/data/3D-EPN_dataset/test-images_dim32_sdf_pc/03001627/point_cloud'

    para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pointnet2/pc2pc/run_3D-EPN/run_chair/pcl2pcl/log_chair_pcl2pcl_gan_3D-EPN_hausdorff_2019-03-07-20-33-29/ckpts/model_230.ckpt'

elif cat_name == 'table':
    para_config_gan['3D-EPN_train_point_cloud_dir'] = '/workspace/pointnet2/pc2pc/data/3D-EPN_dataset/shapenet_dim32_sdf_pc/04379243/point_cloud'
    para_config_gan['3D-EPN_test_point_cloud_dir'] = '/workspace/pointnet2/pc2pc/data/3D-EPN_dataset/test-images_dim32_sdf_pc/04379243/point_cloud'

    para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pointnet2/pc2pc/run_3D-EPN/run_table/pcl2pcl/log_table_pcl2pcl_gan_3D-EPN_hausdorff_2019-03-07-20-42-50/ckpts/model_420.ckpt'

elif cat_name == 'plane':
    para_config_gan['3D-EPN_train_point_cloud_dir'] = '/workspace/pointnet2/pc2pc/data/3D-EPN_dataset/shapenet_dim32_sdf_pc/02691156/point_cloud'
    para_config_gan['3D-EPN_test_point_cloud_dir'] = '/workspace/pointnet2/pc2pc/data/3D-EPN_dataset/test-images_dim32_sdf_pc/02691156/point_cloud'

    #para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pointnet2/pc2pc/run_3D-EPN/run_plane/pcl2pcl/log_plane_pcl2pcl_gan_3D-EPN_hausdorff_2019-03-07-14-14-18/ckpts/model_720.ckpt'
    para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pointnet2/pc2pc/run_3D-EPN/run_plane/pcl2pcl/log_plane_pcl2pcl_gan_3D-EPN-continure_hausdorff_2019-03-10-22-15-02/ckpts/model_720.ckpt' # NOTE: using ckpt from continue training

elif cat_name == 'car':
    para_config_gan['3D-EPN_train_point_cloud_dir'] = '/workspace/pointnet2/pc2pc/data/3D-EPN_dataset/shapenet_dim32_sdf_pc/02958343/point_cloud'
    para_config_gan['3D-EPN_test_point_cloud_dir'] = '/workspace/pointnet2/pc2pc/data/3D-EPN_dataset/test-images_dim32_sdf_pc/02958343/point_cloud'

    para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pointnet2/pc2pc/run_3D-EPN/run_car/pcl2pcl/log_car_pcl2pcl_gan_3D-EPN_hausdorff_2019-03-07-19-59-14/ckpts/model_710.ckpt'

NOISY_TEST_DATASET = shapenet_pc_dataset.ShapeNet_3DEPN_PointsDataset(para_config_gan['3D-EPN_test_point_cloud_dir'], batch_size=para_config_gan['batch_size'], npoint=para_config_gan['point_cloud_shape'][0], shuffle=False, split='all', preprocess=False)

SCAN_PC_DIR = config.ShapeNet_v1_point_cloud_dir

def prepare4test():
    #################### dirs, code backup and etc for this run ##########################
    model_name = para_config_gan['pcl2pcl_gan_ckpt'].split('/')[-1].split('.')[0]
    #para_config_gan['LOG_DIR'] = os.path.join('run_3D-EPN', 'run_%s'%(cat_name), 'pcl2pcl_test', 'log_test_' + para_config_gan['exp_name'] + '_' + model_name + '_' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    para_config_gan['LOG_DIR'] = os.path.join('run_3D-EPN', 'run_%s'%(cat_name), 'pcl2pcl_test', 'all_models_ShapeNetV1-GT', 'log_test_' + para_config_gan['exp_name'] + '_' + model_name + '_' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
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

    for model_idx in range(0, 1230, 10):
        model_ckpt_filename = os.path.join(model_dir, 'model_%d.ckpt'%(model_idx))
        para_config_gan['pcl2pcl_gan_ckpt'] = model_ckpt_filename

        test()

