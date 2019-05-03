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

cat_name = 'table'
pts_remove_percentage = 0.5
partial_portion = 1.0

# paras for autoencoder
para_config_gan = {
    'exp_name': '%s_pcl2pcl_gan_%d-percentage_redo'%(cat_name, pts_remove_percentage*100),
    'random_seed': 0, # None for totally random

    'extra_point_clouds_list': None,

    'batch_size': 1, # important NOTE: batch size should be the same with that of competetor, otherwise, the randomness is not fixed!
    'lr': 0.0001,
    'beta1': 0.5,
    'epoch': 3001,
    'k': 1, # train k times for D each loop when training
    'kk': 1, # train k times for G each loop when training
    'output_interval': 1, # unit in epoch
    'save_interval': 10, # unit in epoch

    'loss': 'emd',
    #'loss': 'hausdorff',
    'lambda': 1.0, # parameter on back-reconstruction loss
    'eval_loss': 'emd',
    #'eval_loss': 'hausdorff',
    #'eval_loss': 'chamfer',

    # noise parameters
    'noise_mu': 0.0, 
    'noise_sigma': 0.01, 
    #'r_min': 0.1, 
    #'r_max': 0.25, 
    'p_min': pts_remove_percentage,
    'p_max': pts_remove_percentage,
    'partial_portion': partial_portion, # 0.25 by default in training

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

if cat_name == 'chair':
    para_config_gan['point_cloud_dir'] = '/workspace/pointnet2/pc2pc/data/ShapeNet_v2_point_cloud/03001627/point_cloud_clean'
    #para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pointnet2/pc2pc/run_chair/pcl2pcl/log_chair_pcl2pcl_gan_percentage_hausdorff_2019-03-02-19-19-18/ckpts/model_410.ckpt'
    para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pointnet2/pc2pc/run_chair/pcl2pcl/log_chair_pcl2pcl_gan_percentage_emd_2019-03-02-19-21-15/ckpts/model_1320.ckpt'

elif cat_name == 'table':
    para_config_gan['point_cloud_dir'] = '/workspace/pointnet2/pc2pc/data/ShapeNet_v2_point_cloud/04379243/point_cloud_clean'
    #para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pointnet2/pc2pc/run_table/pcl2pcl/log_table_pcl2pcl_gan_percentage_hausdorff_2019-03-02-19-41-36/ckpts/model_300.ckpt'
    para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pointnet2/pc2pc/run_table/pcl2pcl/log_table_pcl2pcl_gan_percentage-redo_hausdorff_2019-03-07-20-52-12/ckpts/model_290.ckpt'

elif cat_name == 'motorbike':
    para_config_gan['point_cloud_dir'] = '/workspace/pointnet2/pc2pc/data/ShapeNet_v2_point_cloud/03790512/point_cloud_clean'
    para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pointnet2/pc2pc/run_motorbike/pcl2pcl/log_motorbike_pcl2pcl_gan_percentage_hausdorff_2019-03-04-20-54-10/ckpts/model_870.ckpt'

elif cat_name == 'plane':
    para_config_gan['point_cloud_dir'] = '/workspace/pointnet2/pc2pc/data/ShapeNet_v2_point_cloud/02691156/point_cloud_clean'
    para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pointnet2/pc2pc/run_plane/pcl2pcl/log_plane_pcl2pcl_gan_percentage_hausdorff_2019-03-05-10-55-54/ckpts/model_980.ckpt'

elif cat_name == 'car':
    para_config_gan['point_cloud_dir'] = '/workspace/pointnet2/pc2pc/data/ShapeNet_v2_point_cloud/02958343/point_cloud_clean'
    para_config_gan['pcl2pcl_gan_ckpt'] = '/workspace/pointnet2/pc2pc/run_car/pcl2pcl/log_car_pcl2pcl_gan_percentage_hausdorff_2019-03-07-12-06-59/ckpts/model_710.ckpt'

NOISY_TEST_DATASET = shapenet_pc_dataset.ShapeNetPartPointsDataset(para_config_gan['point_cloud_dir'], batch_size=para_config_gan['batch_size'], npoint=para_config_gan['point_cloud_shape'][0], shuffle=False, split='test', extra_ply_point_clouds_list=para_config_gan['extra_point_clouds_list'], random_seed=para_config_gan['random_seed'], preprocess=False)

#################### dirs, code backup and etc for this run ##########################
model_name = para_config_gan['pcl2pcl_gan_ckpt'].split('/')[-1].split('.')[0]
LOG_DIR = os.path.join('run_%s'%(cat_name), 'pcl2pcl_test', 'log_test_' + para_config_gan['exp_name'] + '_' + model_name + '_' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
print(LOG_DIR)
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

        # print
        log_string('Net layers:')
        log_string(str(latent_gan))

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
            all_gt = []
            all_eval_losses = []
            while NOISY_TEST_DATASET.has_next_batch():

                noise_cur, clean_cur = NOISY_TEST_DATASET.next_batch_noise_partial_by_percentage(noise_mu=para_config_gan['noise_mu'], noise_sigma=para_config_gan['noise_sigma'], p_min=para_config_gan['p_min'], p_max=para_config_gan['p_max'], partial_portion=para_config_gan['partial_portion'], with_gt=True)

                feed_dict={
                            latent_gan.input_noisy_cloud: noise_cur,
                            latent_gan.gt: clean_cur,
                            latent_gan.is_training: False,
                            }
                fake_clean_reconstr_val, eval_losses_val = sess.run([fake_clean_reconstr, eval_loss], feed_dict=feed_dict)

                all_inputs.extend(noise_cur)
                all_recons.extend(fake_clean_reconstr_val)
                all_gt.extend(clean_cur)
                all_eval_losses.append(eval_losses_val)

            NOISY_TEST_DATASET.reset()

            pc_util.write_ply_batch(np.asarray(all_inputs), os.path.join(LOG_DIR, 'pcloud', 'input'))
            pc_util.write_ply_batch(np.asarray(all_gt), os.path.join(LOG_DIR, 'pcloud', 'gt'))
            pc_util.write_ply_batch(np.asarray(all_recons), os.path.join(LOG_DIR, 'pcloud', 'reconstruction'))
            eval_loss_mean = np.mean(all_eval_losses)
            print('Eval loss (%s) on all data: %f'%(para_config_gan['eval_loss'], np.mean(all_eval_losses)))
            return eval_loss_mean
           
if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
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
