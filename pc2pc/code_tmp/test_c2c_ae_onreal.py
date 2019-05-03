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
import shapenet_pc_dataset
import autoencoder
import config

cat_name = 'MP_table'

para_config = {
    'exp_name': '%s_ae_test'%(cat_name),
    'random_seed': 0, # None for totally random
    'ae_type': 'real2c', # 'c2c', 'n2n', 'np2np', 'np2c'

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

if cat_name == 'scannet_chair':
    para_config['test_point_cloud_dir'] = config.real_scannet_chair_aligned_data_dir
    para_config['ckpt'] = config.AE_chair_c2c_ShapeNetV2_ckpt # it is fine to use chair ae trained using shapenet-v2 data
elif cat_name == 'MP_chair':
    para_config['test_point_cloud_dir'] = config.real_MP_chair_aligned_data_dir
    para_config['ckpt'] = config.AE_chair_c2c_ShapeNetV2_ckpt # it is fine to use chair ae trained using shapenet-v2 data

elif cat_name == 'scannet_table':
    para_config['test_point_cloud_dir'] = config.real_scannet_table_aligned_data_dir
    para_config['ckpt'] = config.AE_table_c2c_ShapeNetV2_ckpt # it is fine to use chair ae trained using shapenet-v2 data
elif cat_name == 'MP_table':
    para_config['test_point_cloud_dir'] = config.real_MP_table_aligned_data_dir
    para_config['ckpt'] = config.AE_table_c2c_ShapeNetV2_ckpt # it is fine to use chair ae trained using shapenet-v2 data
'''
elif cat_name == 'car':
    para_config['test_point_cloud_dir'] = os.path.join(config.EPN_dataset_test_dir, '02958343/point_cloud')
    para_config['ckpt'] = config.AE_car_c2c_ShapeNetV1_ckpt # using ae trained with SN-v1
'''
#################### back up code for this run ##########################
LOG_DIR = os.path.join('test_real', 'test_%s'%(cat_name), 'clean_ae_test', 'log_test_' + para_config['exp_name'] +'_'+ para_config['ae_type'] +'_' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)

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

TEST_DATASET = shapenet_pc_dataset.RealWorldPointsDataset(para_config['test_point_cloud_dir'], batch_size=para_config['batch_size'], npoint=para_config['point_cloud_shape'][0], shuffle=False, split='test')

SCAN_PC_DIR = config.ShapeNet_v1_point_cloud_dir

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def test():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(0)):
            ae = autoencoder.AutoEncoder(paras=para_config)

            _, reconstr, latent_code = ae.model()

            saver = tf.train.Saver()

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
            TEST_DATASET.reset()
            all_reconstr_clouds_test = []
            all_input_clouds_test = []
            all_latent_codes_test = []
            eval_loss_mean_test = []
            all_name = []
            while TEST_DATASET.has_next_batch():

                input_batch_test = TEST_DATASET.next_batch()
                gt_batch_test = input_batch_test
                
                latent_code_val_test, reconstr_val_test, eval_loss_val_test = sess.run([latent_code, reconstr, ae.eval_loss],
                                                                feed_dict={
                                                                            ae.input_pl: input_batch_test, 
                                                                            ae.gt: gt_batch_test,
                                                                            ae.is_training: False
                                                                        }
                                                                )

                all_reconstr_clouds_test.extend(reconstr_val_test)
                all_input_clouds_test.extend(input_batch_test)
                all_latent_codes_test.extend(latent_code_val_test)
                eval_loss_mean_test.append(eval_loss_val_test)
            
            eval_loss_mean_test = np.mean(eval_loss_mean_test)
            
            # write out
            pc_util.write_ply_batch(np.asarray(all_reconstr_clouds_test), os.path.join(LOG_DIR, 'pcloud', 'reconstruction'))
            pc_util.write_ply_batch(np.asarray(all_input_clouds_test), os.path.join(LOG_DIR, 'pcloud', 'input'))
            latent_pickle_file = open(os.path.join(LOG_DIR, 'latent_codes.pickle'), 'wb')
            pickle.dump(np.asarray(all_latent_codes_test), latent_pickle_file)
            latent_pickle_file.close()

            log_string('--------- on test split --------')
            log_string('Mean eval loss ({}): {:.6f}'.format(para_config['loss'],eval_loss_mean_test))
            print(LOG_DIR)

             
if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    test()
    LOG_FOUT.close()
