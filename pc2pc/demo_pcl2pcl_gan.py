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
from shapenet_pc_dataset import DemoPointCloudDataset

# paras for autoencoder
para_config_gan = {
    'exp_name': 'pcl2pcl_gan_Nfeed',

    'point_cloud_dir': '/workspace/pointnet2/pc2pc/data/demo_data/N_feed',

    'pcl2pcl_gan_ckpt': '/workspace/pointnet2/pc2pc/run_pcl2pcl/log_pcl2pcl_gan_np2c_dualAE_2019-02-16-15-52-24/ckpts/model_1990.ckpt',

    'batch_size': 1,
    'loss': 'emd',
    'lambda': 1.0, # parameter on back-reconstruction loss
    'eval_loss': 'emd',

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

DEMO_DATASET = DemoPointCloudDataset(para_config_gan['point_cloud_dir'], npoint=para_config_gan['point_cloud_shape'][0], batch_size=para_config_gan['batch_size'])

#################### dirs, code backup and etc for this run ##########################
LOG_DIR = os.path.join('run_demo', para_config_gan['exp_name'] + '_' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
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

def test():  
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(0)):
            latent_gan = PCL2PCLGAN(para_config_gan, para_config_ae)
            print_trainable_vars()
            _, _, _, _, _, _, fake_clean_reconstr, _ = latent_gan.model()
            
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
            while DEMO_DATASET.has_next_batch():

                data_batch = DEMO_DATASET.next_batch()

                feed_dict={
                            latent_gan.input_noisy_cloud: data_batch,
                            latent_gan.is_training: False,
                            }
                fake_clean_reconstr_val = sess.run(fake_clean_reconstr, feed_dict=feed_dict)

                all_inputs.extend(data_batch)
                all_recons.extend(fake_clean_reconstr_val)

            DEMO_DATASET.reset()

            pc_util.write_ply_batch(np.asarray(all_inputs), os.path.join(LOG_DIR, 'pcloud', 'input'))
            pc_util.write_ply_batch(np.asarray(all_recons), os.path.join(LOG_DIR, 'pcloud', 'reconstruction'))
           
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
