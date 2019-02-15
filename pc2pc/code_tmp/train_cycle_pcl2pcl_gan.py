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
    'exp_name': 'cycle_pcl2pcl_gan_PN',
    'point_cloud_dir': '/workspace/pointnet2/pc2pc/data/ShapeNet_v2_point_cloud/03001627/point_cloud_clean',
    'noisy_ae_ckpt': '/workspace/pointnet2/pc2pc/run2/log_ae_chair_2k_noisy-partial_2019-01-30-19-21-35/ckpts/model_1800.ckpt',
    'clean_ae_ckpt': '/workspace/pointnet2/pc2pc/run/log_ae_emd_chair_2048_good/ckpts/model_960.ckpt',

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


NOISY_TRAIN_DATASET = shapenet_pc_dataset.ShapeNetPartPointsDataset(para_config_gan['point_cloud_dir'], batch_size=para_config_gan['batch_size'], npoint=para_config_gan['point_cloud_shape'][0], shuffle=True, split='all')
CLEAN_TRAIN_DATASET = shapenet_pc_dataset.ShapeNetPartPointsDataset(para_config_gan['point_cloud_dir'], batch_size=para_config_gan['batch_size'], npoint=para_config_gan['point_cloud_shape'][0], shuffle=True, split='all')

#################### dirs, code backup and etc for this run ##########################
LOG_DIR = os.path.join('run2', 'log_' + para_config_gan['exp_name'] + '_' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
if not os.path.exists(os.path.join(LOG_DIR, 'fake_cleans')): os.mkdir(os.path.join(LOG_DIR, 'fake_cleans'))

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

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(0)):
            gan = CyclePCL2PCLGAN(para_config_gan, para_config_ae)
            print_trainable_vars()
            cycle_loss, G_loss, F_loss, D_N_loss, D_N_fake_loss, D_N_real_loss, D_C_loss, D_C_fake_loss, D_C_real_loss, fake_noisy_cloud, fake_clean_cloud, eval_loss = gan.model()
            G_optimizer, F_optimizer, D_N_optimizer, D_C_optimizer = gan.optimize(G_loss, F_loss, D_N_loss, D_C_loss)

            # metrics for tensorboard visualization
            with tf.name_scope('metrics'):
                cycle_loss_mean_op, cycle_loss_mean_update_op = tf.metrics.mean(cycle_loss)

                G_loss_mean_op, G_loss_mean_update_op = tf.metrics.mean(G_loss)

                F_loss_mean_op, F_loss_mean_update_op = tf.metrics.mean(F_loss)

                D_N_loss_mean_op, D_N_loss_mean_update_op = tf.metrics.mean(D_N_loss)
                D_N_fake_loss_mean_op, D_N_fake_loss_mean_update_op = tf.metrics.mean(D_N_fake_loss)
                D_N_real_loss_mean_op, D_N_real_loss_mean_update_op = tf.metrics.mean(D_N_real_loss)

                D_C_loss_mean_op, D_C_loss_mean_update_op = tf.metrics.mean(D_C_loss)
                D_C_fake_loss_mean_op, D_C_fake_loss_mean_update_op = tf.metrics.mean(D_C_fake_loss)
                D_C_real_loss_mean_op, D_C_real_loss_mean_update_op = tf.metrics.mean(D_C_real_loss)            
            reset_metrics = tf.variables_initializer([var for var in tf.local_variables()
                                                 if var.name.split('/')[0] == 'metrics'])

            tf.summary.scalar('loss/G/G_loss', G_loss_mean_op, collections=['train'])          
            tf.summary.scalar('loss/F/F_loss', F_loss_mean_op, collections=['train'])
            tf.summary.scalar('loss/D/D_N_loss', D_N_loss_mean_op, collections=['train'])
            tf.summary.scalar('loss/D/D_C_loss', D_C_loss_mean_op, collections=['train'])
            tf.summary.scalar('loss/cycle_loss', cycle_loss_mean_op, collections=['train'])

            summary_op = tf.summary.merge_all('train')
            summary_eval_op = tf.summary.merge_all('eval')
            train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'summary', 'train'))
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
            # Init variables
            init = tf.global_variables_initializer()
            sess.run(init)

            # NOTE: load pre-trained AE weights
            # noisy AE, only pre-trained encoder is used
            noisy_ckpt_vars = tf.contrib.framework.list_variables(para_config_gan['noisy_ae_ckpt'])
            print('Noisy AE pre-trained variables:')
            for vname, _ in noisy_ckpt_vars:
                print(vname)
            restore_dict = get_restore_dict(noisy_ckpt_vars, gan.noisy_encoder.all_variables)
            noisy_saver = tf.train.Saver(restore_dict)
            noisy_saver.restore(sess, para_config_gan['noisy_ae_ckpt'])
            restore_dict = get_restore_dict(noisy_ckpt_vars, gan.noisy_decoder.all_variables)
            noisy_saver = tf.train.Saver(restore_dict)
            noisy_saver.restore(sess, para_config_gan['noisy_ae_ckpt'])
            # clean AE, both pre-trained encoder and decoder are used
            clean_ckpt_vars = tf.contrib.framework.list_variables(para_config_gan['clean_ae_ckpt'])
            print('Clean AE pre-trained variables:')
            for vname, _ in clean_ckpt_vars:
                print(vname)
            restore_dict = get_restore_dict(clean_ckpt_vars, gan.clean_encoder.all_variables)
            clean_saver = tf.train.Saver(restore_dict)
            clean_saver.restore(sess, para_config_gan['clean_ae_ckpt'])
            restore_dict = get_restore_dict(clean_ckpt_vars, gan.clean_decoder.all_variables)
            clean_saver = tf.train.Saver(restore_dict)
            clean_saver.restore(sess, para_config_gan['clean_ae_ckpt'])
            print('Loading pre-trained noisy/clean AE done.')
            # END of weights loading

            for i in range(para_config_gan['epoch']):
                sess.run(reset_metrics)

                while NOISY_TRAIN_DATASET.has_next_batch() and CLEAN_TRAIN_DATASET.has_next_batch():
                    noise_cur = NOISY_TRAIN_DATASET.next_batch_noise_added_with_partial()
                    clean_cur = CLEAN_TRAIN_DATASET.next_batch()
                    feed_dict={
                            gan.input_noisy_cloud: noise_cur,
                            gan.input_clean_cloud: clean_cur,
                            gan.is_training: True,
                            }
                    
                    sess.run([G_optimizer, F_optimizer, D_N_optimizer, D_C_optimizer, 
                              cycle_loss_mean_update_op, 
                              G_loss_mean_update_op, 
                              F_loss_mean_update_op, 
                              D_N_loss_mean_update_op, D_N_fake_loss_mean_update_op, D_N_real_loss_mean_update_op, 
                              D_C_loss_mean_update_op, D_C_fake_loss_mean_update_op, D_C_real_loss_mean_update_op], feed_dict=feed_dict)

                NOISY_TRAIN_DATASET.reset()
                CLEAN_TRAIN_DATASET.reset()

                if i % para_config_gan['output_interval'] == 0:
                    cycle_loss_mean_val, \
                    G_loss_mean_val, \
                    F_loss_mean_val, \
                    D_N_loss_mean_val, D_N_fake_loss_mean_val, D_N_real_loss_mean_val, \
                    D_C_loss_mean_val, D_C_fake_loss_mean_val, D_C_real_loss_mean_val, \
                    fake_noisy_cloud_val, fake_clean_cloud_val, \
                    summary = \
                    sess.run([cycle_loss_mean_op, 
                              G_loss_mean_op, 
                              F_loss_mean_op, 
                              D_N_loss_mean_op, D_N_fake_loss_mean_op, D_N_real_loss_mean_op, 
                              D_C_loss_mean_op, D_C_fake_loss_mean_op, D_C_real_loss_mean_op, 
                              fake_noisy_cloud, fake_clean_cloud, 
                              summary_op], 
                                feed_dict=feed_dict)
                    
                    # save currently generated fake codes
                    pc_util.write_ply_batch(clean_cur, os.path.join(LOG_DIR, 'fake_noisys', 'input_%d'%(i)))
                    pc_util.write_ply_batch(fake_noisy_cloud_val, os.path.join(LOG_DIR, 'fake_noisys', 'reconstr_%d'%(i)))
                    pc_util.write_ply_batch(noise_cur, os.path.join(LOG_DIR, 'fake_cleans', 'input_%d'%(i)))
                    pc_util.write_ply_batch(fake_clean_cloud_val, os.path.join(LOG_DIR, 'fake_cleans', 'reconstr_%d'%(i)))
                    
                    # terminal prints
                    log_string('%s training %d snapshot: '%(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'), i))
                    log_string('        G loss: {:.6f}, F loss: {:.6f}, cycle loss: {:.6f}'.format(G_loss_mean_val, F_loss_mean_val, cycle_loss_mean_val))
                    log_string('        D_N loss: {:.6f} = (f){:.6f}, (r){:.6f}'.format(D_N_loss_mean_val, D_N_fake_loss_mean_val, D_N_real_loss_mean_val))
                    log_string('        D_C loss: {:.6f} = (f){:.6f}, (r){:.6f}'.format(D_C_loss_mean_val, D_C_fake_loss_mean_val, D_C_real_loss_mean_val))
                    
                    # tensorboard output
                    train_writer.add_summary(summary, i)

                if i % para_config_gan['save_interval'] == 0:
                    # test and evaluate on all data
                    '''
                    NOISY_TRAIN_DATASET.reset()
                    while NOISY_TRAIN_DATASET.has_next_batch():
                        noise_cur = NOISY_TRAIN_DATASET.next_batch_noise_added_with_partial()
                        feed_dict={
                                    gan.input_noisy_cloud: noise_cur,
                                    #gan.gt: clean_cur,
                                    gan.is_training: False,
                                    }
                        fake_clean_cloud_val = sess.run(fake_clean_cloud, feed_dict=feed_dict)

                    NOISY_TRAIN_DATASET.reset()
                    summary_eval = sess.run(summary_eval_op, feed_dict=feed_dict)
                    train_writer.add_summary(summary_eval, i)
                    '''
                    
                    # save model
                    save_path = saver.save(sess, os.path.join(LOG_DIR, 'ckpts', 'model_%d.ckpt'%(i)))
                    log_string("Model saved in file: %s" % save_path)


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
