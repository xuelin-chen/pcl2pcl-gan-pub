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
from latent_gan import PCL2PCLGAN_SingleAE
import shapenet_pc_dataset


para_config_gan = {
    'exp_name': 'pcl2pcl_gan_np2c_singleAE',

    'point_cloud_dir': '/workspace/pointnet2/pc2pc/data/ShapeNet_v2_point_cloud/03001627/point_cloud_clean',
    'clean_ae_ckpt': '/workspace/pointnet2/pc2pc/run_ae/log_ae_chair_2019-02-14-20-05-24/ckpts/model_1600.ckpt',

    'batch_size': 48,
    'lr': 0.0001,
    'beta1': 0.5,
    'epoch': 2001,
    'k': 1, # train k times for D each loop when training
    'kk': 1, # train k times for G each loop when training
    'output_interval': 1, # unit in epoch
    'save_interval': 10, # unit in epoch

    'loss': 'emd',
    #'loss': 'hausdorff',
    'lambda': 1.0, # weight on back-reconstruction loss
    'eval_loss': 'emd',

    # noise parameters
    'noise_mu': 0.0, 
    'noise_sigma': 0.01, 
    'r_min': 0.1, 
    'r_max': 0.25, 
    'partial_portion': 0.25,

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

NOISY_TRAIN_DATASET = shapenet_pc_dataset.ShapeNetPartPointsDataset(para_config_gan['point_cloud_dir'], batch_size=para_config_gan['batch_size'], npoint=para_config_gan['point_cloud_shape'][0], shuffle=True, split='trainval')
CLEAN_TRAIN_DATASET = shapenet_pc_dataset.ShapeNetPartPointsDataset(para_config_gan['point_cloud_dir'], batch_size=para_config_gan['batch_size'], npoint=para_config_gan['point_cloud_shape'][0], shuffle=True, split='trainval')
NOISY_TEST_DATASET = shapenet_pc_dataset.ShapeNetPartPointsDataset(para_config_gan['point_cloud_dir'], batch_size=para_config_gan['batch_size'], npoint=para_config_gan['point_cloud_shape'][0], shuffle=False, split='test')

#################### dirs, code backup and etc for this run ##########################
LOG_DIR = os.path.join('run_pcl2pcl', 'log_' + para_config_gan['exp_name'] + '_' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
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
            latent_gan = PCL2PCLGAN_SingleAE(para_config_gan, para_config_ae)
            print_trainable_vars()
            G_loss, G_tofool_loss, reconstr_loss, D_loss, D_fake_loss, D_real_loss, fake_clean_reconstr, eval_loss = latent_gan.model()
            G_optimizer, D_optimizer = latent_gan.optimize(G_loss, D_loss)

            # metrics for tensorboard visualization
            with tf.name_scope('metrics'):
                G_loss_mean_op, G_loss_mean_update_op = tf.metrics.mean(G_loss)
                G_tofool_loss_mean_op, G_tofool_loss_mean_update_op = tf.metrics.mean(G_tofool_loss)
                reconstr_loss_mean_op, reconstr_loss_mean_update_op = tf.metrics.mean(reconstr_loss)
                
                D_loss_mean_op, D_loss_mean_update_op = tf.metrics.mean(D_loss)
                D_fake_loss_mean_op, D_fake_loss_mean_update_op = tf.metrics.mean(D_fake_loss)
                D_real_loss_mean_op, D_real_loss_mean_update_op = tf.metrics.mean(D_real_loss)

                eval_loss_mean_op, eval_loss_mean_update_op = tf.metrics.mean(eval_loss)

            reset_metrics = tf.variables_initializer([var for var in tf.local_variables()
                                                 if var.name.split('/')[0] == 'metrics'])

            tf.summary.scalar('loss/G/G_loss', G_loss_mean_op, collections=['train'])
            tf.summary.scalar('loss/G/G_tofool_loss', G_tofool_loss_mean_op, collections=['train'])
            tf.summary.scalar('loss/G/G_reconstr_loss', reconstr_loss_mean_op, collections=['train'])            
            tf.summary.scalar('loss/D/D_loss', D_loss_mean_op, collections=['train'])
            tf.summary.scalar('loss/D/D_fake_loss', D_fake_loss_mean_op, collections=['train'])
            tf.summary.scalar('loss/D/D_real_loss', D_real_loss_mean_op, collections=['train'])

            tf.summary.scalar('loss/%s_loss'%(para_config_gan['eval_loss']), eval_loss_mean_op, collections=['test'])

            summary_op = tf.summary.merge_all('train')
            summary_eval_op = tf.summary.merge_all('test')
            train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'summary', 'train'))
            test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'summary', 'test'))
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
            # Init variables
            init = tf.global_variables_initializer()
            sess.run(init)

            # NOTE: load pre-trained AE weights
            # clean AE, both pre-trained encoder and decoder are used
            clean_ckpt_vars = tf.contrib.framework.list_variables(para_config_gan['clean_ae_ckpt'])
            print('Clean AE pre-trained variables:')
            for vname, _ in clean_ckpt_vars:
                print(vname)
            restore_dict = get_restore_dict(clean_ckpt_vars, latent_gan.clean_encoder.all_variables)
            clean_saver = tf.train.Saver(restore_dict)
            clean_saver.restore(sess, para_config_gan['clean_ae_ckpt'])
            restore_dict = get_restore_dict(clean_ckpt_vars, latent_gan.clean_decoder.all_variables)
            clean_saver = tf.train.Saver(restore_dict)
            clean_saver.restore(sess, para_config_gan['clean_ae_ckpt'])
            print('Loading pre-trained clean AE done.')
            # END of weights loading

            for i in range(para_config_gan['epoch']):
                sess.run(reset_metrics)

                while NOISY_TRAIN_DATASET.has_next_batch() and CLEAN_TRAIN_DATASET.has_next_batch():

                    noise_cur = NOISY_TRAIN_DATASET.next_batch_noise_added_with_partial(noise_mu=para_config_gan['noise_mu'], noise_sigma=para_config_gan['noise_sigma'], r_min=para_config_gan['r_min'], r_max=para_config_gan['r_max'], partial_portion=para_config_gan['partial_portion'])
                    clean_cur = CLEAN_TRAIN_DATASET.next_batch()

                    feed_dict={
                            latent_gan.input_noisy_cloud: noise_cur,
                            latent_gan.input_clean_cloud: clean_cur,
                            latent_gan.is_training: True,
                            }

                    # train D for k times
                    for _ in range(para_config_gan['k']):
                        sess.run([D_optimizer, D_fake_loss_mean_update_op, D_real_loss_mean_update_op, D_loss_mean_update_op],
                                feed_dict=feed_dict)

                    # train G
                    for _ in range(para_config_gan['kk']):
                        sess.run([G_optimizer, G_tofool_loss_mean_update_op, reconstr_loss_mean_update_op, G_loss_mean_update_op], 
                                feed_dict=feed_dict)

                NOISY_TRAIN_DATASET.reset()
                CLEAN_TRAIN_DATASET.reset()

                if i % para_config_gan['output_interval'] == 0:
                    G_loss_mean_val, G_tofool_loss_mean_val, \
                    reconstr_loss_mean_val, \
                    D_loss_mean_val, D_fake_loss_mean_val, D_real_loss_mean_val, \
                    fake_clean_reconstr_val, summary = \
                    sess.run([G_loss_mean_op, G_tofool_loss_mean_op, \
                              reconstr_loss_mean_op, \
                              D_loss_mean_op, D_fake_loss_mean_op, D_real_loss_mean_op, \
                              fake_clean_reconstr, summary_op], 
                              feed_dict=feed_dict)
                    
                    # save currently generated
                    pc_util.write_ply_batch(fake_clean_reconstr_val, os.path.join(LOG_DIR, 'fake_cleans', 'reconstr_%d'%(i)))
                    pc_util.write_ply_batch(noise_cur, os.path.join(LOG_DIR, 'fake_cleans', 'input_noisy_%d'%(i)))

                    # terminal prints
                    log_string('%s training %d snapshot: '%(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'), i))
                    log_string('        G loss: {:.6f} = (g){:.6f}, (r){:.6f}'.format(G_loss_mean_val, G_tofool_loss_mean_val, reconstr_loss_mean_val))
                    log_string('        D loss: {:.6f} = (f){:.6f}, (r){:.6f}'.format(D_loss_mean_val, D_fake_loss_mean_val, D_real_loss_mean_val))
                    
                    # tensorboard output
                    train_writer.add_summary(summary, i)

                if i % para_config_gan['save_interval'] == 0:
                    # test and evaluate on test set
                    NOISY_TEST_DATASET.reset()
                    while NOISY_TEST_DATASET.has_next_batch():
                        noise_cur, clean_cur = NOISY_TEST_DATASET.next_batch_noise_added_with_partial(noise_mu=para_config_gan['noise_mu'], noise_sigma=para_config_gan['noise_sigma'], r_min=para_config_gan['r_min'], r_max=para_config_gan['r_max'], partial_portion=para_config_gan['partial_portion'], with_gt=True)
                        feed_dict={
                                    latent_gan.input_noisy_cloud: noise_cur,
                                    latent_gan.gt: clean_cur,
                                    latent_gan.is_training: False,
                                    }
                        fake_clean_reconstr_val, _ = sess.run([fake_clean_reconstr, eval_loss_mean_update_op], feed_dict=feed_dict)

                    NOISY_TEST_DATASET.reset()
                    eval_loss_mean_val, summary_eval = sess.run([eval_loss_mean_op, summary_eval_op], feed_dict=feed_dict)
                    test_writer.add_summary(summary_eval, i)
                    log_string('Eval loss (%s) on test set: %f'%(para_config_gan['eval_loss'], np.mean(eval_loss_mean_val)))
                    
                    # save model
                    save_path = saver.save(sess, os.path.join(LOG_DIR, 'ckpts', 'model_%d.ckpt'%(i)))
                    log_string("Model saved in file: %s" % save_path)
           
if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
