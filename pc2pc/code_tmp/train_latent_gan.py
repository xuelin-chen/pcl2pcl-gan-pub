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
    'exp_name': 'latent_gan',
    'ltcode_pkl_filename': '/workspace/pointnet2/pc2pc/run/log_ae_emd_chair_2048_test_good/latent_codes.pickle',

    'batch_size': 50,
    'lr': 0.0001,
    'beta1': 0.5,
    'epoch': 2001,
    'k': 1, # train k times for D each loop when training
    'kk': 1, # train k times for G each loop when training
    'output_interval': 10, # unit in epoch
    'shuffle_lcode_dataset': True,

    'noise_dim': 128,
    'noise_mu': 0.0,
    'noise_sigma': 0.2,
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

latent_code_dataset = LatentCodeDataset(para_config['ltcode_pkl_filename'],
                                             batch_size=para_config['batch_size'],
                                             shuffle=para_config['shuffle_lcode_dataset'])


#################### dirs, code backup and etc for this run ##########################
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
##########################################################################

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def print_trainable_vars():
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    for tv in trainable_vars:
        print(tv.name)

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(0)):
            latent_gan = LatentGAN(para_config)
            print_trainable_vars()
            G_loss, D_fake_loss, D_real_loss, D_loss, fake_code = latent_gan.model()
            G_optimizer, D_optimizer = latent_gan.optimize(G_loss, D_loss)

            # metrics for tensorboard visualization
            with tf.name_scope('metrics'):
                G_loss_mean_op, G_loss_mean_update_op = tf.metrics.mean(G_loss)
                D_fake_loss_mean_op, D_fake_loss_mean_update_op = tf.metrics.mean(D_fake_loss)
                D_real_loss_mean_op, D_real_loss_mean_update_op = tf.metrics.mean(D_real_loss)
                D_loss_mean_op, D_loss_mean_update_op = tf.metrics.mean(D_loss)
            reset_metrics = tf.variables_initializer([var for var in tf.local_variables()
                                                 if var.name.split('/')[0] == 'metrics'])

            tf.summary.scalar('loss/G_loss', G_loss_mean_op)
            tf.summary.scalar('loss/D_fake_loss', D_fake_loss_mean_op)
            tf.summary.scalar('loss/D_real_loss', D_real_loss_mean_op)
            tf.summary.scalar('loss/D_loss', D_loss_mean_op)

            summary_op = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'summary', 'train'))
            saver = tf.train.Saver(max_to_keep=None)

        # print
        log_string('Net layers:')
        log_string(str(latent_gan))

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)

        for i in range(para_config['epoch']):
            sess.run(reset_metrics)

            while latent_code_dataset.has_next_batch():
                noise_cur = latent_gan.generator_noise_distribution()
                real_code_cur = latent_code_dataset.next_batch()
                feed_dict={
                           latent_gan.noise_pl: noise_cur,
                           latent_gan.real_code: real_code_cur,
                           latent_gan.is_training: True,
                          }
                # train D for k times
                for _ in range(para_config['k']):
                    sess.run([D_optimizer, D_fake_loss_mean_update_op, D_real_loss_mean_update_op, D_loss_mean_update_op],
                              feed_dict=feed_dict)

                # train G
                for _ in range(para_config['kk']):
                    sess.run([G_optimizer, G_loss_mean_update_op], 
                              feed_dict=feed_dict)

            latent_code_dataset._reset()

            if i % para_config['output_interval'] == 0:
                G_loss_epoch, D_fake_loss_epoch, D_real_loss_epoch, D_loss_epoch, summary, fake_code_val = \
                 sess.run([G_loss_mean_op, D_fake_loss_mean_op, D_real_loss_mean_op, D_loss_mean_op, summary_op, fake_code], 
                            feed_dict=feed_dict)

                # terminal prints
                log_string('-- training %d snapshot: '%(i))
                log_string('        G loss: {:.6f}'.format(G_loss_epoch))
                log_string('        D loss: {:.6f} = (f){:.6f}, (r){:.6f}'.format(D_loss_epoch, D_fake_loss_epoch, D_real_loss_epoch))
                
                # tensorboard output
                train_writer.add_summary(summary, i)

                # save currently generated fake codes
                with open(os.path.join(LOG_DIR, 'fake_codes', 'gen_%d.pickle'%(i)), 'wb') as pf:
                    pickle.dump(fake_code_val, pf)
                
                # save model
                save_path = saver.save(sess, os.path.join(LOG_DIR, 'ckpts', 'model_%d.ckpt'%(i)))
                log_string("Model saved in file: %s" % save_path)
           
if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
