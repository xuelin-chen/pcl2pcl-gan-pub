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
from pc_gan import PC2PCGAN
import shapenet_pc_dataset

# paras for autoencoder
para_config_gan = {
    'exp_name': 'debug_pc2pc_gan',
    'point_cloud_dir': '/workspace/pointnet2/pc2pc/data/ShapeNet_v2_point_cloud/03001627/point_cloud_clean_1k_objects',

    #'ckpt': '/workspace/pointnet2/pc2pc/run_pc2pc/log_debug_pc2pc_gan_2019-02-09-19-49-25/ckpts/model_2000.ckpt',
    'ckpt': None,

    'batch_size': 12,
    'lr_recon': 0.0005,
    'beta1_recon': 0.9,
    'epoch_recon': 1000,
    'lr': 0.0002,
    'beta1': 0.5,
    'epoch': 4001,
    'k': 1, # train k times for D each loop when training
    'kk': 1, # train k times for G each loop when training
    'output_interval': 1, # unit in epoch
    'save_interval': 50, # unit in epoch
    'shuffle_dataset': True,

    'loss': 'pairwise',
    #'loss': 'emd',

    'G_bn1': True,
    'G_bn2': True,
    'D_bn': False,

    'point_cloud_shape': [2048, 3],
    #'point_cloud_shape': [8192, 3],

    'noise_sigma': 0.05,

}


NOISY_TRAIN_DATASET = shapenet_pc_dataset.ShapeNetPartPointsDataset(para_config_gan['point_cloud_dir'], batch_size=para_config_gan['batch_size'], npoint=para_config_gan['point_cloud_shape'][0], shuffle=True, split='all')
CLEAN_TRAIN_DATASET = shapenet_pc_dataset.ShapeNetPartPointsDataset(para_config_gan['point_cloud_dir'], batch_size=para_config_gan['batch_size'], npoint=para_config_gan['point_cloud_shape'][0], shuffle=True, split='all')

#################### dirs, code backup and etc for this run ##########################
LOG_DIR = os.path.join('run_pc2pc', 'log_' + para_config_gan['exp_name'] + '_' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
if not os.path.exists(os.path.join(LOG_DIR, 'fake_cleans')): os.mkdir(os.path.join(LOG_DIR, 'fake_cleans'))

script_name = os.path.basename(__file__)
bk_filenames = ['pc_gan.py', 
                 script_name,  
                 'pointcnn_utils/pointcnn_util.py',
                 'pointcnn_utils/pointcnn_discriminator_generator.py']
for bf in bk_filenames:
    os.system('cp %s %s' % (bf, LOG_DIR))
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(para_config_gan)+'\n')

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
            gan = PC2PCGAN(para_config_gan)
            print_trainable_vars()
            G_loss, back_recon_loss, D_loss, D_fake_loss, D_real_loss, fake_clean_cloud, eval_recon_loss = gan.model()

            # 
            gan.G.bn1 = para_config_gan['G_bn1']
            gan.G.bn2 = para_config_gan['G_bn2']
            gan.D.bn  = para_config_gan['D_bn']

            G_recon_optimizer = gan.make_optimizer(back_recon_loss, gan.G.variables, para_config_gan['lr_recon'], para_config_gan['beta1_recon'], name='Adam_G_recon')
            G_optimizer = gan.make_optimizer(G_loss+back_recon_loss, gan.G.variables, para_config_gan['lr'], para_config_gan['beta1'], name='Adam_G')
            D_optimizer = gan.make_optimizer(D_loss, gan.D.variables, para_config_gan['lr'], para_config_gan['beta1'], name='Adam_D')
            #G_optimizer, D_optimizer = gan.optimize(G_loss, D_loss)

            # metrics for tensorboard visualization
            with tf.name_scope('metrics'):
                G_loss_mean_op, G_loss_mean_update_op = tf.metrics.mean(G_loss)
                back_recon_loss_mean_op, back_recon_loss_mean_update_op = tf.metrics.mean(back_recon_loss)
                
                D_loss_mean_op, D_loss_mean_update_op = tf.metrics.mean(D_loss)
                D_fake_loss_mean_op, D_fake_loss_mean_update_op = tf.metrics.mean(D_fake_loss)
                D_real_loss_mean_op, D_real_loss_mean_update_op = tf.metrics.mean(D_real_loss)

                eval_recon_loss_mean_op, eval_recon_loss_mean_update_op = tf.metrics.mean(eval_recon_loss)

            reset_metrics = tf.variables_initializer([var for var in tf.local_variables()
                                                 if var.name.split('/')[0] == 'metrics'])

            tf.summary.scalar('loss/G/G_loss', G_loss_mean_op, collections=['train'])          
            tf.summary.scalar('loss/G/back_recon_loss', back_recon_loss_mean_op, collections=['train'])
            tf.summary.scalar('loss/D/D_loss', D_loss_mean_op, collections=['train'])
            tf.summary.scalar('loss/D/D_fake_loss', D_fake_loss_mean_op, collections=['train'])
            tf.summary.scalar('loss/D/D_real_loss', D_real_loss_mean_op, collections=['train'])
            tf.summary.scalar('loss/eval_recon_loss', eval_recon_loss_mean_op, collections=['eval'])

            summary_op = tf.summary.merge_all('train')
            summary_eval_op = tf.summary.merge_all('eval')
            train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'summary', 'train'))
            test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'summary', 'test'))
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

            if para_config_gan['ckpt'] is not None:
                saver.restore(sess, para_config_gan['ckpt'])
                para_config_gan['epoch_recon'] = 0 # do not need to retrain the reconstruction

            # train to reconstruct the input noisy point cloud
            for i in range(para_config_gan['epoch_recon']):
                sess.run(reset_metrics)

                while NOISY_TRAIN_DATASET.has_next_batch():
                    noise_cur, clean_cur = NOISY_TRAIN_DATASET.next_batch_noisy_clean_pair(noise_sigma=para_config_gan['noise_sigma'])
                    feed_dict={
                            gan.input_noisy_cloud: noise_cur,
                            gan.gt: clean_cur,
                            gan.is_training: True,
                            }

                    sess.run([G_recon_optimizer, back_recon_loss_mean_update_op], feed_dict=feed_dict)
                NOISY_TRAIN_DATASET.reset()
                
                if i % para_config_gan['output_interval'] == 0:
                    back_recon_loss_val, fake_clean_cloud_val, summary = sess.run([back_recon_loss_mean_op, fake_clean_cloud, summary_op], feed_dict=feed_dict)

                    # terminal prints
                    log_string('%s recon. training epoch %d snapshot: '%(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'), i))
                    log_string('        Recon. loss: {:.6f}'.format(back_recon_loss_val))

                     # save currently generated fake codes
                    pc_util.write_ply_batch(fake_clean_cloud_val, os.path.join(LOG_DIR, 'fake_cleans', 'recon_%d'%(i)))

                    # tensorboard output
                    train_writer.add_summary(summary, i)

                if i % para_config_gan['save_interval'] == 0:
                    # save model
                    save_path = saver.save(sess, os.path.join(LOG_DIR, 'ckpts', 'recon_model_%d.ckpt'%(i)))
                    log_string("Model saved in file: %s" % save_path)

            for i in range(para_config_gan['epoch']):
                sess.run(reset_metrics)

                while NOISY_TRAIN_DATASET.has_next_batch() and CLEAN_TRAIN_DATASET.has_next_batch():
                    noise_cur = NOISY_TRAIN_DATASET.next_batch_noise_added(noise_sigma=para_config_gan['noise_sigma'])
                    clean_cur = CLEAN_TRAIN_DATASET.next_batch()
                    feed_dict={
                            gan.input_noisy_cloud: noise_cur,
                            gan.input_clean_cloud: clean_cur,
                            gan.is_training: True,
                            }
                    # train D for k times
                    for _ in range(para_config_gan['k']):
                        sess.run([D_optimizer, D_fake_loss_mean_update_op, D_real_loss_mean_update_op, D_loss_mean_update_op],
                                feed_dict=feed_dict)

                    # train G
                    for _ in range(para_config_gan['kk']):
                        sess.run([G_optimizer,
                                  G_loss_mean_update_op, 
                                  back_recon_loss_mean_update_op], 
                                  feed_dict=feed_dict)

                NOISY_TRAIN_DATASET.reset()
                CLEAN_TRAIN_DATASET.reset()

                if i % para_config_gan['output_interval'] == 0:
                    G_loss_mean_val, back_recon_loss_val, D_loss_mean_val, D_fake_loss_mean_val, D_real_loss_mean_val, fake_clean_cloud_val, summary = \
                    sess.run([G_loss_mean_op, back_recon_loss_mean_op, D_loss_mean_op, D_fake_loss_mean_op, D_real_loss_mean_op, fake_clean_cloud, summary_op], 
                                feed_dict=feed_dict)
                    
                    # save currently generated fake codes
                    pc_util.write_ply_batch(fake_clean_cloud_val, os.path.join(LOG_DIR, 'fake_cleans', 'fake_clean_%d'%(i)))
                    pc_util.write_ply_batch(noise_cur, os.path.join(LOG_DIR, 'fake_cleans', 'input_noisy_%d'%(i)))

                    # terminal prints
                    log_string('%s training epoch %d snapshot: '%(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'), i))
                    log_string('        G loss: {:.6f}, (recon){:.6f}'.format(G_loss_mean_val, back_recon_loss_val))
                    log_string('        D loss: {:.6f} = (f){:.6f}, (r){:.6f}'.format(D_loss_mean_val, D_fake_loss_mean_val, D_real_loss_mean_val))
                    
                    # tensorboard output
                    train_writer.add_summary(summary, i)

                if i % para_config_gan['save_interval'] == 0:
                    # test and evaluate on all data
                    NOISY_TRAIN_DATASET.reset()
                    while NOISY_TRAIN_DATASET.has_next_batch():
                        noise_cur, clean_cur = NOISY_TRAIN_DATASET.next_batch_noisy_clean_pair(noise_sigma=para_config_gan['noise_sigma'])
                        feed_dict={
                                    gan.input_noisy_cloud: noise_cur,
                                    gan.gt: clean_cur,
                                    gan.is_training: False,
                                    }
                        sess.run(eval_recon_loss_mean_update_op, feed_dict=feed_dict)

                    NOISY_TRAIN_DATASET.reset()
                    summary_eval = sess.run(summary_eval_op, feed_dict=feed_dict)
                    test_writer.add_summary(summary_eval, i)
                    
                    # save model
                    save_path = saver.save(sess, os.path.join(LOG_DIR, 'ckpts', 'model_%d.ckpt'%(i)))
                    log_string("Model saved in file: %s" % save_path)


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
