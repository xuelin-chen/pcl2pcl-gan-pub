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

# paras for autoencoder on all chairs
para_config = {
    'exp_name': 'ae_chair_real_normed',
    'random_seed': None,

    'point_cloud_dir': '/workspace/pointnet2/pc2pc/data/scannet_v2_chairs_alilgned_v2/point_cloud',

    'batch_size': 48,
    'lr': 0.0005, # base starting learning rate
    'decay_step': 5000000, # in samples, ~800 epoches
    'decay_rate': 0.5,
    'clip_lr': 0.0001, # minimal learning rate for clipping lr
    'epoch': 2001,
    'output_interval': 3, # unit in batch
    'save_interval': 10, # unit in epoch
    
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


TRAIN_DATASET = shapenet_pc_dataset.RealWorldPointsDataset(para_config['point_cloud_dir'], batch_size=para_config['batch_size'], npoint=para_config['point_cloud_shape'][0], shuffle=True, split='trainval')
TEST_DATASET = shapenet_pc_dataset.RealWorldPointsDataset(para_config['point_cloud_dir'], batch_size=para_config['batch_size'], npoint=para_config['point_cloud_shape'][0], shuffle=False, split='test')

#################### back up code for this run ##########################
LOG_DIR = os.path.join('run_ae', 'log_' + para_config['exp_name'] + '_' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)

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
            ae = autoencoder.AutoEncoder(paras=para_config)
            print_trainable_vars()

            reconstr_loss, reconstr, latent_code = ae.model()
            optimizer = ae.make_optimizer(reconstr_loss)

            # metrics for tensorboard visualization
            with tf.name_scope('metrics'):
                reconstr_loss_mean, reconstr_loss_mean_update = tf.metrics.mean(reconstr_loss)
            reset_metrics = tf.variables_initializer([var for var in tf.local_variables()
                                                 if var.name.split('/')[0] == 'metrics'])

            tf.summary.scalar('loss/train', reconstr_loss_mean, collections=['train'])
            tf.summary.scalar('loss/test', reconstr_loss_mean, collections=['test'])

            summary_op = tf.summary.merge_all('train')
            summary_test_op = tf.summary.merge_all('test')
            train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'summary', 'train'))
            test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'summary', 'test'))
            saver = tf.train.Saver(max_to_keep=None)

        # print
        log_string('Net layers:')
        log_string(str(ae))

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)
        sess.run(reset_metrics)

        total_batch_idx = 0
        for ep_idx in range(para_config['epoch']):
            log_string('-----------Epoch %d:-------------' % ep_idx)
            
            # train one epoch
            while TRAIN_DATASET.has_next_batch():
                sess.run(reset_metrics)

                input_batch = TRAIN_DATASET.next_batch()
                
                _, _ = sess.run([optimizer, reconstr_loss_mean_update], 
                                                        feed_dict={
                                                                    ae.input_pl: input_batch, 
                                                                    ae.is_training: True
                                                                    }
                                                        )

                if TRAIN_DATASET.batch_idx % para_config['output_interval'] == 0:
                    reconstr_loss_mean_val, summary = sess.run([reconstr_loss_mean, summary_op])
                    sess.run(reset_metrics)
                    log_string('-----------batch %d statistics snapshot:-------------' % TRAIN_DATASET.batch_idx)
                    log_string('  Reconstruction loss   : {:.6f}'.format(reconstr_loss_mean_val))

                    train_writer.add_summary(summary, total_batch_idx)
                    train_writer.flush()

                total_batch_idx +=  1

            # after each epoch, reset
            TRAIN_DATASET.reset() 

            # test and save
            if ep_idx % para_config['save_interval'] == 0:
                # test on whole test dataset
                sess.run(reset_metrics)
                TEST_DATASET.reset()
                while TEST_DATASET.has_next_batch():

                    input_batch_test = TEST_DATASET.next_batch()

                    reconstr_val_test, _ = sess.run([reconstr, reconstr_loss_mean_update],
                                                                    feed_dict={
                                                                                ae.input_pl: input_batch_test, 
                                                                                ae.is_training: False
                                                                              }
                                                                    )

                log_string('--------- on test split: --------')
                reconstr_loss_mean_val, summary_test = sess.run([reconstr_loss_mean, summary_test_op])
                log_string('Mean Reconstruction loss: {:.6f}'.format(reconstr_loss_mean_val))
                sess.run(reset_metrics) # reset metrics

                # tensorboard
                test_writer.add_summary(summary_test, ep_idx)
                test_writer.flush()

                # write out only one (last) batch for check
                pc_util.write_ply_batch(np.asarray(reconstr_val_test), os.path.join(LOG_DIR, 'pcloud', 'reconstr_%d'%(ep_idx)))
                pc_util.write_ply_batch(np.asarray(input_batch_test), os.path.join(LOG_DIR, 'pcloud', 'input_%d'%(ep_idx)))

                # save model
                save_path = saver.save(sess, os.path.join(LOG_DIR, 'ckpts', 'model_%d.ckpt'%(ep_idx)))
                log_string("Model saved in file: %s" % save_path)
                       
           
if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
