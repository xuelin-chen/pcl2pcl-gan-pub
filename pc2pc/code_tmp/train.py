'''
    Single-GPU training.
'''
import argparse
import math
from datetime import datetime
import h5py
import socket
import importlib
import os
import sys
import glob

import numpy as np
import tensorflow as tf
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
print('ROOT_DIR: ', ROOT_DIR)
sys.path.append(BASE_DIR) # model
sys.path.append(ROOT_DIR) # provider
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
import tf_util
import pc_util
import chair_clean_noisy_patch_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet2_chair_patch_cls_ssg', help='Model name [default: pointnet2_chair_patch_cls_ssg]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
parser.add_argument('--max_epoch', type=int, default=401, help='Epoch to run [default: 401]')
parser.add_argument('--batch_size', type=int, default=24, help='Batch Size during training [default: 24]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 50000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--output_interval', type=int, default=50, help='Statistics output interval [default: 50]')
FLAGS = parser.parse_args()

EPOCH_CNT = 0

OUTPUT_INTERVAL = FLAGS.output_interval
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module(FLAGS.model) # import network module
#################### back up code for this run ##########################
MODEL_FILE = os.path.join(BASE_DIR, FLAGS.model+'.py')
LOG_DIR = os.path.join('run', 'log_' + FLAGS.model + '_' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
script_name = os.path.basename(__file__)
print(script_name)
os.system('cp %s %s' % (script_name, LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

# Shapenet official train/test split
assert(NUM_POINT<=2048)
TRAIN_DATASET = chair_clean_noisy_patch_dataset.ChairCleanNoisyDataset(data_dir='data/ShapeNet_v2_point_cloud/03001627', split='train', batch_size = BATCH_SIZE, npoints = NUM_POINT, shuffle=True)
TEST_DATASET = chair_clean_noisy_patch_dataset.ChairCleanNoisyDataset(data_dir='data/ShapeNet_v2_point_cloud/03001627', split='test', batch_size = BATCH_SIZE, npoints = NUM_POINT, shuffle=True)

NUM_CLASSES = chair_clean_noisy_patch_dataset.NUM_CLASSES
IMG_OUTPUT_DIR = os.path.join(LOG_DIR, 'img')

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def clean_dir(dirname, name_match='*.jpg'):
    if not os.path.exists(dirname):
        return
    files = glob.glob(os.path.join(dirname, name_match))
    for pf in files:
        os.remove(pf)

def vis_point_cloud_preds(point_clouds, labels):
    '''
    input:
        point_clouds: np array, BxNx3
        labels: np array, B
    '''
    if not os.path.exists(IMG_OUTPUT_DIR):
        os.makedirs(IMG_OUTPUT_DIR)

    clean_dir(IMG_OUTPUT_DIR)

    images = []
    for idx, pc in enumerate(point_clouds):
        label = labels[idx]
        im_array = pc_util.point_cloud_three_views(pc)
        img = Image.fromarray(np.uint8(im_array*255.0))
        images.append(img)

        img_name = 'pc%d_pred_as_%d.jpg' % (idx, label)
        im_output_filename = os.path.join(IMG_OUTPUT_DIR, img_name)
        img.save(im_output_filename)

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter
            # for you every time it trains.
            batch = tf.get_variable('batch', [],
                initializer=tf.constant_initializer(0), trainable=False)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay, collections=['train'])

            # Get model and loss 
            logits, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            probs = tf.nn.softmax(logits, name='probs') # BxNUM_CLASSES
            predictions = tf.argmax(probs, axis=-1, name='predictions') # B

            loss_op = MODEL.get_loss(logits, labels_pl, end_points) # B

            # statistics for log
            with tf.name_scope('metrics'):
                loss_mean_op, loss_mean_update_op = tf.metrics.mean(loss_op)
                t_1_acc_op, t_1_acc_update_op = tf.metrics.accuracy(labels_pl, predictions)
                t_1_per_class_acc_op, t_1_per_class_acc_update_op = tf.metrics.mean_per_class_accuracy(labels_pl, predictions, NUM_CLASSES)
                reset_metrics_op = tf.variables_initializer([var for var in tf.local_variables()
                                                        if var.name.split('/')[0] == 'metrics'])

            # tensorboard visualization
            _ = tf.summary.scalar('loss/train', tensor=loss_mean_op, collections=['train'])
            _ = tf.summary.scalar('t_1_acc/train', tensor=t_1_acc_op, collections=['train'])
            _ = tf.summary.scalar('t_1_per_class_acc/train', tensor=t_1_per_class_acc_op, collections=['train'])

            _ = tf.summary.scalar('loss/val', tensor=loss_mean_op, collections=['val'])
            _ = tf.summary.scalar('t_1_acc/val', tensor=t_1_acc_op, collections=['val'])
            _ = tf.summary.scalar('t_1_per_class_acc/val', tensor=t_1_per_class_acc_op, collections=['val'])

            # get learning rate
            learning_rate = get_learning_rate(batch)
            _ = tf.summary.scalar('learning_rate/train', learning_rate, collections=['val'])

            print("--- Get training operator ---")
            # Get training operator
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss_op, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver(max_to_keep=None)
        
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged_train = tf.summary.merge_all('train')
        merged_val = tf.summary.merge_all('val')
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'summary', 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'summary', 'val'), sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)

        ops = {
               'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'train': train_op,
               'predictions': predictions,
               'reset_metrics': reset_metrics_op,
               'loss_mean': loss_mean_op,
               'loss_mean_update': loss_mean_update_op,
               't_1_acc': t_1_acc_op,
               't_1_acc_update': t_1_acc_update_op,
               't_1_per_class_acc': t_1_per_class_acc_op, 
               't_1_per_class_acc_update': t_1_per_class_acc_update_op,
               'merged_train': merged_train,
               'merged_val': merged_val,
               'step': batch,
               'end_points': end_points
               }

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %04d ****' % (epoch))
            sys.stdout.flush()
             
            train_one_epoch(sess, ops, train_writer)
            eval_one_epoch(sess, ops, test_writer)

            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, 'ckpts', "model_"+str(epoch)+".ckpt"))
                log_string("Model saved in file: %s" % save_path)

def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    log_string(str(datetime.now()))

    batch_idx = 0

    sess.run(ops['reset_metrics']) # before this epoch, clean metrics
    while TRAIN_DATASET.has_next_batch():
        batch_data, batch_label = TRAIN_DATASET.next_batch(augment=True)
        #batch_data = provider.random_point_dropout(batch_data)

        feed_dict = {
                     ops['pointclouds_pl']: batch_data,
                     ops['labels_pl']: batch_label,
                     ops['is_training_pl']: is_training
                    }
        step, _, _, _, _ = sess.run([ops['step'],
                                     ops['train'],
                                     ops['loss_mean_update'], 
                                     ops['t_1_acc_update'], 
                                     ops['t_1_per_class_acc_update']], 
                                     feed_dict=feed_dict)

        if (batch_idx+1)%OUTPUT_INTERVAL == 0:
            log_string(' --- Statistics from batch: %03d - %03d ---'
                              % (batch_idx+1-OUTPUT_INTERVAL, batch_idx+1))
            loss, t_1_acc, t_1_per_class_acc, summary = sess.run([ops['loss_mean'],
                                                                  ops['t_1_acc'],
                                                                  ops['t_1_per_class_acc'],
                                                                  ops['merged_train']])
            train_writer.add_summary(summary, step)
            log_string('{}-[Train]  Loss: {:.4f}  T-1 Acc: {:.4f}  T-1 mAcc: {:.4f}'
                      .format(datetime.now(), loss, t_1_acc, t_1_per_class_acc))
            sess.run(ops['reset_metrics']) # clean metrics every 50 batches

        batch_idx += 1

    TRAIN_DATASET.reset()
    sess.run(ops['reset_metrics']) # after this epoch, clean metrics
        
def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False

    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT))
    
    batch_idx = 0

    sess.run(ops['reset_metrics'])
    while TEST_DATASET.has_next_batch():
        batch_data, batch_label = TEST_DATASET.next_batch(augment=False)

        feed_dict = {
                     ops['pointclouds_pl']: batch_data,
                     ops['labels_pl']: batch_label,
                     ops['is_training_pl']: is_training
                    }
        step, point_clouds, predictions, _, _, _ = sess.run([ops['step'],
                                                            ops['pointclouds_pl'],
                                                            ops['predictions'],
                                                            ops['loss_mean_update'], 
                                                            ops['t_1_acc_update'], 
                                                            ops['t_1_per_class_acc_update']], 
                                                            feed_dict=feed_dict)
        
        if batch_idx == 1: vis_point_cloud_preds(point_clouds, predictions)
        batch_idx = batch_idx + 1

    log_string(' ---- Evaluation statistics ----')
    loss, t_1_acc, t_1_per_class_acc, summary = sess.run([ops['loss_mean'], 
                                                          ops['t_1_acc'], 
                                                          ops['t_1_per_class_acc'], 
                                                          ops['merged_val']])
    test_writer.add_summary(summary, step)
    log_string('{}-[test]  Loss: {:.4f}  T-1 Acc: {:.4f}  T-1 mAcc: {:.4f}'
              .format(datetime.now(), loss, t_1_acc, t_1_per_class_acc))
    sess.run(ops['reset_metrics'])

    TEST_DATASET.reset()
    EPOCH_CNT += 1

if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
