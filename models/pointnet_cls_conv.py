'''
    PointNet version 1 Model
    Reference: https://github.com/charlesq34/pointnet
'''
import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../pc2pc/pointcnn_utils'))
import pointfly as pf
import tf_util

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    input_image = tf.expand_dims(point_cloud, -1) # (B, N, 3, 1)
    
    # Point functions (MLP implemented as conv2d)
    net = tf_util.conv2d(input_image, 64, [1,3],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)
                         # BxNx1x1024

    qrs = tf.constant(0.0, shape=(batch_size, 1, 3))
    _, indices = pf.knn_indices_general(qrs, point_cloud, num_point, True)
    indices = pf.sort_points(point_cloud, indices, 'cxyz')

    nn_pts = tf.gather_nd(point_cloud, indices, name='sorted_pts') # (B, 1, npoint, 3)
    nn_pts = tf.squeeze(nn_pts) # (B, npoint, 3)

    nn_fts = tf.gather_nd(tf.squeeze(net), indices, name='sorted_fts')
    nn_fts = tf.squeeze(nn_fts) # (B, npoint, 1024)

    fts2conv = tf.concat([nn_pts, nn_fts], axis=-1) # (B, npoint, 1024+3)
    fts2conv = tf.expand_dims(fts2conv, axis=2) # (B, npoint, 1, 1024+3)
    fts2conv = tf.reshape(fts2conv, (batch_size, -1, 1, 1027))
    
    ch_nb = 128
    net = tf_util.conv2d(fts2conv, ch_nb, [num_point, 1],
                        padding='VALID', stride=[1,1],
                        bn=True, is_training=is_training,
                        scope='conv6', bn_decay=bn_decay) # (B, 1, 1, 1024)
    net = tf.reshape(net, (batch_size, 1, 1, ch_nb)) 

    # MLP on global point cloud vector
    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 128, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 64, bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp1')
    net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')

    '''
    # Symmetric function: max pooling
    net = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='maxpool')
    
    # MLP on global point cloud vector
    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp1')
    net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')
    '''

    return net, end_points


def get_loss(pred, label, end_points):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
