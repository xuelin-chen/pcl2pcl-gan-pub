import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../pc2pc/pointcnn_utils'))
import tensorflow as tf
import numpy as np
import tf_util
from pointcnn_util import point_conv_module, point_deconv_module

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 6))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, bn_decay=None):
    """ Part segmentation PointNet, input is BxNx6 (XYZ NormalX NormalY NormalZ), output Bx50 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = tf.slice(point_cloud, [0,0,0], [-1,-1,3])
    l0_points = tf.slice(point_cloud, [0,0,3], [-1,-1,3])

    sort_mtd = 'cxyz'
    #sort_mtd = 'random'

    center_patch = False

    # Set Abstraction layers
    l1_xyz, l1_points, l1_indices = point_conv_module(l0_xyz, l0_points, npoint=512, c_fts_out=128, k_neighbors=64, d_rate=1, is_training=is_training, scope='conv_layer_1', sorting_method=sort_mtd, center_patch=center_patch)
    l2_xyz, l2_points, l2_indices = point_conv_module(l1_xyz, l1_points, npoint=128, c_fts_out=256, k_neighbors=32, d_rate=1, is_training=is_training, scope='conv_layer_2', sorting_method=sort_mtd, center_patch=center_patch)
    l3_xyz, l3_points, l3_indices = point_conv_module(l2_xyz, l2_points,  npoint=24, c_fts_out=512, k_neighbors=8, d_rate=1, is_training=is_training, scope='conv_layer_3', sorting_method=sort_mtd, center_patch=center_patch)

    # Feature Propagation layers
    '''
    # no Unet
    l2_xyz, l2_points = point_deconv_module(l3_xyz, l3_points, l2_xyz, None, l3_indices, c_fts_out=256, k_neighbors=8, d_rate=1, is_training=is_training, scope='deconv_layer_1')
    l1_xyz, l1_points = point_deconv_module(l2_xyz, l2_points, l1_xyz, None, l2_indices, c_fts_out=128, k_neighbors=32, d_rate=1, is_training=is_training, scope='deconv_layer_2')
    l0_xyz, l0_points = point_deconv_module(l1_xyz, l1_points, l0_xyz, l0_points, l1_indices, c_fts_out=128, k_neighbors=64, d_rate=1, is_training=is_training, scope='deconv_layer_3')
    '''
    # with U-net
    l2_xyz, l2_points = point_deconv_module(l3_xyz, l3_points, l2_xyz, l2_points, l3_indices, c_fts_out=256, k_neighbors=8, d_rate=1, is_training=is_training, scope='deconv_layer_1', center_patch=center_patch)
    l1_xyz, l1_points = point_deconv_module(l2_xyz, l2_points, l1_xyz, l1_points, l2_indices, c_fts_out=128, k_neighbors=32, d_rate=1, is_training=is_training, scope='deconv_layer_2', center_patch=center_patch)
    l0_xyz, l0_points = point_deconv_module(l1_xyz, l1_points, l0_xyz, l0_points, l1_indices, c_fts_out=128, k_neighbors=64, d_rate=1, is_training=is_training, scope='deconv_layer_3', center_patch=center_patch)


    # FC layers
    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    end_points['feats'] = net 
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util.conv1d(net, 50, 1, padding='VALID', activation_fn=None, scope='fc2')

    return net, end_points


def get_loss(pred, label):
    """ pred: BxNxC,
        label: BxN, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,2048,6))
        net, _ = get_model(inputs, tf.constant(True))
        print(net)
