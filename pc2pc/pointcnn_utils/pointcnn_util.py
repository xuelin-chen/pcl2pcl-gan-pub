import os
import sys

BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../../tf_ops/sampling'))

from tf_sampling import farthest_point_sample, gather_point

from pointcnn import xconv, point_conv, point_conv_transpose
import pointfly as pf

import tensorflow as tf

def pointcnn_xconv_module(xyz, points, npoint, c_fts_out, c_x, k_neighbors, d_rate, is_training, depth_multiplier, with_global, scope):
    '''
    input:
        xyz: TF tensor, input point clouds coords, B x N x 3
        points: TF tensor, input point clouds features, B x N x fts_channel
        npoint: int32, number of representative samplings
        c_fts_out: int32, output channels number
        c_x: int32, channels number of lifted features for x-transformation matrix
        k_neighbors: int32, neighbor size
        d_rate: int32, dilation rate
        is_training: TF tensor, flag indicating training status
    output:
        new_xyz: TF tensor, output point clouds coords, B x npoint x 3
        new_points: TF tensor, output point clouds features, B x npoint x c_fts_out
    '''
    with tf.variable_scope(scope):
        new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz)) 

        new_points = xconv(xyz, points, new_xyz, scope, xyz.shape[0], k_neighbors, d_rate, npoint, c_fts_out, c_x, is_training, True, depth_multiplier, sorting_method=None, with_global=with_global)

        return new_xyz, new_points

def pointcnn_xupconv_module(xyz, points, dense_xyz, dense_points_skip, c_fts_out, c_x, k_neighbors, d_rate, is_training, depth_multiplier, with_global, scope):
    '''
    input:
        xyz: TF tensor, input point clouds coords, B x N x 3
        points: TF tensor, input point clouds features, B x N x fts_channel
        dense_xyz: TF tensor, representative points previously sampled, B x npoint x 3
        dense_points_skip: skip link from previously generated features
        c_fts_out: int32, output channels number
        c_x: int32, channels number of lifted features for x-transformation matrix
        k_neighbors: int32, neighbor size
        d_rate: int32, dilation rate
        is_training: TF tensor, flag indicating training status
    output:
        new_xyz: TF tensor, output point clouds coords, B x npoint x 3
        new_points: TF tensor, output point clouds features, B x npoint x c_fts_out
    '''
    with tf.variable_scope(scope):
        new_xyz = dense_xyz 
        npoint = new_xyz.shape[1]

        new_points = xconv(xyz, points, new_xyz, scope, xyz.shape[0], k_neighbors, d_rate, npoint, c_fts_out, c_x, is_training, True, depth_multiplier, sorting_method=None, with_global=with_global)

        if dense_points_skip is not None:
            points_concat = tf.concat([new_points, dense_points_skip], axis=-1, name=scope + '_skip_concat')
            new_points = pf.dense(points_concat, c_fts_out, scope + '_fuse', is_training)

        return new_xyz, new_points



def point_conv_module(pts_in, fts_in, npoint, c_fts_out, k_neighbors, d_rate, is_training, scope, activation=tf.nn.relu, bn=True, sorting_method='cxyz', center_patch=True):
    '''
    input:
        pts_in: TF tensor, input point clouds coords, B x N x 3
        fts_in: TF tensor, input point clouds features, B x N x fts_channel
        npoint: int32, number of representative samplings
        c_fts_out: int32, output channels number
        k_neighbors: int32, neighbor size
        d_rate: int32, dilation rate
        is_training: TF tensor, flag indicating training status
    output:
        new_xyz: TF tensor, output point clouds coords, B x npoint x 3
        new_points: TF tensor, output point clouds features, B x npoint x c_fts_out
    '''
    with tf.variable_scope(scope):
        if npoint != 1:
            new_pts = gather_point(pts_in, farthest_point_sample(npoint, pts_in)) 
        else:
            new_pts = tf.reduce_mean(pts_in, axis=1, keepdims=True)

        new_fts, indices = point_conv(pts_in, fts_in, new_pts, k_neighbors, d_rate, c_fts_out, is_training, activation=activation, bn=bn, sorting_method=sorting_method, center_patch=center_patch, tag=scope)

        return new_pts, new_fts, indices
    
def point_deconv_module(pts_in, fts_in, pts_out, fts_out_skipped, indices, c_fts_out, k_neighbors, d_rate, is_training, activation=tf.nn.relu, bn=True, sorting_method='cxyz', center_patch=True, scope='deconv'):
    '''
    '''
    with tf.variable_scope(scope):

        new_fts = point_conv_transpose(pts_in, fts_in,
                                       pts_out, fts_out_skipped, indices, 
                                       k_neighbors, d_rate, c_fts_out, 
                                       is_training, 
                                       activation=activation, 
                                       bn=bn, 
                                       sorting_method=sorting_method, 
                                       center_patch=center_patch,
                                       tconv='t_conv',
                                       tag=scope)

        return pts_out, new_fts
