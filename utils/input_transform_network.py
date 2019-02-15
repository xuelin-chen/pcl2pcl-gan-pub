import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/3d_interpolation'))
from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point, group_point, knn_point
from tf_interpolate import three_nn, three_interpolate
import tensorflow as tf
import numpy as np
import tf_util
from pointnet_util import sample_and_group_all, sample_and_group

def channel_weight_all(new_points, mlp, is_training, bn_decay, scope, bn=True, pooling='max', use_nchw=False):
    '''
    reweight all input channels
    input:
        new_points: (batch_size, npoint, nsample, 3+channel)
        mlp: list of int32 -- output size for MLP on each point
    return 
        input_mask: (batch_size, npoint, nsample, 3+channel)
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'
    data_channel = new_points.shape[3]
    nsample = new_points.shape[2]
    mlp.append(data_channel) # append 3+channel in the mlp
    with tf.variable_scope(scope) as sc:
        if use_nchw: new_points = tf.transpose(new_points, [0,3,1,2])
        for i, num_out_channel in enumerate(mlp):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, is_training=is_training,
                                            scope='conv%d'%(i), bn_decay=bn_decay,
                                            data_format=data_format) # (batch_size, npoint, nsample, 3+channel)
        if use_nchw: new_points = tf.transpose(new_points, [0,2,3,1])
        
        # Pooling in Local Regions
        if pooling=='max':
            new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')  # (batch_size, npoint, 1, 3+channel)
        elif pooling=='avg':
            new_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
        elif pooling=='weighted_avg':
            with tf.variable_scope('weighted_avg'):
                dists = tf.norm(grouped_xyz,axis=-1,ord=2,keep_dims=True)
                exp_dists = tf.exp(-dists * 5)
                weights = exp_dists/tf.reduce_sum(exp_dists,axis=2,keep_dims=True) # (batch_size, npoint, nsample, 1)
                new_points *= weights # (batch_size, npoint, nsample, mlp[-1])
                new_points = tf.reduce_sum(new_points, axis=2, keep_dims=True)
        elif pooling=='max_and_avg':
            max_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
            avg_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
            new_points = tf.concat([avg_points, max_points], axis=-1)

        input_mask = tf.tile(new_points, [1,1,nsample,1]) # (batch_size, npoints, nsample, 3+channel)

    return input_mask

def input_reweight_all_net(xyz, points, npoint, radius, nsample, mlp, mlp2, group_all, is_training, bn_decay, scope, bn=True, pooling='max', knn=False, use_xyz=True, use_nchw=False):
    ''' from PointNet Set Abstraction (SA) Module
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: float32 -- search radius in local region
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.variable_scope(scope) as sc:
        # Sample and Grouping
        if group_all:
            nsample = xyz.get_shape()[1].value
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group(npoint, radius, nsample, xyz, points, knn, use_xyz)
            #new_points: (batch_size, npoint, nsample, 3+channel)

        ############################# input channel reweight ###################
        channel_weights = channel_weight_all(new_points, mlp, is_training, bn_decay, scope+'_input_reweight') # (batch_size, npoints, nsample, 3+channel)
        new_points = tf.multiply(new_points, channel_weights)
        ########################################################################   
    
        # Point Feature Embedding
        if use_nchw: new_points = tf.transpose(new_points, [0,3,1,2])
        for i, num_out_channel in enumerate(mlp):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(i), bn_decay=bn_decay,
                                        data_format=data_format) # (batch_size, npoint, nsample, mlp[-1])
        if use_nchw: new_points = tf.transpose(new_points, [0,2,3,1])

        # Pooling in Local Regions
        if pooling=='max':
            new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool') 
        elif pooling=='avg':
            new_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
        elif pooling=='weighted_avg':
            with tf.variable_scope('weighted_avg'):
                dists = tf.norm(grouped_xyz,axis=-1,ord=2,keep_dims=True)
                exp_dists = tf.exp(-dists * 5)
                weights = exp_dists/tf.reduce_sum(exp_dists,axis=2,keep_dims=True) # (batch_size, npoint, nsample, 1)
                new_points *= weights # (batch_size, npoint, nsample, mlp[-1])
                new_points = tf.reduce_sum(new_points, axis=2, keep_dims=True)
        elif pooling=='max_and_avg':
            max_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
            avg_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
            new_points = tf.concat([avg_points, max_points], axis=-1)

        # [Optional] Further Processing 
        if mlp2 is not None:
            if use_nchw: new_points = tf.transpose(new_points, [0,3,1,2])
            for i, num_out_channel in enumerate(mlp2):
                new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, is_training=is_training,
                                            scope='conv_post_%d'%(i), bn_decay=bn_decay,
                                            data_format=data_format) 
            if use_nchw: new_points = tf.transpose(new_points, [0,2,3,1])

        new_points = tf.squeeze(new_points, [2]) # (batch_size, npoints, mlp2[-1])
        return new_xyz, new_points, idx

def channel_weight_partial(new_points, mlp, is_training, bn_decay, scope, bn=True, pooling='max', use_nchw=False):
    '''
    reweight all input channels except xyz
    input:
        new_points: (batch_size, npoint, nsample, 3+channel)
        mlp: list of int32 -- output size for MLP on each point
    return 
        input_mask: (batch_size, npoint, nsample, channel)
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'
    data_channel = new_points.shape[3]-3
    nsample = new_points.shape[2]
    mlp.append(data_channel) # append 3+channel in the mlp
    with tf.variable_scope(scope) as sc:
        if use_nchw: new_points = tf.transpose(new_points, [0,3,1,2])
        for i, num_out_channel in enumerate(mlp):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, is_training=is_training,
                                            scope='conv%d'%(i), bn_decay=bn_decay,
                                            data_format=data_format) # (batch_size, npoint, nsample, channel)
        if use_nchw: new_points = tf.transpose(new_points, [0,2,3,1])
        
        # Pooling in Local Regions
        if pooling=='max':
            new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')  # (batch_size, npoint, 1, channel)
        elif pooling=='avg':
            new_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
        elif pooling=='weighted_avg':
            with tf.variable_scope('weighted_avg'):
                dists = tf.norm(grouped_xyz,axis=-1,ord=2,keep_dims=True)
                exp_dists = tf.exp(-dists * 5)
                weights = exp_dists/tf.reduce_sum(exp_dists,axis=2,keep_dims=True) # (batch_size, npoint, nsample, 1)
                new_points *= weights # (batch_size, npoint, nsample, mlp[-1])
                new_points = tf.reduce_sum(new_points, axis=2, keep_dims=True)
        elif pooling=='max_and_avg':
            max_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
            avg_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
            new_points = tf.concat([avg_points, max_points], axis=-1)

        input_mask = tf.tile(new_points, [1,1,nsample,1]) # (batch_size, npoints, nsample, channel)

    return input_mask

def input_reweight_partial_net(xyz, points, npoint, radius, nsample, mlp, mlp2, group_all, is_training, bn_decay, scope, bn=True, pooling='max', knn=False, use_xyz=True, use_nchw=False):
    ''' from PointNet Set Abstraction (SA) Module
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: float32 -- search radius in local region
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.variable_scope(scope) as sc:
        # Sample and Grouping
        if group_all:
            nsample = xyz.get_shape()[1].value
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group(npoint, radius, nsample, xyz, points, knn, use_xyz)
            #new_points: (batch_size, npoint, nsample, 3+channel)

        ############################# input channel reweight ###################
        channel_weights = channel_weight_partial(new_points, mlp, is_training, bn_decay, scope+'_input_reweight') # (batch_size, npoints, nsample, channel)
        new_points = tf.multiply(new_points[...,3:], channel_weights)
        ########################################################################   
    
        # Point Feature Embedding
        if use_nchw: new_points = tf.transpose(new_points, [0,3,1,2])
        for i, num_out_channel in enumerate(mlp):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(i), bn_decay=bn_decay,
                                        data_format=data_format) # (batch_size, npoint, nsample, mlp[-1])
        if use_nchw: new_points = tf.transpose(new_points, [0,2,3,1])

        # Pooling in Local Regions
        if pooling=='max':
            new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool') 
        elif pooling=='avg':
            new_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
        elif pooling=='weighted_avg':
            with tf.variable_scope('weighted_avg'):
                dists = tf.norm(grouped_xyz,axis=-1,ord=2,keep_dims=True)
                exp_dists = tf.exp(-dists * 5)
                weights = exp_dists/tf.reduce_sum(exp_dists,axis=2,keep_dims=True) # (batch_size, npoint, nsample, 1)
                new_points *= weights # (batch_size, npoint, nsample, mlp[-1])
                new_points = tf.reduce_sum(new_points, axis=2, keep_dims=True)
        elif pooling=='max_and_avg':
            max_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
            avg_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
            new_points = tf.concat([avg_points, max_points], axis=-1)

        # [Optional] Further Processing 
        if mlp2 is not None:
            if use_nchw: new_points = tf.transpose(new_points, [0,3,1,2])
            for i, num_out_channel in enumerate(mlp2):
                new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, is_training=is_training,
                                            scope='conv_post_%d'%(i), bn_decay=bn_decay,
                                            data_format=data_format) 
            if use_nchw: new_points = tf.transpose(new_points, [0,2,3,1])

        new_points = tf.squeeze(new_points, [2]) # (batch_size, npoints, mlp2[-1])
        return new_xyz, new_points, idx
