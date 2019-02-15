import tensorflow as tf

import os
import sys

from pointcnn_util import point_conv_module
import pointfly as pf

class EncoderConv:
    def __init__(self, name, is_training, latent_code_dim=128):
        self.name = name
        self.is_training = is_training
        self.latent_code_dim = latent_code_dim
        
    def __call__(self, point_cloud):
        '''
        input:
            point_cloud: BxNx3
        output: 
            latent_code: Bxlatent_code_dim
        '''
        with tf.variable_scope(self.name):
            #batch_size = point_cloud.get_shape()[0].value
            num_point = point_cloud.get_shape()[1].value
            end_points = {}
            l0_xyz = point_cloud
            l0_points = None
            end_points['l0_xyz'] = l0_xyz 

            #sort_mtd = 'cxyz'
            sort_mtd= None

            # Set Abstraction layers
            l1_xyz, l1_points, l1_indices = point_conv_module(l0_xyz, l0_points, npoint=512, c_fts_out=128, k_neighbors=64, d_rate=1, is_training=is_training, scope='conv_layer_1', sorting_method=sort_mtd)
            l2_xyz, l2_points, l2_indices = point_conv_module(l1_xyz, l1_points, npoint=128, c_fts_out=256, k_neighbors=32, d_rate=1, is_training=is_training, scope='conv_layer_2', sorting_method=sort_mtd)
            l3_xyz, l3_points, l3_indices = point_conv_module(l2_xyz, l2_points,  npoint=24, c_fts_out=512, k_neighbors=8, d_rate=1, is_training=is_training, scope='conv_layer_3', sorting_method=sort_mtd)
            
            
            
            # fc
            fc = pf.dense(l5_points, 1024, 'fc1', self.is_training)
            fc = pf.dense(fc, 512, 'fc2', self.is_training)
            latent_code = pf.dense(fc, self.latent_code_dim, 'fc3', self.is_training)
            
            # B x latent_code_dim
            return latent_code

class DecoderConv:
    def __init__(self, name, is_training, npoint=2048):
        self.name = name
        self.is_training = is_training
        self.npoint = npoint
        
    def __call__(self, latent_code):
        '''
        input:
            latent_code: Bxlatent_code_dim
        output: 
            point_cloud: Bxnpointx3
        '''
        with tf.variable_scope(self.name):
            print('')