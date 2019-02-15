from pointnet_util import pointnet_sa_module, pointnet_fp_module
import tf_util

import tensorflow as tf

class Generator:
    def __init__(self, name, is_training, norm_method='instance', use_sigmoid=False):
        self.name = name
        self.is_training = is_training
        self.range_max = 0.02
        self.reuse = False
        self.norm_method = norm_method

        # NOTE: below is not used so far
        self.use_sigmoid = use_sigmoid

    def __call__(self, point_cloud):
        '''
            Generator for translation point cloud from domain A to domain B
            input:
                point_cloud: point cloud from domain A BxNxChannel
            output:
                translated point cloud in domain B: BxNxChannel
        '''
        with tf.variable_scope(self.name, reuse=self.reuse):
            #batch_size = point_cloud.get_shape()[0].value
            #num_point = point_cloud.get_shape()[1].value
            end_points = {}
            l0_xyz = point_cloud
            l0_points = None
            end_points['l0_xyz'] = l0_xyz

            # Layer 1
            l1_xyz, l1_points, _ = pointnet_sa_module(l0_xyz, l0_points, npoint=1024, radius=0.02, nsample=64, mlp=[32,32,64], mlp2=None, group_all=False, is_training=self.is_training, activation_fn=tf.nn.relu, norm_method=self.norm_method, scope='layer1') # not use normalization for the first layer of D
            l2_xyz, l2_points, _ = pointnet_sa_module(l1_xyz, l1_points, npoint=512, radius=0.04, nsample=48, mlp=[64,64,128], mlp2=None, group_all=False, is_training=self.is_training, activation_fn=tf.nn.relu, norm_method=self.norm_method, scope='layer2')
            l3_xyz, l3_points, _ = pointnet_sa_module(l2_xyz, l2_points, npoint=256, radius=0.06, nsample=32, mlp=[128,128,256], mlp2=None, group_all=False, is_training=self.is_training, activation_fn=tf.nn.relu, norm_method=self.norm_method, scope='layer3')
            l4_xyz, l4_points, _ = pointnet_sa_module(l3_xyz, l3_points, npoint=128, radius=0.08, nsample=16, mlp=[256,256,512], mlp2=None, group_all=False, is_training=self.is_training, activation_fn=tf.nn.relu, norm_method=self.norm_method, scope='layer4')
            l5_xyz, l5_points, _ = pointnet_sa_module(l4_xyz, l4_points, npoint=64, radius=0.10, nsample=8, mlp=[512,512,1024], mlp2=None, group_all=False, is_training=self.is_training, activation_fn=tf.nn.relu, norm_method=self.norm_method, scope='layer5')

            # Feature Propagation layers
            l4_points = pointnet_fp_module(l4_xyz, l5_xyz, l4_points, l5_points, [256,256], self.is_training, activation_fn=tf.nn.relu, norm_method=self.norm_method, scope='fa_layer1')
            l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [256,256], self.is_training, activation_fn=tf.nn.relu, norm_method=self.norm_method, scope='fa_layer2')
            l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256,256], self.is_training, activation_fn=tf.nn.relu, norm_method=self.norm_method, scope='fa_layer3')
            l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256,128], self.is_training, activation_fn=tf.nn.relu, norm_method=self.norm_method, scope='fa_layer4')
            l0_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points, [128,128,128], self.is_training, activation_fn=tf.nn.relu, norm_method=self.norm_method, scope='fa_layer5')

            # FC layers
            net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', is_training=self.is_training, scope='fc1', activation_fn=tf.nn.relu, norm_method=self.norm_method )
            net = tf_util.conv1d(net, 64, 1, padding='VALID', is_training=self.is_training, scope='fc2', activation_fn=tf.nn.relu, norm_method=self.norm_method)
            net = tf_util.conv1d(net, 3, 1, padding='VALID', is_training=self.is_training, activation_fn=tf.nn.tanh, scope='output', norm_method=None) # no normalization for the last output layer

            displacements = net * self.range_max
            translated_point_cloud = point_cloud + displacements

            # set reuse=True for next call
            self.reuse = True
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

            return translated_point_cloud

import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'pointcnn_utils'))
from pointcnn_util import pointcnn_xconv_module, pointcnn_xupconv_module
class GeneratorCNN:
    def __init__(self, name, is_training, norm_method='instance', use_sigmoid=False):
        self.name = name
        self.is_training = is_training
        self.range_max = 0.01
        self.reuse = False
        self.norm_method = norm_method

        # NOTE: below is not used so far
        self.use_sigmoid = use_sigmoid

    def __call__(self, point_cloud):
        '''
            Generator for translation point cloud from domain A to domain B
            input:
                point_cloud: point cloud from domain A BxNxChannel
            output:
                translated point cloud in domain B: BxNxChannel
        '''
        with tf.variable_scope(self.name, reuse=self.reuse):
            batch_size = point_cloud.get_shape()[0].value
            num_point = point_cloud.get_shape()[1].value
            end_points = {}
            l0_xyz = point_cloud
            l0_points = None
            end_points['l0_xyz'] = l0_xyz 
            # Set abstraction layers
            l1_xyz, l1_points = pointcnn_xconv_module(l0_xyz,
                                                      l0_points, 
                                                      npoint=num_point, 
                                                      c_fts_out=128, 
                                                      c_x=128, 
                                                      k_neighbors=8, d_rate=1, 
                                                      depth_multiplier=4, 
                                                      is_training=self.is_training,
                                                      with_global=True, scope='xconv_layer1') # not using normalization for the first layer of D
            l2_xyz, l2_points = pointcnn_xconv_module(l1_xyz,
                                                      l1_points, 
                                                      npoint=786, 
                                                      c_fts_out=256, 
                                                      c_x=64, 
                                                      k_neighbors=12, d_rate=2, 
                                                      depth_multiplier=2, 
                                                      is_training=self.is_training,
                                                      with_global=True, scope='xconv_layer2') 
            l3_xyz, l3_points = pointcnn_xconv_module(l2_xyz,
                                                      l2_points, 
                                                      npoint=384, 
                                                      c_fts_out=512, 
                                                      c_x=64, 
                                                      k_neighbors=12, d_rate=2, 
                                                      depth_multiplier=2, 
                                                      is_training=self.is_training,
                                                      with_global=True, scope='xconv_layer3') 
            l4_xyz, l4_points = pointcnn_xconv_module(l3_xyz,
                                                      l3_points, 
                                                      npoint=128, 
                                                      c_fts_out=1024, 
                                                      c_x=64, 
                                                      k_neighbors=12, d_rate=2, 
                                                      depth_multiplier=2, 
                                                      is_training=self.is_training,
                                                      with_global=True, scope='xconv_layer4') 

            # Feature Propagation layers
            l3_xyz, l3_points = pointcnn_xupconv_module(l4_xyz,
                                                        l4_points, 
                                                        l3_xyz,
                                                        l3_points, 
                                                        c_fts_out=512, 
                                                        c_x=64, 
                                                        k_neighbors=12, d_rate=2, 
                                                        depth_multiplier=1, 
                                                        is_training=self.is_training,
                                                        with_global=True, scope='xupconv_layer1')
            l2_xyz, l2_points = pointcnn_xupconv_module(l3_xyz,
                                                        l3_points, 
                                                        l2_xyz, 
                                                        l2_points,
                                                        c_fts_out=512, 
                                                        c_x=64, 
                                                        k_neighbors=12, d_rate=2, 
                                                        depth_multiplier=1, 
                                                        is_training=self.is_training,
                                                        with_global=True, scope='xupconv_layer2')
            l1_xyz, l1_points = pointcnn_xupconv_module(l2_xyz,
                                                        l2_points, 
                                                        l1_xyz, 
                                                        l1_points,
                                                        c_fts_out=256, 
                                                        c_x=64, 
                                                        k_neighbors=12, d_rate=1, 
                                                        depth_multiplier=1, 
                                                        is_training=self.is_training,
                                                        with_global=True, scope='xupconv_layer3')
            l0_xyz, l0_points = pointcnn_xupconv_module(l1_xyz,
                                                        l1_points, 
                                                        l0_xyz, 
                                                        l0_points,
                                                        c_fts_out=128, 
                                                        c_x=128, 
                                                        k_neighbors=8, d_rate=1, 
                                                        depth_multiplier=1, 
                                                        is_training=self.is_training,
                                                        with_global=True, scope='xupconv_layer4')

            # FC layers
            net = tf_util.conv1d(l0_points, 128, 1, 
                                 padding='VALID', is_training=self.is_training, 
                                 scope='conv1d_1', 
                                 activation_fn=tf.nn.elu, 
                                 norm_method='instance' )
            net = tf_util.conv1d(net, 64, 1, 
                                 padding='VALID', is_training=self.is_training, 
                                 scope='conv1d_2', 
                                 activation_fn=tf.nn.elu, 
                                 norm_method='instance')
            net = tf_util.conv1d(net, 3, 1, 
                                 padding='VALID', is_training=self.is_training, 
                                 activation_fn=tf.nn.tanh, 
                                 scope='output', 
                                 norm_method=None) # no normalization for the last output layer

            displacements = net * self.range_max
            translated_point_cloud = point_cloud + displacements

            # set reuse=True for next call
            self.reuse = True
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

            return translated_point_cloud

