from pointnet_util import pointnet_sa_module, pointnet_fp_module
import tf_util

import tensorflow as tf

class Discriminator:
    def __init__(self, name, is_training, norm_method=None, use_sigmoid=False):
      self.name = name
      self.is_training = is_training
      self.reuse = False
      self.norm_method = norm_method    
      # NOTE: below is not used so far
      self.use_sigmoid = use_sigmoid    
    def __call__(self, point_cloud):
      """
       Discriminator to tell if the input point cloud is from domain B
          input:
              point_cloud: BxNx3
          output: 
              logits: B, only single scalar for each point cloud
      """
      with tf.variable_scope(self.name, reuse=self.reuse):
          #batch_size = point_cloud.get_shape()[0].value
          #num_point = point_cloud.get_shape()[1].value
          end_points = {}
          l0_xyz = point_cloud
          l0_points = None
          end_points['l0_xyz'] = l0_xyz 
          # Set abstraction layers
          l1_xyz, l1_points, _ = pointnet_sa_module(l0_xyz, l0_points, npoint=1024, radius=0.02, nsample=64, mlp=[32,32,64], mlp2=None, group_all=False, is_training=self.is_training, activation_fn=tf.nn.leaky_relu, norm_method=None, scope='layer1') # not using normalization for the first layer of D
          l2_xyz, l2_points, _ = pointnet_sa_module(l1_xyz, l1_points, npoint=512, radius=0.04, nsample=48, mlp=[64,64,128], mlp2=None, group_all=False, is_training=self.is_training, activation_fn=tf.nn.leaky_relu, norm_method=self.norm_method, scope='layer2')
          l3_xyz, l3_points, _ = pointnet_sa_module(l2_xyz, l2_points, npoint=256, radius=0.06, nsample=32, mlp=[128,128,256], mlp2=None, group_all=False, is_training=self.is_training, activation_fn=tf.nn.leaky_relu, norm_method=self.norm_method, scope='layer3')
          l4_xyz, l4_points, _ = pointnet_sa_module(l3_xyz, l3_points, npoint=128, radius=0.08, nsample=16, mlp=[256,256,512], mlp2=None, group_all=False, is_training=self.is_training, activation_fn=tf.nn.leaky_relu, norm_method=self.norm_method, scope='layer4')
          l5_xyz, l5_points, _ = pointnet_sa_module(l4_xyz, l4_points, npoint=64, radius=0.10, nsample=8, mlp=[512,512,1024], mlp2=None, group_all=False, is_training=self.is_training, activation_fn=tf.nn.leaky_relu, norm_method=self.norm_method, scope='layer5')   
          # conv to obtain multiple regions with posibility, not using norm here
          # 64 x 1
          output = tf_util.conv1d(l5_points, 1, 1, padding='VALID', is_training=self.is_training, scope='output', activation_fn=None, norm_method=None )    
          # set reuse=True for next call
          self.reuse = True
          self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name) 
          return output

import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'pointcnn_utils'))
from pointcnn_util import pointcnn_xconv_module
class DiscriminatorCNN:
    def __init__(self, name, is_training, norm_method=None, use_sigmoid=False):
      self.name = name
      self.is_training = is_training
      self.reuse = False
      self.norm_method = norm_method    
      # NOTE: below is not used so far
      self.use_sigmoid = use_sigmoid    
    def __call__(self, point_cloud):
        """
         Discriminator to tell if the input point cloud is from domain B
            input:
                point_cloud: BxNx3
            output: 
                logits: B, only single scalar for each point cloud
        """
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
                                                      c_fts_out=256, 
                                                      c_x=128, 
                                                      k_neighbors=8, d_rate=2, 
                                                      depth_multiplier=4, 
                                                      is_training=self.is_training,
                                                      with_global=True, scope='xconv_layer1',
                                                      norm_mthd=None) # not using normalization for the first layer of D
            l2_xyz, l2_points = pointcnn_xconv_module(l1_xyz,
                                                      l1_points, 
                                                      npoint=786, 
                                                      c_fts_out=256, 
                                                      c_x=64, 
                                                      k_neighbors=12, d_rate=2, 
                                                      depth_multiplier=1, 
                                                      is_training=self.is_training,
                                                      with_global=True, scope='xconv_layer2') 
            l3_xyz, l3_points = pointcnn_xconv_module(l2_xyz,
                                                      l2_points, 
                                                      npoint=384, 
                                                      c_fts_out=512, 
                                                      c_x=64, 
                                                      k_neighbors=16, d_rate=2, 
                                                      depth_multiplier=2, 
                                                      is_training=self.is_training,
                                                      with_global=True, scope='xconv_layer3') 
            l4_xyz, l4_points = pointcnn_xconv_module(l3_xyz,
                                                      l3_points, 
                                                      npoint=128, 
                                                      c_fts_out=1024, 
                                                      c_x=64, 
                                                      k_neighbors=16, d_rate=2, 
                                                      depth_multiplier=2, 
                                                      is_training=self.is_training,
                                                      with_global=True, scope='xconv_layer4') 
            # 64 x 1
            _,      output = pointcnn_xconv_module(l4_xyz,
                                                   l4_points, 
                                                   npoint=64, 
                                                   c_fts_out=1, 
                                                   c_x=64, 
                                                   k_neighbors=16, d_rate=1, 
                                                   depth_multiplier=2, 
                                                   is_training=self.is_training,
                                                   with_global=True, scope='xconv_output',
                                                   norm_mthd=None, active_fn=None)
            # set reuse=True for next call
            self.reuse = True
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name) 
            return output

