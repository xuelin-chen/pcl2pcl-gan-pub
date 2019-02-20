import os,sys

BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../../utils'))

import tensorflow as tf
import numpy as np

from pointcnn_util import point_conv_module, point_deconv_module
import pointfly as pf

class DiscriminatorPointConv:
    def __init__(self, name, sorting_method='cxyz', activation_fn=tf.nn.leaky_relu, bn=True):
        self.name = name

        self.sorting_method = sorting_method
        self.activation_fn = activation_fn
        self.bn = bn

        self.reuse = False

    def __call__(self, point_cloud, is_training):
        '''
        input:
            point_cloud: B x N x 3
        return:
            B, each scalar for one point cloud
        '''
        with tf.variable_scope(self.name, reuse=self.reuse):
            batch_size = point_cloud.shape[0]
            npoint = point_cloud.shape[1]
            l0_xyz = point_cloud
            l0_points = None

            sort_mtd= self.sorting_method

            # Set Abstraction layers
            '''
            # point cloud with 2048 points setting
            l1_xyz, l1_points, l1_indices = point_conv_module(l0_xyz, l0_points, npoint=512, c_fts_out=128, k_neighbors=64, d_rate=1, is_training=is_training,  sorting_method=sort_mtd,activation=self.activation_fn,bn=self.bn,scope='conv_layer_1', center_patch=False)
            l2_xyz, l2_points, l2_indices = point_conv_module(l1_xyz, l1_points, npoint=128, c_fts_out=256, k_neighbors=32, d_rate=1, is_training=is_training,  sorting_method=sort_mtd,activation=self.activation_fn,bn=self.bn,scope='conv_layer_2', center_patch=False)
            l3_xyz, l3_points, l3_indices = point_conv_module(l2_xyz, l2_points, npoint=64,  c_fts_out=512, k_neighbors=8,  d_rate=1, is_training=is_training,  sorting_method=sort_mtd,activation=self.activation_fn,bn=self.bn,scope='conv_layer_3', center_patch=False)
            '''
            # point cloud with 8192 points setting
            l1_xyz, l1_points, l1_indices = point_conv_module(l0_xyz, l0_points, npoint=1024, c_fts_out=64,  k_neighbors=16, d_rate=1, is_training=is_training,  sorting_method=sort_mtd,activation=self.activation_fn,bn=self.bn,scope='conv_layer_1', center_patch=False)
            l2_xyz, l2_points, l2_indices = point_conv_module(l1_xyz, l1_points, npoint=1024, c_fts_out=64,  k_neighbors=1,  d_rate=1, is_training=is_training,  sorting_method=sort_mtd,activation=self.activation_fn,bn=self.bn,scope='conv_layer_2', center_patch=False)
            l3_xyz, l3_points, l3_indices = point_conv_module(l2_xyz, l2_points, npoint=512,  c_fts_out=128, k_neighbors=8,  d_rate=1, is_training=is_training,  sorting_method=sort_mtd,activation=self.activation_fn,bn=self.bn,scope='conv_layer_3', center_patch=False)
            l4_xyz, l4_points, l4_indices = point_conv_module(l3_xyz, l3_points, npoint=128,  c_fts_out=256, k_neighbors=4,  d_rate=1, is_training=is_training,  sorting_method=sort_mtd,activation=self.activation_fn,bn=self.bn,scope='conv_layer_4', center_patch=False)
            l5_xyz, l5_points, l5_indices = point_conv_module(l4_xyz, l4_points, npoint=64,   c_fts_out=512, k_neighbors=2,  d_rate=1, is_training=is_training,  sorting_method=sort_mtd,activation=self.activation_fn,bn=self.bn,scope='conv_layer_5', center_patch=False)
            
            # from bs x 24 x 512 -> bs x 24 x 1
            # no activation
            '''
            output = tf.layers.dense(l3_points,
                                        1,
                                        activation=None,
                                        name='output')
            '''
            output = pf.conv1d(l5_points, 1, 
                               is_training=is_training, 
                               name='output', 
                               kernel_size=1, strides=1, 
                               with_bn=self.bn, activation=None)
            
            # bs x 24
            output = tf.reshape(output, (batch_size, -1))
            
            # set reuse=True for next call
            self.reuse = True
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

            return output
    
    def __str__(self):
        res = ''
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        for tv in train_vars:
            res += tv.name + '\n'
        return res[:len(res)-2]

class GeneratorPointConv:
    def __init__(self, name, sorting_method='cxyz', activation_fn=tf.nn.relu, bn=True):
        self.name = name

        self.sorting_method = sorting_method
        self.activation_fn = activation_fn
        self.bn = bn

        self.bn1 = self.bn2 = bn

        self.reuse = False

    def __call__(self, point_cloud, is_training):
        '''
        input:
            point_cloud: B x N x 3
        return:
            B, each scalar for one point cloud
        '''
        with tf.variable_scope(self.name, reuse=self.reuse):
            batch_size = point_cloud.shape[0]
            npoint = point_cloud.shape[1]
            l0_xyz = point_cloud
            l0_points = None

            sort_mtd= self.sorting_method

            
            # 2048
            # Set Abstraction layers
            l1_xyz, l1_points, l1_indices = point_conv_module(l0_xyz, l0_points, npoint=512, c_fts_out=128, k_neighbors=64, d_rate=1, is_training=is_training,  sorting_method=sort_mtd,activation=self.activation_fn,bn=self.bn1,scope='conv_layer_1', center_patch=False)
            l2_xyz, l2_points, l2_indices = point_conv_module(l1_xyz, l1_points, npoint=128, c_fts_out=256, k_neighbors=32, d_rate=1, is_training=is_training,  sorting_method=sort_mtd,activation=self.activation_fn,bn=self.bn1,scope='conv_layer_2', center_patch=False)
            l3_xyz, l3_points, l3_indices = point_conv_module(l2_xyz, l2_points, npoint=24,  c_fts_out=512, k_neighbors=8,  d_rate=1, is_training=is_training,  sorting_method=sort_mtd,activation=self.activation_fn,bn=self.bn1,scope='conv_layer_3', center_patch=False)
            
            # from bs x 24 x 512 
            # feature propagation
            l2_xyz, l2_points = point_deconv_module(l3_xyz, l3_points, l2_xyz, l2_points, l3_indices, c_fts_out=256, k_neighbors=8,  d_rate=1, is_training=is_training, activation=self.activation_fn,bn=self.bn2, center_patch=False,scope='deconv_layer_1')
            l1_xyz, l1_points = point_deconv_module(l2_xyz, l2_points, l1_xyz, l1_points, l2_indices, c_fts_out=128, k_neighbors=32, d_rate=1, is_training=is_training, activation=self.activation_fn,bn=self.bn2, center_patch=False,scope='deconv_layer_2')
            l0_xyz, l0_points = point_deconv_module(l1_xyz, l1_points, l0_xyz, l0_points, l1_indices, c_fts_out=128, k_neighbors=64, d_rate=1, is_training=is_training, activation=self.activation_fn,bn=self.bn2, center_patch=False,scope='deconv_layer_3')
            # bs x npoint x 128
            '''

            # Set Abstraction layers
            # 8192
            l1_xyz, l1_points, l1_indices = point_conv_module(l0_xyz, l0_points, npoint=1024, c_fts_out=64,  k_neighbors=16, d_rate=1, is_training=is_training,  sorting_method=sort_mtd,activation=self.activation_fn,bn=self.bn1,scope='conv_layer_1', center_patch=False)
            l2_xyz, l2_points, l2_indices = point_conv_module(l1_xyz, l1_points, npoint=1024, c_fts_out=64,  k_neighbors=1,  d_rate=1, is_training=is_training,  sorting_method=sort_mtd,activation=self.activation_fn,bn=self.bn1,scope='conv_layer_2', center_patch=False)
            l3_xyz, l3_points, l3_indices = point_conv_module(l2_xyz, l2_points, npoint=512,  c_fts_out=128, k_neighbors=8,  d_rate=1, is_training=is_training,  sorting_method=sort_mtd,activation=self.activation_fn,bn=self.bn1,scope='conv_layer_3', center_patch=False)
            l4_xyz, l4_points, l4_indices = point_conv_module(l3_xyz, l3_points, npoint=128,  c_fts_out=256, k_neighbors=4,  d_rate=1, is_training=is_training,  sorting_method=sort_mtd,activation=self.activation_fn,bn=self.bn1,scope='conv_layer_4', center_patch=False)
            l5_xyz, l5_points, l5_indices = point_conv_module(l4_xyz, l4_points, npoint=64,   c_fts_out=512, k_neighbors=2,  d_rate=1, is_training=is_training,  sorting_method=sort_mtd,activation=self.activation_fn,bn=self.bn1,scope='conv_layer_5', center_patch=False)
            # feature prop
            l4_xyz, l4_points = point_deconv_module(l5_xyz, l5_points, l4_xyz, l4_points, l5_indices, c_fts_out=256, k_neighbors=2,  d_rate=1, is_training=is_training, activation=self.activation_fn,bn=self.bn2, center_patch=False,scope='deconv_layer_1')
            l3_xyz, l3_points = point_deconv_module(l4_xyz, l4_points, l3_xyz, l3_points, l4_indices, c_fts_out=128, k_neighbors=4,  d_rate=1, is_training=is_training, activation=self.activation_fn,bn=self.bn2, center_patch=False,scope='deconv_layer_2')
            l2_xyz, l2_points = point_deconv_module(l3_xyz, l3_points, l2_xyz, l2_points, l3_indices, c_fts_out=64,  k_neighbors=8,  d_rate=1, is_training=is_training, activation=self.activation_fn,bn=self.bn2, center_patch=False,scope='deconv_layer_3')
            l1_xyz, l1_points = point_deconv_module(l2_xyz, l2_points, l1_xyz, l1_points, l2_indices, c_fts_out=64,  k_neighbors=1,  d_rate=1, is_training=is_training, activation=self.activation_fn,bn=self.bn2, center_patch=False,scope='deconv_layer_4')
            l0_xyz, l0_points = point_deconv_module(l1_xyz, l1_points, l0_xyz, l0_points, l1_indices, c_fts_out=64,  k_neighbors=16, d_rate=1, is_training=is_training, activation=self.activation_fn,bn=self.bn2, center_patch=False,scope='deconv_layer_5')
            '''

            output = l0_points

            # bs x 2048 x 128
            # get bsx 2048 x 3
            #output = tf.layers.dense(l0_points, 3, activation=tf.nn.tanh, name='replacements')
            #l4_points = tf.tile(l4_points, [1,point_cloud.shape[1],1]) # Bx1x1024 -> Bx npoint x 1024
            #fts_final = tf.concat([l4_points, l0_points], axis=-1)
            output = pf.conv1d(output, 3, 
                               is_training=is_training, 
                               name='output', 
                               kernel_size=1, strides=1, 
                               with_bn=False, activation=tf.nn.tanh)

            output = point_cloud + output # use replacements

            # set reuse=True for next call
            self.reuse = True
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

            return output
    
    def __str__(self):
        res = ''
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        for tv in train_vars:
            res += tv.name + '\n'
        return res[:len(res)-2]


if __name__ == '__main__':
    pts = np.zeros([32,2048,3])

    pts_pl = tf.placeholder(tf.float32, shape=(32,2048,3))
    is_training_pl = tf.placeholder(tf.bool, shape=())

    D = DiscriminatorPointConv('D')
    G = GeneratorPointConv('G')

    d_val = D(pts_pl, is_training_pl)
    f_val = G(pts_pl, is_training_pl)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        d, g = sess.run([d_val, f_val], feed_dict={pts_pl: pts, is_training_pl: True})
        print(d.shape, g.shape)


