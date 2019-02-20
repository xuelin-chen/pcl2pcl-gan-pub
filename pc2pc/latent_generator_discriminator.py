import os,sys

BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))

import tensorflow as tf
import numpy as np

class GeneratorLatentFromNoise:
    def __init__(self, name, fc_sizes=[128], latent_dim=128, activation_fn=tf.nn.relu, bn=True):
        self.name = name
        self.fc_sizes = fc_sizes.copy()
        self.latent_dim = latent_dim
        self.activation_fn = activation_fn
        self.bn = bn

        self.reuse = False
    
    def __call__(self, noise, is_training):
        '''
        input:
            noise: B x noise_dim
        return:
            latent: B x latent_dim
        '''
        with tf.variable_scope(self.name, reuse=self.reuse):
            layer = noise
            
            for fc_id, _ in enumerate(self.fc_sizes):
                layer = tf.layers.dense(layer,
                                        self.fc_sizes[fc_id],
                                        activation=self.activation_fn,
                                        name='fc_%d'%(fc_id))
                if self.bn and fc_id != 0:
                    layer = tf.layers.batch_normalization(layer, 
                                                          momentum=0.99,
                                                          training=is_training,
                                                          name='bn_%d'%(fc_id))
            
            layer = tf.layers.dense(layer,
                                    self.latent_dim,
                                    activation=None,
                                    name='fc_output')

            # set reuse=True for next call
            self.reuse = True
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

            return layer
    
    def __str__(self):
        res = ''
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        for tv in train_vars:
            res += tv.name + '\n'
        return res[:len(res)-2]

class DiscriminatorFromLatent:
    def __init__(self, name, fc_sizes=[256, 512], activation_fn=tf.nn.leaky_relu, bn=True):
        self.name = name
        self.fc_sizes = fc_sizes.copy()
        self.activation_fn = activation_fn
        self.bn = bn

        self.reuse = False

    def __call__(self, latent, is_training):
        '''
        input:
            latent: B x latent_dim
        return:
            B, each scalar for one latent code
        '''
        with tf.variable_scope(self.name, reuse=self.reuse):
            layer = latent
            for fc_id, _ in enumerate(self.fc_sizes):
                layer = tf.layers.dense(layer,
                                        self.fc_sizes[fc_id],
                                        activation=None,
                                        name='fc_%d'%(fc_id))
                
                if self.bn and fc_id != 0: # not for the first fc layer
                    layer = tf.layers.batch_normalization(layer, 
                                                          momentum=0.99,
                                                          training=is_training,
                                                          name='bn_%d'%(fc_id))
                
                if self.activation_fn is not None:
                    layer = self.activation_fn(layer, name='activation_fn_%d'%(fc_id))
            
            # fc: fc_size[-1] -> 1, no activation function
            layer = tf.layers.dense(layer,
                                    1,
                                    activation=None,
                                    name='output')
            
            # set reuse=True for next call
            self.reuse = True
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

            return layer
    
    def __str__(self):
        res = ''
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        for tv in train_vars:
            res += tv.name + '\n'
        return res[:len(res)-2]

class GeneratorLatentFromCloud:
    '''
    Point net generator: point cloud -> latent code.
    Consists of several fc layers, which is implemented by conv1d operation.
    Each point is lifted up to higher dimension *independently*.
    A feature-wise symmetry function (max pool) is applied at the end to obtain the final code
    '''
    def __init__(self, name, n_filters=[64, 128, 128, 256], filter_size=1, stride=1, activation_fn=tf.nn.relu, bn=True, latent_dim=128):
        self.name = name
        self.n_filters = n_filters.copy()
        self.n_filters.append(latent_dim)
        self.filter_size = 1
        self.stride = stride
        self.activation_fn = activation_fn
        self.bn = bn
        self.latent_dim = latent_dim

        self.reuse = False
    
    def __call__(self, point_cloud, is_training):
        '''
        input:
            point_cloud: BxNx3
        output: 
            latent_code: Bxlatent_code_dim
        '''
        with tf.variable_scope(self.name, reuse=self.reuse):
            layer = point_cloud
            for f_id, _ in enumerate(self.n_filters):
                layer = tf.layers.conv1d(layer, 
                                         self.n_filters[f_id], 
                                         self.filter_size, 
                                         self.stride, 
                                         padding='same', 
                                         activation=self.activation_fn,
                                         name='conv1d_%d'%(f_id))
                
                if self.bn:
                    layer = tf.layers.batch_normalization(layer, 
                                                          momentum=0.99,
                                                          training=is_training,
                                                          name='bn_%d'%(f_id))
                    
            layer = tf.reduce_max(layer, axis=1, name='max_pool')

            # set reuse=True for next call
            self.reuse = True
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

            return layer
    
    def __str__(self):
        res = ''
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        for tv in train_vars:
            res += tv.name + '\n'
        return res[:len(res)-2]

class GeneratorLatentFromLatent:
    def __init__(self, name, fc_sizes=[128], latent_dim=128, activation_fn=tf.nn.relu, bn=True):
        self.name = name
        self.fc_sizes = fc_sizes.copy()
        self.latent_dim = latent_dim
        self.activation_fn = activation_fn
        self.bn = bn

        self.reuse = False
    
    def __call__(self, noise, is_training):
        '''
        input:
            noise: B x noise_dim
        return:
            latent: B x latent_dim
        '''
        with tf.variable_scope(self.name, reuse=self.reuse):
            layer = noise
            
            for fc_id, _ in enumerate(self.fc_sizes):
                layer = tf.layers.dense(layer,
                                        self.fc_sizes[fc_id],
                                        activation=None,
                                        name='fc_%d'%(fc_id))
                if self.bn:
                    layer = tf.layers.batch_normalization(layer, 
                                                          momentum=0.99,
                                                          training=is_training,
                                                          name='bn_%d'%(fc_id))
                
                if self.activation_fn is not None:
                    layer = self.activation_fn(layer, name='activation_fn_%d'%(fc_id))
            
            # no activation, no bn
            layer = tf.layers.dense(layer,
                                    self.latent_dim,
                                    activation=None,
                                    name='fc_output')

            # set reuse=True for next call
            self.reuse = True
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

            return layer
    
    def __str__(self):
        res = ''
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        for tv in train_vars:
            res += tv.name + '\n'
        return res[:len(res)-2]

class GeneratorCloud2Cloud:
    def __init__(self, name, en_n_filters=[64, 128, 128, 256], en_filter_size=1, en_stride=1, en_activation_fn=tf.nn.relu, en_norm_mtd='instance_norm', en_latent_code_dim=128, de_fc_sizes=[256, 256], de_activation_fn=tf.nn.relu, de_norm_mtd='instance_norm', de_output_shape=[2048, 3]):
        self.name = name

        # 'encoder' paras
        self.en_n_filters = en_n_filters.copy()
        self.en_n_filters.append(en_latent_code_dim) # add last layer of code dim
        self.en_filter_size = en_filter_size
        self.en_stride = en_stride
        self.en_activation_fn = en_activation_fn
        self.en_norm_mtd = en_norm_mtd
        self.en_latent_code_dim = en_latent_code_dim

        # 'decoder' paras
        self.de_fc_sizes = de_fc_sizes.copy()
        self.de_activation_fn = de_activation_fn
        self.de_norm_mtd = de_norm_mtd
        self.de_output_shape = de_output_shape

        self.reuse = False
    
    def __call__(self, input_cloud, is_training):
        '''
        input:
            noise: B x N x 3
        return:
            latent: B x N x 3
        '''
        with tf.variable_scope(self.name, reuse=self.reuse):
            layer = input_cloud
            
            # going down
            for f_id, _ in enumerate(self.en_n_filters):
                layer = tf.layers.conv1d(layer, 
                                         self.en_n_filters[f_id], 
                                         self.en_filter_size, 
                                         self.en_stride, 
                                         padding='same', 
                                         activation=None,
                                         name='en_conv1d_%d'%(f_id))

                # normalization, if applicable
                if self.en_norm_mtd is not None:
                    if self.en_norm_mtd == 'batch_norm':
                        layer = tf.layers.batch_normalization(layer, momentum=0.99, training=is_training, name='en_bn_%d'%(f_id))
                    elif self.en_norm_mtd == 'instance_norm':
                        layer = tf.contrib.layers.instance_norm(layer, scope='en_in_%d'%(f_id))

                # non-linear activation, if applicable
                if self.en_activation_fn is not None:
                    layer = self.en_activation_fn(layer, name='en_activation_fn_%d'%(f_id))


            # bottleneck
            layer = tf.reduce_max(layer, axis=1, name='en_max_pool') 

            # going up
            for fc_id, _ in enumerate(self.de_fc_sizes):
                layer = tf.layers.dense(layer,
                                        self.de_fc_sizes[fc_id],
                                        activation=None,
                                        name='de_fc_%d'%(fc_id))

                # normalization, if applicable
                if self.de_norm_mtd is not None:
                    if self.de_norm_mtd == 'batch_norm':
                        layer = tf.layers.batch_normalization(layer, momentum=0.99, training=is_training, name='de_bn_%d'%(fc_id))
                    elif self.de_norm_mtd == 'instance_norm':
                        layer = tf.contrib.layers.instance_norm(layer, scope='de_in_%d'%(fc_id))

                # non-linear activation, if applicable
                if self.de_activation_fn is not None:
                    layer = self.de_activation_fn(layer, name='de_activation_fn_%d'%(fc_id))
            
            # last layer, no activation, no normalization
            layer = tf.layers.dense(layer,
                                    np.prod(self.de_output_shape),
                                    activation=None,
                                    name='de_fc_output')
            layer = tf.reshape(layer, [-1, self.de_output_shape[0], self.de_output_shape[1]], name='de_reshape')

            # set reuse=True for next call
            self.reuse = True
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

            return layer
    
    def __str__(self):
        res = ''
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        for tv in train_vars:
            res += tv.name + '\n'
        return res[:len(res)-2]
  
class DiscriminatorFromCloud:
    def __init__(self, name, n_filters=[64, 128, 128, 256], filter_size=1, stride=1, activation_fn=tf.nn.leaky_relu, norm_mtd='instance_norm', latent_code_dim=128):
        self.name = name
        self.n_filters = n_filters.copy()
        self.n_filters.append(latent_code_dim) # add last layer of code dim
        self.filter_size = filter_size
        self.stride = stride
        self.activation_fn = activation_fn
        self.norm_mtd = norm_mtd
        self.latent_code_dim = latent_code_dim

        self.reuse = False

    def __call__(self, input_cloud, is_training):
        '''
        input:
            latent: B x N x 3
        return:
            B, each scalar for one latent code
        '''
        with tf.variable_scope(self.name, reuse=self.reuse):
            layer = input_cloud
            for f_id, _ in enumerate(self.n_filters):
                layer = tf.layers.conv1d(layer, 
                                         self.n_filters[f_id], 
                                         self.filter_size, 
                                         self.stride, 
                                         padding='same', 
                                         activation=None,
                                         name='conv1d_%d'%(f_id))
                
                if f_id!=0 and self.norm_mtd is not None:
                    if self.norm_mtd == 'batch_norm':
                        layer = tf.layers.batch_normalization(layer, momentum=0.99, training=is_training, name='bn_%d'%(f_id))
                    elif self.norm_mtd == 'instance_norm':
                        layer = tf.contrib.layers.instance_norm(layer, scope='in_%d'%(f_id))
                
                if self.activation_fn is not None:
                    layer = self.activation_fn(layer, name='activation_fn_%d'%(f_id))
                    
            layer = tf.reduce_max(layer, axis=1, name='max_pool')

            # no activation, no norm
            layer = tf.layers.dense(layer,
                                    1,
                                    activation=None,
                                    name='output')
            
            # set reuse=True for next call
            self.reuse = True
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

            return layer
    
    def __str__(self):
        res = ''
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        for tv in train_vars:
            res += tv.name + '\n'
        return res[:len(res)-2]
