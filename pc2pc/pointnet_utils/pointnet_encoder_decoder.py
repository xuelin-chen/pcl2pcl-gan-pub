import os,sys

BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../../utils'))

import tensorflow as tf
import numpy as np

class EncoderPointnet:
    '''
    Point net encoder.
    Consists of several fc layers, which is implemented by conv1d operation.
    Each point is lifted up to higher dimension *independently*.
    A feature-wise symmetry function (max pool) is applied at the end to obtain the final code
    '''
    def __init__(self, name, n_filters=[64, 128, 128, 256], filter_size=1, stride=1, activation_fn=tf.nn.relu, bn=True, latent_code_dim=128):
        self.name = name
        self.n_filters = n_filters.copy()
        self.n_filters.append(latent_code_dim) # add last layer of code dim
        self.filter_size = filter_size
        self.stride = stride
        self.activation_fn = activation_fn
        self.bn = bn
        self.latent_code_dim = latent_code_dim

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

            self.reuse = True
            # trainable variables
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
            # all variables
            self.all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

            return layer
    
    def __call__reorder(self, point_cloud, is_training):
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
                                         activation=None,
                                         name='conv1d_%d'%(f_id))
                
                if self.bn:
                    layer = tf.layers.batch_normalization(layer, 
                                                          momentum=0.99,
                                                          training=is_training,
                                                          name='bn_%d'%(f_id))
                    
                if self.activation_fn is not None:
                    layer = self.activation_fn(layer, name='activation_fn_%d'%(f_id))
                    
            layer = tf.reduce_max(layer, axis=1, name='max_pool')

            self.reuse = True
            # trainable variables
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
            # all variables
            self.all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

            return layer

    def __str__(self):
        res = ''
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        for tv in train_vars:
            res += tv.name + '\n'
        return res[:len(res)-2]


class DecoderFC:
    '''
    Decoder that consists several fully connected layers
    No activation at the last layer.
    '''
    def __init__(self, name, fc_sizes=[256, 256], activation_fn=tf.nn.relu, bn=True, output_shape=[2048, 3]):
        self.name = name
        self.fc_sizes = fc_sizes.copy()
        self.activation_fn = activation_fn
        self.bn = bn
        self.output_shape = output_shape

        self.reuse = False

    def __call__(self, latent_code, is_training):
        '''
        input:
            latent_code: B x latent_code_dim
        output:
            layer: B x output_shape[0] x output_shape[1]
        '''
        with tf.variable_scope(self.name, reuse=self.reuse):
            layer = latent_code

            for fc_id, _ in enumerate(self.fc_sizes):
                layer = tf.layers.dense(layer,
                                        self.fc_sizes[fc_id],
                                        activation=self.activation_fn,
                                        name='fc_%d'%(fc_id))
                if self.bn:
                    layer = tf.layers.batch_normalization(layer, 
                                                          momentum=0.99,
                                                          training=is_training,
                                                          name='bn_%d'%(fc_id))
            
            # last layer, no activation, no bn
            layer = tf.layers.dense(layer,
                                    np.prod(self.output_shape),
                                    activation=None,
                                    name='fc_output')
        layer = tf.reshape(layer, [-1, self.output_shape[0], self.output_shape[1]], name='reshape')

        self.reuse = True
        # trainable variables
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        # all variables
        self.all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

        return layer

    def __call__reorder(self, latent_code, is_training):
        '''
        input:
            latent_code: B x latent_code_dim
        output:
            layer: B x output_shape[0] x output_shape[1]
        '''
        with tf.variable_scope(self.name, reuse=self.reuse):
            layer = latent_code

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
            
            # last layer, no activation, no bn
            layer = tf.layers.dense(layer,
                                    np.prod(self.output_shape),
                                    activation=None,
                                    name='fc_output')
        layer = tf.reshape(layer, [-1, self.output_shape[0], self.output_shape[1]], name='reshape')

        self.reuse = True
        # trainable variables
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        # all variables
        self.all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

        return layer

    def __str__(self):
        res = ''
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        for tv in train_vars:
            res += tv.name + '\n'
        return res[:len(res)-2]

if __name__ == "__main__":
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        encoder = EncoderPointnet('encoder')
        latent_code = encoder(inputs, tf.constant(True))
        print(encoder)
        print(latent_code)

        decoder = DecoderFC('decoder')
        point_cloud = decoder(latent_code, tf.constant(True))
        print(decoder)
        print(point_cloud)

