"""
    PointNet++ Model for point clouds classification
"""

import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import random

import pc_util
from discriminator import Discriminator, DiscriminatorCNN
from generator import Generator, GeneratorCNN

REAL_LABEL = 1.0

class CycleGAN3D:
    def __init__(self, batch_size=1, npoint=2048, data_channel=3, use_lsgan=True, norm_method='instance', lambda1=10, lambda2=10, learning_rate=2e-4, beta1=0.5):
        self.batch_size = batch_size
        self.npoint = npoint
        self.data_channel = data_channel
        self.use_lsgan = use_lsgan
        self.norm_method = norm_method
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.learning_rate = learning_rate
        self.beta1 = beta1

        self.is_training = tf.placeholder_with_default(True, shape=[], name='is_training')

        self.G = GeneratorCNN('G', self.is_training, norm_method=self.norm_method)
        self.D_Y = DiscriminatorCNN('D_Y', self.is_training, norm_method=self.norm_method)
        self.F = GeneratorCNN('F', self.is_training, norm_method=self.norm_method)
        self.D_X = DiscriminatorCNN('D_X', self.is_training, norm_method=self.norm_method)

        self.fake_x_prev = tf.placeholder(tf.float32, shape=[batch_size, npoint, data_channel]) # fake x from previous
        self.fake_y_prev = tf.placeholder(tf.float32, shape=[batch_size, npoint, data_channel]) # fake y from previous

        self.x = tf.placeholder(tf.float32, shape=[batch_size, npoint, data_channel]) # x input
        self.y = tf.placeholder(tf.float32, shape=[batch_size, npoint, data_channel]) # y input

    def model(self):
        
        cycle_loss = self._cycle_consistency_loss(self.G, self.F, self.x, self.y)

        # X -> Y
        fake_y = self.G(self.x)
        G_gan_loss = self._generator_loss(self.D_Y, fake_y, use_lsgan=self.use_lsgan)
        G_loss = G_gan_loss + cycle_loss
        D_Y_loss, D_Y_fake_loss, D_Y_real_loss = self._discriminator_loss(self.D_Y, self.y, self.fake_y_prev, use_lsgan=self.use_lsgan)

        # Y -> X
        fake_x = self.F(self.y)
        F_gan_loss = self._generator_loss(self.D_X, fake_x, use_lsgan=self.use_lsgan)
        F_loss = F_gan_loss + cycle_loss
        D_X_loss, D_X_fake_loss, D_X_real_loss = self._discriminator_loss(self.D_X, self.x, self.fake_x_prev, use_lsgan=self.use_lsgan)

        # loss visualization
        tf.summary.scalar('loss/G', G_gan_loss)
        tf.summary.scalar('loss/D_Y', D_Y_loss)
        tf.summary.scalar('loss/F', F_gan_loss)
        tf.summary.scalar('loss/D_X', D_X_loss)
        tf.summary.scalar('loss/cycle', cycle_loss)
        # gradient visualization
        '''
        grads_op = tf.gradients(G_gan_loss, self.G.variables)
        for  grad in (grads_op):
            tf.summary.histogram('gradients/{}'.format(grad.name), grad)
            g_mean = tf.reduce_mean(tf.abs(grad))
            tf.summary.scalar('gradients_mean/{}'.format(grad.name), g_mean)
            g_norm = tf.norm(grad)
            tf.summary.scalar('gradients_norm/{}'.format(grad.name), g_norm)
        '''

        return G_loss, D_Y_loss, F_loss, D_X_loss, \
               cycle_loss, D_Y_fake_loss, D_Y_real_loss, D_X_fake_loss, D_X_real_loss, \
               fake_y, fake_x
    
    def _cycle_consistency_loss(self, G, F, x, y):
        x_recon = F(G(x))
        forward_loss = tf.reduce_mean(tf.abs(x_recon - x))
        y_recon = G(F(y))
        backward_loss = tf.reduce_mean(tf.abs(y_recon - y))
        loss = self.lambda1 * forward_loss + self.lambda2 * backward_loss
        return loss
    
    def _generator_loss(self, D, fake_y, use_lsgan=True):
        if use_lsgan:
            # use mean squared error
            loss = tf.reduce_mean(tf.squared_difference(D(fake_y), REAL_LABEL))
        return loss
    
    def _discriminator_loss(self, D, y, fake_y, use_lsgan=True):
        if use_lsgan:
            # use mean squared error
            error_real = tf.reduce_mean(tf.squared_difference(D(y), REAL_LABEL))
            error_fake = tf.reduce_mean(tf.square(D(fake_y)))
        loss = (error_fake + error_real) / 2
        return loss, error_fake, error_real

    def optimize(self, G_loss, D_Y_loss, F_loss, D_X_loss):
        def make_optimizer(loss, variables, name='Adam'):
            """ Adam optimizer with learning rate 0.0002 for the first 100k steps (~100 epochs)
                and a linearly decaying rate that goes to zero over the next 100k steps
            """
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = self.learning_rate
            end_learning_rate = 0.0
            start_decay_step = 100000
            decay_steps = 100000
            beta1 = self.beta1
            learning_rate = (
                tf.where(
                        tf.greater_equal(global_step, start_decay_step),
                        tf.train.polynomial_decay(starter_learning_rate, global_step-start_decay_step,
                                                    decay_steps, end_learning_rate,
                                                    power=1.0),
                        starter_learning_rate
                )

            )
            tf.summary.scalar('learning_rate/{}'.format(name), learning_rate)

            learning_step = (
                tf.train.AdamOptimizer(learning_rate, beta1=beta1, name=name)
                        .minimize(loss, global_step=global_step, var_list=variables)
            )
            return learning_step

        G_optimizer = make_optimizer(G_loss, self.G.variables, name='Adam_G')
        D_Y_optimizer = make_optimizer(D_Y_loss, self.D_Y.variables, name='Adam_D_Y')
        F_optimizer =  make_optimizer(F_loss, self.F.variables, name='Adam_F')
        D_X_optimizer = make_optimizer(D_X_loss, self.D_X.variables, name='Adam_D_X')

        with tf.control_dependencies([G_optimizer, D_Y_optimizer, F_optimizer, D_X_optimizer]):
            return tf.no_op(name='optimizers')

class PointCloudPool:
    '''
    History of generated point clouds
    NOTE: each element in point_clouds and point_clouds_ori is in batch
    '''
    def __init__(self, pool_name, pool_size=50):
        self.pool_size = pool_size
        self.pool_name = pool_name
        self.point_clouds = []
        self.point_clouds_ori = [] # for original point clouds used to generate fake point clouds
    
    def query(self, point_cloud, point_cloud_ori):
        if self.pool_size == 0:
            return point_cloud

        if len(self.point_clouds) < self.pool_size:
            self.point_clouds.append(point_cloud)
            self.point_clouds_ori.append(point_cloud)
            return point_cloud
        else:
            p = random.random()
            if p > 0.5:
                # use olde image
                random_id = random.randrange(0, self.pool_size)
                tmp = self.point_clouds[random_id].copy()
                self.point_clouds[random_id] = point_cloud.copy()
                self.point_clouds_ori[random_id] = point_cloud_ori.copy()
                return tmp
            else:
                return point_cloud
        
    def dump2disk(self, dir_name):

        for batch_idx, pc_batch_fake in enumerate(self.point_clouds):
            for pc_idx, pc_fake in enumerate(pc_batch_fake):
                pc_fake_fielname = os.path.join(dir_name, str(batch_idx)+'_'+str(pc_idx)+'_'+self.pool_name+'.ply')
                pc_util.write_ply(pc_fake, pc_fake_fielname)

            pc_batch_ori = self.point_clouds_ori[batch_idx]
            for pc_idx, pc_ori in enumerate(pc_batch_ori):
                pc_ori_filename = os.path.join(dir_name, str(batch_idx)+'_'+str(pc_idx)+'_'+self.pool_name+'_ori.ply')
                pc_util.write_ply(pc_ori, pc_ori_filename)

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        output, _ = get_model(inputs, tf.constant(True))
        print(output)
