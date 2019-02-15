import os,sys
import numpy as np

import tensorflow as tf

BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'pointnet_utils'))
sys.path.append(os.path.join(BASE_DIR, 'pointcnn_utils'))

from pointcnn_discriminator_generator import DiscriminatorPointConv, GeneratorPointConv

sys.path.append(os.path.join(BASE_DIR, 'structural_losses_utils'))
from tf_nndistance import nn_distance
from tf_approxmatch import approx_match, match_cost

REAL_LABEL = 1.0
FAKE_LABEL = 0.0

import pickle

class PC2PCGAN:
    '''
    '''
    def __init__(self, para_config):
        self.para_config = para_config

        self.G = GeneratorPointConv('G', bn=False)

        self.D = DiscriminatorPointConv('D', bn=False)
                            
        self.input_noisy_cloud = tf.placeholder(tf.float32, shape=[para_config['batch_size'], para_config['point_cloud_shape'][0], para_config['point_cloud_shape'][1]])
        self.input_clean_cloud = tf.placeholder(tf.float32, shape=[para_config['batch_size'], para_config['point_cloud_shape'][0], para_config['point_cloud_shape'][1]])
        self.is_training = tf.placeholder(tf.bool, shape=())

        # used when evaluation only
        self.gt = tf.placeholder(tf.float32, shape=[para_config['batch_size'], para_config['point_cloud_shape'][0], para_config['point_cloud_shape'][1]])
        #self.eval_loss = tf.placeholder(tf.float32, shape=())

    def model(self):

        fake_clean_cloud = self.G(self.input_noisy_cloud, self.is_training)

        # for check
        eval_recon_loss = self._reconstruction_loss(fake_clean_cloud, self.gt)

        G_loss = self._generator_loss(self.D, fake_clean_cloud)
        back_recon_loss = self._reconstruction_loss(fake_clean_cloud, self.gt)
        #G_loss = G_2fool_loss + 10 * back_recon_loss
        #G_loss = G_2fool_loss

        D_fake_loss, D_real_loss, D_loss = self._discriminator_loss(self.D, fake_clean_cloud, self.input_clean_cloud)
        
        return G_loss, back_recon_loss, D_loss, D_fake_loss, D_real_loss, fake_clean_cloud, eval_recon_loss
    
    def _reconstruction_loss(self, recon, input):
        if self.para_config['loss'] == 'chamfer':
            cost_p1_p2, _, cost_p2_p1, _ = nn_distance(recon, input)
            loss = tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(cost_p2_p1)
        elif self.para_config['loss'] == 'emd':
            match = approx_match(recon, input)
            loss = match_cost(recon, input, match)
            loss = tf.reduce_mean(loss)
            loss = tf.div(loss, self.para_config['point_cloud_shape'][0]) # return point-wise loss
        elif self.para_config['loss'] == 'pairwise':
            loss = tf.reduce_mean(tf.squared_difference(recon, input))

        return loss

    def _generator_loss(self, D, fake_cloud):
        g_loss = tf.reduce_mean(tf.squared_difference(D(fake_cloud, self.is_training), REAL_LABEL))
        return g_loss
    
    def _discriminator_loss(self, D, fake_cloud, real_cloud):
        d_fake_loss = tf.reduce_mean(tf.squared_difference(D(fake_cloud, self.is_training), FAKE_LABEL))
        d_real_loss = tf.reduce_mean(tf.squared_difference(D(real_cloud, self.is_training), REAL_LABEL))
        d_loss = (d_fake_loss + d_real_loss) / 2.0
        return d_fake_loss, d_real_loss, d_loss
    
    def make_optimizer(self, loss, variables, lr, beta1, name='Adam'):
        """ Adam optimizer with learning rate 0.0001 for the first 100ksteps (~100 epochs)
            and a linearly decaying rate that goes to zero over the next 100k steps
        """
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = lr
        end_learning_rate = 0.0
        start_decay_step = 1000000
        decay_steps = 1000000
        beta1 = beta1
        learning_rate = (
            tf.where(
                    tf.greater_equal(global_step, start_decay_step),
                    tf.train.polynomial_decay(starter_learning_rate, global_step-start_decay_step,
                                                decay_steps, end_learning_rate,
                                                power=1.0),
                    starter_learning_rate
            )

        )
        tf.summary.scalar('learning_rate/{}'.format(name),learning_rate, collections=['train'])

        optimizer_here = tf.train.AdamOptimizer(learning_rate,beta1=beta1, name=name)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            learning_step = optimizer_here.minimize(loss, global_step=global_step, var_list=variables)
            return learning_step

    def optimize(self, g_loss, d_loss):
        
        G_optimizer = self.make_optimizer(g_loss, self.G.variables, name='Adam_G')
        D_optimizer = self.make_optimizer(d_loss, self.D.variables, name='Adam_D')

        return G_optimizer, D_optimizer
    
    def __str__(self):
        res = str(self.G) + '\n' + str(self.D)
        return res

class CyclePC2PCGAN:
    '''
    point cloud latent to point cloud latent GAN
    input: 
        noisy point cloud, encode it using pre-trained noisy point cloud AE,
        clean point cloud, encode it using pre-trained clean point cloud AE,
    '''
    def __init__(self, para_config):
        self.para_config = para_config
        self.G_paras = para_config['G_paras']
        self.D_paras = para_config['D_paras']

        self.G = GeneratorCloud2Cloud('G_N2C', 
                                      en_n_filters=self.G_paras['en_n_filters'].copy(), 
                                      en_filter_size=self.G_paras['en_filter_size'], 
                                      en_stride=self.G_paras['en_stride'], 
                                      en_activation_fn=self.G_paras['en_activation_fn'], 
                                      en_norm_mtd=self.G_paras['en_norm_mtd'], 
                                      en_latent_code_dim=self.G_paras['en_latent_code_dim'], 
                                      de_fc_sizes=self.G_paras['de_fc_sizes'], 
                                      de_activation_fn=self.G_paras['de_activation_fn'], 
                                      de_norm_mtd=self.G_paras['de_norm_mtd'], 
                                      de_output_shape=self.G_paras['de_output_shape'].copy())

        self.D_N = DiscriminatorFromCloud('D_N', 
                                        n_filters=self.D_paras['n_filters'].copy(), 
                                        filter_size=self.D_paras['filter_size'], 
                                        stride=self.D_paras['stride'],
                                        activation_fn=self.D_paras['activation_fn'], 
                                        norm_mtd=self.D_paras['norm_mtd'], 
                                        latent_code_dim=self.D_paras['latent_code_dim'])

        self.F = GeneratorCloud2Cloud('F_C2N', 
                                      en_n_filters=self.G_paras['en_n_filters'].copy(), 
                                      en_filter_size=self.G_paras['en_filter_size'], 
                                      en_stride=self.G_paras['en_stride'], 
                                      en_activation_fn=self.G_paras['en_activation_fn'], 
                                      en_norm_mtd=self.G_paras['en_norm_mtd'], 
                                      en_latent_code_dim=self.G_paras['en_latent_code_dim'], 
                                      de_fc_sizes=self.G_paras['de_fc_sizes'], 
                                      de_activation_fn=self.G_paras['de_activation_fn'], 
                                      de_norm_mtd=self.G_paras['de_norm_mtd'], 
                                      de_output_shape=self.G_paras['de_output_shape'].copy())
        
        self.D_C = DiscriminatorFromCloud('D_C', 
                                        n_filters=self.D_paras['n_filters'].copy(), 
                                        filter_size=self.D_paras['filter_size'], 
                                        stride=self.D_paras['stride'],
                                        activation_fn=self.D_paras['activation_fn'], 
                                        norm_mtd=self.D_paras['norm_mtd'], 
                                        latent_code_dim=self.D_paras['latent_code_dim'])
                            
        self.input_noisy_cloud = tf.placeholder(tf.float32, shape=[para_config['batch_size'], para_config['point_cloud_shape'][0], para_config['point_cloud_shape'][1]])
        self.input_clean_cloud = tf.placeholder(tf.float32, shape=[para_config['batch_size'], para_config['point_cloud_shape'][0], para_config['point_cloud_shape'][1]])
        self.is_training = tf.placeholder(tf.bool, shape=())

        # used when evaluation only
        #self.gt = tf.placeholder(tf.float32, shape=[para_config['batch_size'], para_config['point_cloud_shape'][0], para_config['point_cloud_shape'][1]])
        #self.eval_loss = tf.placeholder(tf.float32, shape=())

    def model(self):

        # noisy -> fake clean -> reconstructed noisy
        fake_clean_cloud = self.G(self.input_noisy_cloud, self.is_training)
        recon_noisy_cloud = self.F(fake_clean_cloud, self.is_training)

        # clean -> fake noisy -> reconstructed clean
        fake_noisy_cloud = self.F(self.input_clean_cloud, self.is_training)
        recon_clean_cloud = self.G(fake_noisy_cloud, self.is_training)

        cycle_loss = self._cycle_loss(self.input_noisy_cloud, recon_noisy_cloud, self.input_clean_cloud, recon_clean_cloud)

        # G loss
        G_gan_loss = self._generator_loss(self.D_C, fake_clean_cloud)
        G_loss = G_gan_loss + cycle_loss

        # F loss
        F_gan_loss = self._generator_loss(self.D_N, fake_noisy_cloud)
        F_loss = F_gan_loss + cycle_loss

        # D_N loss
        D_N_fake_loss, D_N_real_loss, D_N_loss = self._discriminator_loss(self.D_N, fake_noisy_cloud, self.input_noisy_cloud)

        # D_C loss
        D_C_fake_loss, D_C_real_loss, D_C_loss = self._discriminator_loss(self.D_C, fake_clean_cloud, self.input_clean_cloud)
        
        return cycle_loss, G_loss, F_loss, D_N_loss, D_N_fake_loss, D_N_real_loss, D_C_loss, D_C_fake_loss, D_C_real_loss, fake_noisy_cloud, fake_clean_cloud
    
    def _cycle_loss(self, noisy_cloud, noisy_cloud_recon, clean_cloud, clean_cloud_recon):
        recon_loss_noisy = self._reconstruction_loss(noisy_cloud_recon, noisy_cloud)
        recon_loss_clean = self._reconstruction_loss(clean_cloud_recon, clean_cloud)

        return 10 * recon_loss_noisy + 10 * recon_loss_clean

    def _reconstruction_loss(self, recon, input):
        if self.para_config['loss'] == 'chamfer':
            cost_p1_p2, _, cost_p2_p1, _ = nn_distance(recon, input)
            loss = tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(cost_p2_p1)
        elif self.para_config['loss'] == 'emd':
            match = approx_match(recon, input)
            loss = match_cost(recon, input, match)
            loss = tf.reduce_mean(loss)
            loss = tf.div(loss, self.para_config['point_cloud_shape'][0]) # return point-wise loss

        return loss

    def _generator_loss(self, D, fake_cloud):
        g_loss = tf.reduce_mean(tf.squared_difference(D(fake_cloud, self.is_training), REAL_LABEL))
        return g_loss
    
    def _discriminator_loss(self, D, fake_cloud, real_cloud):
        d_fake_loss = tf.reduce_mean(tf.squared_difference(D(fake_cloud, self.is_training), FAKE_LABEL))
        d_real_loss = tf.reduce_mean(tf.squared_difference(D(real_cloud, self.is_training), REAL_LABEL))
        d_loss = (d_fake_loss + d_real_loss) / 2.0
        return d_fake_loss, d_real_loss, d_loss
    
    def optimize(self, G_loss, F_loss, D_N_loss, D_C_loss):
        def make_optimizer(loss, variables, name='Adam'):
            """ Adam optimizer with learning rate 0.0001 for the first 100k steps (~100 epochs)
                and a linearly decaying rate that goes to zero over the next 100k steps
            """
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = self.para_config['lr']
            end_learning_rate = 0.0
            start_decay_step = 1000000
            decay_steps = 1000000
            beta1 = self.para_config['beta1']
            learning_rate = (
                tf.where(
                        tf.greater_equal(global_step, start_decay_step),
                        tf.train.polynomial_decay(starter_learning_rate, global_step-start_decay_step,
                                                    decay_steps, end_learning_rate,
                                                    power=1.0),
                        starter_learning_rate
                )

            )
            tf.summary.scalar('learning_rate/{}'.format(name), learning_rate, collections=['train'])

            optimizer_here = tf.train.AdamOptimizer(learning_rate, beta1=beta1, name=name)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                learning_step = optimizer_here.minimize(loss, global_step=global_step, var_list=variables)
                return learning_step

        G_optimizer = make_optimizer(G_loss, self.G.variables, name='Adam_G')
        F_optimizer = make_optimizer(F_loss, self.F.variables, name='Adam_F')
        D_N_optimizer = make_optimizer(D_N_loss, self.D_N.variables, name='Adam_D_N')
        D_C_optimizer = make_optimizer(D_C_loss, self.D_C.variables, name='Adam_D_C')

        return G_optimizer, F_optimizer, D_N_optimizer, D_C_optimizer
    
    def __str__(self):
        res = str(self.G) + '\n' + str(self.F) + '\n' + str(self.D_N) + '\n' + str(self.D_C)
        return res

