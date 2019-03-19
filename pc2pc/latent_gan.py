import os,sys
import numpy as np

import tensorflow as tf

BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'pointnet_utils'))
from pointnet_encoder_decoder import EncoderPointnet, DecoderFC
from latent_generator_discriminator import GeneratorLatentFromNoise, GeneratorLatentFromLatent, DiscriminatorFromLatent, GeneratorLatentFromCloud, GeneratorCloud2Cloud, DiscriminatorFromCloud

sys.path.append(os.path.join(BASE_DIR, 'structural_losses_utils'))
from tf_nndistance import nn_distance
from tf_approxmatch import approx_match, match_cost
from tf_hausdorff_distance import directed_hausdorff

default_para_config = {
    'exp_name': 'latent_gan',
    'ltcode_pkl_filename': '',

    'batch_size': 50,
    'lr': 0.0001,
    'beta1': 0.5,
    'training_loops': 1000000,
    'k': 5, # train k times for D each loop when training
    'output_interval': 50, # unit in batch

    'noise_dim': 128,
    'noise_mu': 0.0,
    'noise_sigma': 0.2,
    'latent_dim': 128,

    # G paras
    'g_fc_sizes': [128],
    'g_activation_fn': tf.nn.relu,
    'g_bn': True,

    #D paras
    'd_fc_sizes': [256, 512],
    'd_activation_fn': tf.nn.leaky_relu,
    'd_bn': True,
}

REAL_LABEL = 1.0
FAKE_LABEL = 0.0

class PCL2PCLGAN:
    '''
    point cloud latent to point cloud latent GAN
    input: 
        noisy point cloud, encode it using pre-trained noisy point cloud AE,
        clean point cloud, encode it using pre-trained clean point cloud AE,
    '''
    def __init__(self, para_config, para_config_ae):
        self.para_config = para_config
        self.para_config_ae = para_config_ae

        self.G = GeneratorLatentFromLatent('G', fc_sizes=para_config['g_fc_sizes'].copy(), 
                                               latent_dim=para_config['latent_dim'],
                                               activation_fn=para_config['g_activation_fn'],
                                               bn=para_config['g_bn'])

        self.D = DiscriminatorFromLatent('D', fc_sizes=para_config['d_fc_sizes'].copy(),
                                              activation_fn=para_config['d_activation_fn'],
                                              bn=para_config['d_bn'])

        # loaded pre-trained noisy cloud encoder
        self.noisy_encoder = EncoderPointnet('noisy_Encoder', 
                                              n_filters=para_config_ae['n_filters'].copy(),
                                              filter_size=para_config_ae['filter_size'],
                                              stride=para_config_ae['stride'],
                                              activation_fn=para_config_ae['activation_fn'],
                                              bn=para_config_ae['encoder_bn'],
                                              latent_code_dim=para_config_ae['latent_code_dim'])
        
        # loaded pre-trained clean cloud decoder
        self.clean_decoder = DecoderFC('clean_Decoder',
                                        fc_sizes=para_config_ae['fc_sizes'].copy(),
                                        activation_fn=para_config_ae['activation_fn'],
                                        bn=para_config_ae['decoder_bn'],
                                        output_shape=para_config_ae['point_cloud_shape'].copy())
        # loaded pre-trained clean cloud encoder
        self.clean_encoder = EncoderPointnet('clean_Encoder', 
                                              n_filters=para_config_ae['n_filters'].copy(),
                                              filter_size=para_config_ae['filter_size'],
                                              stride=para_config_ae['stride'],
                                              activation_fn=para_config_ae['activation_fn'],
                                              bn=para_config_ae['encoder_bn'],
                                              latent_code_dim=para_config_ae['latent_code_dim'])
                            
        self.input_noisy_cloud = tf.placeholder(tf.float32, shape=[para_config['batch_size'], para_config['point_cloud_shape'][0], para_config['point_cloud_shape'][1]])
        self.input_clean_cloud = tf.placeholder(tf.float32, shape=[para_config['batch_size'], para_config['point_cloud_shape'][0], para_config['point_cloud_shape'][1]])
        self.is_training = tf.placeholder(tf.bool, shape=())
        
        # stored intermediate stuff
        self.noisy_code = tf.placeholder(tf.float32, shape=[para_config['batch_size'], para_config['latent_dim']])
        self.fake_code = tf.placeholder(tf.float32, shape=[para_config['batch_size'], para_config['latent_dim']])
        self.real_code = tf.placeholder(tf.float32, shape=[para_config['batch_size'], para_config['latent_dim']])

        self.gt = tf.placeholder(tf.float32, shape=[para_config['batch_size'], para_config['point_cloud_shape'][0], para_config['point_cloud_shape'][1]])

    def model(self):

        self.noisy_code = self.noisy_encoder(self.input_noisy_cloud, tf.constant(False, shape=()))

        self.fake_code = self.G(self.noisy_code, self.is_training)

        fake_clean_reconstr = self.clean_decoder(self.fake_code, tf.constant(False, shape=()))

        reconstr_loss = self._reconstruction_loss(fake_clean_reconstr, self.input_noisy_cloud) # comput loss against the input
        G_tofool_loss = self._generator_loss(self.D, self.fake_code)
        G_loss = G_tofool_loss + self.para_config['lambda'] * reconstr_loss

        self.real_code = self.clean_encoder(self.input_clean_cloud, tf.constant(False, shape=()))

        D_fake_loss, D_real_loss, D_loss = self._discriminator_loss(self.D, self.fake_code, self.real_code)

        # eval only loss
        eval_loss = self._reconstruction_loss(fake_clean_reconstr, self.gt, eval_loss=self.para_config['eval_loss'])

        return G_loss, G_tofool_loss, reconstr_loss, D_loss, D_fake_loss, D_real_loss, fake_clean_reconstr, eval_loss
    
    def model_wGT(self):

        self.noisy_code = self.noisy_encoder(self.input_noisy_cloud, tf.constant(False, shape=()))

        self.fake_code = self.G(self.noisy_code, self.is_training)

        fake_clean_reconstr = self.clean_decoder(self.fake_code, tf.constant(False, shape=()))

        reconstr_loss = self._reconstruction_loss(fake_clean_reconstr, self.gt) # compute loss against gt
        G_tofool_loss = self._generator_loss(self.D, self.fake_code)
        G_loss = G_tofool_loss + self.para_config['lambda'] * reconstr_loss

        self.real_code = self.clean_encoder(self.input_clean_cloud, tf.constant(False, shape=()))

        D_fake_loss, D_real_loss, D_loss = self._discriminator_loss(self.D, self.fake_code, self.real_code)

        # eval only loss
        eval_loss = self._reconstruction_loss(fake_clean_reconstr, self.gt, eval_loss=self.para_config['eval_loss'])

        return G_loss, G_tofool_loss, reconstr_loss, D_loss, D_fake_loss, D_real_loss, fake_clean_reconstr, eval_loss
    
    def model_wGT_noGAN(self):

        self.noisy_code = self.noisy_encoder(self.input_noisy_cloud, tf.constant(False, shape=()))

        self.fake_code = self.G(self.noisy_code, self.is_training)

        fake_clean_reconstr = self.clean_decoder(self.fake_code, tf.constant(False, shape=()))

        reconstr_loss = self._reconstruction_loss(fake_clean_reconstr, self.gt) # compute loss against gt
        G_tofool_loss = self._generator_loss(self.D, self.fake_code)
        #G_loss = G_tofool_loss + self.para_config['lambda'] * reconstr_loss
        G_loss = self.para_config['lambda'] * reconstr_loss # no GAN loss, only EMD loss against GT

        self.real_code = self.clean_encoder(self.input_clean_cloud, tf.constant(False, shape=()))

        D_fake_loss, D_real_loss, D_loss = self._discriminator_loss(self.D, self.fake_code, self.real_code)

        # eval only loss
        eval_loss = self._reconstruction_loss(fake_clean_reconstr, self.gt, eval_loss=self.para_config['eval_loss'])

        return G_loss, G_tofool_loss, reconstr_loss, D_loss, D_fake_loss, D_real_loss, fake_clean_reconstr, eval_loss

    def model_noReconLoss(self):

        self.noisy_code = self.noisy_encoder(self.input_noisy_cloud, tf.constant(False, shape=()))

        self.fake_code = self.G(self.noisy_code, self.is_training)

        fake_clean_reconstr = self.clean_decoder(self.fake_code, tf.constant(False, shape=()))

        G_tofool_loss = self._generator_loss(self.D, self.fake_code)
        reconstr_loss = G_tofool_loss
        G_loss = G_tofool_loss

        self.real_code = self.clean_encoder(self.input_clean_cloud, tf.constant(False, shape=()))

        D_fake_loss, D_real_loss, D_loss = self._discriminator_loss(self.D, self.fake_code, self.real_code)

        # eval only loss
        eval_loss = self._reconstruction_loss(fake_clean_reconstr, self.gt, eval_loss=self.para_config['eval_loss'])

        return G_loss, G_tofool_loss, reconstr_loss, D_loss, D_fake_loss, D_real_loss, fake_clean_reconstr, eval_loss
    
    def model_noGAN(self):

        self.noisy_code = self.noisy_encoder(self.input_noisy_cloud, tf.constant(False, shape=()))

        self.fake_code = self.G(self.noisy_code, self.is_training)

        fake_clean_reconstr = self.clean_decoder(self.fake_code, tf.constant(False, shape=()))

        reconstr_loss = self._reconstruction_loss(fake_clean_reconstr, self.input_noisy_cloud) # comput loss against the input
        #G_tofool_loss = self._generator_loss(self.D, self.fake_code)
        #G_loss = G_tofool_loss + self.para_config['lambda'] * reconstr_loss
        G_loss = self.para_config['lambda'] * reconstr_loss # no GAN loss

        self.real_code = self.clean_encoder(self.input_clean_cloud, tf.constant(False, shape=()))

        D_fake_loss, D_real_loss, D_loss = self._discriminator_loss(self.D, self.fake_code, self.real_code)

        # eval only loss
        eval_loss = self._reconstruction_loss(fake_clean_reconstr, self.gt, eval_loss=self.para_config['eval_loss'])

        return G_loss, G_loss, reconstr_loss, D_loss, D_fake_loss, D_real_loss, fake_clean_reconstr, eval_loss
    
    def model_varying_weight(self):
        self.noisy_code = self.noisy_encoder(self.input_noisy_cloud, tf.constant(False, shape=()))

        self.fake_code = self.G(self.noisy_code, self.is_training)

        fake_clean_reconstr = self.clean_decoder(self.fake_code, tf.constant(False, shape=()))

        reconstr_loss = self._reconstruction_loss(fake_clean_reconstr, self.input_noisy_cloud) # comput loss against the input
        G_tofool_loss = self._generator_loss(self.D, self.fake_code)
        G_loss = (1-self.para_config['lambda']) * G_tofool_loss + self.para_config['lambda'] * reconstr_loss

        self.real_code = self.clean_encoder(self.input_clean_cloud, tf.constant(False, shape=()))

        D_fake_loss, D_real_loss, D_loss = self._discriminator_loss(self.D, self.fake_code, self.real_code)

        # eval only loss
        eval_loss = self._reconstruction_loss(fake_clean_reconstr, self.gt, eval_loss=self.para_config['eval_loss'])

        return G_loss, G_tofool_loss, reconstr_loss, D_loss, D_fake_loss, D_real_loss, fake_clean_reconstr, eval_loss

    def _reconstruction_loss(self, recon, input, eval_loss=None):
        if eval_loss == None:
            if self.para_config['loss'] == 'chamfer':
                cost_p1_p2, _, cost_p2_p1, _ = nn_distance(recon, input)
                loss = tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(cost_p2_p1)
            elif self.para_config['loss'] == 'emd':
                match = approx_match(recon, input)
                loss = match_cost(recon, input, match)
                loss = tf.reduce_mean(loss)
                loss = tf.div(loss, self.para_config['point_cloud_shape'][0]) # return point-wise loss
            elif self.para_config['loss'] == 'hausdorff':
                distances = directed_hausdorff(input, recon) # partial-noisy -> fake_clean
                loss = tf.reduce_mean(distances)

            return loss
        else:
            if eval_loss == 'chamfer':
                cost_p1_p2, _, cost_p2_p1, _ = nn_distance(recon, input)
                loss = tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(cost_p2_p1)
            elif eval_loss == 'emd':
                match = approx_match(recon, input)
                loss = match_cost(recon, input, match)
                loss = tf.reduce_mean(loss)
                loss = tf.div(loss, self.para_config['point_cloud_shape'][0]) # return point-wise loss
            elif eval_loss == 'hausdorff':
                distances = directed_hausdorff(input, recon) # partial-noisy -> fake_clean
                loss = tf.reduce_mean(distances)

            return loss

    def _generator_loss(self, D, fake_code):
        g_loss = tf.reduce_mean(tf.squared_difference(D(fake_code, self.is_training), REAL_LABEL))
        return g_loss
    
    def _discriminator_loss(self, D, fake_code, real_code):
        d_fake_loss = tf.reduce_mean(tf.squared_difference(D(fake_code, self.is_training), FAKE_LABEL))
        d_real_loss = tf.reduce_mean(tf.squared_difference(D(real_code, self.is_training), REAL_LABEL))
        d_loss = (d_fake_loss + d_real_loss) / 2.0
        return d_fake_loss, d_real_loss, d_loss
    
    def optimize(self, g_loss, d_loss):
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

        G_optimizer = make_optimizer(g_loss, self.G.variables, name='Adam_G')
        D_optimizer = make_optimizer(d_loss, self.D.variables, name='Adam_D')

        return G_optimizer, D_optimizer
    
    def __str__(self):
        res = str(self.G) + '\n' + str(self.D)
        return res


#############################################################################

class PCL2PCLGAN_SingleAE:
    '''
    point cloud latent to point cloud latent GAN, only clean autoencoder
    input: 
        clean point cloud, encode it using pre-trained clean point cloud AE,
    '''
    def __init__(self, para_config, para_config_ae):
        self.para_config = para_config
        self.para_config_ae = para_config_ae

        self.G = GeneratorLatentFromLatent('G', fc_sizes=para_config['g_fc_sizes'].copy(), 
                                               latent_dim=para_config['latent_dim'],
                                               activation_fn=para_config['g_activation_fn'],
                                               bn=para_config['g_bn'])

        self.D = DiscriminatorFromLatent('D', fc_sizes=para_config['d_fc_sizes'].copy(),
                                              activation_fn=para_config['d_activation_fn'],
                                              bn=para_config['d_bn'])

        # loaded pre-trained clean cloud decoder
        self.clean_decoder = DecoderFC('clean_Decoder',
                                        fc_sizes=para_config_ae['fc_sizes'].copy(),
                                        activation_fn=para_config_ae['activation_fn'],
                                        bn=para_config_ae['decoder_bn'],
                                        output_shape=para_config_ae['point_cloud_shape'].copy())
        # loaded pre-trained clean cloud encoder
        self.clean_encoder = EncoderPointnet('clean_Encoder', 
                                              n_filters=para_config_ae['n_filters'].copy(),
                                              filter_size=para_config_ae['filter_size'],
                                              stride=para_config_ae['stride'],
                                              activation_fn=para_config_ae['activation_fn'],
                                              bn=para_config_ae['encoder_bn'],
                                              latent_code_dim=para_config_ae['latent_code_dim'])
                            
        self.input_noisy_cloud = tf.placeholder(tf.float32, shape=[para_config['batch_size'], para_config['point_cloud_shape'][0], para_config['point_cloud_shape'][1]])
        self.input_clean_cloud = tf.placeholder(tf.float32, shape=[para_config['batch_size'], para_config['point_cloud_shape'][0], para_config['point_cloud_shape'][1]])
        self.is_training = tf.placeholder(tf.bool, shape=())
        
        # stored intermediate stuff
        self.noisy_code = tf.placeholder(tf.float32, shape=[para_config['batch_size'], para_config['latent_dim']])
        self.fake_code = tf.placeholder(tf.float32, shape=[para_config['batch_size'], para_config['latent_dim']])
        self.real_code = tf.placeholder(tf.float32, shape=[para_config['batch_size'], para_config['latent_dim']])

        # used when evaluation only
        self.gt = tf.placeholder(tf.float32, shape=[para_config['batch_size'], para_config['point_cloud_shape'][0], para_config['point_cloud_shape'][1]])

    def model(self):

        self.noisy_code = self.clean_encoder(self.input_noisy_cloud, tf.constant(False, shape=()))

        self.fake_code = self.G(self.noisy_code, self.is_training)

        fake_clean_reconstr = self.clean_decoder(self.fake_code, tf.constant(False, shape=()))

        reconstr_loss = self._reconstruction_loss(fake_clean_reconstr, self.input_noisy_cloud)
        G_tofool_loss = self._generator_loss(self.D, self.fake_code)
        G_loss = G_tofool_loss + self.para_config['lambda'] * reconstr_loss

        self.real_code = self.clean_encoder(self.input_clean_cloud, tf.constant(False, shape=()))

        D_fake_loss, D_real_loss, D_loss = self._discriminator_loss(self.D, self.fake_code, self.real_code)

        # eval only loss
        eval_loss = self._reconstruction_loss(fake_clean_reconstr, self.gt, eval_loss=self.para_config['eval_loss'])

        return G_loss, G_tofool_loss, reconstr_loss, D_loss, D_fake_loss, D_real_loss, fake_clean_reconstr, eval_loss
    
    def _reconstruction_loss(self, recon, input, eval_loss=None):
        if eval_loss == None:
            if self.para_config['loss'] == 'chamfer':
                cost_p1_p2, _, cost_p2_p1, _ = nn_distance(recon, input)
                loss = tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(cost_p2_p1)
            elif self.para_config['loss'] == 'emd':
                match = approx_match(recon, input)
                loss = match_cost(recon, input, match)
                loss = tf.reduce_mean(loss)
                loss = tf.div(loss, self.para_config['point_cloud_shape'][0]) # return point-wise loss
            elif self.para_config['loss'] == 'hausdorff':
                distances = directed_hausdorff(input, recon) # partial-noisy -> fake_clean
                loss = tf.reduce_mean(distances)

            return loss
        else:
            if eval_loss == 'chamfer':
                cost_p1_p2, _, cost_p2_p1, _ = nn_distance(recon, input)
                loss = tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(cost_p2_p1)
            elif eval_loss == 'emd':
                match = approx_match(recon, input)
                loss = match_cost(recon, input, match)
                loss = tf.reduce_mean(loss)
                loss = tf.div(loss, self.para_config['point_cloud_shape'][0]) # return point-wise loss
            elif eval_loss == 'hausdorff':
                distances = directed_hausdorff(input, recon) # partial-noisy -> fake_clean
                loss = tf.reduce_mean(distances)

            return loss

    def _generator_loss(self, D, fake_code):
        g_loss = tf.reduce_mean(tf.squared_difference(D(fake_code, self.is_training), REAL_LABEL))
        return g_loss
    
    def _discriminator_loss(self, D, fake_code, real_code):
        d_fake_loss = tf.reduce_mean(tf.squared_difference(D(fake_code, self.is_training), FAKE_LABEL))
        d_real_loss = tf.reduce_mean(tf.squared_difference(D(real_code, self.is_training), REAL_LABEL))
        d_loss = (d_fake_loss + d_real_loss) / 2.0
        return d_fake_loss, d_real_loss, d_loss
    
    def optimize(self, g_loss, d_loss):
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

        G_optimizer = make_optimizer(g_loss, self.G.variables, name='Adam_G')
        D_optimizer = make_optimizer(d_loss, self.D.variables, name='Adam_D')

        return G_optimizer, D_optimizer
    
    def __str__(self):
        res = str(self.G) + '\n' + str(self.D)
        return res

import pickle
class LatentCodeDataset:
    def __init__(self, latent_pickle_filename, batch_size, shuffle=True):
        self.latent_pickle_filename = latent_pickle_filename
        self.batch_size = batch_size
        self.shuffle = shuffle

        with open(latent_pickle_filename, 'rb') as pf:
            self.latent_codes = pickle.load(pf) # np.array Num_codes x code_dim
            print('Loaded latent codes: ', self.latent_codes.shape)
        
        self._reset()

    def _shuffle_codes(self, codes):
        idx = np.arange(codes.shape[0])
        np.random.shuffle(idx)
        return codes[idx, ...]
    
    def _reset(self):
        if self.shuffle:
            self._shuffle_codes(self.latent_codes)
        
        self.batch_idx = 0

    def has_next_batch(self):
        num_batch = self.get_num_batches()
        if self.batch_idx < num_batch:
            return True
        return False

    def next_batch_autoreset(self):
        if not self.has_next_batch():
            self._reset()

        start_idx = self.batch_idx * self.batch_size
        end_idx = (self.batch_idx+1) * self.batch_size
        data_batch = self.latent_codes[start_idx:end_idx, :]

        self.batch_idx += 1
        return data_batch
    
    def next_batch(self):
        start_idx = self.batch_idx * self.batch_size
        end_idx = (self.batch_idx+1) * self.batch_size
        data_batch = self.latent_codes[start_idx:end_idx, :]

        self.batch_idx += 1
        return data_batch
    
    def get_num_batches(self):
        return np.floor(self.latent_codes.shape[0] / self.batch_size)

class LatentGAN:
    def __init__(self, para_config):
        self.para_config = para_config

        self.G = GeneratorLatentFromNoise('G', fc_sizes=para_config['g_fc_sizes'], 
                                               latent_dim=para_config['latent_dim'],
                                               activation_fn=para_config['g_activation_fn'],
                                               bn=para_config['g_bn'])
        self.D = DiscriminatorFromLatent('D', fc_sizes=para_config['d_fc_sizes'],
                                              activation_fn=para_config['d_activation_fn'],
                                              bn=para_config['d_bn'])

        self.noise_pl = tf.placeholder(tf.float32, shape=[para_config['batch_size'], para_config['noise_dim']])
        self.fake_code = tf.placeholder(tf.float32, shape=[para_config['batch_size'], para_config['latent_dim']])
        self.real_code = tf.placeholder(tf.float32, shape=[para_config['batch_size'], para_config['noise_dim']])
        self.is_training = tf.placeholder(tf.bool, shape=())

    def model(self):

        self.fake_code = self.G(self.noise_pl, self.is_training)

        G_loss = self._generator_loss(self.D, self.fake_code)

        D_fake_loss, D_real_loss, D_loss = self._discriminator_loss(self.D, self.fake_code, self.real_code)

        # tensorboard visualization
        '''
        tf.summary.scalar('loss/G', G_loss)
        tf.summary.scalar('loss/D', D_loss)
        tf.summary.scalar('loss/D_fake', D_fake_loss)
        tf.summary.scalar('loss/D_real', D_real_loss)
        '''
        
        return G_loss, D_fake_loss, D_real_loss, D_loss, self.fake_code
    
    def _generator_loss(self, D, fake_code):
        g_loss = tf.reduce_mean(tf.squared_difference(D(fake_code, self.is_training), REAL_LABEL))
        return g_loss
    
    def _discriminator_loss(self, D, fake_code, real_code):
        d_fake_loss = tf.reduce_mean(tf.squared_difference(D(fake_code, self.is_training), FAKE_LABEL))
        d_real_loss = tf.reduce_mean(tf.squared_difference(D(real_code, self.is_training), REAL_LABEL))
        d_loss = (d_fake_loss + d_real_loss) / 2.0
        return d_fake_loss, d_real_loss, d_loss
    
    def optimize(self, g_loss, d_loss):
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
            tf.summary.scalar('learning_rate/{}'.format(name), learning_rate)

            optimizer_here = tf.train.AdamOptimizer(learning_rate, beta1=beta1, name=name)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                learning_step = optimizer_here.minimize(loss, global_step=global_step, var_list=variables)
                return learning_step

        G_optimizer = make_optimizer(g_loss, self.G.variables, name='Adam_G')
        D_optimizer = make_optimizer(d_loss, self.D.variables, name='Adam_D')

        return G_optimizer, D_optimizer
    
    def generator_noise_distribution(self):
        n_samples = self.para_config['batch_size'] 
        ndims = self.para_config['noise_dim'] 
        mu = self.para_config['noise_mu']
        sigma = self.para_config['noise_sigma']
        return np.random.normal(mu, sigma, (n_samples, ndims))

    def __str__(self):
        res = str(self.G) + '\n' + str(self.D)
        return res

class CloudLatentGAN:
    def __init__(self, para_config):
        self.para_config = para_config

        self.G = GeneratorLatentFromCloud('G', n_filters=para_config['g_n_filters'], 
                                               filter_size=para_config['filter_size'],
                                               stride=para_config['stride'],
                                               activation_fn=para_config['g_activation_fn'],
                                               bn=para_config['g_bn'],
                                               latent_dim=para_config['latent_dim'],)
        self.D = DiscriminatorFromLatent('D', fc_sizes=para_config['d_fc_sizes'],
                                              activation_fn=para_config['d_activation_fn'],
                                              bn=para_config['d_bn'])

        self.noise_cloud_pl = tf.placeholder(tf.float32, shape=[para_config['batch_size'], para_config['point_cloud_shape'][0], para_config['point_cloud_shape'][1]])
        self.fake_code = tf.placeholder(tf.float32, shape=[para_config['batch_size'], para_config['latent_dim']])
        self.real_code = tf.placeholder(tf.float32, shape=[para_config['batch_size'], para_config['latent_dim']])
        self.is_training = tf.placeholder(tf.bool, shape=())

    def model(self):

        self.fake_code = self.G(self.noise_cloud_pl, self.is_training)

        G_loss = self._generator_loss(self.D, self.fake_code)

        D_fake_loss, D_real_loss, D_loss = self._discriminator_loss(self.D, self.fake_code, self.real_code)

        # tensorboard visualization
        '''
        tf.summary.scalar('loss/G', G_loss)
        tf.summary.scalar('loss/D', D_loss)
        tf.summary.scalar('loss/D_fake', D_fake_loss)
        tf.summary.scalar('loss/D_real', D_real_loss)
        '''
        
        return G_loss, D_fake_loss, D_real_loss, D_loss, self.fake_code
    
    def _generator_loss(self, D, fake_code):
        g_loss = tf.reduce_mean(tf.squared_difference(D(fake_code, self.is_training), REAL_LABEL))
        return g_loss
    
    def _discriminator_loss(self, D, fake_code, real_code):
        d_fake_loss = tf.reduce_mean(tf.squared_difference(D(fake_code, self.is_training), FAKE_LABEL))
        d_real_loss = tf.reduce_mean(tf.squared_difference(D(real_code, self.is_training), REAL_LABEL))
        d_loss = (d_fake_loss + d_real_loss) / 2.0
        return d_fake_loss, d_real_loss, d_loss
    
    def optimize(self, g_loss, d_loss):
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
            tf.summary.scalar('learning_rate/{}'.format(name), learning_rate)

            optimizer_here = tf.train.AdamOptimizer(learning_rate, beta1=beta1, name=name)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                learning_step = optimizer_here.minimize(loss, global_step=global_step, var_list=variables)
                return learning_step

        G_optimizer = make_optimizer(g_loss, self.G.variables, name='Adam_G')
        D_optimizer = make_optimizer(d_loss, self.D.variables, name='Adam_D')

        return G_optimizer, D_optimizer
    
    def generator_noise_distribution(self):
        n_samples = self.para_config['batch_size'] 
        ndims = self.para_config['noise_dim'] 
        mu = self.para_config['noise_mu']
        sigma = self.para_config['noise_sigma']
        return np.random.normal(mu, sigma, (n_samples, ndims))

    def __str__(self):
        res = str(self.G) + '\n' + str(self.D)
        return res

class CyclePCL2PCLGAN:
    '''
    point cloud latent to point cloud latent GAN
    input: 
        noisy point cloud, encode it using pre-trained noisy point cloud AE,
        clean point cloud, encode it using pre-trained clean point cloud AE,
    '''
    def __init__(self, para_config, para_config_ae):
        self.para_config = para_config
        self.para_config_ae = para_config_ae

        self.G = GeneratorLatentFromLatent('G_N2C', fc_sizes=para_config['g_fc_sizes'].copy(), 
                                               latent_dim=para_config['latent_dim'],
                                               activation_fn=para_config['g_activation_fn'],
                                               bn=para_config['g_bn'])
        self.F = GeneratorLatentFromLatent('F_C2N', fc_sizes=para_config['g_fc_sizes'].copy(), 
                                               latent_dim=para_config['latent_dim'],
                                               activation_fn=para_config['g_activation_fn'],
                                               bn=para_config['g_bn'])

        self.D_N = DiscriminatorFromLatent('D_N', fc_sizes=para_config['d_fc_sizes'].copy(),
                                              activation_fn=para_config['d_activation_fn'],
                                              bn=para_config['d_bn'])
        self.D_C = DiscriminatorFromLatent('D_C', fc_sizes=para_config['d_fc_sizes'].copy(),
                                              activation_fn=para_config['d_activation_fn'],
                                              bn=para_config['d_bn'])

        # loaded pre-trained noisy cloud encoder
        self.noisy_encoder = EncoderPointnet('noisy_Encoder', 
                                              n_filters=para_config_ae['n_filters'].copy(),
                                              filter_size=para_config_ae['filter_size'],
                                              stride=para_config_ae['stride'],
                                              activation_fn=para_config_ae['activation_fn'],
                                              bn=para_config_ae['encoder_bn'],
                                              latent_code_dim=para_config_ae['latent_code_dim'])
        # loaded pre-trained noisy cloud decoder
        self.noisy_decoder = DecoderFC('noisy_Decoder',
                                        fc_sizes=para_config_ae['fc_sizes'].copy(),
                                        activation_fn=para_config_ae['activation_fn'],
                                        bn=para_config_ae['decoder_bn'],
                                        output_shape=para_config_ae['point_cloud_shape'].copy())
        # loaded pre-trained clean cloud encoder
        self.clean_encoder = EncoderPointnet('clean_Encoder', 
                                              n_filters=para_config_ae['n_filters'].copy(),
                                              filter_size=para_config_ae['filter_size'],
                                              stride=para_config_ae['stride'],
                                              activation_fn=para_config_ae['activation_fn'],
                                              bn=para_config_ae['encoder_bn'],
                                              latent_code_dim=para_config_ae['latent_code_dim'])
        # loaded pre-trained clean cloud decoder
        self.clean_decoder = DecoderFC('clean_Decoder',
                                        fc_sizes=para_config_ae['fc_sizes'].copy(),
                                        activation_fn=para_config_ae['activation_fn'],
                                        bn=para_config_ae['decoder_bn'],
                                        output_shape=para_config_ae['point_cloud_shape'].copy())
                            
        self.input_noisy_cloud = tf.placeholder(tf.float32, shape=[para_config['batch_size'], para_config['point_cloud_shape'][0], para_config['point_cloud_shape'][1]])
        self.input_clean_cloud = tf.placeholder(tf.float32, shape=[para_config['batch_size'], para_config['point_cloud_shape'][0], para_config['point_cloud_shape'][1]])
        self.is_training = tf.placeholder(tf.bool, shape=())

        # used when evaluation only
        self.gt = tf.placeholder(tf.float32, shape=[para_config['batch_size'], para_config['point_cloud_shape'][0], para_config['point_cloud_shape'][1]])

    def model(self):

        # noisy point cloud -> noisy code -> fake clean code -> reconstructed noisy code -> reconstructed noisy point cloud
        noisy_code = self.noisy_encoder(self.input_noisy_cloud, tf.constant(False, shape=()))
        fake_clean_code = self.G(noisy_code, self.is_training)
        noisy_code_reconstructed = self.F(fake_clean_code, self.is_training)
        noisy_cloud_reconstructed = self.noisy_decoder(noisy_code_reconstructed, tf.constant(False, shape=()))

        # clean point cloud -> clean code -> fake noisy code -> reconstructed clean code _> reconstructed clean point cloud
        clean_code = self.clean_encoder(self.input_clean_cloud, tf.constant(False, shape=()))
        fake_noisy_code = self.F(clean_code, self.is_training)
        clean_code_reconstructed = self.G(fake_noisy_code, self.is_training)
        clean_cloud_reconstructed = self.clean_decoder(clean_code_reconstructed, tf.constant(False, shape=()))

        cycle_loss = self._cycle_loss(self.input_noisy_cloud, noisy_cloud_reconstructed, self.input_clean_cloud, clean_cloud_reconstructed)

        # G loss
        G_gan_loss = self._generator_loss(self.D_C, fake_clean_code)
        G_loss = G_gan_loss + cycle_loss

        # F loss
        F_gan_loss = self._generator_loss(self.D_N, fake_noisy_code)
        F_loss = F_gan_loss + cycle_loss

        # D_N loss
        D_N_fake_loss, D_N_real_loss, D_N_loss = self._discriminator_loss(self.D_N, fake_noisy_code, noisy_code)

        # D_C loss
        D_C_fake_loss, D_C_real_loss, D_C_loss = self._discriminator_loss(self.D_C, fake_clean_code, clean_code)

        # visualize the fake
        fake_clean_cloud = self.clean_decoder(fake_clean_code, tf.constant(False, shape=()))
        fake_noisy_cloud = self.clean_decoder(fake_noisy_code, tf.constant(False, shape=()))

        # eval only loss
        eval_loss = self._reconstruction_loss(fake_clean_cloud, self.gt)
        
        return cycle_loss, G_loss, F_loss, D_N_loss, D_N_fake_loss, D_N_real_loss, D_C_loss, D_C_fake_loss, D_C_real_loss, fake_noisy_cloud, fake_clean_cloud, eval_loss
    
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

    def _generator_loss(self, D, fake_code):
        g_loss = tf.reduce_mean(tf.squared_difference(D(fake_code, self.is_training), REAL_LABEL))
        return g_loss
    
    def _discriminator_loss(self, D, fake_code, real_code):
        d_fake_loss = tf.reduce_mean(tf.squared_difference(D(fake_code, self.is_training), FAKE_LABEL))
        d_real_loss = tf.reduce_mean(tf.squared_difference(D(real_code, self.is_training), REAL_LABEL))
        d_loss = (d_fake_loss + d_real_loss) / 2.0
        return d_fake_loss, d_real_loss, d_loss
    
    def optimize(self, G_loss, F_loss, D_N_loss, D_C_loss):
        def make_optimizer(loss, variables, name='Adam'):
            """ Adam optimizer with learning rate 0.0001 for the first 100k steps (~100 epochs)
                and a linearly decaying rate that goes to zero over the next 100k steps
            """
            global_step = tf.Variable(0, trainable=False, name=name+'_gobal_step')
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

class PC2PCGAN:
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

        self.G = GeneratorCloud2Cloud('G', 
                                      en_n_filters=self.G_paras['en_n_filters'].copy(), 
                                      en_filter_size=self.G_paras['en_filter_size'], 
                                      en_stride=self.G_paras['en_stride'], 
                                      en_activation_fn=self.G_paras['en_activation_fn'], 
                                      en_bn=self.G_paras['en_bn'], 
                                      en_latent_code_dim=self.G_paras['en_latent_code_dim'], 
                                      de_fc_sizes=self.G_paras['de_fc_sizes'], 
                                      de_activation_fn=self.G_paras['de_activation_fn'], 
                                      de_bn=self.G_paras['de_bn'], 
                                      de_output_shape=self.G_paras['de_output_shape'].copy())

        self.D = DiscriminatorFromCloud('D', 
                                        n_filters=self.D_paras['n_filters'].copy(), 
                                        filter_size=self.D_paras['filter_size'], 
                                        stride=self.D_paras['stride'],
                                        activation_fn=self.D_paras['activation_fn'], 
                                        bn=self.D_paras['bn'], 
                                        latent_code_dim=self.D_paras['latent_code_dim'])
                            
        self.input_noisy_cloud = tf.placeholder(tf.float32, shape=[para_config['batch_size'], para_config['point_cloud_shape'][0], para_config['point_cloud_shape'][1]])
        self.input_clean_cloud = tf.placeholder(tf.float32, shape=[para_config['batch_size'], para_config['point_cloud_shape'][0], para_config['point_cloud_shape'][1]])
        self.is_training = tf.placeholder(tf.bool, shape=())

        # used when evaluation only
        #self.gt = tf.placeholder(tf.float32, shape=[para_config['batch_size'], para_config['point_cloud_shape'][0], para_config['point_cloud_shape'][1]])
        #self.eval_loss = tf.placeholder(tf.float32, shape=())

    def model(self):

        fake_clean_cloud = self.G(self.input_noisy_cloud, self.is_training)

        G_loss = self._generator_loss(self.D, fake_clean_cloud)

        D_fake_loss, D_real_loss, D_loss = self._discriminator_loss(self.D, fake_clean_cloud, self.input_clean_cloud)
        
        return G_loss, D_loss, D_fake_loss, D_real_loss, fake_clean_cloud
    
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
    
    def optimize(self, g_loss, d_loss):
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

        G_optimizer = make_optimizer(g_loss, self.G.variables, name='Adam_G')
        D_optimizer = make_optimizer(d_loss, self.D.variables, name='Adam_D')

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

