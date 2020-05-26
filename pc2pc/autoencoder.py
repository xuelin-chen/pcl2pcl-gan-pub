import os,sys

import tensorflow as tf

BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'pointnet_utils'))
from pointnet_encoder_decoder import EncoderPointnet, DecoderFC

sys.path.append(os.path.join(BASE_DIR, 'structural_losses_utils'))
from tf_nndistance import nn_distance
from tf_approxmatch import approx_match, match_cost
from tf_hausdorff_distance import directed_hausdorff

default_para_config = {
    'exp_name': 'ae',

    'batch_size': 200,
    'lr': 0.0005, # base starting learning rate
    'decay_step': 7000000, # in samples, for chair data: 5000000 (~800 epoches), for table data: 7000000 (~800 epoches) 
    'decay_rate': 0.5,
    'clip_lr': 0.0001, # minimal learning rate for clipping lr
    'epoch': 2001,
    
    'loss': 'emd',

    # encoder
    'latent_code_dim': 128,
    'n_filters': [64,128,128,256],
    'filter_size': 1,
    'stride': 1,
    'encoder_bn': True,

    # decoder
    'point_cloud_shape': [2048, 3],
    'fc_sizes': [256, 256], 
    'decoder_bn': False,

    'activation_fn': tf.nn.relu,
}

def get_learning_rate(base_lr, global_step, batch_size, decay_step, decay_rate, clip_lr):
    '''
    global_step: measured in batch
    decay_step: measured in sample
    '''
    learning_rate = tf.train.exponential_decay(
                        base_lr,  # Base learning rate.
                        global_step * batch_size,  # Current index into the dataset.
                        decay_step,          # Decay step.
                        decay_rate,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, clip_lr) # CLIP THE LEARNING RATE!
    return learning_rate   

class AutoEncoder:
    def __init__(self, paras=default_para_config):
        self.paras = paras
        self.batch_size = paras['batch_size']
        self.lr = paras['lr']
        self.activation_fn = paras['activation_fn']
        self.latent_code_dim = paras['latent_code_dim']
        self.point_cloud_shape = paras['point_cloud_shape']
        self.loss = paras['loss']
        
        self.encoder = EncoderPointnet('Encoder', 
                                        n_filters=paras['n_filters'],
                                        filter_size=paras['filter_size'],
                                        stride=paras['stride'],
                                        activation_fn=paras['activation_fn'],
                                        bn=paras['encoder_bn'],
                                        latent_code_dim=paras['latent_code_dim'])
        self.decoder = DecoderFC('Decoder',
                                 fc_sizes=paras['fc_sizes'],
                                 activation_fn=paras['activation_fn'],
                                 bn=paras['decoder_bn'],
                                 output_shape=paras['point_cloud_shape'])

        self.input_pl = tf.placeholder(tf.float32, shape=[paras['batch_size'], self.point_cloud_shape[0], self.point_cloud_shape[1]])
        self.is_training = tf.placeholder(tf.bool, shape=())

        self.latent_code = tf.placeholder(tf.float32, shape=[paras['batch_size'], paras['latent_code_dim']])

        # eval only, to compute loss against gt
        self.gt = tf.placeholder(tf.float32, shape=[paras['batch_size'], self.point_cloud_shape[0], self.point_cloud_shape[1]])

    def model(self):
        self.latent_code = self.encoder(self.input_pl, self.is_training)
        reconstr = self.decoder(self.latent_code, self.is_training)

        ae_loss = self._reconstruction_loss(reconstr, self.input_pl)

        # eval
        self.eval_loss = self._reconstruction_loss(reconstr, self.gt)

        return ae_loss, reconstr, self.latent_code
    
    def make_optimizer(self, loss):
        def make_optimizer(loss, name='Adam'):
            """ Adam optimizer with learning rate 0.0001 for the first 100k steps (~100 epochs)
                and a linearly decaying rate that goes to zero over the next 100k steps
            """
            global_step = tf.Variable(0, trainable=False, name=name+'_gobal_step')
            base_lr = self.paras['lr']
            decay_step = self.paras['decay_step']
            decay_rate = self.paras['decay_rate']
            clip_lr = self.paras['clip_lr']

            lr_cur = get_learning_rate(base_lr, global_step, self.paras['batch_size'], decay_step, decay_rate, clip_lr)

            tf.summary.scalar('learning_rate/{}'.format(name), lr_cur, collections=['train'])

            optimizer_here = tf.train.AdamOptimizer(lr_cur, name=name)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                learning_step = optimizer_here.minimize(loss, global_step=global_step)
                return learning_step

        train_step = make_optimizer(loss, name='Adam')

        return train_step

    def _reconstruction_loss(self, recon, input):

        if self.loss == 'chamfer':
            cost_p1_p2, _, cost_p2_p1, _ = nn_distance(recon, input)
            loss = tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(cost_p2_p1)
        elif self.loss == 'emd':
            match = approx_match(recon, input)
            loss = match_cost(recon, input, match)
            loss = tf.reduce_mean(loss)
            loss = tf.div(loss, self.point_cloud_shape[0]) # return point-wise loss
        elif self.loss == 'hausdorff':
            distances = directed_hausdorff(input, recon) # partial-noisy ->fake_clean
            loss = tf.reduce_mean(distances)
        return loss

    def __str__(self):
        res = str(self.encoder) + '\n' + str(self.decoder)
        return res

class AutoEncoderDenoise:
    def __init__(self, paras=default_para_config):
        self.paras = paras
        self.batch_size = paras['batch_size']
        self.lr = paras['lr']
        self.activation_fn = paras['activation_fn']
        self.latent_code_dim = paras['latent_code_dim']
        self.point_cloud_shape = paras['point_cloud_shape']
        self.loss = paras['loss']
        
        self.encoder = EncoderPointnet('Encoder', 
                                        n_filters=paras['n_filters'],
                                        filter_size=paras['filter_size'],
                                        stride=paras['stride'],
                                        activation_fn=paras['activation_fn'],
                                        bn=paras['encoder_bn'],
                                        latent_code_dim=paras['latent_code_dim'])
        self.decoder = DecoderFC('Decoder',
                                 fc_sizes=paras['fc_sizes'],
                                 activation_fn=paras['activation_fn'],
                                 bn=paras['decoder_bn'],
                                 output_shape=paras['point_cloud_shape'])

        self.input_pl = tf.placeholder(tf.float32, shape=[paras['batch_size'], self.point_cloud_shape[0], self.point_cloud_shape[1]])
        self.gt = tf.placeholder(tf.float32, shape=[paras['batch_size'], self.point_cloud_shape[0], self.point_cloud_shape[1]])
        self.is_training = tf.placeholder(tf.bool, shape=())

        self.latent_code = tf.placeholder(tf.float32, shape=[paras['batch_size'], paras['latent_code_dim']])

    def model(self):
        self.latent_code = self.encoder(self.input_pl, self.is_training)
        self.reconstr = self.decoder(self.latent_code, self.is_training)

        ae_loss = self._reconstruction_loss(self.reconstr, self.gt)

        # loss visualization
        #tf.summary.scalar('loss', ae_loss)

        return ae_loss, self.reconstr, self.latent_code
    
    def make_optimizer(self, loss):

        if False:
            #self.lr = tf.train.exponential_decay(c.learning_rate, self.epoch, c.decay_steps, decay_rate=0.5, staircase=True, name="learning_rate_decay")
            self.lr = tf.maximum(self.lr, 1e-5)
        
        tf.summary.scalar('learning_rate', self.lr, collections=['train'])

        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = optimizer.minimize(loss)

        return train_step


    def _reconstruction_loss(self, recon, input):
        #latent_code = encoder(input, self.is_training)
        #recon = decoder(latent_code, self.is_training)

        if self.loss == 'chamfer':
            cost_p1_p2, _, cost_p2_p1, _ = nn_distance(recon, input)
            loss = tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(cost_p2_p1)
        elif self.loss == 'emd':
            match = approx_match(recon, input)
            loss = match_cost(recon, input, match)
            loss = tf.reduce_mean(loss)
            loss = tf.div(loss, self.point_cloud_shape[0]) # return point-wise loss
        
        return loss

    def __str__(self):
        res = str(self.encoder) + '\n' + str(self.decoder)
        return res

