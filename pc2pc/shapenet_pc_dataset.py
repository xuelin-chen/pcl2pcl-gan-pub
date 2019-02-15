import os
import sys

import math
import numpy as np
import pickle
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pc_util

snc_synth_id_to_category = {
    '02691156': 'airplane',  '02773838': 'bag',        '02801938': 'basket',
    '02808440': 'bathtub',   '02818832': 'bed',        '02828884': 'bench',
    '02834778': 'bicycle',   '02843684': 'birdhouse',  '02871439': 'bookshelf',
    '02876657': 'bottle',    '02880940': 'bowl',       '02924116': 'bus',
    '02933112': 'cabinet',   '02747177': 'can',        '02942699': 'camera',
    '02954340': 'cap',       '02958343': 'car',        '03001627': 'chair',
    '03046257': 'clock',     '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table',     '04401088': 'telephone',  '02946921': 'tin_can',
    '04460130': 'tower',     '04468005': 'train',      '03085013': 'keyboard',
    '03261776': 'earphone',  '03325088': 'faucet',     '03337140': 'file',
    '03467517': 'guitar',    '03513137': 'helmet',     '03593526': 'jar',
    '03624134': 'knife',     '03636649': 'lamp',       '03642806': 'laptop',
    '03691459': 'speaker',   '03710193': 'mailbox',    '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano',     '03938244': 'pillow',     '03948459': 'pistol',
    '03991062': 'pot',       '04004475': 'printer',    '04074963': 'remote_control',
    '04090263': 'rifle',     '04099429': 'rocket',     '04225987': 'skateboard',
    '04256520': 'sofa',      '04330267': 'stove',      '04530566': 'vessel',
    '04554684': 'washer',    '02858304': 'boat',       '02992529': 'cellphone'
}

snc_synth_category_to_id = {
    'airplane' : '02691156' ,  'bag'       : '02773838' ,  'basket'        : '02801938' ,
    'bathtub'  : '02808440' ,  'bed'       : '02818832' ,  'bench'         : '02828884' ,
    'bicycle'  : '02834778' ,  'birdhouse' : '02843684' ,  'bookshelf'     : '02871439' ,
    'bottle'   : '02876657' ,  'bowl'      : '02880940' ,  'bus'           : '02924116' ,
    'cabinet'  : '02933112' ,  'can'       : '02747177' ,  'camera'        : '02942699' ,
    'cap'      : '02954340' ,  'car'       : '02958343' ,  'chair'         : '03001627' ,
    'clock'    : '03046257' ,  'dishwasher': '03207941' ,  'monitor'       : '03211117' ,
    'table'    : '04379243' ,  'telephone' : '04401088' ,  'tin_can'       : '02946921' ,
    'tower'    : '04460130' ,  'train'     : '04468005' ,  'keyboard'      : '03085013' ,
    'earphone' : '03261776' ,  'faucet'    : '03325088' ,  'file'          : '03337140' ,
    'guitar'   : '03467517' ,  'helmet'    : '03513137' ,  'jar'           : '03593526' ,
    'knife'    : '03624134' ,  'lamp'      : '03636649' ,  'laptop'        : '03642806' ,
    'speaker'  : '03691459' ,  'mailbox'   : '03710193' ,  'microphone'    : '03759954' ,
    'microwave': '03761084' ,  'motorcycle': '03790512' ,  'mug'           : '03797390' ,
    'piano'    : '03928116' ,  'pillow'    : '03938244' ,  'pistol'        : '03948459' ,
    'pot'      : '03991062' ,  'printer'   : '04004475' ,  'remote_control': '04074963' ,
    'rifle'    : '04090263' ,  'rocket'    : '04099429' ,  'skateboard'    : '04225987' ,
    'sofa'     : '04256520' ,  'stove'     : '04330267' ,  'vessel'        : '04530566' ,
    'washer'   : '04554684' ,  'boat'      : '02858304' ,  'cellphone'     : '02992529' 
}

# not used for now
class ShapeNetPointsDataset:
    def __init__(self, shapenet_points_root, cat_name='chair', batch_size=50, npoint=2048, shuffle=True):
        self.shapenet_points_root = shapenet_points_root
        self.cat_name = cat_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.syth_cat_id = snc_synth_category_to_id[cat_name]
        self.point_cloud_dir = os.path.join(self.shapenet_points_root, self.syth_cat_id, 'point_cloud_clean')
        self.npoint = npoint

        self.point_clouds = self._read_all_pointclouds(self.point_cloud_dir)
        if self.shuffle:
            self.point_clouds = self._shuffle_array(self.point_clouds)

        self.reset()

    def _shuffle_array(self, arr):
        idx = np.arange(arr.shape[0])
        np.random.shuffle(idx)
        return arr[idx, ...]
    
    def _read_all_pointclouds(self, dir):
        '''
        return a numpy array
        '''
        pickle_filename = os.path.join(os.path.dirname(dir), os.path.basename(dir)+'.pickle')
        if os.path.exists(pickle_filename):
            print('Loading cached pickle file.')
            p_f = open(pickle_filename, 'rb')
            point_clouds = pickle.load(p_f)
            p_f.close()
        else:
            print('Reading and caching pickle file.')
            point_clouds = pc_util.read_all_ply_under_dir(dir) # a list of arrays
            p_f = open(pickle_filename, 'wb')
            pickle.dump(point_clouds, p_f)
            print('Cache to %s'%(pickle_filename))
            p_f.close()
        
        print('Loaded #point clouds: ', len(point_clouds))
        res = []
        if True:
            print('Warning: randomly duplicate/downsample point clouds!')
            for pc in point_clouds:
                choice = np.random.choice(len(pc), self.npoint, replace=True)
                pc = pc[choice, :]
                res.append(pc)
        res = np.array(res)
        print('Data shape: ', res.shape)
        return res

    def reset(self):
        self.batch_idx = 0
        self.point_clouds = self._shuffle_array(self.point_clouds)

    def has_next_batch(self):
        num_batch = np.floor(self.point_clouds.shape[0] / self.batch_size)
        if self.batch_idx < num_batch:
            return True
        return False
    
    def next_batch(self):
        start_idx = self.batch_idx * self.batch_size
        end_idx = (self.batch_idx+1) * self.batch_size
        data_batch = self.point_clouds[start_idx:end_idx, :, :]

        self.batch_idx += 1
        return data_batch

    # to be removed
    def get_a_random_batch(self):
        num_batch = np.floor(self.point_clouds.shape[0] / self.batch_size)
        all_batch_idx = np.arange(num_batch, dtype=int)
        np.random.shuffle(all_batch_idx)
        idx_random = all_batch_idx[0]

        start_idx = idx_random * self.batch_size
        end_idx = (idx_random+1) * self.batch_size
        data_batch = self.point_clouds[start_idx:end_idx, :, :]

        return data_batch

    def get_npoint(self):
        return self.point_clouds.shape[1]

class ShapeNetPartPointsDataset:
    def __init__(self, part_point_cloud_dir, batch_size=50, npoint=2048, shuffle=True, normalize=False, split='train', extra_ply_point_clouds_list=None):
        '''
        part_point_cloud_dir: the directory contains the oringal ply point clouds
        batch_size:
        npoint: a fix number of points that will sample from the point clouds
        shuffle: whether to shuffle the order of point clouds
        normalize: whether to normalize the point clouds
        split: 
        extra_ply_point_clouds_list: a list contains some extra point cloud file names, 
                                     note that only use it in test time, 
                                     these point clouds will be inserted in front of the point cloud list,
                                     which means extra clouds get to be tested first
        '''
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.point_cloud_dir = part_point_cloud_dir
        self.npoint = npoint
        self.normalize = normalize
        self.split = split

        # list of numpy arrays
        self.point_clouds = self._read_all_pointclouds(self.point_cloud_dir)
        
        if extra_ply_point_clouds_list is not None:
            print('Reading extra point clouds...')
            extra_point_clouds = pc_util.read_ply_from_file_list(extra_ply_point_clouds_list)
            for e_pc in reversed(extra_point_clouds):
                self.point_clouds.insert(0, e_pc)

        self.reset()

    def _shuffle_array(self, arr):
        idx = np.arange(arr.shape[0])
        np.random.shuffle(idx)
        return arr[idx, ...]

    def _shuffle_list(self, l):
        random.shuffle(l)
    
    def _read_all_pointclouds(self, dir):
        '''
        return a list of point clouds
        '''
        # prepare file names
        split_filename = os.path.join(os.path.dirname(dir), os.path.basename(dir)+'_%s_split.pickle'%(self.split))
        with open(split_filename, 'rb') as pf:
            pc_name_list = pickle.load(pf)
        pc_filenames = []
        for pc_n in pc_name_list:
            pc_filenames.append(os.path.join(dir, pc_n))
        pc_filenames.sort() # NOTE: sort the file names here!

        #pickle_filename = os.path.join(os.path.dirname(dir), 'OLD_'+os.path.basename(dir)+'_%s.pickle'%(self.split))
        pickle_filename = os.path.join(os.path.dirname(dir), os.path.basename(dir)+'_%s.pickle'%(self.split))
        if os.path.exists(pickle_filename):
            print('Loading cached pickle file: %s'%(pickle_filename))
            p_f = open(pickle_filename, 'rb')
            point_clouds = pickle.load(p_f)
            p_f.close()
        else:
            print('Reading and caching pickle file.')
            point_clouds = pc_util.read_ply_from_file_list(pc_filenames) # a list of arrays
            p_f = open(pickle_filename, 'wb')
            pickle.dump(point_clouds, p_f)
            print('Cache to %s'%(pickle_filename))
            p_f.close()

        print('Loaded #point clouds: ', len(point_clouds))

        return point_clouds

    def reset(self):
        self.batch_idx = 0
        if self.shuffle:
            self._shuffle_list(self.point_clouds)

    def has_next_batch(self):
        num_batch = np.floor(len(self.point_clouds) / self.batch_size)
        if self.batch_idx < num_batch:
            return True
        return False
    
    def next_batch(self):
        start_idx = self.batch_idx * self.batch_size
        end_idx = (self.batch_idx+1) * self.batch_size

        data_batch = np.zeros((self.batch_size, self.npoint, 3))
        for i in range(start_idx, end_idx):
            pc_cur = self.point_clouds[i] # M x 3
            choice_cur = np.random.choice(pc_cur.shape[0], self.npoint, replace=True)
            idx_cur = i % self.batch_size
            data_batch[idx_cur] = pc_cur[choice_cur, :]

        if self.normalize:
            data_batch = pc_util.point_cloud_normalized(data_batch)

        self.batch_idx += 1
        return data_batch

    def next_batch_noise_added(self, noise_mu=0.0, noise_sigma=0.01):
        data_batch = self.next_batch()

        # noise
        noise_here = np.random.normal(noise_mu, noise_sigma, data_batch.shape)
        data_batch = data_batch + noise_here

        return data_batch
    
    def next_batch_noisy_clean_pair(self, noise_mu=0.0, noise_sigma=0.01):
        data_batch = self.next_batch()

        # noise
        noise_here = np.random.normal(noise_mu, noise_sigma, data_batch.shape)
        noisy_batch = data_batch + noise_here

        return noisy_batch, data_batch

    def next_batch_noise_added_with_partial(self, noise_mu=0.0, noise_sigma=0.01, r_min=0.1, r_max=0.25, partial_portion=0.25, with_gt=False):
        '''
        r_max: the max radius for carving out the point around a chosen center
        partial_portion: the portion of partial data being generated
        '''
        data_batch = self.next_batch()

        # randomly carve out some points
        data_res = []
        for _, data in enumerate(data_batch):
            do_partial_odd = np.random.rand()
            if do_partial_odd < partial_portion:
                center_idx = np.random.randint(self.get_npoint(), size=1)
                center = data[center_idx]

                distances = np.linalg.norm(data - center, axis=1)

                clip_r = np.random.uniform(r_min, r_max)
                #clip_r = np.random.rand() * r_max

                remain_points = data[distances > clip_r]
                
                choice = np.random.choice(len(remain_points), self.npoint, replace=True)
                remain_points = remain_points[choice, :]

                data_res.append(remain_points)
            else:
                data_res.append(data)
            
        data_res = np.asarray(data_res)

        # noise
        noise_here = np.random.normal(noise_mu, noise_sigma, data_res.shape)
        noisy_batch = data_res + noise_here

        if with_gt:
            return noisy_batch, data_batch
        return noisy_batch

    # for test, deprecated
    def get_a_random_batch(self, noise_mu, noise_sigma):
        num_batch = np.floor(self.point_clouds.shape[0] / self.batch_size)
        all_batch_idx = np.arange(num_batch, dtype=int)
        np.random.shuffle(all_batch_idx)
        idx_random = all_batch_idx[0]

        start_idx = idx_random * self.batch_size
        end_idx = (idx_random+1) * self.batch_size
        data_batch = self.point_clouds[start_idx:end_idx, :, :]

        # noise
        noise_here = np.random.normal(noise_mu, noise_sigma, data_batch.shape)
        data_batch = data_batch + noise_here

        return data_batch

    def get_npoint(self):
        return self.npoint

if __name__=='__main__':

    TRAIN_DATASET = ShapeNetPartPointsDataset('/workspace/pointnet2/pc2pc/data/ShapeNet_v2_point_cloud/03001627/point_cloud_clean', batch_size=6, npoint=8192, normalize=False, split='test')
    #n_batch, c_batch = TRAIN_DATASET.nex_batch_noisy_clean_pair(0, 0.01)

    data_batch = TRAIN_DATASET.next_batch()
    pc_util.write_ply(data_batch[0], 'test_clean.ply')

    noisy_data_batch = TRAIN_DATASET.next_batch_noise_added(noise_sigma=0.05)
    pc_util.write_ply(noisy_data_batch[0], 'test_noisy.ply')

    noisy_partial_data_batch = TRAIN_DATASET.next_batch_noise_added_with_partial(partial_portion=1)
    pc_util.write_ply(noisy_partial_data_batch[0], 'test_noisy_partial.ply')

    noisy_data_batch, clean_data_batch = TRAIN_DATASET.next_batch_noisy_clean_pair()
    pc_util.write_ply(noisy_data_batch[0], 'test_pair_noisy.ply')
    pc_util.write_ply(clean_data_batch[0], 'test_pair_clean.ply')

    '''
    epoch = 10
    for i in range(epoch):
        print(i)
        while dataset.has_next_batch():
            data_batch = dataset.next_batch()
            print(data_batch.shape)
        dataset.reset()
    '''