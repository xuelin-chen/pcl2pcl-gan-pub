import os
import sys
from tqdm import tqdm

import math
import numpy as np
import pickle
from numpy.random import RandomState

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
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

class DemoPointCloudDataset:
    def __init__(self, part_point_cloud_dir, batch_size=1, npoint=2048, random_seed=None):
        self.batch_size = batch_size
        self.point_cloud_dir = part_point_cloud_dir
        self.npoint = npoint
        self.random_seed = random_seed

        # make a random generator
        self.rand_gen = RandomState(self.random_seed)

        # list of numpy arrays
        self.point_clouds = self._read_all_pointclouds(self.point_cloud_dir)

        self.reset()
    
    def reset(self):
        self.batch_idx = 0
        self.rotate_angle_deg = 0

    def _read_all_pointclouds(self, dir):
        '''
        return a list of point clouds
        '''
        point_clouds = pc_util.read_all_ply_under_dir(dir)

        return point_clouds

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

        self.batch_idx += 1
        return data_batch
    
    def next_batch_noise_added(self, noise_mu=0.0, noise_sigma=0.01):

        data_batch = self.next_batch()

        # noise
        noise_here = self.rand_gen.normal(noise_mu, noise_sigma, data_batch.shape)
        data_batch = data_batch + noise_here

        return data_batch
    def get_cur_batch(self):
        start_idx = self.batch_idx * self.batch_size
        end_idx = (self.batch_idx+1) * self.batch_size

        data_batch = np.zeros((self.batch_size, self.npoint, 3))
        for i in range(start_idx, end_idx):
            pc_cur = self.point_clouds[i] # M x 3
            choice_cur = np.random.choice(pc_cur.shape[0], self.npoint, replace=True)
            idx_cur = i % self.batch_size
            data_batch[idx_cur] = pc_cur[choice_cur, :]

        return data_batch

    def next_batch_rotated(self, rotated_angle_deg=30):

        if self.rotate_angle_deg < 360:
            data_batch = self.get_cur_batch()
            # rotation for current point cloud not finished
            data_batch = provider.rotate_point_cloud_by_angle(data_batch, self.rotate_angle_deg/180.0 * np.pi)

            self.rotate_angle_deg += rotated_angle_deg
        elif self.rotate_angle_deg == 360:
            self.rotate_angle_deg = 0
            self.batch_idx += 1

            data_batch = self.get_cur_batch()
            # rotation for current point cloud not finished
            data_batch = provider.rotate_point_cloud_by_angle(data_batch, self.rotate_angle_deg/180.0 * np.pi)

            self.rotate_angle_deg += rotated_angle_deg

        return data_batch

class ShapeNetPartPointsDataset:
    def __init__(self, part_point_cloud_dir, batch_size=50, npoint=2048, shuffle=True,  split='train', extra_ply_point_clouds_list=None, random_seed=None, preprocess=True):
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
        random_seed: not used for now, debug needed
        '''
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.point_cloud_dir = part_point_cloud_dir
        self.npoint = npoint
        #self.normalize = normalize
        self.split = split
        self.random_seed = random_seed
        self.preprocess = preprocess

        # make a random generator
        self.rand_gen = RandomState(self.random_seed)
        #self.rand_gen = np.random

        # list of numpy arrays
        self.point_clouds = self._read_all_pointclouds(self.point_cloud_dir)
        if self.preprocess:
            self._preprocess_point_clouds(self.point_clouds)

        if extra_ply_point_clouds_list is not None:
            print('Reading extra point clouds...')
            extra_point_clouds = pc_util.read_ply_from_file_list(extra_ply_point_clouds_list)
            for e_pc in reversed(extra_point_clouds):
                self.point_clouds.insert(0, e_pc)

        self.reset()

    def _shuffle_array(self, arr):

        idx = np.arange(arr.shape[0])
        self.rand_gen.shuffle(idx)
        return arr[idx, ...]

    def _shuffle_list(self, l):
        self.rand_gen.shuffle(l)
    
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

    def _preprocess_point_clouds(self, point_clouds):
        '''
        1. normalize to fit within a unit cube
        2. snap to the ground
        '''
        print('Pre-processing point clouds...')
        for i, pc in enumerate(point_clouds):
            pts_min = np.amin(pc, axis=0)
            pts_max = np.amax(pc, axis=0)

            bbox_size = pts_max - pts_min

            scale_factor = 1.0 / np.amax(bbox_size)

            bbox_center = (pts_max + pts_min) / 2.0
            bbox_bot_center = bbox_center - np.array([0,bbox_size[1]/2.0,0])

            pc = pc - bbox_bot_center
            pc = pc * scale_factor

            point_clouds[i] = pc

        print('Pre-processing done.')

    def reset(self):
        self.batch_idx = 0
        if self.shuffle:
            self._shuffle_list(self.point_clouds)

    def has_next_batch(self):
        num_batch = np.floor(len(self.point_clouds) / self.batch_size) + 1
        if self.batch_idx < num_batch:
            return True
        return False
    
    def next_batch(self):
        start_idx = self.batch_idx * self.batch_size
        end_idx = (self.batch_idx+1) * self.batch_size

        data_batch = np.zeros((self.batch_size, self.npoint, 3))
        for i in range(start_idx, end_idx):

            if i >= len(self.point_clouds):
                i_tmp = i % len(self.point_clouds)
                pc_cur = self.point_clouds[i_tmp]
            else:
                pc_cur = self.point_clouds[i] # M x 3

            choice_cur = self.rand_gen.choice(pc_cur.shape[0], self.npoint, replace=True)
            idx_cur = i % self.batch_size
            data_batch[idx_cur] = pc_cur[choice_cur, :]

        self.batch_idx += 1
        return data_batch

    def next_batch_noise_partial_by_percentage(self, noise_mu=0.0, noise_sigma=0.01, p_min=0.05, p_max=0.5, partial_portion=0.25, with_gt=False):
        '''
        p_min and p_max: the min and max percentage of removed points
        partial_portion: the portion of partial data being generated
        '''
        data_batch = self.next_batch()

        # randomly carve out some points
        data_res = []
        for _, data in enumerate(data_batch):
            do_partial_odd = self.rand_gen.rand()
            if do_partial_odd < partial_portion:
                center_idx = self.rand_gen.randint(self.get_npoint(), size=1)
                center = data[center_idx]

                distances = np.linalg.norm(data - center, axis=1)

                p_cur = self.rand_gen.uniform(p_min, p_max)

                nb_pts2remove = int(self.npoint * p_cur)
                sorted_indices = np.argsort(distances) # ascending order

                remain_points = data[sorted_indices[nb_pts2remove:]]
                
                choice = self.rand_gen.choice(len(remain_points), self.npoint, replace=True)
                remain_points = remain_points[choice, :]

                data_res.append(remain_points)
            else:
                data_res.append(data)
            
        data_res = np.asarray(data_res)

        # noise
        noise_here = self.rand_gen.normal(noise_mu, noise_sigma, data_res.shape)
        noisy_batch = data_res + noise_here

        if with_gt:
            return noisy_batch, data_batch
        return noisy_batch   

    def next_batch_noise_added_with_partial(self, noise_mu=0.0, noise_sigma=0.01, r_min=0.1, r_max=0.25, partial_portion=0.25, with_gt=False):
        '''
        r_max: the max radius for carving out the point around a chosen center
        partial_portion: the portion of partial data being generated
        '''
        data_batch = self.next_batch()

        # randomly carve out some points
        data_res = []
        for _, data in enumerate(data_batch):
            do_partial_odd = self.rand_gen.rand()
            if do_partial_odd < partial_portion:
                center_idx = self.rand_gen.randint(self.get_npoint(), size=1)
                center = data[center_idx]

                distances = np.linalg.norm(data - center, axis=1)

                clip_r = self.rand_gen.uniform(r_min, r_max)

                remain_points = data[distances > clip_r]

                if len(remain_points) < 0.2 * self.npoint:
                    remain_points = data
                    print('WARNING: too partial data. Using complete data instead.')
                
                choice = self.rand_gen.choice(len(remain_points), self.npoint, replace=True)
                remain_points = remain_points[choice, :]

                data_res.append(remain_points)
            else:
                data_res.append(data)
            
        data_res = np.asarray(data_res)

        # noise
        noise_here = self.rand_gen.normal(noise_mu, noise_sigma, data_res.shape)
        noisy_batch = data_res + noise_here

        if with_gt:
            return noisy_batch, data_batch
        return noisy_batch

    def next_batch_noise_added(self, noise_mu=0.0, noise_sigma=0.01):

        data_batch = self.next_batch()

        # noise
        noise_here = self.rand_gen.normal(noise_mu, noise_sigma, data_batch.shape)
        data_batch = data_batch + noise_here

        return data_batch

    def aug_data_batch(self, data_batch, scale_low=0.8, scale_high=1.25, rot=True, snap2ground=True, trans=0.1):
        res_batch = data_batch
        if True:
            res_batch = provider.random_scale_point_cloud(data_batch, scale_low=scale_low, scale_high=scale_high)
        if rot:
            res_batch = provider.rotate_point_cloud(res_batch)
        if trans is not None:
            res_batch = provider.shift_point_cloud(res_batch, shift_range=trans)
        if snap2ground:
            res_batch = provider.lift_point_cloud_to_ground(res_batch)
        return res_batch

    def get_npoint(self):
        return self.npoint

class ShapeNet_3DEPN_PointsDataset:
    def __init__(self, part_point_cloud_dir, batch_size=50, npoint=2048, shuffle=True,  split='train', extra_ply_point_clouds_list=None, random_seed=None, preprocess=True):
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
        random_seed: not used for now, debug needed
        '''
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.point_cloud_dir = part_point_cloud_dir
        self.npoint = npoint
        #self.normalize = normalize
        self.split = split
        self.random_seed = random_seed
        self.preprocess = preprocess

        # make a random generator
        self.rand_gen = RandomState(self.random_seed)
        #self.rand_gen = np.random

        # list of numpy arrays
        self.point_clouds = self._read_all_pointclouds(self.point_cloud_dir)
        if self.preprocess:
            self._preprocess_point_clouds(self.point_clouds)

        self.reset()

    def _shuffle_list(self, l):
        self.rand_gen.shuffle(l)
    
    def _read_all_pointclouds(self, dir):
        '''
        return a list of point clouds
        '''
        # prepare file names
        split_filename = os.path.join(os.path.dirname(dir), os.path.basename(dir)+'_%s_split.pickle'%(self.split))
        with open(split_filename, 'rb') as pf:
            pc_name_list = pickle.load(pf)
        self.pc_filenames = []
        for pc_n in pc_name_list:
            self.pc_filenames.append(os.path.join(dir, pc_n))
        self.pc_filenames.sort() # NOTE: sort the file names here!

        pickle_filename = os.path.join(os.path.dirname(dir), os.path.basename(dir)+'_%s.pickle'%(self.split))
        if os.path.exists(pickle_filename):
            print('Loading cached pickle file: %s'%(pickle_filename))
            p_f = open(pickle_filename, 'rb')
            point_clouds = pickle.load(p_f)
            p_f.close()
        else:
            print('Reading and caching pickle file.')
            point_clouds = pc_util.read_ply_from_file_list(self.pc_filenames) # a list of arrays

            # NOTE!!!: rotate the point clouds here, to align with our data
            for pc_id, pc in enumerate(point_clouds):
                rotated_points = pc_util.rotate_point_cloud_by_axis_angle(pc, [0,1,0], 90)
                point_clouds[pc_id] = rotated_points

            p_f = open(pickle_filename, 'wb')
            pickle.dump(point_clouds, p_f)
            print('Cache to %s'%(pickle_filename))
            p_f.close()

        print('Loaded #point clouds: ', len(point_clouds))

        return point_clouds

    def _preprocess_point_clouds(self, point_clouds):
        '''
        1. normalize to fit within a unit cube
        2. snap to the ground
        '''
        print('Pre-processing point clouds...')
        for i, pc in enumerate(point_clouds):
            pts_min = np.amin(pc, axis=0)
            pts_max = np.amax(pc, axis=0)

            bbox_size = pts_max - pts_min

            scale_factor = 1.0 / np.amax(bbox_size)

            bbox_center = (pts_max + pts_min) / 2.0
            bbox_bot_center = bbox_center - np.array([0,bbox_size[1]/2.0,0])

            pc = pc - bbox_bot_center
            pc = pc * scale_factor

            point_clouds[i] = pc

        print('Pre-processing done.')

    def reset(self):
        self.batch_idx = 0
        if self.shuffle:
            self._shuffle_list(self.point_clouds)

    def has_next_batch(self):
        num_batch = np.floor(len(self.point_clouds) / self.batch_size) + 1
        if self.batch_idx < num_batch:
            return True
        return False
    
    def next_batch(self):
        start_idx = self.batch_idx * self.batch_size
        end_idx = (self.batch_idx+1) * self.batch_size

        data_batch = np.zeros((self.batch_size, self.npoint, 3))
        for i in range(start_idx, end_idx):

            if i >= len(self.point_clouds):
                i_tmp = i % len(self.point_clouds)
                pc_cur = self.point_clouds[i_tmp]
            else:
                pc_cur = self.point_clouds[i] # M x 3

            choice_cur = self.rand_gen.choice(pc_cur.shape[0], self.npoint, replace=True)
            idx_cur = i % self.batch_size
            data_batch[idx_cur] = pc_cur[choice_cur, :]

        self.batch_idx += 1
        return data_batch

    def next_batch_with_name(self):
        start_idx = self.batch_idx * self.batch_size
        end_idx = (self.batch_idx+1) * self.batch_size

        data_batch = np.zeros((self.batch_size, self.npoint, 3))
        name_batch = []
        for bs_idx in range(self.batch_size):
            name_batch.append('name_placeholder')
        for i in range(start_idx, end_idx):

            if i >= len(self.point_clouds):
                i_tmp = i % len(self.point_clouds)
                pc_cur = self.point_clouds[i_tmp]
                pc_name_cur = self.pc_filenames[i_tmp].split('/')[-1]
            else:
                pc_cur = self.point_clouds[i] # M x 3
                pc_name_cur = self.pc_filenames[i].split('/')[-1]

            choice_cur = self.rand_gen.choice(pc_cur.shape[0], self.npoint, replace=True)
            idx_cur = i % self.batch_size
            data_batch[idx_cur] = pc_cur[choice_cur, :]
            name_batch[idx_cur] = pc_name_cur

        self.batch_idx += 1
        return data_batch, name_batch

    def get_npoint(self):
        return self.npoint


import trimesh
class RealWorldPointsDataset:
    def __init__(self, mesh_dir, batch_size=50, npoint=2048, shuffle=True, split='train', random_seed=None):
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
        random_seed: 
        '''
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mesh_dir = mesh_dir
        self.npoint = npoint
        self.split = split
        self.random_seed = random_seed

        # make a random generator
        self.rand_gen = RandomState(self.random_seed)
        #self.rand_gen = np.random

        # list of meshes
        self.meshes = self._read_all_meshes(self.mesh_dir) # a list of trimeshes
        self._preprocess_meshes(self.meshes)

        self.point_clouds = self._pre_sample_points(self.meshes)

        self.reset()

    def _shuffle_list(self, l):
        self.rand_gen.shuffle(l)
    
    def _preprocess_meshes_old(self, meshes):
        '''
        currently, just normalize all meshes, according to the height
        also, snap chairs to the ground
        '''

        max_height = -1

        for mesh in meshes:
            bbox = mesh.bounding_box # 8 pts
            height = np.max(bbox.vertices[:,1])

            if height > max_height:
                max_height = height
            
        for mesh in meshes:
            bbox = mesh.bounding_box # 8 pts
            height = np.max(bbox.vertices[:,1])
            scale_factor = height / max_height

            bbox_center = np.mean(bbox.vertices, axis=0)
            bbox_center[1] = height / 2.0 # assume that the object is alreay snapped to ground

            trans_v = -bbox_center 
            trans_v[1] += mesh.bounding_box.extents[1]/2.
            mesh.apply_translation(trans_v) # translate the bottom center bbox center to ori

            mesh.apply_scale(scale_factor) # do scaling

        return 0
    
    def _preprocess_meshes(self, meshes):
        '''
        assume the input mesh has already been snapped to the ground
        1. normalize to fit within a unit cube
        2. center the bottom center to the original
        '''

        for mesh in meshes:
            bbox = mesh.bounding_box # 8 pts
            height = np.max(bbox.vertices[:,1])
            extents = mesh.bounding_box.extents.copy()
            extents[1] = height

            scale_factor = 1.0 / np.amax(extents)

            bbox_center = np.mean(bbox.vertices, axis=0)

            trans_v = -bbox_center 
            trans_v[1] = 0 # assume already snap to the ground, so do not translate along y
            mesh.apply_translation(trans_v) # translate the center bbox bottom to ori

            mesh.apply_scale(scale_factor)
    
    def _read_all_meshes(self, mesh_dir):
        meshes_cache_filename = os.path.join(os.path.dirname(mesh_dir), 'meshes_cache_%s.pickle'%(self.split))

        if os.path.exists(meshes_cache_filename):
            print('Loading cached pickle file: %s'%(meshes_cache_filename))
            p_f = open(meshes_cache_filename, 'rb')
            mesh_list = pickle.load(p_f)
            p_f.close()
        else:
            split_filename = os.path.join(os.path.dirname(mesh_dir), os.path.basename(mesh_dir)+'_%s_split.pickle'%(self.split))
            with open(split_filename, 'rb') as pf:
                mesh_name_list = pickle.load(pf)
            mesh_filenames = []
            for mesh_n in mesh_name_list:
                mesh_filenames.append(os.path.join(mesh_dir, mesh_n))
            mesh_filenames.sort() # NOTE: sort the file names here!

            print('Reading and caching...')
            mesh_list = []
            for mn in tqdm(mesh_filenames):
                m_fn = os.path.join(mesh_dir, mn)
                mesh = trimesh.load(m_fn)
            
                mesh_list.append(mesh)
            
            p_f = open(meshes_cache_filename, 'wb')
            pickle.dump(mesh_list, p_f)
            print('Cache to %s'%(meshes_cache_filename))
            p_f.close()

        return mesh_list

    def _pre_sample_points(self, meshes):
        presamples_cache_filename = os.path.join(os.path.dirname(self.mesh_dir), 'presamples_cache_%s.pickle'%(self.split))
        if os.path.exists(presamples_cache_filename):
            print('Loading cached pickle file: %s'%(presamples_cache_filename))
            p_f = open(presamples_cache_filename, 'rb')
            points_list = pickle.load(p_f)
            p_f.close()
        else:
            print('Pre-sampling...')
            points_list = []
            for m in tqdm(meshes):
                samples, _ = trimesh.sample.sample_surface_even(m, self.npoint * 10)
                points_list.append(np.array(samples))

            p_f = open(presamples_cache_filename, 'wb')
            pickle.dump(points_list, p_f)
            p_f.close()

            print('Pre-sampling done and cached.')

        return points_list

    def reset(self):
        self.batch_idx = 0
        if self.shuffle:
            self._shuffle_list(self.meshes)

    def has_next_batch(self):
        num_batch = np.floor(len(self.meshes) / self.batch_size) + 1
        if self.batch_idx < num_batch:
            return True
        return False
    
    def next_batch(self):
        start_idx = self.batch_idx * self.batch_size
        end_idx = (self.batch_idx+1) * self.batch_size

        data_batch = np.zeros((self.batch_size, self.npoint, 3))
        for i in range(start_idx, end_idx):

            if i >= len(self.point_clouds):
                i_tmp = i % len(self.point_clouds)
                pc_cur = self.point_clouds[i_tmp]
            else:
                pc_cur = self.point_clouds[i] # M x 3
                
            choice_cur = self.rand_gen.choice(pc_cur.shape[0], self.npoint, replace=True)
            idx_cur = i % self.batch_size
            data_batch[idx_cur] = pc_cur[choice_cur, :]

        self.batch_idx += 1
        return data_batch

    def get_npoint(self):
        return self.npoint


if __name__=='__main__':
    '''
    REALDATASET = RealWorldPointsDataset('/workspace/pointnet2/pc2pc/data/scannet_v2_chairs_alilgned_v2/point_cloud', batch_size=6, npoint=2048,  shuffle=False, split='trainval', random_seed=0)

    data_batch = REALDATASET.next_batch()
    pc_util.write_ply(data_batch[0], 'real_data.ply')
    pc_util.write_ply(data_batch[1], 'real_data_1.ply')
    pc_util.write_ply(data_batch[2], 'real_data_2.ply')
    '''

    
    TRAIN_DATASET = ShapeNetPartPointsDataset('/workspace/pointnet2/pc2pc/data/ShapeNet_v2_point_cloud/03001627/point_cloud_clean', batch_size=6, npoint=2048,  shuffle=False, split='test', random_seed=0)

    data_batch = TRAIN_DATASET.next_batch_noise_partial_by_percentage()
    pc_util.write_ply(data_batch[0], 'data.ply')
    #data_batch = TRAIN_DATASET.aug_data_batch(data_batch, rot=False, trans=0.1, snap2ground=False)
    #pc_util.write_ply(data_batch[0], 'data_aug.ply')


    '''
    noisy_partial_data_batch = TRAIN_DATASET.next_batch_noise_added_with_partial(partial_portion=1)
    pc_util.write_ply(noisy_partial_data_batch[0], 'test_noisy_partial.ply')

    TEST_DATASET = ShapeNetPartPointsDataset('/workspace/pointnet2/pc2pc/data/ShapeNet_v2_point_cloud/03001627/point_cloud_clean', batch_size=6, npoint=8192, normalize=False, shuffle=False, split='test', random_seed=0)

    noisy_partial_data_batch = TEST_DATASET.next_batch_noise_added_with_partial(partial_portion=1)
    pc_util.write_ply(noisy_partial_data_batch[0], 'test_noisy_partial_1.ply')
    '''

    '''
    epoch = 10
    for i in range(epoch):
        print(i)
        while dataset.has_next_batch():
            data_batch = dataset.next_batch()
            print(data_batch.shape)
        dataset.reset()
    '''