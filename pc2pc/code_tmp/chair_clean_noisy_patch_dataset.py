'''
    ModelNet dataset. Support ModelNet40, XYZ channels. Up to 2048 points.
    Faster IO than ModelNetDataset in the first epoch.
'''

import os
import sys

import math
import numpy as np
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider

CHAIR_CLEAN_NOISY_DATA_DIR = 'data/ShapeNet_v2_point_cloud/03001627'
NUM_CLASSES = 2

# ! this label assignment should be consistent with the script for generating pickle files!
NOISY_LABEL = 0
CLEAN_LABEL = 1

def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: Nx2048x3 numpy array
          label: N numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx

class ChairCleanNoisyDataset(object):
    def __init__(self, data_dir=CHAIR_CLEAN_NOISY_DATA_DIR, split='train', batch_size = 32, npoints = 2048, shuffle=True):
        self.data_dir = data_dir
        self.split = split
        self._read_pickle_file()

        self.batch_size = batch_size
        self.npoints = npoints
        self.shuffle = shuffle
        if self.shuffle:
            self.patch, self.label, _ = shuffle_data(self.patch, self.label)
        self.num_batches = math.floor(len(self.label)*1.0 / self.batch_size) # drop the last incomplete batch
        
        self.reset()

    def _center_patches(self):
        centers = np.mean(self.patch, axis=1) # Nx3
        centers = np.expand_dims(centers, 1)
        self.patch = self.patch - centers

    def _read_pickle_file(self):
        pickle_filename = os.path.join(self.data_dir, 'chair_clean_noisy_%s.pickle'%(self.split))
        p_file = open(pickle_filename, 'rb')
        self.patch = pickle.load(p_file) # np array, Nx2048x3
        self.label = pickle.load(p_file) # np array, N
        self._center_patches()

        num_cleans = np.sum(self.label==0)
        num_noisys = np.sum(self.label==1)
        print('Load data: %s'%(pickle_filename))
        print('Number of samples: %d, num_cleans: %d num_noisys: %d'%(len(self.label), num_cleans, num_noisys))
        
    
    def __len__(self):
        return self.num_batches

    def _augment_batch_data(self, batch_data):
        rotated_data = provider.rotate_point_cloud(batch_data)
        rotated_data = provider.rotate_perturbation_point_cloud(rotated_data)
        jittered_data = provider.random_scale_point_cloud(rotated_data[:,:,0:3])
        jittered_data = provider.shift_point_cloud(jittered_data)
        #jittered_data = provider.jitter_point_cloud(jittered_data)
        rotated_data[:,:,0:3] = jittered_data
        return provider.shuffle_points(rotated_data)

    def reset(self):
        self.batch_idx = 0
        if self.shuffle:
            self.patch, self.label, _ = shuffle_data(self.patch, self.label)

    def num_channel(self):
        return self.patch.shape[2]

    def has_next_batch(self):
        if self.batch_idx >= 0 and self.batch_idx < self.num_batches:
            return True
        return False

    def next_batch(self, augment=False):
        ''' returned dimension may be smaller than self.batch_size '''
        start_idx = self.batch_idx * self.batch_size
        end_idx = (self.batch_idx+1) * self.batch_size

        data_batch = self.patch[start_idx:end_idx, 0:self.npoints, :]
        label_batch = self.label[start_idx:end_idx]

        #if augment: data_batch = self._augment_batch_data(data_batch)

        self.batch_idx += 1
        
        return data_batch, label_batch     

class ChairSeparatedCleanNoisyDataset(object):
    def __init__(self, data_dir=CHAIR_CLEAN_NOISY_DATA_DIR, split='train', batch_size = 32, npoints = 2048, shuffle=True, nb_clean=2000, nb_noisy=2000):
        self.data_dir = data_dir
        self.split = split
        self.shuffle = shuffle
        self.nb_clean = nb_clean
        self.nb_noisy = nb_noisy

        self._read_pickle_file()

        self.batch_size = batch_size
        self.npoints = npoints

        self._reset_clean()
        self._reset_noisy()

    def _reset_clean(self):
        self.clean_batch_idx = 0
        if self.shuffle:
            self.clean_patches, _ = self._shuffle_patches(self.clean_patches)
    
    def _reset_noisy(self):
        self.noisy_batch_idx = 0
        if self.shuffle:
            self.noisy_patches, _ = self._shuffle_patches(self.noisy_patches)

    def _center_patches(self, patches):
        centers = np.mean(patches, axis=1) # Nx3
        centers = np.expand_dims(centers, 1)
        patches = patches - centers
        return patches
    
    def _shuffle_patches(self, patches):
        idx = np.arange(patches.shape[0])
        np.random.shuffle(idx)
        return patches[idx, ...], idx

    def _read_pickle_file(self):
        pickle_filename = os.path.join(self.data_dir, 'chair_clean_noisy_%s.pickle'%(self.split))
        p_file = open(pickle_filename, 'rb')
        all_patches = pickle.load(p_file) # np array, Nx2048x3
        print('Input all patches: ', all_patches.shape)
        all_labels = pickle.load(p_file) # np array, N
        print('Input all labels: ', all_labels.shape)
        all_patches = self._center_patches(all_patches)

        num_cleans = np.sum(all_labels==CLEAN_LABEL)
        num_noisys = np.sum(all_labels==NOISY_LABEL)
        print('Load data: %s'%(pickle_filename))
        print('Number of samples: %d, num_cleans: %d num_noisys: %d'%(len(all_labels), num_cleans, num_noisys))

        # separate clean from noisy
        self.clean_patches = all_patches[np.argwhere(all_labels==CLEAN_LABEL)]
        self.clean_patches = np.squeeze(self.clean_patches)
        self.noisy_patches = all_patches[np.argwhere(all_labels==NOISY_LABEL)]
        self.noisy_patches = np.squeeze(self.noisy_patches)

        if self.nb_clean is not None and self.nb_noisy:
            self.clean_patches = self.clean_patches[:self.nb_clean]
            self.noisy_patches = self.noisy_patches[:self.nb_noisy]

        if self.shuffle:
            self.clean_patches, _ = self._shuffle_patches(self.clean_patches)
            self.noisy_patches, _ = self._shuffle_patches(self.noisy_patches)
        
        print('Final samples, num_cleans: %d, num_noisys: %d'%(self.clean_patches.shape[0], self.noisy_patches.shape[0]))

    def has_next_clean_batch(self):
        num_clean_patches = np.floor(self.clean_patches.shape[0] / self.batch_size)

        if self.clean_batch_idx < num_clean_patches:
            return True
        return False

    def next_clean_batch(self):
        num_clean_patches = np.floor(self.clean_patches.shape[0] / self.batch_size)

        if self.clean_batch_idx >= num_clean_patches:
            self._reset_clean()

        start_idx = self.clean_batch_idx * self.batch_size
        end_idx = (self.clean_batch_idx+1) * self.batch_size

        data_batch = self.clean_patches[start_idx:end_idx, 0:self.npoints, :]

        self.clean_batch_idx += 1
        return data_batch
    
    def next_noisy_batch(self):
        num_noisy_patches = np.floor(self.noisy_patches.shape[0] / self.batch_size)

        if self.noisy_batch_idx >= num_noisy_patches:
            self._reset_noisy()

        start_idx = self.noisy_batch_idx * self.batch_size
        end_idx = (self.noisy_batch_idx+1) * self.batch_size

        data_batch = self.noisy_patches[start_idx:end_idx, 0:self.npoints, :]

        self.noisy_batch_idx += 1
        return data_batch

if __name__=='__main__':
    d = ChairSeparatedCleanNoisyDataset('data/ShapeNet_v2_point_cloud/03001627')
    print(d.shuffle)

    for i in range(0, 1000):
        print(i)
        ps_batch = d.next_noisy_batch()
        #print(ps_batch.shape)
        ps_batch = d.next_clean_batch()
        #print(ps_batch.shape)
    print('One epoch over.')