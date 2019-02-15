import os
import sys
import pickle

import numpy as np
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UTIL_DIR = os.path.join(BASE_DIR, '../../utils')
sys.path.append(UTIL_DIR)
import pc_util

output_pickle_dir = '../data/ShapeNet_v2_point_cloud/03001627'
patch_clean_dir = '../data/ShapeNet_v2_point_cloud/03001627/patch_clean'
patch_noisy_dir = '../data/ShapeNet_v2_point_cloud/03001627/patch_noisy_0.0012'
point_cloud_file_extension = '.ply'
read_pc = pc_util.read_ply
if point_cloud_file_extension == '.ply':
    read_pc = pc_util.read_ply

NOISY_LABEL = 0
CLEAN_LABEL = 1

train_ratio = 0.7
test_ratio = 0.3

if __name__ == "__main__":

    clean_files = [os.path.join(patch_clean_dir, f) for f in os.listdir(patch_clean_dir)]
    noisy_files = [os.path.join(patch_noisy_dir, f) for f in os.listdir(patch_noisy_dir)]
    all_files = clean_files + noisy_files

    all_patches = []
    all_labels = []
    for f in tqdm(all_files):
        if f.endswith(point_cloud_file_extension):
            patch_points = read_pc(f) # 2048x3
            label = -1
            if 'clean' in f:
                label = CLEAN_LABEL
            elif 'noisy' in f:
                label = NOISY_LABEL
            if label < 0:
                print('Error! Skip. %s'%(f))
                continue

            all_patches.append(patch_points)
            all_labels.append(label)

    all_patches = np.array(all_patches, dtype=float)
    all_labels = np.array(all_labels, dtype=int)

    # shuffle data
    idx = np.arange(len(all_labels))
    np.random.shuffle(idx)
    all_patches = all_patches[idx, :, :]
    all_labels = all_labels[idx]

    # train/test split
    num_trains = int(train_ratio * len(all_labels))
    train_patches = all_patches[0:num_trains, :, :]
    train_labels = all_labels[0:num_trains]

    test_patches = all_patches[num_trains:,:,:]
    test_labels = all_labels[num_trains:]

    # dump into pickle files
    train_pikle_f = open(os.path.join(output_pickle_dir, 'chair_clean_noisy_train.pickle'), 'wb')
    pickle.dump(train_patches, train_pikle_f)
    pickle.dump(train_labels, train_pikle_f)

    test_pickle_f = open(os.path.join(output_pickle_dir, 'chair_clean_noisy_test.pickle'), 'wb')
    pickle.dump(test_patches, test_pickle_f)
    pickle.dump(test_labels, test_pickle_f)

