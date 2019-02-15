import os
import sys
import pickle
import math


import numpy as np
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UTIL_DIR = os.path.join(BASE_DIR, '../../utils')
sys.path.append(UTIL_DIR)
import pc_util

noise_mu = 0
noise_sigma = 0.0012

all_patches_train_pfile = '../data/ShapeNet_v2_point_cloud/03001627/chair_clean_noisy_train.pickle'
all_patches_test_pfile = '../data/ShapeNet_v2_point_cloud/03001627/chair_clean_noisy_test.pickle'

# ! this label assignment should be consistent with the script for generating pickle files!
NOISY_LABEL = 0
CLEAN_LABEL = 1

def generate_camera_view_target_points():
    '''
    gen some samples on a circle on XZ and YZ plane
    '''
    r = 2. # 2 meters away from the center
    angle_step = 12 # in degree
    
    cam_view_points = []
    cam_target_points = []
    for azi_angle in range(0, 360, angle_step):
        azi_radian = math.radians(azi_angle)
        x = r * math.cos(azi_radian)
        y = 0
        z = r * math.sin(azi_radian)
        cam_view_points.extend([x,y,z])
        cam_target_points.extend([0,0,0])

        x = 0
        y = r * math.cos(azi_radian)
        z = r * math.sin(azi_radian)
        cam_view_points.extend([x,y,z])
        cam_target_points.extend([0,0,0])

        x = r * math.cos(azi_radian)
        y = r * math.sin(azi_radian)
        z = 0
        cam_view_points.extend([x,y,z])
        cam_target_points.extend([0,0,0])
    
    return cam_view_points, cam_target_points

if __name__ == "__main__":

    cam_view_points, cam_target_points = generate_camera_view_target_points()
    cam_view_points = np.asarray(cam_view_points).reshape([-1,3])
    cam_target_points = np.asarray(cam_target_points).reshape([-1,3])
    pc_util.write_ply(cam_view_points, 'cam_views.ply')
    pc_util.write_ply(cam_target_points, 'cam_targets.ply')
    

