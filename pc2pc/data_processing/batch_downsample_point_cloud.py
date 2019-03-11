import os,sys
import numpy as np
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, '../utils'))

import pc_util

point_cloud_dir = '/workspace/pointnet2/pc2pc/data/ShapeNet_v1_point_cloud/04379243/point_cloud_clean_full'
output_dir = '/workspace/pointnet2/pc2pc/data/ShapeNet_v1_point_cloud/04379243/point_cloud_clean'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
down_sample_rate = 0.25

ply_filename_list = [os.path.join(point_cloud_dir, f) for f in os.listdir(point_cloud_dir)]
ply_filename_list.sort()

for pf in tqdm(ply_filename_list):
    points = pc_util.read_ply_xyz(pf)

    choice = np.random.choice(points.shape[0], int(points.shape[0]*down_sample_rate))

    sampled_points = points[choice]

    if sampled_points.shape[0] < 1000:
        print('Skip, probably empty scan. %s'%(pf))
        continue

    # ensure that the bbox is centerred at the original
    pts_min = np.amin(sampled_points, axis=0, keepdims=True)
    pts_max = np.amax(sampled_points, axis=0, keepdims=True)
    bbox_center = (pts_min + pts_max) / 2.0
    sampled_points = sampled_points - bbox_center

    output_filename = os.path.join(output_dir, os.path.basename(pf))
    pc_util.write_ply(sampled_points, output_filename)
    