import os,sys
import numpy as np
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, '../utils'))
import pc_util

dataset_dir = '/workspace/pointnet2/pc2pc/data/3D-EPN_dataset/test-images_dim32_sdf_pc'
output_dir = '/workspace/pointnet2/pc2pc/data/3D-EPN_dataset/test-images_dim32_sdf_pc_processed'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cls_ids = os.listdir(dataset_dir)

for cls_id in cls_ids:

    cls_dir = os.path.join(dataset_dir, cls_id)
    output_cls_dir = os.path.join(output_dir, cls_id, 'point_cloud')
    if not os.path.exists(output_cls_dir):
        os.makedirs(output_cls_dir)

    point_cloud_names = os.listdir(cls_dir)
    for pc_n in tqdm(point_cloud_names):

        pc_filename = os.path.join(cls_dir, pc_n)
        out_pc_filename = os.path.join(output_cls_dir, pc_n)

        points = pc_util.read_ply_xyz(pc_filename)
        rotated_points = pc_util.rotate_point_cloud_by_axis_angle(points, [0,1,0], 90)

        pc_util.write_ply(rotated_points, out_pc_filename)

