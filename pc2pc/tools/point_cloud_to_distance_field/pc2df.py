import pc2df_utils
import numpy as np
import os,sys
import pickle
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../../../utils'))
import pc_util

reconstructed_pc_dir = '/workspace/pointnet2/pc2pc/data/ShapeNet_v2_point_cloud/02691156/point_cloud_clean'
output_dir = os.path.join(os.path.dirname(reconstructed_pc_dir), os.path.basename(reconstructed_pc_dir)+'_gt_df')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

resolution = 32

all_recon_ply_names = os.listdir(reconstructed_pc_dir)
for rpn in tqdm(all_recon_ply_names):

    if '1d63eb2b1f78aa88acf77e718d93f3e1' not in rpn:
        continue
    
    ply_filename = os.path.join(reconstructed_pc_dir, rpn)
    recon_pc = pc_util.read_ply_xyz(ply_filename)

    recon_pc = pc_util.rotate_point_cloud_by_axis_angle(recon_pc, [0,1,0], -90)

    recon_df, recon_df_arr = pc2df_utils.convert_pc2df(recon_pc, resolution=resolution)
    
    #output_filename = os.path.join(output_dir, rpn[:-4]+'.txt')
    output_filename = os.path.join(output_dir, rpn[:-10]+'__0__.txt')
    with open(output_filename, 'w') as file:
        file.write('%d %d %d '%(resolution, resolution, resolution))
        for ele in recon_df_arr:
            file.write('%f '%(ele))