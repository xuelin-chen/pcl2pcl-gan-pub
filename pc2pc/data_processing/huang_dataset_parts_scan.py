import sys
import os
import multiprocessing
import math

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UTIL_DIR = os.path.join(BASE_DIR, '../../utils')
sys.path.append(UTIL_DIR)
import mesh_util
import pc_util

extracted_parts_dir = '/workspace/dataset/ShapeNetCore.v2/03001627'
otuput_points_dir = '/workspace/pointnet2/pc2pc/data/ShapeNet_v2_point_cloud/03001627/point_cloud_clean_partial'
if not os.path.exists(otuput_points_dir):
    os.makedirs(otuput_points_dir)

####### paras for virtual scanner #########
EXE_VIRTUAL_SCANNER = '/workspace/pcl/build/bin/pcl_virtual_scanner'
VERT_DEGREE_RES = 0.125
HOR_DEGREE_RES = 0.125
VERT_NB_SCANS = 900
HOR_BN_SCANS = 900
cam_mu = 0
cam_sigma = 0.2 # deviation in m
CMD_POSTFIX = '-customized_views 1 -view_points {} -target_points {}' + ' -nr_scans' + ' ' +str(VERT_NB_SCANS) + ' -pts_in_scan' + ' ' + str(HOR_BN_SCANS) + ' -vert_res' + ' ' + str(VERT_DEGREE_RES) + ' -hor_res' + ' ' + str(HOR_DEGREE_RES) + ' -output_name {}'

def generate_camera_view_target_points():
    '''
    gen some samples on a circle on 3 planes
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
        
    # adding noise
    noise_viewpoints = np.random.normal(cam_mu, cam_sigma, len(cam_view_points))
    noise_targpoints = np.random.normal(cam_mu, cam_sigma, len(cam_target_points))

    cam_view_points = cam_view_points + noise_viewpoints
    cam_target_points = cam_target_points + noise_targpoints
    
    return cam_view_points, cam_target_points

def generate_camera_view_target_points_for_partial_scan():
    '''
    gen some samples on a circle on 3 planes
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
        
    # adding noise
    noise_viewpoints = np.random.normal(cam_mu, cam_sigma, len(cam_view_points))
    noise_targpoints = np.random.normal(cam_mu, cam_sigma, len(cam_target_points))

    cam_view_points = cam_view_points + noise_viewpoints
    cam_target_points = cam_target_points + noise_targpoints
    
    cam_views_output = cam_view_points.reshape(-1,3)
    cam_targets_output = cam_target_points.reshape(-1,3)
    pc_util.write_ply(np.concatenate([cam_views_output, cam_targets_output], axis=0), 'cams.ply')
    return cam_view_points, cam_target_points

def scan_one_part(part_ply_filename):
    file_basename = os.path.basename(os.path.dirname(os.path.dirname(part_ply_filename)))
    output_pcd_name = os.path.join(otuput_points_dir, file_basename+'.pcd')
    print(output_pcd_name)

    cam_view_points, cam_target_points = generate_camera_view_target_points_for_partial_scan()

    cmd_str = EXE_VIRTUAL_SCANNER + ' ' + part_ply_filename + ' ' + CMD_POSTFIX.format(','.join(str(e) for e in cam_view_points), ','.join(str(e) for e in cam_target_points), output_pcd_name)
    #print(cmd_str)
    os.system(cmd_str)
    print('Scanning done.')

    all_points = pc_util.read_pcd(output_pcd_name)
    all_points = pc_util.remove_duplicated_points(all_points)
    output_ply_name = os.path.join(otuput_points_dir, file_basename+'.ply')
    pc_util.write_ply(all_points, output_ply_name)
    print(output_ply_name)
    os.remove(output_pcd_name)

if __name__ == "__main__":
    all_parts_filename = [os.path.join(extracted_parts_dir, pf, 'models', 'model_normalized.obj') for pf in os.listdir(extracted_parts_dir)]

    for pf in all_parts_filename:
        scan_one_part(pf)
        break
