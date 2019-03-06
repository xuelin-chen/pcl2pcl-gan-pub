import os, sys
import glob
import numpy as np
import math
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UTIL_DIR = os.path.join(BASE_DIR, '../../utils')
sys.path.append(UTIL_DIR)
import mesh_util
import pc_util

############################ useful vars ############################
SHAPENET_V2_PATH = '/workspace/dataset/ShapeNetCore.v2'
#cat_synset_id = '03001627' # chair synset number for a specific category in shapnetv2
#cat_synset_id = '02691156' # airplane
#cat_synset_id = '04379243' # table
#cat_synset_id = '03790512' # motorcycle
cat_synset_id = '02958343' # car

OUTPUT_DATA_PATH = os.path.join('../data/ShapeNet_v2_point_cloud', cat_synset_id, 'point_cloud_clean')

EXE_VIRTUAL_SCANNER = '/workspace/pcl/build/bin/pcl_virtual_scanner'
VERT_DEGREE_RES = 0.125
HOR_DEGREE_RES = 0.125
VERT_NB_SCANS = 900
HOR_BN_SCANS = 900
'''
point_mu = 0
point_sigma = 0.0012 # deviation in m
'''
cam_mu = 0
cam_sigma = 0.1 # deviation

CMD_POSTFIX = '-customized_views 1 -view_points {} -target_points {}' + ' '+ '-nr_scans' + ' ' +str(VERT_NB_SCANS) + ' ' + '-pts_in_scan' + ' ' + str(HOR_BN_SCANS) + ' ' + '-vert_res' + ' ' + str(VERT_DEGREE_RES) + ' ' + '-hor_res' + ' ' + str(HOR_DEGREE_RES)

####################################################################

def clean_dir(dirname):
    if not os.path.exists(dirname):
        return
    pcd_files = glob.glob(os.path.join(dirname, '*.pcd'))
    for pf in pcd_files:
        os.remove(pf)

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

def virtual_scane_one_model(model_dir, worker_id):
    print('Scanning ' + model_dir)

    ######### prepare some folder for tmp work #############
    tmp_model_name = 'tmp'+ str(worker_id) +'.ply'
    TMP_DATA_PATH = './tmp' + str(worker_id)
    TMP_PLY_POINTCLOUD_PATH =  './tmp'+str(worker_id)+'.ply_output'
    if not os.path.exists(TMP_DATA_PATH):
        os.makedirs(TMP_DATA_PATH)
    clean_dir(TMP_PLY_POINTCLOUD_PATH)

    # generate camera parameters
    cam_view_points, cam_target_points = generate_camera_view_target_points()

    model_filename = os.path.join(model_dir, 'models/model_normalized.obj')
    if not os.path.exists(model_filename):
        print('File not found: %s'%(model_filename))
        return
    model_basename = os.path.basename(model_dir)

    ply_tmp_name = os.path.join(TMP_DATA_PATH, tmp_model_name)
    mesh_util.convert_obj2ply(model_filename, ply_tmp_name, recenter=True, center_mode='box_center')

    cmd_str = EXE_VIRTUAL_SCANNER + ' ' + ply_tmp_name + ' ' + CMD_POSTFIX.format(','.join(str(e) for e in cam_view_points), ','.join(str(e) for e in cam_target_points))
    os.system(cmd_str)

    # collect all scanned point clouds
    all_xyz = []
    pcd_files = glob.glob(TMP_PLY_POINTCLOUD_PATH + '/*.pcd')
    for pf in pcd_files:
        xyz = pc_util.read_pcd(pf)
        all_xyz.extend(xyz)
    all_points = np.array(all_xyz)
    all_points = pc_util.remove_duplicated_points(all_points)
    print('Total points: %d' % (all_points.shape[0]))
    clean_output_filename = os.path.join(OUTPUT_DATA_PATH, model_basename+'_clean.ply')
    pc_util.write_ply(all_points, clean_output_filename)
    print('Save point cloud to ' + clean_output_filename)

    return

def do_virtual_scan(cat_dir, worker_id, num_workers):
    object_folders = [dir for dir in os.listdir(cat_dir)]
    object_folders.sort()
    print('#Model: %d' % (len(object_folders)))

    # clip out a portion of the folders
    worker_size = int(math.ceil(len(object_folders) / num_workers))
    print('Worker size: ' + str(worker_size))
    start_idx = worker_id * worker_size
    end_idx = start_idx + worker_size

    for o_idx, obj_f in enumerate(object_folders):
        if o_idx >= start_idx and o_idx < end_idx:
            virtual_scane_one_model(os.path.join(cat_dir, obj_f), worker_id)
    print('Done!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker_id', type=int, default=0, help='worker id to work separately')
    parser.add_argument('--num_workers', type=int, default=1, help='worker id to work separately')
    FLAGS = parser.parse_args()
    
    if not os.path.exists(OUTPUT_DATA_PATH):
        os.makedirs(OUTPUT_DATA_PATH)    

    cat_dir = os.path.join(SHAPENET_V2_PATH, cat_synset_id)
    do_virtual_scan(cat_dir, FLAGS.worker_id, FLAGS.num_workers)

