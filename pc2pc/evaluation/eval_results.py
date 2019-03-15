import os,sys
import numpy as np
import evaluation_utils
from tqdm import tqdm
import multiprocessing

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, '../utils'))
import pc_util

num_workers = 70
thre = 0.03
#cat_name = 'car'
#test_name = 'vanilla_ae_test'
#test_name = 'N2N_ae_test'
#test_name = 'pcl2pcl_test'
#keyword2filter = '0-partial'
#keyword2filter = '-perc'
#keyword2filter = 'redo'
#keyword2filter = '_gt-retrieved'
keyword2filter = None

#test_dir = '/workspace/pointnet2/pc2pc/run_%s/%s'%(cat_name, test_name)
#test_dir = '/workspace/pointnet2/pc2pc/run_3D-EPN/run_%s/%s'%(cat_name, test_name)
test_dir = '/workspace/pointnet2/pc2pc/test_3D-EPN/test_plane/clean_ae_test'

def gt_isvalid(gt_points):
    pts_max = np.max(gt_points)
    if pts_max < 0.01:
        return False
    return True

def get_3D_EPN_GT_dir(result_dir):
    if 'car' in result_dir:
        return '/workspace/pointnet2/pc2pc/test_3D-EPN/GT/3D-EPN_car_gt'
    elif 'chair' in result_dir:
        return '/workspace/pointnet2/pc2pc/test_3D-EPN/GT/3D-EPN_chair_gt'
    elif 'table' in result_dir:
        return '/workspace/pointnet2/pc2pc/test_3D-EPN/GT/3D-EPN_table_gt'
    elif 'plane' in result_dir:
        return '/workspace/pointnet2/pc2pc/test_3D-EPN/GT/3D-EPN_plane_gt'

def eval_result_folder(result_dir):
    #gt_point_cloud_dir = get_3D_EPN_GT_dir(result_dir)
    gt_point_cloud_dir = os.path.join(result_dir, 'pcloud', 'gt')
    result_point_cloud_dir = os.path.join(result_dir, 'pcloud', 'reconstruction')

    re_pc_names = os.listdir(result_point_cloud_dir)
    re_pc_names.sort()

    all_avg_dist = []
    all_comp_percentage = []
    all_comp_avg_dist = []
    for re_pc_n  in (re_pc_names):

        gt_pc_filename = os.path.join(gt_point_cloud_dir, re_pc_n)
        re_pc_filename = os.path.join(result_point_cloud_dir, re_pc_n)

        gt_pc_pts = pc_util.read_ply_xyz(gt_pc_filename)
        if not gt_isvalid(gt_pc_pts):
            print('Invalid gt point cloud, skip.')
            continue
        
        re_pc_pts = pc_util.read_ply_xyz(re_pc_filename)
        if re_pc_pts.shape[0] < 2048:
            re_pc_pts = pc_util.sample_point_cloud(re_pc_pts, 2048)

        avg_d = evaluation_utils.avg_dist(re_pc_pts, gt_pc_pts)
        comp_perct, comp_avg_dist = evaluation_utils.completeness(re_pc_pts, gt_pc_pts, thre=thre)

        all_avg_dist.append(avg_d)
        all_comp_percentage.append(comp_perct)
        all_comp_avg_dist.append(comp_avg_dist)

    avg_acc_dist = np.mean(all_avg_dist)
    avg_comp_perct = np.mean(all_comp_percentage)
    avg_comp_avg_dist = np.mean(all_comp_avg_dist)

    print('%s - avg_acc_distance, completeness-avg_distance, completeness-percentage: %s,%s, %s'%(result_dir.split('/')[-1], str(avg_acc_dist), str(avg_comp_avg_dist), str(avg_comp_perct)))

result_folders = os.listdir(test_dir)
result_folders.sort()

if keyword2filter is not None:
    result_folders_tmp = []
    for rs in result_folders:
        if keyword2filter in rs:
            result_folders_tmp.append(rs)
    result_folders = result_folders_tmp

print('#folders: ', len(result_folders))

if len(result_folders) <= num_workers:
    num_workers = len(result_folders)
    work_round_nb = 1
else:
    work_round_nb = int(np.ceil(len(result_folders) / num_workers))

count = 0
for wround_idx in range(work_round_nb):
    pros = []
    for worker_id in range(num_workers):

        result_f_idx = worker_id + wround_idx * num_workers
        if result_f_idx >= len(result_folders):
            continue

        res_folder = os.path.join(test_dir, result_folders[result_f_idx])
        pros.append(multiprocessing.Process(target=eval_result_folder, args=(res_folder,)))
        pros[worker_id].start()
        print('start to work on:', res_folder.split('/')[-1])
        count += 1
    
    for worker_id in range(len(pros)):
        pros[worker_id].join()
    
print('#worked folders:', count)
print('Done!')