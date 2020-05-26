import os,sys
import numpy as np
import evaluation_utils
from tqdm import tqdm
import multiprocessing

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, '../utils'))
import pc_util

num_workers = 10
thre = 0.03
keyword2filter = None

test_dir = '/workspace/pcl2pcl-gan/pc2pc/results/test_3D-EPN_pcl2pcl_sharedAE'
#test_dir = '/workspace/pcn/results'

def gt_isvalid(gt_points):
    pts_max = np.max(gt_points)
    if pts_max < 0.01:
        return False
    return True

def eval_result_folder(result_dir):
    #gt_point_cloud_dir = get_3D_EPN_GT_dir(result_dir)
    gt_point_cloud_dir = os.path.join(result_dir, 'pcloud', 'gt')
    result_point_cloud_dir = os.path.join(result_dir, 'pcloud', 'reconstruction')

    re_pc_names = os.listdir(result_point_cloud_dir)
    re_pc_names.sort()

    all_acc_percentage = []
    all_acc_avg_dist = []
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

        acc_perct, acc_avg_dist = evaluation_utils.accuracy(re_pc_pts, gt_pc_pts, thre=thre)
        comp_perct, comp_avg_dist = evaluation_utils.completeness(re_pc_pts, gt_pc_pts, thre=thre)

        all_acc_percentage.append(acc_perct)
        all_acc_avg_dist.append(acc_avg_dist)
        all_comp_percentage.append(comp_perct)
        all_comp_avg_dist.append(comp_avg_dist)

    avg_acc_perct = np.mean(all_acc_percentage)
    avg_acc_avg_dist = np.mean(all_acc_avg_dist)
    avg_comp_perct = np.mean(all_comp_percentage)
    avg_comp_avg_dist = np.mean(all_comp_avg_dist)

    f1_score = evaluation_utils.compute_F1_score(avg_acc_perct, avg_comp_perct)

    print('%s:'%(result_dir.split('/')[-1]))
    print('\tacc_avg_distance, completeness-avg_distance: %s,%s'%(str(avg_acc_avg_dist), str(avg_comp_avg_dist)))
    print('\tacc_percentage, completeness-percentage, F1: %s,%s,%s'%(str(avg_acc_perct*100.), str(avg_comp_perct*100.), str(f1_score*100.)))

print('Working directory:', test_dir)
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