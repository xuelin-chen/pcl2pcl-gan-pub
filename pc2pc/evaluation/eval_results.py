import os,sys
import numpy as np
import evaluation_utils
from tqdm import tqdm
import multiprocessing
from random import sample

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PC2PC_DIR = os.path.dirname(BASE_DIR)
print(BASE_DIR)
print('BASE_DIR:', BASE_DIR)
print('PC2PC_DIR: ', PC2PC_DIR)
sys.path.append(os.path.join(PC2PC_DIR, '../utils'))
import pc_util

# result folder to evaluate
test_dir = os.path.join(PC2PC_DIR, 'test_3D-EPN/val_pcl2pcl_car/all_models_ShapeNetV1-GT')

num_samples = None
num_workers = 70
thre = 0.03

keyword2filter = None

def gt_isvalid(gt_points):
    pts_max = np.max(gt_points)
    if pts_max < 0.01:
        return False
    return True

def eval_result_folder(result_dir):
    gt_point_cloud_dir = os.path.join(result_dir, 'pcloud', 'gt')
    result_point_cloud_dir = os.path.join(result_dir, 'pcloud', 'reconstruction')

    re_pc_names = os.listdir(result_point_cloud_dir)
    re_pc_names.sort()

    # randomly evaluate a part of the results
    if num_samples is not None:
        re_pc_names = sample(re_pc_names, num_samples)

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
    print('\tacc_avg_distance, acc_percentage, completeness-avg_distance, completeness-percentage, F1: %s,%s,%s,%s,%s'%(str(avg_acc_avg_dist), str(avg_acc_perct), str(avg_comp_avg_dist), str(avg_comp_perct), str(f1_score)))


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