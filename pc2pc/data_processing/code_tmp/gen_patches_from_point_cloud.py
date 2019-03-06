import sys
import os
import multiprocessing

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UTIL_DIR = os.path.join(BASE_DIR, '../../utils')
sys.path.append(UTIL_DIR)
from GKNN import GKNN


def start_work(gpu_id, worker_id, pc_file_list, output_dir, patch_size=2048, patch_num=100, use_dijkstra=True, scale_ratio=1):

    for pc_file in pc_file_list:
        gm = GKNN(point_path=pc_file, patch_size=patch_size, patch_num=patch_num)
        gm.crop_patch(output_dir, use_dijkstra=use_dijkstra, scale_ratio=scale_ratio, gpu_id=gpu_id)


if __name__ == "__main__":
    point_cloud_ply_dir = '../data/ShapeNet_v2_point_cloud/03001627/point_cloud_clean'
    patch_extraction_out_dir = '../data/ShapeNet_v2_point_cloud/03001627/patch_clean'
    gpu_id = 0

    num_workers = 8
    patch_size = 2048
    patch_num = 100
    use_dijkstra = True
    scale_ratio = 1

    max_num_pointcloud = 200
    num_pc_per_worker = int(int(max_num_pointcloud) / int(num_workers))

    all_pc_files = [os.path.join(point_cloud_ply_dir, pc_f) for pc_f in os.listdir(point_cloud_ply_dir)]
    pc_indices = np.random.permutation(len(all_pc_files))[:max_num_pointcloud]
    pc_files = [all_pc_files[idx] for idx in pc_indices]
    
    pros = []
    for worker_id in range(num_workers):
        workers_file_list = [pc_files[idx] for idx in range(worker_id*num_pc_per_worker, (worker_id+1)*num_pc_per_worker)]
        
        pros.append(multiprocessing.Process(target = start_work, args = (gpu_id, worker_id, workers_file_list, patch_extraction_out_dir,)))
        pros[worker_id].start()

    for worker_id in range(num_workers):
        pros[worker_id].join()

    print('Done!')

