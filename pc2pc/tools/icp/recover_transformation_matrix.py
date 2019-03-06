import os,sys
import trimesh
from ICP_from_global import get_ICP_transform
import numpy as np
import pickle
from tqdm import tqdm

original_mesh_dir = '../../data/scannet_v2_chairs_Y_extracted'
aligned_mesh_dir = '../../data/scannet_v2_chairs_alilgned_v2/point_cloud'
output_dir = 'scannet_v2_chairs_transformed_back'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

obj_mesh_names = os.listdir(aligned_mesh_dir)
obj_mesh_names.sort()

def merge_two_mehses(v1, f1, v2, f2):
    v = []
    v.extend(v1)
    v.extend(v2)
    v = np.asarray(v)

    f2_new = f2 + v1.shape[0]
    f = []
    f.extend(f1)
    f.extend(f2_new)
    f = np.asarray(f)

    return v, f

trans_mat_dict = {}
for om_name in tqdm(obj_mesh_names):
    ori_obj_filename = os.path.join(original_mesh_dir, om_name)
    aligned_obj_filename = os.path.join(aligned_mesh_dir, om_name)

    trans_mat, fitness = get_ICP_transform(aligned_obj_filename, ori_obj_filename)
    if fitness < 0.99:
        print('!WARN: not a good ICP! fitness: %f, obj name: %s'%(fitness, om_name))

    trans_mat_dict[om_name] = trans_mat

    ori_obj_mesh = trimesh.load(ori_obj_filename)
    aligned_obj_mesh = trimesh.load(aligned_obj_filename)
    aligned_obj_mesh.apply_transform(trans_mat)

    aligned_obj_mesh.export(os.path.join(output_dir, om_name))

    new_v, new_f = merge_two_mehses(aligned_obj_mesh.vertices, aligned_obj_mesh.faces, ori_obj_mesh.vertices, ori_obj_mesh.faces)
    new_trimesh = trimesh.Trimesh(new_v, new_f)
    new_trimesh.export(os.path.join(output_dir, om_name))

with open('transformations_dict.pickle', 'wb') as pf:
    pickle.dump(trans_mat_dict, pf)
