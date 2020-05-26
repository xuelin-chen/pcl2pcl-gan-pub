import os, sys
import numpy as np
from tqdm import tqdm
import mcubes
import trimesh
from transforms3d.axangles import axangle2aff
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pc_util
from shutil import copyfile

cat_name = 'plane'
isoval = 0.5

EPN_test_result_dir = '/workspace/cnncomplete/results_synthetic_'+cat_name
EPN_test_name_filename = '/workspace/pcl2pcl-gan/pc2pc/data_processing/synthetic_input_sdf/%s/names.txt'%(cat_name)
if cat_name == 'boat':
    EPN_input_gt_dir = '/workspace/pcl2pcl-gan/pc2pc/results/test_synthetic_pcl2pcl/test_boat/test_boat_pcl2pcl_gan_synthetic_model_960_-1.0_2019-09-22-14-14-43/pcloud'
elif cat_name == 'car':
    EPN_input_gt_dir = '/workspace/pcl2pcl-gan/pc2pc/results/test_synthetic_pcl2pcl/test_car/test_car_pcl2pcl_gan_synthetic_model_960_-1.0_2019-09-22-14-14-13/pcloud'
elif cat_name == 'chair':
    EPN_input_gt_dir = '/workspace/pcl2pcl-gan/pc2pc/results/test_synthetic_pcl2pcl/test_chair/test_chair_pcl2pcl_gan_synthetic_model_960_-1.0_2019-09-22-14-10-58/pcloud'
elif cat_name == 'dresser':
    EPN_input_gt_dir = '/workspace/pcl2pcl-gan/pc2pc/results/test_synthetic_pcl2pcl/test_dresser/test_dresser_pcl2pcl_gan_synthetic_model_960_-1.0_2019-09-22-14-15-52/pcloud'
elif cat_name == 'lamp':
    EPN_input_gt_dir = '/workspace/pcl2pcl-gan/pc2pc/results/test_synthetic_pcl2pcl/test_lamp/test_lamp_pcl2pcl_gan_synthetic_model_1000_-1.0_2019-09-23-11-33-38/pcloud'
elif cat_name == 'plane':
    EPN_input_gt_dir = '/workspace/pcl2pcl-gan/pc2pc/results/test_synthetic_pcl2pcl/test_plane/test_plane_pcl2pcl_gan_synthetic_model_960_-1.0_2019-09-22-14-13-40/pcloud'
elif cat_name == 'sofa':
    EPN_input_gt_dir = '/workspace/pcl2pcl-gan/pc2pc/results/test_synthetic_pcl2pcl/test_sofa/test_sofa_pcl2pcl_gan_synthetic_model_960_-1.0_2019-09-22-14-15-29/pcloud'
elif cat_name == 'table':
    EPN_input_gt_dir = '/workspace/pcl2pcl-gan/pc2pc/results/test_synthetic_pcl2pcl/test_table/test_table_pcl2pcl_gan_synthetic_model_960_-1.0_2019-09-22-14-12-31/pcloud'
EPN_input_dir = os.path.join(EPN_input_gt_dir, 'input')
EPN_gt_dir = os.path.join(EPN_input_gt_dir, 'gt')
output_dir = os.path.join(ROOT_DIR, 'pc2pc', 'results', 'EPN_method_results_synthetic', cat_name)
if not os.path.exists(output_dir): os.makedirs(output_dir)

def align_dfmesh_scanpc(df_mesh, df_resolution):
    '''
    df_mesh: trimesh
    df_resolution: distance field resolution
    scan_pc: Nx3, np array
    '''
    pc_bbox_center = np.array([0.,0.,0.])

    df_mesh_extents = df_mesh.bounding_box.extents
    max_mesh_size = np.max(df_mesh_extents)
    
    scale_factor = 1.0 / df_resolution
    trans_v = pc_bbox_center -  np.array([df_resolution/2.0, df_resolution/2.0, df_resolution/2.0])
    
    df_mesh.apply_translation(trans_v)
    df_mesh.apply_scale(scale_factor)

    # rotate to make the face -z, from +z to -z
    rot_m = axangle2aff([0,1,0], np.pi)
    df_mesh.apply_transform(rot_m)

    return df_mesh

with open(EPN_test_name_filename) as f:
    model_name_list = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    model_name_list = [x.strip()+'_clean.ply' for x in model_name_list]
volumn_name_list = os.listdir(EPN_test_result_dir)
for vname in volumn_name_list:
    if not vname.endswith('.npy'): continue

    v_idxname = int(vname.split('.')[0])
    model_name = model_name_list[v_idxname]

    v_filename = os.path.join(EPN_test_result_dir, vname)
    v = np.load(v_filename)
    vertices, triangles = mcubes.marching_cubes(v, isoval)
    # isosuface from distance field
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    mesh = align_dfmesh_scanpc(mesh, v.shape[0])
    
    # samples from isosurface
    recon_pc, _ = trimesh.sample.sample_surface(mesh, 2048)
    recon_pc = np.array(recon_pc)

    output_pcloud_in_dir = os.path.join(output_dir, 'pcloud', 'input')
    if not os.path.exists(output_pcloud_in_dir): os.makedirs(output_pcloud_in_dir)
    in_output_filename = os.path.join(output_pcloud_in_dir, model_name)
    in_source_filename = os.path.join(EPN_input_dir, model_name)
    if not os.path.exists(in_source_filename):
        print('Skip: ', in_source_filename)
        continue
    copyfile(in_source_filename, in_output_filename)

    output_pcloud_gt_dir = os.path.join(output_dir, 'pcloud', 'gt')
    if not os.path.exists(output_pcloud_gt_dir): os.makedirs(output_pcloud_gt_dir)
    gt_output_filename = os.path.join(output_pcloud_gt_dir, model_name)
    gt_source_filename = os.path.join(EPN_gt_dir, model_name)
    if not os.path.exists(gt_source_filename):
        print('Skip: ', gt_source_filename)
        continue
    copyfile(gt_source_filename, gt_output_filename)

    output_pcloud_re_dir = os.path.join(output_dir, 'pcloud', 'reconstruction')
    if not os.path.exists(output_pcloud_re_dir): os.makedirs(output_pcloud_re_dir)
    re_output_filename = os.path.join(output_pcloud_re_dir, model_name)
    pc_util.write_ply(recon_pc, re_output_filename)
    