import os, sys
import numpy as np
from tqdm import tqdm
import mcubes
import trimesh
from transforms3d.axangles import axangle2aff
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pc_util

isoval = 0.5

EPN_test_result_dir = '/workspace/cnncomplete/results_mp_table'
output_dir = os.path.join(ROOT_DIR, 'pc2pc', 'results', 'EPN_method_results_real', 'mp_table')
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

volumn_name_list = os.listdir(EPN_test_result_dir)
for vname in volumn_name_list:
    if not vname.endswith('.npy'): continue
    v_filename = os.path.join(EPN_test_result_dir, vname)
    v = np.load(v_filename)
    vertices, triangles = mcubes.marching_cubes(v, isoval)
    # isosuface from distance field
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    mesh = align_dfmesh_scanpc(mesh, v.shape[0])
    
    # samples from isosurface
    recon_pc, _ = trimesh.sample.sample_surface(mesh, 2048)
    recon_pc = np.array(recon_pc)

    output_pcloud_re_dir = os.path.join(output_dir, 'pcloud', 'reconstruction')
    if not os.path.exists(output_pcloud_re_dir): os.makedirs(output_pcloud_re_dir)
    re_output_filename = os.path.join(output_pcloud_re_dir, vname.split('.')[0]+'.ply')
    pc_util.write_ply(recon_pc, re_output_filename)
    