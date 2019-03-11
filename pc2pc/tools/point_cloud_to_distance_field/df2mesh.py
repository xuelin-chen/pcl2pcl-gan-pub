import os, sys

import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff

import numpy as np
from skimage import measure

import trimesh
import pymesh
from transforms3d.axangles import axangle2aff
import glob
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../../../utils'))
import pc_util

dim = 128
points_sample_nb = 2048
distance_field_dir = '/workspace/pointnet2/pc2pc/data/3D-EPN_dataset/EPN_results/converted_txt_dim%d'%(dim)
if dim == 128:
    iso_val = 1.0

SHAPENET_POINTCLOUD_DIR = '/workspace/pointnet2/pc2pc/data/ShapeNet_v1_point_cloud'

def find_files(dir, extension='.txt', recursive=True):
    filenames = glob.glob(os.path.join(dir, '**', '*'+extension), recursive=True)
    return filenames

def read_df_from_txt(df_txt_filename):
    
    with open(df_txt_filename, 'r') as file:
        long_line = file.readline()
        numbers = long_line.split(' ')

        dimx, dimy, dimz = int(numbers[0]), int(numbers[1]), int(numbers[2])
        volume_data = np.zeros((dimx, dimy, dimz))
        data = numbers[3:dimx*dimy*dimz+3]
        data_idx = 0
        for z_i in range(dimz):
            for y_i in range(dimy):
                for x_i in range(dimx):
                    volume_data[x_i, y_i, z_i] = float(data[data_idx])
                    data_idx += 1
    return volume_data

def get_clsId_modelId(df_filename):
    cls_id = os.path.dirname(df_filename).split('/')[-1]
    mdl_id = os.path.basename(df_filename).split('_')[0]
    return cls_id, mdl_id

def get_isosurface(volume_data, iso_val):
    vs, fs, _, _ = measure.marching_cubes_lewiner(df, iso_val)
    
    final_mesh = trimesh.Trimesh(vertices=vs, faces=fs, validate=True)
    return final_mesh

def align_dfmesh_scanpc(df_mesh, df_resolution, scan_pc):
    '''
    df_mesh: trimesh
    df_resolution: distance field resolution
    scan_pc: Nx3, np array
    '''
    pts_min = np.amin(scan_pc, axis=0)
    pts_max = np.amax(scan_pc, axis=0)
    pc_extents = pts_max - pts_min
    pc_bbox_center = (pts_max + pts_max) / 2.0
    max_pc_size = np.max(pc_extents)

    df_mesh_extents = df_mesh.bounding_box.extents
    max_mesh_size = np.max(df_mesh_extents)
    
    #scale_factor = max_pc_size / df_resolution
    #scale_factor = max_pc_size / max_mesh_size
    scale_factor = 1.0 / dim
    trans_v = pc_bbox_center -  np.array([df_resolution/2.0, df_resolution/2.0, df_resolution/2.0])
    
    df_mesh.apply_translation(trans_v)
    df_mesh.apply_scale(scale_factor)

    # rotate to align the face direction
    rot_m = axangle2aff([0,1,0], 90/180.0*np.pi)
    df_mesh.apply_transform(rot_m)

    return df_mesh

df_txt_filenames = find_files(distance_field_dir)
for df_txt_fn in tqdm(df_txt_filenames):

    out_dir = os.path.dirname(df_txt_fn) + '_results_log_dimscale-align_iso-%f/pcloud'%(float(iso_val))
    out_gt_dir = os.path.join(out_dir, 'gt')
    out_recon_dir = os.path.join(out_dir, 'reconstruction')
    out_recon_mesh_dir = os.path.join(out_dir, 'reconstruction_mesh')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(out_gt_dir):
        os.mkdir(out_gt_dir)
    if not os.path.exists(out_recon_dir):
        os.mkdir(out_recon_dir)
    if not os.path.exists(out_recon_mesh_dir):
        os.mkdir(out_recon_mesh_dir)

    df = read_df_from_txt(df_txt_fn)
    df_mesh = get_isosurface(df, iso_val)

    cls_id, mdl_id = get_clsId_modelId(df_txt_fn)
    scanned_pc_filename = os.path.join(SHAPENET_POINTCLOUD_DIR, cls_id, 'point_cloud_clean', mdl_id+'_clean.ply')
    if not os.path.exists(scanned_pc_filename):
        print('No scanning available: %s'%(scanned_pc_filename))
        continue
    scan_pc = pc_util.read_ply_xyz(scanned_pc_filename)
    scan_pc =pc_util.sample_point_cloud(scan_pc, points_sample_nb)
    pc_util.write_ply(scan_pc, os.path.join(out_gt_dir, mdl_id+'.ply'))

    df_mesh = align_dfmesh_scanpc(df_mesh, dim, scan_pc)
    df_mesh.export(os.path.join(out_recon_mesh_dir, mdl_id+'.ply'))

    #recon_samples, _ = trimesh.sample.sample_surface_even(df_mesh, points_sample_nb)
    recon_samples, _ = trimesh.sample.sample_surface(df_mesh, points_sample_nb)
    pc_util.write_ply(np.array(recon_samples), os.path.join(out_recon_dir, mdl_id+'.ply'))

