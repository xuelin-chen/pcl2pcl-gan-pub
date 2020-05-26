import os, sys
import numpy as np
from transforms3d.axangles import axangle2aff
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(ROOT_DIR, 'pc2pc'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pc_util
import shapenet_pc_dataset
import config
import trimesh
from scipy import spatial
import skimage.measure
from tqdm import tqdm
import h5py
from math import radians
import pymesh

cat_name = 'lamp'
sdf_resolution = 32
sdf_scale = 1.0

cat_name2id = {
                'plane': '02691156',
                'car': '02958343',
                'chair': '03001627',
                'table': '04379243',
                'lamp': '03636649',
                'sofa': '04256520',
                'boat': '04530566',
                'dresser': '02933112'
              }
if cat_name == 'boat':
    input_point_cloud_dir = '/workspace/pcl2pcl-gan/pc2pc/results/test_synthetic_pcl2pcl/test_boat/test_boat_pcl2pcl_gan_synthetic_model_960_-1.0_2019-09-22-14-14-43/pcloud/input'
    mesh_dir = '/workspace/dataset/ShapeNetCore.v1/'+cat_name2id['boat']
    output_dir = 'synthetic_input_sdf/boat'
elif cat_name == 'car':
    input_point_cloud_dir = '/workspace/pcl2pcl-gan/pc2pc/results/test_synthetic_pcl2pcl/test_car/test_car_pcl2pcl_gan_synthetic_model_960_-1.0_2019-09-22-14-14-13/pcloud/input'
    mesh_dir = '/workspace/dataset/ShapeNetCore.v1/'+cat_name2id['car']
    output_dir = 'synthetic_input_sdf/car'
elif cat_name == 'chair':
    input_point_cloud_dir = '/workspace/pcl2pcl-gan/pc2pc/results/test_synthetic_pcl2pcl/test_chair/test_chair_pcl2pcl_gan_synthetic_model_960_-1.0_2019-09-22-14-10-58/pcloud/input'
    mesh_dir = '/workspace/dataset/ShapeNetCore.v1/'+cat_name2id['chair']
    output_dir = 'synthetic_input_sdf/chair'
elif cat_name == 'dresser':
    input_point_cloud_dir = '/workspace/pcl2pcl-gan/pc2pc/results/test_synthetic_pcl2pcl/test_dresser/test_dresser_pcl2pcl_gan_synthetic_model_960_-1.0_2019-09-22-14-15-52/pcloud/input'
    mesh_dir = '/workspace/dataset/ShapeNetCore.v1/'+cat_name2id['dresser']
    output_dir = 'synthetic_input_sdf/dresser'
elif cat_name == 'lamp':
    input_point_cloud_dir = '/workspace/pcl2pcl-gan/pc2pc/results/test_synthetic_pcl2pcl/test_lamp/test_lamp_pcl2pcl_gan_synthetic_model_1000_-1.0_2019-09-23-11-33-38/pcloud/input'
    mesh_dir = '/workspace/dataset/ShapeNetCore.v1/'+cat_name2id['lamp']
    output_dir = 'synthetic_input_sdf/lamp'
elif cat_name == 'plane':
    input_point_cloud_dir = '/workspace/pcl2pcl-gan/pc2pc/results/test_synthetic_pcl2pcl/test_plane/test_plane_pcl2pcl_gan_synthetic_model_960_-1.0_2019-09-22-14-13-40/pcloud/input'
    mesh_dir = '/workspace/dataset/ShapeNetCore.v1/'+cat_name2id['plane']
    output_dir = 'synthetic_input_sdf/plane'
elif cat_name == 'sofa':
    input_point_cloud_dir = '/workspace/pcl2pcl-gan/pc2pc/results/test_synthetic_pcl2pcl/test_sofa/test_sofa_pcl2pcl_gan_synthetic_model_960_-1.0_2019-09-22-14-15-29/pcloud/input'
    mesh_dir = '/workspace/dataset/ShapeNetCore.v1/'+cat_name2id['sofa']
    output_dir = 'synthetic_input_sdf/sofa'
elif cat_name == 'table':
    input_point_cloud_dir = '/workspace/pcl2pcl-gan/pc2pc/results/test_synthetic_pcl2pcl/test_table/test_table_pcl2pcl_gan_synthetic_model_960_-1.0_2019-09-22-14-12-31/pcloud/input'
    mesh_dir = '/workspace/dataset/ShapeNetCore.v1/'+cat_name2id['table']
    output_dir = 'synthetic_input_sdf/table'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def plane_which_side(pts_on_plane, plane_normal, query_pts):
    d = - (plane_normal[0]*pts_on_plane[0] + plane_normal[1]*pts_on_plane[1] + plane_normal[2]*pts_on_plane[2])
    res = np.dot(np.array([plane_normal[0], plane_normal[1], plane_normal[2], d]), np.array([query_pts[0], query_pts[1], query_pts[2], 1]))

    if res > 0: 
        return 1 # positive side
    elif res == 0:
        return 0 # on
    else:
        return -1 # negtive side

def pc2df(pc, df_scale, df_resolution):
    vox_len = df_scale / df_resolution
    x_coords = np.linspace(-df_scale/2. + vox_len/2.,  df_scale/2. - vox_len/2., df_resolution)
    y_coords = np.linspace(-df_scale/2. + vox_len/2.,  df_scale/2. - vox_len/2., df_resolution)
    z_coords = np.linspace(-df_scale/2. + vox_len/2.,  df_scale/2. - vox_len/2., df_resolution)
    xv, yv, zv = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    xv = np.expand_dims(xv, axis=-1)
    yv = np.expand_dims(yv, axis=-1)
    zv = np.expand_dims(zv, axis=-1)
    xyz_center_arr = np.concatenate((xv, yv, zv), axis=-1)

    sdf_volumn = np.ones((df_resolution, df_resolution, df_resolution)) # channels for distance and sign

    tree = spatial.cKDTree(pc)
    # 
    print('Querying all points...')
    distances, pts_indices = tree.query(np.reshape(xyz_center_arr, (-1, 3)))
    distances = np.reshape(distances, (df_resolution, df_resolution, df_resolution))
    pts_indices = np.reshape(pts_indices, (df_resolution, df_resolution, df_resolution))
    print('Querying done.')

    for i in range(df_resolution):
        for j in range(df_resolution):
            for k in range(df_resolution):
                xyz_center = xyz_center_arr[i, j, k]
                dist, pts_idx = distances[i,j,k], pts_indices[i,j,k]
                target_pts = pc[pts_idx, :3]
                sdf_volumn[i, j, k] = dist
    
    return sdf_volumn

def mesh_from_volumedata(v, level=0., spacing=[1.]*3):
    verts, faces, normals_, values = skimage.measure.marching_cubes_lewiner(v, level=level, spacing=spacing)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    return mesh

sdf_data = []
model_id_list = []
model_list = os.listdir(input_point_cloud_dir)
model_list.sort()
for model_name in model_list:
    #if '90c6d1df1f83329fe1181b0e584cdf9b_clean' not in model_name: continue

    model_id = model_name.split('_')[0]
    input_pc_filename = os.path.join(input_point_cloud_dir, model_name)
    gt_mesh_filename = os.path.join(mesh_dir, model_id, 'model.obj')

    # pc from -z to +z face
    input_pc = pc_util.read_ply(input_pc_filename)
    input_pc = pc_util.rotate_point_cloud_by_axis_angle(input_pc, [0,1,0], 180)

    # mesh from +x to +z face
    gt_mesh_pymesh = pymesh.load_mesh(gt_mesh_filename)
    gt_mesh = trimesh.Trimesh(vertices=(gt_mesh_pymesh.vertices).copy(), faces=(gt_mesh_pymesh.faces).copy())
    gt_mesh_pts_min = np.amin(gt_mesh.vertices, axis=0)
    gt_mesh_pts_max = np.amax(gt_mesh.vertices, axis=0)
    bbox_center = (gt_mesh_pts_max + gt_mesh_pts_min) / 2.0
    trans_v = -bbox_center
    gt_mesh.apply_translation(trans_v)
    gt_mesh_points, gt_mesh_sample_fidx  = trimesh.sample.sample_surface(gt_mesh, 20480)
    gt_mesh_point_normals = gt_mesh.face_normals[gt_mesh_sample_fidx]
    gt_mesh_points = pc_util.rotate_point_cloud_by_axis_angle(gt_mesh_points, [0,1,0], -90)
    gt_mesh_point_normals = pc_util.rotate_point_cloud_by_axis_angle(gt_mesh_point_normals, [0,1,0], -90)
    #pc_util.write_ply_versatile(gt_mesh_points, os.path.join(output_dir, model_id+'_gt_points_0.ply'), normals=gt_mesh_point_normals)

    gt_df = pc2df(gt_mesh_points, 1, 128)
    gt_mesh_recon = mesh_from_volumedata(gt_df, level=0.01, spacing=[1/128]*3)
    gt_mesh_recon.apply_translation([-0.5,-0.5,-0.5])
    #gt_mesh_recon.export(os.path.join(output_dir, model_id+'_meshfromdf.ply'))
    gt_mesh_points, gt_mesh_sample_fidx  = trimesh.sample.sample_surface(gt_mesh_recon, 20480)
    gt_mesh_point_normals = gt_mesh_recon.face_normals[gt_mesh_sample_fidx]
    #pc_util.write_ply_versatile(gt_mesh_points, os.path.join(output_dir, model_id+'_gt_points_1.ply'), normals=gt_mesh_point_normals)
    
    # get normals for input pc from gt points
    #print('Querying all input points...')
    gt_tree = spatial.cKDTree(gt_mesh_points)
    distances, pts_indices = gt_tree.query(input_pc)
    input_pc_normals = gt_mesh_point_normals[pts_indices]
    #pc_util.write_ply_versatile(input_pc, os.path.join(output_dir, model_id+'_input_points.ply'), normals=input_pc_normals)
    print('Querying done.')

    # get centers of volumn voxels
    vox_len = sdf_scale / sdf_resolution
    x_coords = np.linspace(-sdf_scale/2. + vox_len/2.,  sdf_scale/2. - vox_len/2., sdf_resolution)
    y_coords = np.linspace(-sdf_scale/2. + vox_len/2.,  sdf_scale/2. - vox_len/2., sdf_resolution)
    z_coords = np.linspace(-sdf_scale/2. + vox_len/2.,  sdf_scale/2. - vox_len/2., sdf_resolution)
    xv, yv, zv = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    xv = np.expand_dims(xv, axis=-1)
    yv = np.expand_dims(yv, axis=-1)
    zv = np.expand_dims(zv, axis=-1)
    xyz_center_arr = np.concatenate((xv, yv, zv), axis=-1)

    sdf_volumn = np.ones((sdf_resolution, sdf_resolution, sdf_resolution, 2)) # channels for distance and sign

    tree = spatial.cKDTree(input_pc) 
    # 
    print('Querying all points...')
    distances, pts_indices = tree.query(np.reshape(xyz_center_arr, (-1, 3)))
    distances = np.reshape(distances, (sdf_resolution, sdf_resolution, sdf_resolution))
    pts_indices = np.reshape(pts_indices, (sdf_resolution, sdf_resolution, sdf_resolution))
    print('Querying done.')

    for i in tqdm(range(sdf_resolution)):
        for j in range(sdf_resolution):
            for k in range(sdf_resolution):
                xyz_center = xyz_center_arr[i, j, k]
                dist, pts_idx = distances[i,j,k], pts_indices[i,j,k]

                # debug
                target_pts = input_pc[pts_idx, :3]
                #target_pts = gt_mesh_points[pts_idx, :3]
                target_nor = input_pc_normals[pts_idx, :3]
                #target_nor = gt_mesh_point_normals[pts_idx, :3]

                # check the sign of distance
                side = plane_which_side(target_pts, target_nor, xyz_center)

                sdf_volumn[i, j, k, 0] = dist / vox_len # sdf in voxel space
                sdf_volumn[i, j, k, 1] = 1.
                if side == 1: # positive side
                    sdf_volumn[i, j, k, 0] *= -1. 
                    sdf_volumn[i, j, k, 1] = -1.
                else:
                    sdf_volumn[i, j, k, 0] *= 1.
                    sdf_volumn[i, j, k, 1] = 1.
    
    verts, faces, normals_, values = skimage.measure.marching_cubes_lewiner(sdf_volumn[:,:,:,0], level=0.02, spacing=[1.] * 3)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.export(os.path.join(output_dir, model_id+'_sdf_recon.ply'))
    
    sdf_for_epn = np.moveaxis(sdf_volumn, -1, 0)
    sdf_data.append(sdf_for_epn)
    model_id_list.append(model_id)

sdf_data = np.array(sdf_data)
print(sdf_data.shape)
hf = h5py.File(os.path.join(output_dir, 'data.h5'), 'w')
hf.create_dataset('data', data=sdf_data)
with open(os.path.join(output_dir, 'names.txt'), 'w') as f:
    for item in model_id_list:
        f.write("%s\n" % item)
hf.close()
