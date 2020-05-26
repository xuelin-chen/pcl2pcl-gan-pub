import os, sys
import numpy as np
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(ROOT_DIR, 'pc2pc'))
import shapenet_pc_dataset
import config
import trimesh
from scipy import spatial
import skimage.measure
from tqdm import tqdm
import h5py
from transforms3d.axangles import axangle2aff
from math import radians

cat_name = 'MP_chair'
sdf_resolution = 32
sdf_scale = 1.0

if cat_name == 'scannet_chair':
    real_point_cloud_dir = config.real_scannet_chair_aligned_data_dir
elif cat_name == 'scannet_table':
    real_point_cloud_dir = config.real_scannet_table_aligned_data_dir
elif cat_name == 'MP_chair':
    real_point_cloud_dir = config.real_MP_chair_aligned_data_dir
elif cat_name == 'MP_table':
    real_point_cloud_dir = config.real_MP_table_aligned_data_dir
output_dir = os.path.dirname(real_point_cloud_dir)+'_sdf'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

REAL_TEST_DATASET = shapenet_pc_dataset.RealWorldPointsDataset(real_point_cloud_dir, batch_size=1, npoint=2048, shuffle=False, split='test')

def plane_which_side(pts_on_plane, plane_normal, query_pts):
    d = - (plane_normal[0]*pts_on_plane[0] + plane_normal[1]*pts_on_plane[1] + plane_normal[2]*pts_on_plane[2])
    res = np.dot(np.array([plane_normal[0], plane_normal[1], plane_normal[2], d]), np.array([query_pts[0], query_pts[1], query_pts[2], 1]))

    if res > 0: 
        return 1 # positive side
    elif res == 0:
        return 0 # on
    else:
        return -1 # negtive side

sdf_data = []
for mesh in REAL_TEST_DATASET.meshes:

    # v2 (-z) -> 3D-EPN (+z)
    rot_affmat = axangle2aff([0,1,0], radians(180))
    mesh.apply_transform(rot_affmat)

    mesh_points, mesh_sample_fidx  = trimesh.sample.sample_surface(mesh, 20480)
    mesh_point_normals = mesh.face_normals[mesh_sample_fidx]
    print('Construct cKD-tree from %d points.'%(mesh_points.shape[0]))
    tree = spatial.cKDTree(mesh_points)
    
    # get centers of volumn voxels
    vox_len = sdf_scale / sdf_resolution
    x_coords = np.linspace(-sdf_scale/2. + vox_len/2.,  sdf_scale/2. - vox_len/2., sdf_resolution)
    y_coords = np.linspace(-sdf_scale/2. + vox_len/2.,  sdf_scale/2. - vox_len/2., sdf_resolution)
    z_coords = np.linspace(-sdf_scale/2. + vox_len/2.,  sdf_scale/2. - vox_len/2., sdf_resolution)
    xv, yv, zv = np.meshgrid(x_coords, y_coords, z_coords, indexing='xy')
    xv = np.expand_dims(xv, axis=-1)
    yv = np.expand_dims(yv, axis=-1)
    zv = np.expand_dims(zv, axis=-1)
    xyz_center_arr = np.concatenate((xv, yv, zv), axis=-1)

    sdf_volumn = np.ones((sdf_resolution, sdf_resolution, sdf_resolution, 2)) # channels for distance and sign

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

                target_pts = mesh_points[pts_idx, :3]
                target_nor = mesh_point_normals[pts_idx, :3]

                # check the sign of distance
                side = plane_which_side(target_pts, target_nor, xyz_center)

                sdf_volumn[i, j, k, 0] = dist / vox_len # sdf in voxel space
                sdf_volumn[i, j, k, 1] = 1.
                if side == 1: # positive side
                    sdf_volumn[i, j, k, 0] *= -1. # using -1 for positive side in sdf representation
                    sdf_volumn[i, j, k, 1] = -1.
                else:
                    sdf_volumn[i, j, k, 0] *= 1.
                    sdf_volumn[i, j, k, 1] = 1.
    '''     
    verts, faces, normals_, values = skimage.measure.marching_cubes_lewiner(sdf_volumn[:,:,:,0], level=0.0, spacing=[vox_len] * 3)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    '''

    sdf_for_epn = np.moveaxis(sdf_volumn, -1, 0)
    sdf_data.append(sdf_for_epn)

sdf_data = np.array(sdf_data)
print(sdf_data.shape)
hf = h5py.File(os.path.join(output_dir, 'data.h5'), 'w')
hf.create_dataset('data', data=sdf_data)
hf.close()
