import os,sys
import trimesh
from plyfile import PlyData, PlyElement
import numpy as np
from tqdm import tqdm

#sphere_r = 0.01
sphere_r = 0.008
#sphere_r = 0.005


def read_ply_xyz(filename):
    """ read XYZ point cloud from filename PLY file """
    assert(os.path.isfile(filename))
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 3], dtype=np.float32)
        vertices[:,0] = plydata['vertex'].data['x']
        vertices[:,1] = plydata['vertex'].data['y']
        vertices[:,2] = plydata['vertex'].data['z']
    return vertices

def convert_point_cloud_to_balls(pc_ply_filename, output_filename):
    pc = read_ply_xyz(pc_ply_filename)

    points = []
    faces = []

    for pts in tqdm(pc):
        sphere_m = trimesh.creation.uv_sphere(radius=sphere_r, count=[16,16])
        sphere_m.apply_translation(pts)

        faces_offset = np.array(sphere_m.faces) + len(points)
        faces.extend(faces_offset)
        points.extend(np.array(sphere_m.vertices))
    
    points = np.array(points)
    faces = np.array(faces)
    print(points.shape, faces.shape)
    finale_mesh = trimesh.Trimesh(vertices=points, faces=faces)
    finale_mesh.export(output_filename)
    #write_obj(points, faces, output_filename)

convert_point_cloud_to_balls('pcl2pcl_motor_30P_25_gt.ply', 'pcl2pcl_motor_30P_25_gt_spheres.obj')
convert_point_cloud_to_balls('pcl2pcl_motor_30P_25_recon.ply', 'pcl2pcl_motor_30P_25_recon_spheres.obj')
convert_point_cloud_to_balls('pcl2pcl_motor_30P_25_input.ply', 'pcl2pcl_motor_30P_25_input_spheres.obj')