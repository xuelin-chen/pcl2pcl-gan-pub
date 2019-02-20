# excute it with python2 !

import os,sys
import numpy as np
import trimesh
from PIL import Image
from tqdm import tqdm
from pyrender import PerspectiveCamera,\
                     DirectionalLight, SpotLight, PointLight,\
                     MetallicRoughnessMaterial,\
                     Primitive, Mesh, Node, Scene,\
                     Viewer, OffscreenRenderer, RenderFlags
from plyfile import PlyData, PlyElement

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

def get_transformation_matrix(rx, ry, rz, tx, ty, tz):
    angles = np.array([rx, ry, rz])
    angles = angles / 180.0 * np.pi
    Rx = np.array([[1,0,0, 0],
                [0,np.cos(angles[0]),-np.sin(angles[0]),0],
                [0,np.sin(angles[0]),np.cos(angles[0]),0],
                [0,0,0,1]])
    Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1]),0],
                        [0,1,0,0],
                        [-np.sin(angles[1]),0,np.cos(angles[1]),0],
                        [0,0,0,1]])
    Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0,0],
                        [np.sin(angles[2]),np.cos(angles[2]),0,0],
                        [0,0,1,0],
                        [0,0,0,1]])
    T = np.array([
                [1.0, 0.0,  0.0, tx],
                [0.0, 1.0,  0.0, ty],
                [0.0, 0.0,  1.0, tz],
                [0.0, 0.0,  0.0, 1.0],
            ])
    pose = np.dot(Rz, np.dot(Ry,Rx))
    pose = np.dot(T, pose)
    return pose

def add_point_cloud_mesh_to_scene(point_cloud, scene, pose):
    '''
    point_cloud: NX3, np array
    '''
    points = point_cloud

    colors = np.zeros(points.shape)
    colors += 0.5
    mn = Mesh.from_points(points, colors=colors)
    pc_node = scene.add(mn, pose=pose)
    return pc_node

def get_all_filnames(dir, nb=50):
    all_filenames = [ os.path.join(dir, f) for f in os.listdir(dir)]
    all_filenames.sort()
    return all_filenames[:nb]

def render_big_gallery(results_dir):
    '''
    return np array of a big image
    '''
    cam = PerspectiveCamera(yfov=(np.pi / 3.0))
    cam_pose = get_transformation_matrix(-20,0,0,0,0.3,0)
    
    point_l = PointLight(color=np.ones(3), intensity=0.5)
    scene = Scene(bg_color=np.array([1,1,1,0]))

    # cam and light
    _ = scene.add(cam, pose=cam_pose)
    _ = scene.add(point_l, pose=cam_pose)

    input_ply_filenames = get_all_filnames(results_dir)

    r = OffscreenRenderer(viewport_width=640*2, viewport_height=480*2, point_size=5)
    pc_pose = get_transformation_matrix(0,125,0,0,0,-1)

    images = []
    for _, input_pf in enumerate(input_ply_filenames):

        input_pc = read_ply_xyz(input_pf)

        input_pc_node = add_point_cloud_mesh_to_scene(input_pc, scene, pc_pose)
        input_color, _ = r.render(scene)
        scene.remove_node(input_pc_node)

        images.append(input_color)

    big_gallery = np.concatenate(images, axis=0)

    r.delete()

    return big_gallery

if __name__=='__main__':

    test_results_log_dir_1 = '/workspace/pointnet2/pc2pc/run_ae/log_ae_chair_m1850_rotation_np2c_test_2019-02-20-11-41-32'
    test_results_log_dir_2 = '/workspace/pointnet2/pc2pc/run_pcl2pcl/log_pcl2pcl_gan_np2c_dualAE_rotation_test_2019-02-20-11-37-03'

    input_color_arr_1 = render_big_gallery(os.path.join(test_results_log_dir_1, 'pcloud','input'))
    recon_color_arr_1 = render_big_gallery(os.path.join(test_results_log_dir_1, 'pcloud','reconstruction'))

    input_color_arr_2 = render_big_gallery(os.path.join(test_results_log_dir_2, 'pcloud','input'))
    recon_color_arr_2 = render_big_gallery(os.path.join(test_results_log_dir_2, 'pcloud', 'reconstruction'))

    bi_im = np.concatenate([input_color_arr_1, recon_color_arr_1, input_color_arr_2, recon_color_arr_2], axis=1)

    big_img = Image.fromarray(bi_im)
    big_img.save('n2n_vs_ours_rotation_test.png')


    
    
