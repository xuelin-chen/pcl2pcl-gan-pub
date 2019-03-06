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

############ useful vars in rendering #################
YFOV = np.pi / 3.0
POINT_LIGHT_INTENSITY = 5.
POINT_SIZE = 8

# for real data
#CAM_POSE = get_transformation_matrix(-20,0,0,0,1.0,0)
#PC_POSE = get_transformation_matrix(0,125,0,0,0,-1.5)

#for synthetic data
CAM_POSE = get_transformation_matrix(-20,0,0,0,0.6,0)
PC_POSE = get_transformation_matrix(0,125,0,0,0,-1.0)

#######################################################

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

def get_all_filnames(dir, nb=30):
    all_filenames = [ os.path.join(dir, f) for f in os.listdir(dir)]
    all_filenames.sort()
    return all_filenames[:nb]

def add_point_cloud_mesh_to_scene(point_cloud, scene, pose, pts_colors):
    '''
    pts_colors: np array, Nx3
    point_cloud: NX3, np array
    '''
    points = point_cloud

    mn = Mesh.from_points(points, colors=pts_colors)
    pc_node = scene.add(mn, pose=pose)

    return pc_node

def render_big_gallery_overlay(dir_1, dir_2, pts_color_1=[0.5,0.5,0.5], pts_color_2=[0.5,0.5,0.5], nb=30):
    '''
    return np array of a big image
    '''
    cam = PerspectiveCamera(yfov=(YFOV))
    cam_pose = CAM_POSE
    
    point_l = PointLight(color=np.ones(3), intensity=POINT_LIGHT_INTENSITY)
    scene = Scene(bg_color=np.array([1,1,1,0]))

    # cam and light
    _ = scene.add(cam, pose=cam_pose)
    _ = scene.add(point_l, pose=cam_pose)

    input_ply_filenames_1 = get_all_filnames(dir_1, nb)
    input_ply_filenames_2 = get_all_filnames(dir_2, nb)

    r = OffscreenRenderer(viewport_width=640*2, viewport_height=480*2, point_size=POINT_SIZE)
    pc_pose = PC_POSE

    images = []
    for idx, input_pf in enumerate(input_ply_filenames_1):

        input_pc_1 = read_ply_xyz(input_pf)
        input_pc_2 = read_ply_xyz(input_ply_filenames_2[idx])

        color_1 = np.array(pts_color_1)
        color_1 = np.tile(color_1, (input_pc_1.shape[0], 1))

        color_2 = np.array(pts_color_2)
        color_2 = np.tile(color_2, (input_pc_2.shape[0], 1))

        input_pc_node_1 = add_point_cloud_mesh_to_scene(input_pc_1, scene, pc_pose, color_1)
        input_pc_node_2 = add_point_cloud_mesh_to_scene(input_pc_2, scene, pc_pose, color_2)

        renderred_color, _ = r.render(scene)

        scene.remove_node(input_pc_node_1)
        scene.remove_node(input_pc_node_2)

        images.append(renderred_color)

    big_gallery = np.concatenate(images, axis=0)

    r.delete()

    return big_gallery

def render_big_gallery(results_dir, nb=30, pts_colors=[0.5,0.5,0.5]):
    '''
    pts_colors: [0,0,0]
    return np array of a big image
    '''

    cam = PerspectiveCamera(yfov=(YFOV))
    cam_pose = CAM_POSE
    
    point_l = PointLight(color=np.ones(3), intensity=POINT_LIGHT_INTENSITY)
    scene = Scene(bg_color=np.array([1,1,1,0]))

    # cam and light
    _ = scene.add(cam, pose=cam_pose)
    _ = scene.add(point_l, pose=cam_pose)

    input_ply_filenames = get_all_filnames(results_dir, nb)

    r = OffscreenRenderer(viewport_width=640*2, viewport_height=480*2, point_size=POINT_SIZE)
    pc_pose = PC_POSE

    images = []
    for _, input_pf in enumerate(input_ply_filenames):

        input_pc = read_ply_xyz(input_pf)

        colors = np.array(pts_colors)
        colors = np.tile(colors, (input_pc.shape[0], 1))

        input_pc_node = add_point_cloud_mesh_to_scene(input_pc, scene, pc_pose, colors)

        renderred_color, _ = r.render(scene)
        
        scene.remove_node(input_pc_node)

        images.append(renderred_color)

    big_gallery = np.concatenate(images, axis=0)

    r.delete()

    return big_gallery

if __name__=='__main__':
    nb_chairs_to_show = 50
    cat_name = 'plane'
    test_results_log_dir_1 = '/workspace/pointnet2/pc2pc/run_%s/N2N_ae_test/log_test_ae_plane_50-percentage_np2c_model_1810_2019-03-06-12-07-02'%(cat_name)
    #test_results_log_dir_2 = '/workspace/pointnet2/pc2pc/run_pcl2pcl/log_pcl2pcl_gan_np2c_dualAE_rotation_test_2019-02-20-11-37-03'

    gt_color_arr_1 = render_big_gallery(os.path.join(test_results_log_dir_1, 'pcloud','gt'), nb_chairs_to_show, [.5,0.5,.5])
    input_color_arr_1 = render_big_gallery(os.path.join(test_results_log_dir_1, 'pcloud','input'), nb_chairs_to_show, [.5,0.5,.5])
    recon_color_arr_1 = render_big_gallery(os.path.join(test_results_log_dir_1, 'pcloud','reconstruction'), nb_chairs_to_show, [0,0,1])
    overlaid_1 = render_big_gallery_overlay(os.path.join(test_results_log_dir_1, 'pcloud','input'), os.path.join(test_results_log_dir_1, 'pcloud','reconstruction'), [.5,.5,.5], [0,0,1], nb_chairs_to_show)

    big_1_im = np.concatenate([gt_color_arr_1, input_color_arr_1, recon_color_arr_1, overlaid_1], axis=1)
    big_1_img = Image.fromarray(big_1_im)
    big_1_img.save('exp_completion_test_N2N_%s_50P.png'%(cat_name))

    #input_color_arr_2 = render_big_gallery(os.path.join(test_results_log_dir_2, 'pcloud','input'))
    #recon_color_arr_2 = render_big_gallery(os.path.join(test_results_log_dir_2, 'pcloud', 'reconstruction'))

    #bi_im = np.concatenate([input_color_arr_1, recon_color_arr_1, input_color_arr_2, recon_color_arr_2], axis=1)

    #big_img = Image.fromarray(bi_im)
    #big_img.save('n2n_vs_ours_rotation_test.png')   
    