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

EPN_test_result_dir = '/workspace/cnncomplete/results'
EP_test_label_filename = '/workspace/cnncomplete/data/h5_shapenet_dim32_sdf/test_file_label.txt'
output_dir = os.path.join(ROOT_DIR, 'pc2pc', 'results', 'EPN_method_results')
if not os.path.exists(output_dir): os.makedirs(output_dir)

def get_gt_point_clouds(cls_id, model_name, sample_nb=2048):
    SCAN_PC_DIR = '/workspace/pcl2pcl-gan/pc2pc/data/ShapeNet_v1_point_cloud'
    pc_dir = os.path.join(SCAN_PC_DIR, cls_id, 'point_cloud_clean')

    mn = model_name
    gt_pc_filename = os.path.join(pc_dir, mn+'_clean.ply')
    if not os.path.exists(gt_pc_filename):
        print('GT points not found: %s'%(gt_pc_filename))        
        return np.zeros((sample_nb, 3))

    gt_pc = pc_util.read_ply_xyz(gt_pc_filename)
    if 'v1' in SCAN_PC_DIR:
        # for v1 data, rotate it to align with v2 (-z face)
        gt_pc = pc_util.rotate_point_cloud_by_axis_angle(gt_pc, [0,1,0], 90)
    gt_pc = pc_util.sample_point_cloud(gt_pc, sample_nb)
    
    return gt_pc

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
    
    scale_factor = 1.0 / df_resolution
    trans_v = pc_bbox_center -  np.array([df_resolution/2.0, df_resolution/2.0, df_resolution/2.0])
    
    df_mesh.apply_translation(trans_v)
    df_mesh.apply_scale(scale_factor)

    # rotate to make the face -z
    rot_m = axangle2aff([0,1,0], np.pi)
    df_mesh.apply_transform(rot_m)

    return df_mesh

test_lineList = [line.rstrip('\n') for line in open(EP_test_label_filename)]
# parse model name and class id for each test sample
model_name_list = []
cls_id_list = []
scan_name_list = []
for tl in test_lineList:
    sdf_filename = tl.split(' ')[0]
    scan_name = sdf_filename.split('/')[-1]
    model_name = scan_name.split('_')[0]
    cls_id = sdf_filename.split('/')[1]

    model_name_list.append(model_name)
    cls_id_list.append(cls_id)
    scan_name_list.append(scan_name.split('.')[0])

test_name_list = os.listdir(EPN_test_result_dir)
test_name_list.sort()
for tn in tqdm(test_name_list):
    test_idx = int(tn.split('.')[0])

    model_name = model_name_list[test_idx]
    cls_id = cls_id_list[test_idx]

    volumn_filename = os.path.join(EPN_test_result_dir, tn)
    v = np.load(volumn_filename)
    vertices, triangles = mcubes.marching_cubes(v, isoval)
    # isosuface from distance field
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    # gt pc
    gt_pc = get_gt_point_clouds(cls_id, model_name)
    new_mesh = align_dfmesh_scanpc(mesh, v.shape[0], gt_pc)
    
    # samples from isosurface
    recon_pc, _ = trimesh.sample.sample_surface(new_mesh, 2048)
    recon_pc = np.array(recon_pc)

    # prepare to write out
    output_pcloud_gt_dir = os.path.join(output_dir, cls_id, 'pcloud', 'gt')
    output_pcloud_re_dir = os.path.join(output_dir, cls_id, 'pcloud', 'reconstruction')
    if not os.path.exists(output_pcloud_gt_dir): os.makedirs(output_pcloud_gt_dir)
    if not os.path.exists(output_pcloud_re_dir): os.makedirs(output_pcloud_re_dir)
    gt_output_filename = os.path.join(output_pcloud_gt_dir, scan_name_list[test_idx]+'.ply')
    re_output_filename = os.path.join(output_pcloud_re_dir, scan_name_list[test_idx]+'.ply')
    pc_util.write_ply(gt_pc, gt_output_filename)
    pc_util.write_ply(recon_pc, re_output_filename)
