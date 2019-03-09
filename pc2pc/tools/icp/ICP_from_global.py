# examples/Python/Tutorial/Advanced/global_registration.py

from open3d import *
import numpy as np
import copy

import trimesh

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.0, 0])
    target_temp.paint_uniform_color([0, 0.0, 0.0])
    source_temp.transform(transformation)
    draw_geometries([source_temp, target_temp])

def preprocess_point_cloud(pcd, voxel_size):
    #print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = voxel_down_sample(pcd, voxel_size)

    radius_normal = voxel_size * 2
    #print(":: Estimate normal with search radius %.3f." % radius_normal)
    estimate_normals(pcd_down, KDTreeSearchParamHybrid(
            radius = radius_normal, max_nn = 30))

    radius_feature = voxel_size * 5
    #print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = compute_fpfh_feature(pcd_down,
            KDTreeSearchParamHybrid(radius = radius_feature, max_nn = 100))
    return pcd_down, pcd_fpfh

def prepare_dataset(obj_filename_1, obj_filename_2, voxel_size):
    #print(":: Load two point clouds and disturb initial pose.")
    source_mesh = trimesh.load(obj_filename_1)
    target_mesh = trimesh.load(obj_filename_2)
    source_pcd = PointCloud()
    target_pcd = PointCloud()
    source_pcd.points = Vector3dVector(source_mesh.vertices)
    source_pcd.normals = Vector3dVector(source_mesh.vertex_normals)
    target_pcd.points = Vector3dVector(target_mesh.vertices)
    target_pcd.normals = Vector3dVector(target_mesh.vertex_normals)

    source = source_pcd
    target = target_pcd

    trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    #draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source_mesh, target_mesh, source, target, source_down, target_down, source_fpfh, target_fpfh

def execute_global_registration(
        source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    #print(":: RANSAC registration on downsampled point clouds.")
    #print("   Since the downsampling voxel size is %.3f," % voxel_size)
    #print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            distance_threshold,
            TransformationEstimationPointToPoint(False), 4,
            [CorrespondenceCheckerBasedOnEdgeLength(0.9),
            CorrespondenceCheckerBasedOnDistance(distance_threshold)],
            RANSACConvergenceCriteria(4000000, 500))
    return result

def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_ransac):
    distance_threshold = voxel_size * 0.4
    #print(":: Point-to-plane ICP registration is applied on original point")
    #print("   clouds to refine the alignment. This time we use a strict")
    #print("   distance threshold %.3f." % distance_threshold)
    result = registration_icp(source, target, distance_threshold,
            result_ransac.transformation,
            TransformationEstimationPointToPlane())
    return result

def get_ICP_transform(source_filename_1, target_filename2, voxel_size=0.02):
    #voxel_size = 0.02 # means 1cm for the dataset
    source_mesh, target_mesh, source, target, source_down, target_down, source_fpfh, target_fpfh = \
            prepare_dataset(source_filename_1,target_filename2,voxel_size)

    result_ransac = execute_global_registration(source_down, target_down,
            source_fpfh, target_fpfh, voxel_size)
    #print(result_ransac)
    
    #draw_registration_result(source_down, target_down, result_ransac.transformation)
    
    result_icp = refine_registration(source, target,
            source_fpfh, target_fpfh, voxel_size, result_ransac)
    return result_icp.transformation, result_icp.fitness
    #draw_registration_result(source, target, result_icp.transformation)
    

if __name__ == "__main__":
    voxel_size = 0.02 # means 1cm for the dataset
    source_mesh, target_mesh, source, target, source_down, target_down, source_fpfh, target_fpfh = \
            prepare_dataset('scene0000_02_0_aligned.obj', 'scene0000_02_0_ori.obj', voxel_size)

    result_ransac = execute_global_registration(source_down, target_down,
            source_fpfh, target_fpfh, voxel_size)
    print(result_ransac)
    
    #draw_registration_result(source_down, target_down, result_ransac.transformation)
    
    result_icp = refine_registration(source, target,
            source_fpfh, target_fpfh, voxel_size, result_ransac)
    print(result_icp)
    #draw_registration_result(source, target, result_icp.transformation)

    source_mesh.apply_transform(result_icp.transformation)
    source_mesh.export('transformed.ply')
