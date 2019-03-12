import os

#################### datasets ###########################
ShapeNet_v1_point_cloud_dir = '/workspace/pointnet2/pc2pc/data/ShapeNet_v1_point_cloud'
ShapeNet_v2_point_cloud_dir = '/workspace/pointnet2/pc2pc/data/ShapeNet_v2_point_cloud'

EPN_dataset_train_dir = '/workspace/pointnet2/pc2pc/data/3D-EPN_dataset/shapenet_dim32_sdf_pc'
EPN_dataset_test_dir = '/workspace/pointnet2/pc2pc/data/3D-EPN_dataset/test-images_dim32_sdf_pc'
EPN_results_dir = '/workspace/pointnet2/pc2pc/data/3D-EPN_dataset/EPN_results'

real_chair_aligned_data_dir = '/workspace/pointnet2/pc2pc/data/scannet_v2_chairs_alilgned_v2/point_cloud'
real_chair_ori_data_dir = '/workspace/pointnet2/pc2pc/data/scannet_v2_chairs_Y_extracted'

#################### models for synthetic data ################################

################ AE ckpt - ShapeNet v2
AE_car_c2c_ShapeNetV2_ckpt = '/workspace/pointnet2/pc2pc/run_synthetic/run_car/ae/log_ae_car_percentage_c2c_2019-03-06-15-50-06/ckpts/model_1900.ckpt'
AE_car_np2np_ShapeNetV2_ckpt = '/workspace/pointnet2/pc2pc/run_synthetic/run_car/ae/log_ae_car_percentage_np2np_2019-03-06-15-50-22/ckpts/model_1920.ckpt'

AE_chair_c2c_ShapeNetV2_ckpt = '/workspace/pointnet2/pc2pc/run_synthetic/run_chair/ae/log_ae_chair_c2c_2019-02-14-20-05-24/ckpts/model_1600.ckpt'
AE_chair_np2np_ShapeNetV2_ckpt = '/workspace/pointnet2/pc2pc/run_synthetic/run_chair/ae/log_ae_chair_percent_np2np_2019-03-01-21-13-28/ckpts/model_1800.ckpt'

AE_table_c2c_ShapeNetV2_ckpt = '/workspace/pointnet2/pc2pc/run_synthetic/run_table/ae/log_ae_table_c2c_2019-02-28-14-52-10/ckpts/model_1810.ckpt'
AE_table_np2np_ShapeNetV2_ckpt = '/workspace/pointnet2/pc2pc/run_synthetic/run_table/ae/log_ae_table_percent_np2np_2019-03-01-21-20-13/ckpts/model_1540.ckpt'

AE_plane_c2c_ShapeNetV2_ckpt = '/workspace/pointnet2/pc2pc/run_synthetic/run_plane/ae/log_ae_plane_percent_c2c_2019-03-04-16-22-26/ckpts/model_1820.ckpt'
AE_plane_np2np_ShapeNetV2_ckpt = '/workspace/pointnet2/pc2pc/run_synthetic/run_plane/ae/log_ae_plane_percent_np2np_2019-03-04-16-22-53/ckpts/model_1810.ckpt'

################ pcl2pcl ckpt - ShapeNet v2
pcl2pcl_car_ShapeNetV2_ckpt = '/workspace/pointnet2/pc2pc/run_synthetic/run_car/pcl2pcl/log_car_pcl2pcl_gan_percentage_hausdorff_2019-03-07-12-06-59/ckpts/model_710.ckpt'
pcl2pcl_chair_ShapeNetV2_ckpt = '/workspace/pointnet2/pc2pc/run_synthetic/run_chair/pcl2pcl/log_chair_pcl2pcl_gan_percentage_hausdorff_2019-03-02-19-19-18/ckpts/model_410.ckpt'
pcl2pcl_table_ShapeNetV2_ckpt = '/workspace/pointnet2/pc2pc/run_synthetic/run_table/pcl2pcl/log_table_pcl2pcl_gan_percentage-redo_hausdorff_2019-03-07-20-52-12/ckpts/model_290.ckpt'
pcl2pcl_plane_ShapeNetV2_ckpt = '/workspace/pointnet2/pc2pc/run_synthetic/run_plane/pcl2pcl/log_plane_pcl2pcl_gan_percentage_hausdorff_2019-03-05-10-55-54/ckpts/model_980.ckpt'


################ AE ckpt - ShapeNet v1
AE_car_c2c_ShapeNetV1_ckpt = '/workspace/pointnet2/pc2pc/run_synthetic/run_car/ae/log_ae_car_ShapeNet-V1_c2c_2019-03-11-15-37-32/ckpts/model_2000.ckpt'

######################## AE ckpt - 3D-EPN
AE_car_np2np_EPN_ckpt = '/workspace/pointnet2/pc2pc/run_3D-EPN/run_car/ae/log_3DEPN_ae_car_2019-03-06-22-51-09/ckpts/model_325.ckpt'

AE_chair_np2np_EPN_ckpt = '/workspace/pointnet2/pc2pc/run_3D-EPN/run_chair/ae/log_3DEPN_ae_chair_2019-03-06-16-08-06/ckpts/model_420.ckpt'

AE_table_np2np_EPN_ckpt = '/workspace/pointnet2/pc2pc/run_3D-EPN/run_table/ae/log_3DEPN_ae_table_2019-03-06-16-10-10/ckpts/model_485.ckpt'

AE_plane_np2np_EPN_ckpt = '/workspace/pointnet2/pc2pc/run_3D-EPN/run_plane/ae/log_3DEPN_ae_plane_2019-03-06-17-21-52/ckpts/model_450.ckpt'

if __name__ == '__main__':
        
    all_local_vars = locals().copy()
    for k, v in all_local_vars.items():
        if k[0] != '_' and k != 'os':
            if 'ckpt' in v:
                v += '.index'
            path_exists = os.path.exists(v)
            print(k, '\t\t', path_exists)
