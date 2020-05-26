import os

PC2PC_DIR = os.path.dirname(os.path.abspath(__file__))
print('PC2PC DIR: ', PC2PC_DIR)

data_root = os.path.join(PC2PC_DIR, 'data')

#################### datasets ###########################
ShapeNet_v1_point_cloud_dir = os.path.join(data_root, 'ShapeNet_v1_point_cloud')
ShapeNet_v1_car_point_cloud = os.path.join(ShapeNet_v1_point_cloud_dir, '02958343', 'point_cloud_clean')
ShapeNet_v1_chair_point_cloud = os.path.join(ShapeNet_v1_point_cloud_dir, '03001627', 'point_cloud_clean')
ShapeNet_v1_plane_point_cloud = os.path.join(ShapeNet_v1_point_cloud_dir, '02691156', 'point_cloud_clean')
ShapeNet_v1_table_point_cloud = os.path.join(ShapeNet_v1_point_cloud_dir, '04379243', 'point_cloud_clean')
ShapeNet_v1_lamp_point_cloud = os.path.join(ShapeNet_v1_point_cloud_dir, '03636649', 'point_cloud_clean')
ShapeNet_v1_sofa_point_cloud = os.path.join(ShapeNet_v1_point_cloud_dir, '04256520', 'point_cloud_clean')
ShapeNet_v1_boat_point_cloud = os.path.join(ShapeNet_v1_point_cloud_dir, '04530566', 'point_cloud_clean')
ShapeNet_v1_dresser_point_cloud = os.path.join(ShapeNet_v1_point_cloud_dir, '02933112', 'point_cloud_clean')

#ShapeNet_v2_point_cloud_dir = os.path.join(data_root, 'ShapeNet_v2_point_cloud')

EPN_dataset_root_dir = os.path.join(data_root, '3D-EPN_dataset/shapenet_dim32_sdf_pc')
EPN_car_point_cloud_dir = os.path.join(EPN_dataset_root_dir, '02958343', 'point_cloud')
EPN_chair_point_cloud_dir = os.path.join(EPN_dataset_root_dir, '03001627', 'point_cloud')
EPN_plane_point_cloud_dir = os.path.join(EPN_dataset_root_dir, '02691156', 'point_cloud')
EPN_table_point_cloud_dir = os.path.join(EPN_dataset_root_dir, '04379243', 'point_cloud')
EPN_lamp_point_cloud_dir = os.path.join(EPN_dataset_root_dir, '03636649', 'point_cloud')
EPN_sofa_point_cloud_dir = os.path.join(EPN_dataset_root_dir, '04256520', 'point_cloud')
EPN_boat_point_cloud_dir = os.path.join(EPN_dataset_root_dir, '04530566', 'point_cloud')
EPN_dresser_point_cloud_dir = os.path.join(EPN_dataset_root_dir, '02933112', 'point_cloud')
#EPN_dataset_test_dir = os.path.join(data_root, '3D-EPN_dataset/test-images_dim32_sdf_pc')
#EPN_results_dir = os.path.join(data_root, '3D-EPN_dataset/EPN_results')

real_scannet_chair_aligned_data_dir = os.path.join(data_root, 'scannet_v2_chairs_aligned/point_cloud')
real_scannet_chair_ori_data_dir = os.path.join(data_root, 'scannet_v2_chairs_Y_extracted')

real_scannet_table_aligned_data_dir = os.path.join(data_root, 'scannet_v2_tables_aligned/point_cloud')

real_MP_chair_aligned_data_dir = os.path.join(data_root, 'MatterPort_v1_chair_Yup_aligned/point_cloud')
real_MP_table_aligned_data_dir = os.path.join(data_root, 'MatterPort_v1_table_Yup_aligned/point_cloud')

kitti_car_data_train_dir = os.path.join(data_root, 'kitti_3D_detection/frustum_data_for_pcl2pcl/point_cloud_train')
kitti_car_data_test_dir = os.path.join(data_root, 'kitti_3D_detection/frustum_data_for_pcl2pcl/point_cloud_val')

#################### ckpts ################################

################ AE ckpt - ShapeNet v2
#AE_car_c2c_ShapeNetV2_ckpt = os.path.join(PC2PC_DIR, 'run/run_shapenet_v1_clean_ae/run_car/log_ae_car_percentage_c2c_2019-03-06-15-50-06/ckpts/model_1900.ckpt')
#AE_car_np2np_ShapeNetV2_ckpt = os.path.join(PC2PC_DIR, 'run/run_shapenet_v1_clean_ae/run_car/log_ae_car_percentage_np2np_2019-03-06-15-50-22/ckpts/model_1920.ckpt')

AE_chair_c2c_ShapeNetV2_ckpt = os.path.join(PC2PC_DIR, 'run/run_shapenet_v1_clean_ae/run_chair/log_ae_chair_c2c_2019-02-14-20-05-24/ckpts/model_1600.ckpt')
#AE_chair_np2np_ShapeNetV2_ckpt = os.path.join(PC2PC_DIR, 'run/run_shapenet_v1_clean_ae/run_chair/log_ae_chair_percent_np2np_2019-03-01-21-13-28/ckpts/model_1800.ckpt')

AE_table_c2c_ShapeNetV2_ckpt = os.path.join(PC2PC_DIR, 'run/run_shapenet_v1_clean_ae/run_table/log_ae_table_c2c_2019-02-28-14-52-10/ckpts/model_1810.ckpt')
#AE_table_np2np_ShapeNetV2_ckpt = os.path.join(PC2PC_DIR, 'run/run_shapenet_v1_clean_ae/run_table/log_ae_table_percent_np2np_2019-03-01-21-20-13/ckpts/model_1540.ckpt')

AE_plane_c2c_ShapeNetV2_ckpt = os.path.join(PC2PC_DIR, 'run/run_shapenet_v1_clean_ae/run_plane/log_ae_plane_percent_c2c_2019-03-04-16-22-26/ckpts/model_1820.ckpt')
#AE_plane_np2np_ShapeNetV2_ckpt = os.path.join(PC2PC_DIR, 'run/run_shapenet_v1_clean_ae/run_plane/log_ae_plane_percent_np2np_2019-03-04-16-22-53/ckpts/model_1810.ckpt')

################ AE ckpt - ShapeNet v1
AE_car_c2c_ShapeNetV1_ckpt = os.path.join(PC2PC_DIR, 'run/run_shapenet_v1_clean_ae/run_car/log_ae_car_ShapeNet-V1_c2c_2019-03-11-15-37-32/ckpts/model_2000.ckpt')

AE_chair_c2c_ShapeNetV1_ckpt = AE_chair_c2c_ShapeNetV2_ckpt

AE_table_c2c_ShapeNetV1_ckpt = AE_table_c2c_ShapeNetV2_ckpt

AE_plane_c2c_ShapeNetV1_ckpt = AE_plane_c2c_ShapeNetV2_ckpt

AE_lamp_c2c_ShapeNetV1_ckpt = os.path.join(PC2PC_DIR, 'run/run_shapenet_v1_clean_ae/run_lamp/log_ae_lamp_ShapeNet-V1_c2c_2019-07-24-15-18-24/ckpts/model_2000.ckpt')

AE_sofa_c2c_ShapeNetV1_ckpt = os.path.join(PC2PC_DIR, 'run/run_shapenet_v1_clean_ae/run_sofa/log_ae_sofa_ShapeNet-V1_c2c_2019-07-24-15-18-39/ckpts/model_2000.ckpt')

AE_boat_c2c_ShapeNetV1_ckpt = os.path.join(PC2PC_DIR, 'run/run_shapenet_v1_clean_ae/run_boat/log_ae_boat_ShapeNet-V1_c2c_2019-07-24-15-18-44/ckpts/model_2000.ckpt')

AE_dresser_c2c_ShapeNetV1_ckpt = os.path.join(PC2PC_DIR, 'run/run_shapenet_v1_clean_ae/run_dresser/log_ae_dresser_ShapeNet-V1_c2c_2019-07-24-15-20-12/ckpts/model_2000.ckpt')

######################## AE ckpt - 3D-EPN
AE_car_np2np_EPN_ckpt = os.path.join(PC2PC_DIR, 'run/run_3D-EPN_ae/run_car/log_3DEPN_ae_car_2019-07-24-22-05-30/ckpts/model_500.ckpt')
AE_chair_np2np_EPN_ckpt = os.path.join(PC2PC_DIR, 'run/run_3D-EPN_ae/run_chair/log_3DEPN_ae_chair_2019-07-24-21-57-33/ckpts/model_500.ckpt')
AE_table_np2np_EPN_ckpt = os.path.join(PC2PC_DIR, 'run/run_3D-EPN_ae/run_table/log_3DEPN_ae_table_2019-07-24-22-03-25/ckpts/model_500.ckpt')
AE_plane_np2np_EPN_ckpt = os.path.join(PC2PC_DIR, 'run/run_3D-EPN_ae/run_plane/log_3DEPN_ae_plane_2019-07-24-22-43-08/ckpts/model_500.ckpt')
AE_lamp_np2np_EPN_ckpt = os.path.join(PC2PC_DIR, 'run/run_3D-EPN_ae/run_lamp/log_3DEPN_ae_lamp_2019-07-25-17-59-20/ckpts/model_500.ckpt')
AE_sofa_np2np_EPN_ckpt = os.path.join(PC2PC_DIR, 'run/run_3D-EPN_ae/run_sofa/log_3DEPN_ae_sofa_2019-07-25-12-14-09/ckpts/model_500.ckpt')
AE_boat_np2np_EPN_ckpt = os.path.join(PC2PC_DIR, 'run/run_3D-EPN_ae/run_boat/log_3DEPN_ae_boat_2019-07-25-17-46-26/ckpts/model_500.ckpt')
AE_dresser_np2np_EPN_ckpt = os.path.join(PC2PC_DIR, 'run/run_3D-EPN_ae/run_dresser/log_3DEPN_ae_dresser_2019-07-25-18-04-21/ckpts/model_500.ckpt')


######################### AE ckpt - scannet real chairs and table
AE_scannet_chair_ckpt = os.path.join(PC2PC_DIR, 'run_real/run_scannet_chair/ae/log_scannet_chair_real_ae_2019-03-16-19-13-39/ckpts/model_1720.ckpt')

AE_scannet_table_ckpt = os.path.join(PC2PC_DIR, 'run_real/run_scannet_table/ae/log_scannet_table_real_ae_2019-03-16-17-01-39/ckpts/model_1630.ckpt')


########################## AE ckpt - kitti car
AE_kitti_car_ckpt = os.path.join(PC2PC_DIR, 'run_kitti/run_car/ae/log_kitti_ae_car_2019-03-18-21-31-29/ckpts/model_455.ckpt')

if __name__ == '__main__':
        
    all_local_vars = locals().copy()
    for k, v in all_local_vars.items():
        if k[0] != '_' and k != 'os':
            if 'ckpt' in v:
                v += '.index'
            path_exists = os.path.exists(v)
            print(k, '\t\t', path_exists)
