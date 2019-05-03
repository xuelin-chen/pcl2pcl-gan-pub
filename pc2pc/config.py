import os
PC2PC_DIR = os.path.dirname(os.path.abspath(__file__))
print('PC2PC DIR: ', PC2PC_DIR)

data_root = os.path.join(PC2PC_DIR, 'data')

#################### datasets ###########################
ShapeNet_v1_point_cloud_dir = os.path.join(data_root, 'ShapeNet_v1_point_cloud')

EPN_dataset_train_dir = os.path.join(data_root, '3D-EPN_dataset/shapenet_dim32_sdf_pc')
EPN_dataset_test_dir = os.path.join(data_root, '3D-EPN_dataset/test-images_dim32_sdf_pc')
#EPN_results_dir = os.path.join(data_root, '3D-EPN_dataset/EPN_results')

real_scannet_chair_aligned_data_dir = os.path.join(data_root, 'scannet_v2_chairs_aligned/point_cloud')
real_scannet_chair_ori_data_dir = os.path.join(data_root, 'scannet_v2_chairs_Y_extracted')

real_scannet_table_aligned_data_dir = os.path.join(data_root, 'scannet_v2_tables_aligned/point_cloud')

real_MP_chair_aligned_data_dir = os.path.join(data_root, 'MatterPort_v1_chair_Yup_aligned/point_cloud')
real_MP_table_aligned_data_dir = os.path.join(data_root, 'MatterPort_v1_table_Yup_aligned/point_cloud')

kitti_car_data_train_dir = os.path.join(data_root, 'kitti_3D_detection/frustum_data_for_pcl2pcl/point_cloud_train')
kitti_car_data_test_dir = os.path.join(data_root, 'kitti_3D_detection/frustum_data_for_pcl2pcl/point_cloud_val')

#################### models for synthetic data ################################

################ AE ckpt - ShapeNet v1
# NOTE: by default, train AE on synthetic shapenet data for 2000 epochs, and pick a model at around 2000.
AE_chair_c2c_ShapeNetV1_ckpt = os.path.join(PC2PC_DIR, 'run_synthetic/run_chair/ae/log_ae_chair_ShapeNet-V1_c2c/ckpts/model_2000.ckpt') 

AE_table_c2c_ShapeNetV1_ckpt = os.path.join(PC2PC_DIR, 'run_synthetic/run_table/ae/log_ae_table_ShapeNet-V1_c2c/ckpts/model_2000.ckpt')

AE_plane_c2c_ShapeNetV1_ckpt = os.path.join(PC2PC_DIR, 'run_synthetic/run_plane/ae/log_ae_plane_ShapeNet-V1_c2c/ckpts/model_2000.ckpt')

AE_car_c2c_ShapeNetV1_ckpt = os.path.join(PC2PC_DIR, 'run_synthetic/run_car/ae/log_ae_car_ShapeNet-V1_c2c/ckpts/model_2000.ckpt') 

######################## AE ckpt - 3D-EPN
# NOTE: pick a model at around 500 epoch, since 3D-EPN has too more training data compared to our synthetic ShapeNet data, training AE on 3D-EPN for less epochs, stop at 500 empirically.
AE_chair_np2np_EPN_ckpt = os.path.join(PC2PC_DIR, 'run_3D-EPN/run_chair/ae/log_3DEPN_ae_chair/ckpts/model_500.ckpt')

AE_table_np2np_EPN_ckpt = os.path.join(PC2PC_DIR, 'run_3D-EPN/run_table/ae/log_3DEPN_ae_table/ckpts/model_500.ckpt')

AE_plane_np2np_EPN_ckpt = os.path.join(PC2PC_DIR, 'run_3D-EPN/run_plane/ae/log_3DEPN_ae_plane/ckpts/model_500.ckpt')

AE_car_np2np_EPN_ckpt = os.path.join(PC2PC_DIR, 'run_3D-EPN/run_car/ae/log_3DEPN_ae_car/ckpts/model_500.ckpt')


######################### AE ckpt - scannet real chairs and table
AE_scannet_chair_ckpt = '/workspace/pointnet2/pc2pc/run_real/run_scannet_chair/ae/log_scannet_chair_real_ae_2019-03-16-19-13-39/ckpts/model_1720.ckpt'

AE_scannet_table_ckpt = '/workspace/pointnet2/pc2pc/run_real/run_scannet_table/ae/log_scannet_table_real_ae_2019-03-16-17-01-39/ckpts/model_1630.ckpt'


########################## AE ckpt - kitti car
AE_kitti_car_ckpt = '/workspace/pointnet2/pc2pc/run_kitti/run_car/ae/log_kitti_ae_car_2019-03-18-21-31-29/ckpts/model_455.ckpt'



if __name__ == '__main__':
        
    all_local_vars = locals().copy()
    for k, v in all_local_vars.items():
        if k[0] != '_' and k != 'os':
            if 'ckpt' in v:
                v += '.index'
            path_exists = os.path.exists(v)
            print(k, '\t\t', path_exists)
