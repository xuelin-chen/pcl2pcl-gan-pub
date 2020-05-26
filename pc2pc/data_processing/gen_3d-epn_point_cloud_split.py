import os, sys
import numpy as np
import pickle
import shutil
from tqdm import tqdm

train_model_list_filename = '../data/3D-EPN_dataset/completion_train.txt'
test_model_list_filename = '../data/3D-EPN_dataset/completion_test.txt'
point_cloud_root = '../data/3D-EPN_dataset/tmp/shapenet_dim32_sdf_pc'

def gen_split_for(cls_id):
    ori_class_point_cloud_dir = os.path.join(point_cloud_root, cls_id)
    class_point_cloud_dir = os.path.join(point_cloud_root, cls_id, 'point_cloud')
    if not os.path.exists(class_point_cloud_dir): os.makedirs(class_point_cloud_dir)
    all_ply_filenames = [f for f in os.listdir(ori_class_point_cloud_dir)]
    for pf in tqdm(all_ply_filenames):
        src_filename = os.path.join(ori_class_point_cloud_dir, pf)
        dst_filename = os.path.join(class_point_cloud_dir, pf)
        if '.ply' in pf: shutil.move(src_filename, dst_filename)
    print('Total: %d'%(len(all_ply_filenames)))

    train_line_list = [line.rstrip('\n') for line in open(train_model_list_filename)]
    train_modelname_list = []
    for tl in train_line_list:
        if cls_id in tl:
            train_modelname_list.append(tl.split('\\')[-1])
    
    test_line_list = [line.rstrip('\n') for line in open(test_model_list_filename)]
    test_modelname_list = []
    for tl in test_line_list:
        if cls_id in tl:
            test_modelname_list.append(tl.split('\\')[-1])

    print('#train_model', len(train_modelname_list), '#test_model', len(test_modelname_list))
    
    train_ply_filenames = []
    val_ply_filenames = []
    trainval_ply_filenames = []
    test_ply_filenames = []

    for pf in all_ply_filenames:
        cur_modelname = pf.split('_')[0]
        
        if cur_modelname in train_modelname_list:
            odd = np.random.rand()
            trainval_ply_filenames.append(pf)
            if odd < 0.9:
                train_ply_filenames.append(pf)
            else:
                val_ply_filenames.append(pf)
        else:
            test_ply_filenames.append(pf)

    print('#trains: %d'%(len(train_ply_filenames)))
    print('#vals: %d'%(len(val_ply_filenames)))
    print('#trainvals: %d'%(len(trainval_ply_filenames)))
    print('#tests: %d'%(len(test_ply_filenames)))

    base_dir = os.path.dirname(class_point_cloud_dir)
    base_name = os.path.basename(class_point_cloud_dir)
    with open(os.path.join(base_dir, base_name+'_train_split.pickle'), 'wb') as pf:
        pickle.dump(train_ply_filenames, pf)

    with open(os.path.join(base_dir, base_name+'_val_split.pickle'), 'wb') as pf:
        pickle.dump(val_ply_filenames, pf)

    with open(os.path.join(base_dir, base_name+'_trainval_split.pickle'), 'wb') as pf:
        pickle.dump(trainval_ply_filenames, pf)

    with open(os.path.join(base_dir, base_name+'_test_split.pickle'), 'wb') as pf:
        pickle.dump(test_ply_filenames, pf)

    with open(os.path.join(base_dir, base_name+'_all_split.pickle'), 'wb') as pf:
        pickle.dump(all_ply_filenames, pf)
    return

gen_split_for('02958343')
gen_split_for('03001627')
#gen_split_for('02691156')
gen_split_for('04379243')
gen_split_for('03636649')
gen_split_for('04256520')
gen_split_for('04530566')
gen_split_for('02933112')

'''
train_portion = 0.85
val_portion = 0.05
test_portion = 0.10

train_ply_filenames = []
val_ply_filenames = []
trainval_ply_filenames = []
test_ply_filenames = []
for pf in all_ply_filenames:
    odd = np.random.rand()
    if odd <= train_portion: # train
        train_ply_filenames.append(pf)
        trainval_ply_filenames.append(pf)
    elif odd > train_portion and odd < (train_portion+val_portion):
        val_ply_filenames.append(pf)
        trainval_ply_filenames.append(pf)
    else:
        test_ply_filenames.append(pf)

print('#trains: %d'%(len(train_ply_filenames)))
print('#vals: %d'%(len(val_ply_filenames)))
print('#trainvals: %d'%(len(trainval_ply_filenames)))
print('#tests: %d'%(len(test_ply_filenames)))

base_dir = os.path.dirname(point_cloud_dir)
base_name = os.path.basename(point_cloud_dir)
with open(os.path.join(base_dir, base_name+'_train_split.pickle'), 'wb') as pf:
    pickle.dump(train_ply_filenames, pf)

with open(os.path.join(base_dir, base_name+'_val_split.pickle'), 'wb') as pf:
    pickle.dump(val_ply_filenames, pf)

with open(os.path.join(base_dir, base_name+'_trainval_split.pickle'), 'wb') as pf:
    pickle.dump(trainval_ply_filenames, pf)

with open(os.path.join(base_dir, base_name+'_test_split.pickle'), 'wb') as pf:
    pickle.dump(test_ply_filenames, pf)

with open(os.path.join(base_dir, base_name+'_all_split.pickle'), 'wb') as pf:
    pickle.dump(all_ply_filenames, pf)
'''
