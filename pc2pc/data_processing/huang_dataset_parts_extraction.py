import os,sys
import pymesh
import numpy as np

#### input paras #####
cls_name = 'chair'
label_map = {0:'seat', 
             1:'back', 
             2:'leg', 
             3:'armrest',
             4:'leg',
             5:'leg',
             6:'leg'}


###### useful vars #######
huang_dataset_download_dir = '/workspace/pointnet2/pc2pc/data/shapenet_part_seg_Huang/downloaded'
output_dir = '/workspace/pointnet2/pc2pc/data/shapenet_part_seg_Huang/processed/'+cls_name
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model_dir = os.path.join(huang_dataset_download_dir, cls_name, 'model')
seg_dir = os.path.join(huang_dataset_download_dir, cls_name, 'segmentation_gt')

def load_seg_file(seg_filename):
    labels = []
    with open(seg_filename, 'r') as sf:
        all_lines = sf.readlines()
        for line in all_lines:
            label = int(line.split()[0])
            if label >=4:
                label = 2
            labels.append(label)
    return np.array(labels)

def extract_parts_from_object(obj_basename):
    print(obj_basename)
    mesh_filename = os.path.join(model_dir, obj_basename+'.obj')
    seg_filename = os.path.join(seg_dir, obj_basename+'.seg')

    mesh = pymesh.load_mesh(mesh_filename)
    seg = load_seg_file(seg_filename)
    
    label_set = set(seg)
    #print(label_set)
    
    for part_label in label_set:
        f_idx = np.argwhere(seg==part_label)
        cur_points = mesh.vertices
        cur_faces = np.squeeze(mesh.faces[f_idx])
        cur_part = pymesh.form_mesh(cur_points, cur_faces)
        cur_part, _ = pymesh.remove_isolated_vertices(cur_part)

        cur_part_filename = os.path.join(output_dir, obj_basename+'_'+label_map[part_label]+'.ply')
        #cur_part_filename = os.path.join(output_dir, obj_basename+'_'+str(part_label)+'.ply')
        pymesh.save_mesh(cur_part_filename, cur_part)


if __name__ == "__main__":

    seg_filenames = os.listdir(seg_dir)
    seg_filenames.sort()
    
    for sf in seg_filenames:
        extract_parts_from_object(sf[:-4])

