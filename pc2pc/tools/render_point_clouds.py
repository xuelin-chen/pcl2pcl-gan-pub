import os, sys
import math
from PIL import Image
import numpy as np
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../../utils'))
import pc_util

point_cloud_dir = '/workspace/pointnet2/pc2pc/run/log_ae_emd_chair_2048_test_good/pcloud/input'
output_img_dir = './renderings'
if not os.path.exists(output_img_dir):
    os.makedirs(output_img_dir)

all_ply_filenames = [ os.path.join(point_cloud_dir, f) for f in os.listdir(point_cloud_dir)]
all_ply_filenames.sort()

for ply_f in tqdm(all_ply_filenames):
    pc_basename = os.path.basename(ply_f)[:-4]
    pc = pc_util.read_ply(ply_f)
    img = pc_util.draw_point_cloud(pc, diameter=10, xrot=-math.pi/6.0, yrot=-math.pi/8.0, zrot=math.pi/2.0)
    img = Image.fromarray(np.uint8(img*255))
    img.save(os.path.join(output_img_dir, pc_basename+'.png'))
