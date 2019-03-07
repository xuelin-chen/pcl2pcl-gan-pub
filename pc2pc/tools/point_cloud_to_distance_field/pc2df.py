import pc2df_utils
import numpy as np
import os,sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../../../utils'))
import pc_util

ply_filename = '0.ply'

pc = pc_util.read_ply_xyz(ply_filename)

df = pc2df_utils.convert_pc2df(pc)
np.save('0_df', df)
