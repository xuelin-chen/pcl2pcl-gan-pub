import os

import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff

import numpy as np
from skimage import measure

import trimesh
import glob

distance_field_dir = ''

def find_files(dir, extension='.txt'):
    filenames = glob.glob(os.path.join(dir, '**', '*'+extension))
    print(filenames)
    print(len(filenames))
    return filenames

def read_df_from_txt(df_txt_filename):
    
    with open(df_txt_filename, 'r') as file:
        long_line = file.readline()
        numbers = long_line.split(' ')

        dimx, dimy, dimz = int(numbers[0]), int(numbers[1]), int(numbers[2])
        volume_data = np.zeros((dimx, dimy, dimz))
        data = numbers[3:dimx*dimy*dimz+3]
        data_idx = 0
        for z_i in range(dimz):
            for y_i in range(dimy):
                for x_i in range(dimx):
                    volume_data[x_i, y_i, z_i] = float(data[data_idx])
                    data_idx += 1
    return volume_data

def get_isosurface(volume_data, iso_val):
    vs, fs, _, _ = measure.marching_cubes_lewiner(df, iso_val)

    final_mesh = trimesh.Trimesh(vertices=vs, faces=fs)
    return final_mesh

df = read_df_from_txt('1d63eb2b1f78aa88acf77e718d93f3e1__0__recon_128.txt')
print(df)
print(np.min(df), np.max(df))

final_mesh = get_isosurface(df, 0.1)
final_mesh.export('test_0.1.obj')
final_mesh = get_isosurface(df, 0.2)
final_mesh.export('test_0.2.obj')
final_mesh = get_isosurface(df, 0.3)
final_mesh.export('test_0.3.obj')
final_mesh = get_isosurface(df, 0.4)
final_mesh.export('test_0.4.obj')
final_mesh = get_isosurface(df, 0.5)
final_mesh.export('test_0.5.obj')

