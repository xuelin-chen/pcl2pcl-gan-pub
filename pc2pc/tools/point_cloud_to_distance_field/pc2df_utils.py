import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
import numpy as np
from skimage import measure
import trimesh
from scipy import spatial

def get_cell_center(min_pts, x_index, y_index, z_index, x_grid_size, y_grid_size, z_grid_size):

    x_value = min_pts[0] + (x_index + 0.5) * x_grid_size
    y_value = min_pts[1] + (y_index + 0.5) * y_grid_size
    z_value = min_pts[2] + (z_index + 0.5) * z_grid_size

    center = np.array([x_value, y_value, z_value])

    return center

def convert_pc2df(points, resolution=32):
    '''
    the range is defined by the bbox of points
    points: Nx3, np array
    '''

    tree = spatial.KDTree(points)

    pts_min = np.amin(points, axis=0)
    pts_max = np.amax(points, axis=0)

    extents = pts_max - pts_min

    x_grid_size, y_grid_size, z_grid_size = extents[0]/float(resolution), extents[1]/float(resolution), extents[2]/float(resolution)

    df_mat = np.zeros((resolution, resolution, resolution))
    for x_i in range(resolution):
        for y_i in range(resolution):
            for z_i in range(resolution):

                center_cur = get_cell_center(pts_min, x_i, y_i, z_i, x_grid_size, y_grid_size, z_grid_size)

                nearest_dist, _ = tree.query(center_cur)

                df_mat[x_i, y_i, z_i] = nearest_dist

    return df_mat

def convert_df2mesh(df_volume_data, out_filename, thre=0.02):
    vs, fs, _, _ = measure.marching_cubes_lewiner(df_volume_data, 0.02)

    final_mesh = trimesh.Trimesh(vertices=vs, faces=fs)
    final_mesh.export(out_filename)