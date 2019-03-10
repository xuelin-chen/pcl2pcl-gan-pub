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
    # scale points to fit within a cube of resolution (32) size length
    # move points to center at (resolution/2, resolution/2, resolution/2)
    pts_min = np.amin(points, axis=0)
    pts_max = np.amax(points, axis=0)
    extents = pts_max - pts_min
    max_size = np.max(extents)
    scale_factor = resolution / max_size

    bbox_center = (pts_max + pts_min) / 2.0
    trans_v = np.array([resolution/2.0, resolution/2.0, resolution/2.0]) - bbox_center
    
    for pidx, p in enumerate(points):
        points[pidx] = p * scale_factor
        points[pidx] = points[pidx] + trans_v    
    ####

    x_grid_size, y_grid_size, z_grid_size = 1, 1, 1

    tree = spatial.KDTree(points)

    df_mat = np.zeros((resolution, resolution, resolution))
    df_arr = []
    for z_i in range(resolution):
        for y_i in range(resolution):
            for x_i in range(resolution):

                center_cur = get_cell_center(np.array([0,0,0]), x_i, y_i, z_i, x_grid_size, y_grid_size, z_grid_size)

                nearest_dist, _ = tree.query(center_cur)

                df_mat[x_i, y_i, z_i] = nearest_dist
                df_arr.append(nearest_dist)
    df_arr = np.array(df_arr)
    return df_mat, df_arr
