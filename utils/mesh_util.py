import os
import sys
import numpy as np
from plyfile import PlyData, PlyElement
import pymesh

############## mesh I/O ####################
def read_obj(filename):
    '''
    return pymesh mesh
    '''
    mesh = pymesh.load_mesh(filename)
    return mesh

def convert_obj2ply(obj_filename, ply_filename, recenter=False, center_mode='pt_center'):
    mesh = pymesh.load_mesh(obj_filename)

    if recenter:
        if center_mode == 'pt_center':
            center = np.mean(mesh.vertices, axis=0)
        elif center_mode == 'box_center':
            min_p = np.amin(mesh.vertices, axis=0)  
            max_p = np.amax(mesh.vertices, axis=0)
            center = (min_p + max_p) / 2.0
            
        new_vertices = mesh.vertices - center
        mesh = pymesh.form_mesh(new_vertices, mesh.faces)

    pymesh.save_mesh(ply_filename, mesh)
