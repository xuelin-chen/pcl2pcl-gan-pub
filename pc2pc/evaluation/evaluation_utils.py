import os,sys

import numpy as np

def avg_dist(P_recon, P_gt):
    '''
    ACCURACY
    compute the average distance between ground truth point cloud and reconstructed point cloud
    P_gt: N x 3, np array
    P_recon: N x 3, np array
    '''
    npoint = P_recon.shape[0]

    P_recon_here = np.expand_dims(P_recon, axis=1) # N x 1 x 3
    P_recon_here = np.tile(P_recon_here, (1, npoint, 1)) # N x N x 3
    #print(P_recon_here)
    #print(P_recon_here.shape)

    P_gt_here = np.tile(P_gt, (npoint,1)) 
    P_gt_here =  np.reshape(P_gt_here, (npoint, npoint, 3)) # N x N x 3
    #print(P_gt_here)
    #print(P_gt_here.shape)

    dists = np.linalg.norm(P_recon_here - P_gt_here, axis=-1) # N x N x 1
    dists = np.squeeze(dists) # N x N
    #print(dists)
    #print(dists.shape)

    min_dists = np.amin(dists, axis=1)
    #print(min_dists)
    #print(min_dists.shape)

    avg_dist = np.mean(min_dists)

    return avg_dist

def accuracy(P_recon, P_gt, thre=0.01):
    '''
    ACCURACY
    P_gt: N x 3, np array
    P_recon: N x 3, np array
    '''

    npoint = P_recon.shape[0]

    P_recon_here = np.expand_dims(P_recon, axis=1) # N x 1 x 3
    P_recon_here = np.tile(P_recon_here, (1, npoint, 1)) # N x N x 3

    P_gt_here = np.tile(P_gt, (npoint,1)) 
    P_gt_here =  np.reshape(P_gt_here, (npoint, npoint, 3)) # N x N x 3

    dists = np.linalg.norm(P_recon_here - P_gt_here, axis=-1) # N x N x 1
    dists = np.squeeze(dists) # N x N

    min_dists = np.amin(dists, axis=1) # 1 x N

    matched = min_dists[min_dists < thre]
    fraction = matched.shape[0] / npoint
    return fraction

def completeness(P_recon, P_gt, thre=0.01):
    '''
    COMPLETENESS
    P_gt: N x 3, np array
    P_recon: N x 3, np array
    '''

    npoint = P_recon.shape[0]

    P_gt_here = np.expand_dims(P_gt, axis=1) # N x 1 x 3
    P_gt_here = np.tile(P_gt_here, (1, npoint, 1)) # N x N x 3

    P_recon_here = np.tile(P_recon, (npoint,1)) 
    P_recon_here =  np.reshape(P_recon_here, (npoint, npoint, 3)) # N x N x 3

    dists = np.linalg.norm(P_gt_here - P_recon_here, axis=-1) # N x N x 1
    dists = np.squeeze(dists) # N x N

    min_dists = np.amin(dists, axis=1) # 1 x N

    matched = min_dists[min_dists < thre]
    fraction = matched.shape[0] / npoint
    return fraction

if __name__=='__main__':

    p_gt = np.array([[0,1,0],[1,0,0], [2,-1,0]])

    p_re = np.array([[0,-1,0],[1,0,0], [2,1,0]])

    #p_gt = np.zeros((2048,3))
    #p_re = np.zeros((2048,3))

    avg_d = avg_dist(p_re, p_gt)
    print(avg_d)

    comp_f = completeness(p_re, p_gt)
    print(comp_f)
