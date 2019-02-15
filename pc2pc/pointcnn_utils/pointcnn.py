from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import pointfly as pf
import tensorflow as tf

def point_conv(pts, fts, qrs, K, D, C, is_training, activation=tf.nn.relu, bn=True, sorting_method='cxyz', center_patch=True, tag='point_conv'):
    '''
    pts: xyz points, Tensor, (batch_size, npoint, 3)
    fts: features, Tensor, (batch_size, npoint, channel_fts)
    qrs: xyz points as queries, Tensor, (batch_size, n_qrs, 3)
    K: number of neighbors for one query, scalar, int
    D: dilation rate
    C: output feature channel number
    '''
    with tf.variable_scope(tag):
        _, indices_dilated = pf.knn_indices_general(qrs, pts, K * D, True)
        indices = indices_dilated[:, :, ::D, :]

        if sorting_method is not None:
            indices = pf.sort_points(pts, indices, sorting_method)

        nn_pts = tf.gather_nd(pts, indices, name='nn_pts')  # (N, P, K, 3)
        if center_patch:
            nn_pts_center = tf.expand_dims(qrs, axis=2, name='nn_pts_center')  # (N, P, 1, 3)
            nn_pts_local = tf.subtract(nn_pts, nn_pts_center, name='nn_pts_local')  # (N, P, K, 3), move to local frame
        else:
            nn_pts_local = nn_pts

        if fts is None:
            nn_fts_input = nn_pts_local
        else:
            nn_fts_from_prev = tf.gather_nd(fts, indices, 'nn_fts_from_prev')
            nn_fts_input = tf.concat([nn_pts_local, nn_fts_from_prev], axis=-1, name='nn_fts_input') # (N, P, K, 3+C_prev_fts)

        fts_conv = pf.conv2d(nn_fts_input, C, 'fts_conv', is_training, (1, K), with_bn=bn, activation=activation) # (N, P, 1, C)
        fts_conv_3d = tf.squeeze(fts_conv, axis=2, name='fts_conv_3d') # (N, P, C)

        return fts_conv_3d, indices

def point_conv_transpose(pts_in, fts_in, pts_out, fts_out_skipped, indices, K, D, C, is_training, activation=tf.nn.relu, bn=True, sorting_method='cxyz', center_patch=True, tconv='fc', tag='point_deconv'):
    '''
    pts_in: xyz points, Tensor, (batch_size, npoint, 3)
    fts_in: features, Tensor, (batch_size, npoint, channel_fts)
    pts_out: xyz points, Tensor, (batch_size, n_pts_out, 3), NOTE: pts_out is denser than pts.
    K: number of neighbors for one query, scalar, int
    D: dilation rate
    C: output feature channel number
    tconv: fc or t_conv
    in the comments:
        N - batch size
        P - number of pts_in
        K - number of neighbors
    '''
    with tf.variable_scope(tag):
        batch_size = pts_out.shape[0]
        n_pts_out = pts_out.shape[1]
        n_pts_in = pts_in.shape[1]

        if indices is None:
            _, indices_dilated = pf.knn_indices_general(pts_in, pts_out, K * D, True)
            indices = indices_dilated[:, :, ::D, :]
            
            if sorting_method is not None:
                indices = pf.sort_points(pts_out, indices, sorting_method) # (N, P, K, 2)
        
        # move the gathered pts_out to local frame or not
        nn_pts_out = tf.gather_nd(pts_out, indices, name='nn_pts_out')  # (N, P, K, 3)
        if center_patch:
            nn_pts_out_center = tf.expand_dims(pts_in, axis=2, name='nn_pts_out_center')  # (N, P, 1, 3)
            nn_pts_out_local = tf.subtract(nn_pts_out, nn_pts_out_center, name='nn_pts_out_local')  # (N, P, K, 3)
        else:
            nn_pts_out_local = nn_pts_out

        if fts_in is None:
            nn_fts_input = pts_in # (N, P, 3)
        else:
            # NOTE: do not concat pts_in with fts_in now, since fts_in already contains information of pts_in
            nn_fts_input = fts_in # (N, P, C_fts_in)

        nn_fts_input = tf.expand_dims(nn_fts_input, axis=2)  # (N, P, 1, C_fts_in)

        # using fc
        if tconv == 'fc':
            fts_fc = pf.dense(nn_fts_input, K*C, 'fc_fts_prop', is_training) # (N, P, 1, K*C)
            fts_fc = tf.reshape(fts_fc, (batch_size, n_pts_in, K, C)) # (N, P, K, C)
        elif tconv == 't_conv':
            fts_fc = pf.conv2d_transpose(nn_fts_input, C, tag + '_fts_deconv', is_training, kernel_size=(1, K), strides=(1, 1), with_bn=bn, activation=activation) # (N, P, K, C)

        if fts_out_skipped is not None:
            nn_fts_out_skipped = tf.gather_nd(fts_out_skipped, indices, name='nn_fts_out_from_skip')
            fts_all_concat = tf.concat([fts_fc, nn_fts_out_skipped, nn_pts_out_local], axis=-1) # (N, P, K, C+3+channel_skipped)
        else:
            fts_all_concat = tf.concat([fts_fc, nn_pts_out_local], axis=-1) # (N, P, K, C+3)

        fts_pts_conv = pf.conv2d(fts_all_concat, C, 'fts_pts_conv1D', is_training, kernel_size=(1,1), strides=(1,1), with_bn=bn, activation=activation) # (N, P, K, C+3+[ch_skip]) -> (N, P, K, C)

        # summing up feature for each point
        fts_dense = tf.scatter_nd(tf.reshape(indices, (-1, 2)), tf.reshape(fts_pts_conv, (-1, C)), (batch_size, n_pts_out, C)) # (N, N_PTS_OUT, c)
        
        return fts_dense


###########################################################################################
def point_conv_transpose_conv2d(pts_in, fts_in, pts_out, fts_out_skipped, indices, K, D, C, is_training, activation=tf.nn.relu, bn=True, sorting_method='cxyz', tag='point_deconv'):
    '''
    pts_in: xyz points, Tensor, (batch_size, npoint, 3)
    fts_in: features, Tensor, (batch_size, npoint, channel_fts)
    pts_out: xyz points, Tensor, (batch_size, n_pts_out, 3), NOTE: pts_out is denser than pts.
    K: number of neighbors for one query, scalar, int
    D: dilation rate
    C: output feature channel number

    in the comments:
        N - batch size
        P - number of pts_in
        K - number of neighbors
    '''
    with tf.variable_scope(tag):
        batch_size = pts_out.shape[0]
        n_pts_out = pts_out.shape[1]
        n_pts_in = pts_in.shape[1]

        if indices is None:
            _, indices_dilated = pf.knn_indices_general(pts_in, pts_out, K * D, True)
            indices = indices_dilated[:, :, ::D, :]
            
            if sorting_method is not None:
                indices = pf.sort_points(pts_out, indices, sorting_method) # (N, P, K, 2)
        
        # move the gathered pts_out to local frame
        nn_pts_out = tf.gather_nd(pts_out, indices, name='nn_pts_out')  # (N, P, K, 3)
        nn_pts_out_center = tf.expand_dims(pts_in, axis=2, name='nn_pts_out_center')  # (N, P, 1, 3)
        nn_pts_out_local = tf.subtract(nn_pts_out, nn_pts_out_center, name='nn_pts_out_local')  # (N, P, K, 3)

        if fts_in is None:
            nn_fts_input = pts_in # (N, P, 3)
            #print('Error: fts_in is None.')
        else:
            #nn_fts_input = tf.concat([pts_in, fts_in], axis=-1, name=tag + 'nn_fts_input') # (N, P, 3+C_fts_in)
            nn_fts_input = fts_in # (N, P, C_fts_in)

        nn_fts_input = tf.expand_dims(nn_fts_input, axis=2)  # (N, P, 1, C_fts_in)

        # using deconv
        fts_fc = pf.conv2d_transpose(nn_fts_input, C, tag + '_fts_deconv', is_training, kernel_size=(1, K), strides=(1, 1), with_bn=bn, activation=activation) # # (N, P, K, C)
        
        if fts_out_skipped is not None:
            nn_fts_out_skipped = tf.gather_nd(fts_out_skipped, indices, name='nn_fts_out_from_skip')
            fts_all_concat = tf.concat([fts_fc, nn_fts_out_skipped, nn_pts_out_local], axis=-1) # (N, P, K, C+3+channel_skipped)
        else:
            fts_all_concat = tf.concat([fts_fc, nn_pts_out_local], axis=-1) # (N, P, K, C+3)

        fts_pts_conv = pf.conv2d(fts_all_concat, C, 'fts_pts_conv1D', is_training, kernel_size=(1,1), strides=(1,1), with_bn=bn, activation=activation) # (N, P, K, C+3_ch_skip?) -> (N, P, K, C)

        # summing up feature for each point
        fts_dense = tf.scatter_nd(tf.reshape(indices, (-1, 2)), tf.reshape(fts_pts_conv, (-1, C)), (batch_size, n_pts_out, C)) # (N, N_PTS_OUT, c)
        
        return fts_dense

def point_conv_transpose_fc_no_skip(pts_in, fts_in, indices, np_out, K, C, is_training, activation=tf.nn.relu, bn=True, tag='point_deconv'):
    '''
    NOTE: no skipped link of output points and features
    pts_in: xyz points, Tensor, (batch_size, npoint, 3)
    fts_in: features, Tensor, (batch_size, npoint, channel_fts)
    np_out: number of points that is expected
    K: neighbor size
    C: output feature channel number

    '''
    with tf.variable_scope(tag):
        batch_size = fts_in.shape[0]
        n_pts_in = fts_in.shape[1]

        if pts_in is None:
            nn_fts_input = fts_in # (N, P, C_fts_in)
            #print('Error: fts_in is None.')
        else:
            nn_fts_input = tf.concat([pts_in, fts_in], axis=-1, name='_nn_fts_input') # (N, P, C_fts_in+3)

        nn_fts_input = tf.expand_dims(nn_fts_input, axis=2)  # (N, P, 1, C_fts_in)

        # using fc
        fts_fc = pf.dense(nn_fts_input, K*C, 'fc_fts_prop', is_training) # (N, P, 1, K*C)
        fts_fc = tf.reshape(fts_fc, (batch_size, n_pts_in, K, C)) # (N, P, K, C)

        fts_pts_conv = pf.conv2d(fts_all_concat, C, tag + '_fts_pts_conv1D', is_training, kernel_size=(1,1), strides=(1,1), with_bn=bn, activation=activation) # (N, P, K, C+3_ch_skip?) -> (N, P, K, C)

        # summing up feature for each point
        fts_dense = tf.scatter_nd(tf.reshape(indices, (-1, 2)), tf.reshape(fts_pts_conv, (-1, C)), (batch_size, n_pts_out, C)) # (N, N_PTS_OUT, c)
        
        return fts_dense

def point_conv_transpose_1(pts_in, fts_in, pts_out, indices, K, D, C, is_training, activation=tf.nn.relu, bn=True, sorting_method='cxyz', tag='point_conv'):
    '''
    pts_in: xyz points, Tensor, (batch_size, npoint, 3)
    fts_in: features, Tensor, (batch_size, npoint, channel_fts)
    pts_out: xyz points, Tensor, (batch_size, n_pts_out, 3), NOTE: pts_out is denser than pts.
    K: number of neighbors for one query, scalar, int
    D: dilation rate
    C: output feature channel number

    in the comments:
        N - batch size
        P - number of pts_in
        K - number of neighbors
    '''
    batch_size = pts_out.shape[0]
    n_pts_out = pts_out.shape[1]
    n_pts_in = pts_in.shape[1]

    if indices == None:
        _, indices_dilated = pf.knn_indices_general(pts_in, pts_out, K * D, True)
        indices = indices_dilated[:, :, ::D, :]
        
        if sorting_method is not None:
            indices = pf.sort_points(pts_out, indices, sorting_method) # (N, P, K, 2)
    
    # move the gathered pts_out to local frame
    nn_pts = tf.gather_nd(pts_out, indices, name=tag + '_nn_pts')  # (N, P, K, 3)
    nn_pts_center = tf.expand_dims(pts_in, axis=2, name=tag + '_nn_pts_center')  # (N, P, 1, 3)
    nn_pts_local = tf.subtract(nn_pts, nn_pts_center, name=tag + '_nn_pts_local')  # (N, P, K, 3)

    if fts_in is None:
        #nn_fts_input = pts_in # (N, P, 3)
        print('Error: fts_in is None.')
    else:
        #nn_fts_input = tf.concat([pts_in, fts_in], axis=-1, name=tag + 'nn_fts_input') # (N, P, 3+C_fts_in)
        nn_fts_input = fts_in # (N, P, C_fts_in)

    nn_fts_input = tf.expand_dims(nn_fts_input, axis=2)  # (N, P, 1, C_fts_in)

    # TODO: maybe try other conv1d? fc? there
    fts_conv = pf.conv2d_transpose(nn_fts_input, C, tag + '_fts_deconv', is_training, kernel_size=(1, 1), strides=(1, K), with_bn=bn, activation=activation) # (N, P, 1, C_fts_in) -> (N, P, K, C)


    fts_conv_pts_out_concat = tf.concat([fts_conv, nn_pts_local], axis=-1) # (N, P, K, C+3)

    fts_pts_conv = pf.conv2d(fts_conv_pts_out_concat, C, tag + '_fts_pts_conv1D', is_training, kernel_size=(1,1), strides=(1,1), with_bn=bn, activation=activation) # (N, P, K, C+3) -> (N, P, K, C)

    ######## collect fts for points in the output #######################
    ##########################################################################
    npoint_indices = tf.reshape(tf.range(n_pts_in), [-1, 1]) # (P, 1)
    npoint_indices = tf.tile(npoint_indices, (1, K)) # (P, K)
    npoint_indices = tf.expand_dims(npoint_indices, axis=0) # (1, P, K)
    npoint_indices = tf.tile(npoint_indices, (batch_size, 1, 1)) # (N, P, K)
    npoint_indices = tf.expand_dims(npoint_indices, axis=3) # (N, P, K, 1)
    NPK_indices = tf.concat([tf.expand_dims(indices[:,:,:,0], axis=-1), npoint_indices, tf.expand_dims(indices[:,:,:,1], axis=-1)], axis=3) # (N, P, K, 3) last dim is 3 for (batch_idx, point_idx, neighbor_idx)

    NPK_indices = tf.expand_dims(NPK_indices, axis=3) # (N, P, K, 1, 3)
    NPK_indices = tf.tile(NPK_indices, (1,1,1,C,1)) # (N, P, K, C, 3)

    channel_indices = tf.reshape(tf.range(C), [1, 1, 1, -1, 1]) # (1, 1, 1, C, 1)
    channel_indices = tf.tile(channel_indices, (batch_size, n_pts_in, K, 1, 1)) # (N, P, K, C, 1)

    final_indices = tf.concat([NPK_indices, channel_indices], axis=4) # (N, P, K, C, 4) last dim: (batch_idx, rep_point_idx, neighbor_idx_in_pts_out, channel_idx)

    ######## reshape for constructing sparse tensor ################
    final_indices = tf.reshape(final_indices, [-1, 4])
    final_value = tf.reshape(fts_pts_conv, [-1])

    # using sparse tensor and sparse ops
    # TODO: in tf 1.4, the sparse_reduce_sum cannot infer the shape even if the input shape is known.
    #       so, try to use latest tf, a version of after Oct. 2018.
    #fts_sparse = tf.SparseTensor(indices=tf.cast(final_indices, tf.int64), values=final_value, dense_shape=[batch_size, n_pts_in, n_pts_out, C]) # (N, P, N_PTS_OUT, C) in sparse representation
    #fts_out = tf.sparse_reduce_sum(fts_sparse, axis=1) # (N, N_PTS_OUT, C)
    #fts_dense = tf.sparse_tensor_to_dense(fts_sparse, default_value=0, validate_indices=False) # (N, P, N_PTS_OUT, C)
    
    # using dense tensor, memory-consuming
    fts_dense = tf.scatter_nd(final_indices, final_value, [batch_size, n_pts_in, n_pts_out, C]) # (N, P, N_PTS_OUT, C)
    fts_out = tf.reduce_sum(fts_dense, axis=1) # (N, N_PTS_OUT, C)
    
    return fts_out

def point_conv_transpose_2(pts_in, fts_in, pts_out, fts_out_skipped, indices, K, D, C, is_training, activation=tf.nn.relu, bn=True, sorting_method='cxyz', tag='point_conv'):
    '''
    pts_in: xyz points, Tensor, (batch_size, npoint, 3)
    fts_in: features, Tensor, (batch_size, npoint, channel_fts)
    pts_out: xyz points, Tensor, (batch_size, n_pts_out, 3), NOTE: pts_out is denser than pts.
    K: number of neighbors for one query, scalar, int
    D: dilation rate
    C: output feature channel number

    in the comments:
        N - batch size
        P - number of pts_in
        K - number of neighbors
    '''
    batch_size = pts_out.shape[0]
    n_pts_out = pts_out.shape[1]
    n_pts_in = pts_in.shape[1]

    if indices is None:
        _, indices_dilated = pf.knn_indices_general(pts_in, pts_out, K * D, True)
        indices = indices_dilated[:, :, ::D, :]
        
        if sorting_method is not None:
            indices = pf.sort_points(pts_out, indices, sorting_method) # (N, P, K, 2)
    
    # move the gathered pts_out to local frame
    nn_pts_out = tf.gather_nd(pts_out, indices, name=tag + 'nn_pts_out')  # (N, P, K, 3)
    nn_pts_out_center = tf.expand_dims(pts_in, axis=2, name=tag + 'nn_pts_out_center')  # (N, P, 1, 3)
    nn_pts_out_local = tf.subtract(nn_pts_out, nn_pts_out_center, name=tag + 'nn_pts_out_local')  # (N, P, K, 3)

    if fts_in is None:
        nn_fts_input = pts_in # (N, P, 3)
        #print('Error: fts_in is None.')
    else:
        #nn_fts_input = tf.concat([pts_in, fts_in], axis=-1, name=tag + 'nn_fts_input') # (N, P, 3+C_fts_in)
        nn_fts_input = fts_in # (N, P, C_fts_in)

    nn_fts_input = tf.expand_dims(nn_fts_input, axis=2)  # (N, P, 1, C_fts_in)

    # TODO: maybe try other conv1d? fc? there
    fts_conv = pf.conv2d_transpose(nn_fts_input, C, tag + 'fts_deconv', is_training, kernel_size=(1, 1), strides=(1, K), with_bn=bn, activation=activation) # (N, P, 1, C_fts_in) -> (N, P, K, C)

    if fts_out_skipped is not None:
        nn_fts_out_skipped = tf.gather_nd(fts_out_skipped, indices, name=tag + 'nn_fts_out_from_skip')
        fts_all_concat = tf.concat([fts_conv, nn_fts_out_skipped, nn_pts_out_local], axis=-1) # (N, P, K, C+3+channel_skipped)
    else:
        fts_all_concat = tf.concat([fts_conv, nn_pts_out_local], axis=-1) # (N, P, K, C+3)

    fts_pts_conv = pf.conv2d(fts_all_concat, C, tag + 'fts_pts_conv1D', is_training, kernel_size=(1,1), strides=(1,1), with_bn=bn, activation=activation) # (N, P, K, C+3_ch_skip?) -> (N, P, K, C)

    # summing up feature for each point
    fts_dense = tf.scatter_nd(tf.reshape(indices, (-1, 2)), tf.reshape(fts_pts_conv, (-1, C)), (batch_size, n_pts_out, C)) # (N, N_PTS_OUT, c)
    
    return fts_dense

def xconv(pts, fts, qrs, tag, N, K, D, P, C, C_pts_fts, is_training, with_X_transformation, depth_multiplier,
          sorting_method=None, with_global=False):
    _, indices_dilated = pf.knn_indices_general(qrs, pts, K * D, True)
    indices = indices_dilated[:, :, ::D, :]

    if sorting_method is not None:
        indices = pf.sort_points(pts, indices, sorting_method)

    nn_pts = tf.gather_nd(pts, indices, name=tag + 'nn_pts')  # (N, P, K, 3)
    nn_pts_center = tf.expand_dims(qrs, axis=2, name=tag + 'nn_pts_center')  # (N, P, 1, 3)
    nn_pts_local = tf.subtract(nn_pts, nn_pts_center, name=tag + 'nn_pts_local')  # (N, P, K, 3)

    # Prepare features to be transformed
    nn_fts_from_pts_0 = pf.dense(nn_pts_local, C_pts_fts, tag + 'nn_fts_from_pts_0', is_training) # (N, P, K, C_pts_fts)
    nn_fts_from_pts = pf.dense(nn_fts_from_pts_0, C_pts_fts, tag + 'nn_fts_from_pts', is_training) # (N, P, K, C_pts_fts) 
    if fts is None:
        nn_fts_input = nn_fts_from_pts
    else:
        nn_fts_from_prev = tf.gather_nd(fts, indices, name=tag + 'nn_fts_from_prev')
        nn_fts_input = tf.concat([nn_fts_from_pts, nn_fts_from_prev], axis=-1, name=tag + 'nn_fts_input')

    if with_X_transformation:
        ######################## X-transformation #########################
        X_0 = pf.conv2d(nn_pts_local, K * K, tag + 'X_0', is_training, (1, K)) # from (N, P, K, 3) -> (N, P, 1, K*K)
        X_0_KK = tf.reshape(X_0, (N, P, K, K), name=tag + 'X_0_KK') # (N, P, K, K)
        X_1 = pf.depthwise_conv2d(X_0_KK, K, tag + 'X_1', is_training, (1, K))
        X_1_KK = tf.reshape(X_1, (N, P, K, K), name=tag + 'X_1_KK')
        X_2 = pf.depthwise_conv2d(X_1_KK, K, tag + 'X_2', is_training, (1, K), activation=None)
        X_2_KK = tf.reshape(X_2, (N, P, K, K), name=tag + 'X_2_KK') # (N, P, K, K)
        fts_X = tf.matmul(X_2_KK, nn_fts_input, name=tag + 'fts_X') # (N, P, K, K) * (N, P, K, C_pts_fts?)
        ###################################################################
    else:
        fts_X = nn_fts_input

    fts_conv = pf.separable_conv2d(fts_X, C, tag + 'fts_conv', is_training, (1, K), depth_multiplier=depth_multiplier)
    fts_conv_3d = tf.squeeze(fts_conv, axis=2, name=tag + 'fts_conv_3d')

    if with_global:
        fts_global_0 = pf.dense(qrs, C // 4, tag + 'fts_global_0', is_training)
        fts_global = pf.dense(fts_global_0, C // 4, tag + 'fts_global', is_training)
        fts_conv_3d =  tf.concat([fts_global, fts_conv_3d], axis=-1, name=tag + 'fts_conv_3d_with_global')

    return fts_conv_3d

class PointCNN:
    def __init__(self, points, features, is_training, setting):
        xconv_params = setting.xconv_params
        fc_params = setting.fc_params
        with_X_transformation = setting.with_X_transformation
        sorting_method = setting.sorting_method
        N = tf.shape(points)[0]

        if setting.sampling == 'fps':
            from sampling import tf_sampling

        self.layer_pts = [points]
        if features is None:
            self.layer_fts = [features]
        else:
            features = tf.reshape(features, (N, -1, setting.data_dim - 3), name='features_reshape')
            C_fts = xconv_params[0]['C'] // 2
            features_hd = pf.dense(features, C_fts, 'features_hd', is_training)
            self.layer_fts = [features_hd]

        for layer_idx, layer_param in enumerate(xconv_params):
            tag = 'xconv_' + str(layer_idx + 1) + '_'
            K = layer_param['K'] # neighbor size
            D = layer_param['D'] # delation rate
            P = layer_param['P'] # representative point number in the out put (-1 indicates all input points are output as representative points)
            C = layer_param['C'] # channel number
            links = layer_param['links']
            if setting.sampling != 'random' and links:
                print('Error: flexible links are supported only when random sampling is used!')
                exit()

            # get k-nearest points
            pts = self.layer_pts[-1]
            fts = self.layer_fts[-1]
            if P == -1 or (layer_idx > 0 and P == xconv_params[layer_idx - 1]['P']): # first layer or last layer
                qrs = self.layer_pts[-1]
            else:
                if setting.sampling == 'fps':
                    fps_indices = tf_sampling.farthest_point_sample(P, pts)
                    batch_indices = tf.tile(tf.reshape(tf.range(N), (-1, 1, 1)), (1, P, 1))
                    indices = tf.concat([batch_indices, tf.expand_dims(fps_indices,-1)], axis=-1)
                    qrs = tf.gather_nd(pts, indices, name= tag + 'qrs') # (N, P, 3)
                elif setting.sampling == 'ids':
                    indices = pf.inverse_density_sampling(pts, K, P)
                    qrs = tf.gather_nd(pts, indices)
                elif setting.sampling == 'random':
                    qrs = tf.slice(pts, (0, 0, 0), (-1, P, -1), name=tag + 'qrs')  # (N, P, 3)
                else:
                    print('Unknown sampling method!')
                    exit()
            self.layer_pts.append(qrs)

            if layer_idx == 0:
                C_pts_fts = C // 2 if fts is None else C // 4
                depth_multiplier = 4
            else:
                C_prev = xconv_params[layer_idx - 1]['C']
                C_pts_fts = C_prev // 4
                depth_multiplier = math.ceil(C / C_prev)
            with_global = (setting.with_global and layer_idx == len(xconv_params) - 1)
            fts_xconv = xconv(pts, fts, qrs, tag, N, K, D, P, C, C_pts_fts, is_training, with_X_transformation,
                              depth_multiplier, sorting_method, with_global)
            fts_list = []
            for link in links:
                fts_from_link = self.layer_fts[link]
                if fts_from_link is not None:
                    fts_slice = tf.slice(fts_from_link, (0, 0, 0), (-1, P, -1), name=tag + 'fts_slice_' + str(-link))
                    fts_list.append(fts_slice)
            if fts_list:
                fts_list.append(fts_xconv)
                self.layer_fts.append(tf.concat(fts_list, axis=-1, name=tag + 'fts_list_concat'))
            else:
                self.layer_fts.append(fts_xconv)

        if hasattr(setting, 'xdconv_params'):
            for layer_idx, layer_param in enumerate(setting.xdconv_params):
                tag = 'xdconv_' + str(layer_idx + 1) + '_'
                K = layer_param['K']
                D = layer_param['D']
                pts_layer_idx = layer_param['pts_layer_idx']
                qrs_layer_idx = layer_param['qrs_layer_idx']

                pts = self.layer_pts[pts_layer_idx + 1]
                fts = self.layer_fts[pts_layer_idx + 1] if layer_idx == 0 else self.layer_fts[-1]
                qrs = self.layer_pts[qrs_layer_idx + 1]
                fts_qrs = self.layer_fts[qrs_layer_idx + 1]
                P = xconv_params[qrs_layer_idx]['P']
                C = xconv_params[qrs_layer_idx]['C']
                C_prev = xconv_params[pts_layer_idx]['C']
                C_pts_fts = C_prev // 4
                depth_multiplier = 1
                fts_xdconv = xconv(pts, fts, qrs, tag, N, K, D, P, C, C_pts_fts, is_training, with_X_transformation,
                                   depth_multiplier, sorting_method)
                fts_concat = tf.concat([fts_xdconv, fts_qrs], axis=-1, name=tag + 'fts_concat')
                fts_fuse = pf.dense(fts_concat, C, tag + 'fts_fuse', is_training)
                self.layer_pts.append(qrs)
                self.layer_fts.append(fts_fuse)

        self.fc_layers = [self.layer_fts[-1]]
        for layer_idx, layer_param in enumerate(fc_params):
            C = layer_param['C']
            dropout_rate = layer_param['dropout_rate']
            fc = pf.dense(self.fc_layers[-1], C, 'fc{:d}'.format(layer_idx), is_training)
            fc_drop = tf.layers.dropout(fc, dropout_rate, training=is_training, name='fc{:d}_drop'.format(layer_idx))
            self.fc_layers.append(fc_drop)

import numpy as np
if __name__=='__main__':
    pts = np.asarray([
                        [
                            [0, 0, 0],
                            [1, 0, 0],
                            [2, 0, 0],
                            [3, 0, 0],
                            [4, 0, 0],
                            [5, 0, 0],
                            [6, 0, 0],
                        ],

                        [
                            [0, 0, 0],
                            [1, 0, 0],
                            [2, 0, 0],
                            [3, 0, 0],
                            [4, 0, 0],
                            [5, 0, 0],
                            [6, 0, 0],
                        ]
            ], dtype=float)
    print(pts.shape)

    qrs = np.asarray([
                        [
                            [1, 0, 0],
                            [3, 0, 0],
                            [5, 0, 0],
                        ],

                        [
                            [1, 0, 0],
                            [3, 0, 0],
                            [5, 0, 0],
                        ]
            ], dtype=float)

    pts_pl = tf.placeholder(tf.float32, shape=(2, 7, 3))
    qrs_pl = tf.placeholder(tf.float32, shape=(2, 3, 3))

    fts, fts_fc = point_conv_transpose_conv2d(qrs_pl, None, pts_pl, None, None, K=3, D=1, C=2, is_training=tf.constant(False, shape=()))

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        pts_val = sess.run(pts_pl, {pts_pl: pts, qrs_pl: qrs})
        qrs_val = sess.run(qrs_pl, {pts_pl: pts, qrs_pl: qrs})


        fts_val, fts_fc_val = sess.run([fts, fts_fc], {pts_pl: pts, qrs_pl: qrs})
        print(fts_val)
        print('_')
        print( fts_fc_val)
        

