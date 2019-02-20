import numpy as np
import tensorflow as tf

def directed_hausdorff(point_cloud_A, point_cloud_B):
  '''
  input:
    point_cloud_A: Tensor, B x N x 3
    point_cloud_B: Tensor, B x N x 3
  return:
    Tensor, B, directed hausdorff distance, A -> B
  '''
  npoint = point_cloud_A.shape[1]

  A = tf.expand_dims(point_cloud_A, axis=2) # (B, N, 1, 3)
  A = tf.tile(A, (1, 1, npoint, 1)) # (B, N, N, 3)

  B = tf.expand_dims(point_cloud_B, axis=1) # (B, 1, N, 3)
  B = tf.tile(B, (1, npoint, 1, 1)) # (B, N, N, 3)

  distances = tf.squared_difference(B, A) # (B, N, N, 3)
  distances = tf.reduce_sum(distances, axis=-1) # (B, N, N, 1)
  distances = tf.sqrt(distances) # (B, N, N)

  shortest_dists, _ = tf.nn.top_k(-distances)
  shortest_dists = tf.squeeze(-shortest_dists) # (B, N)

  hausdorff_dists, _ = tf.nn.top_k(shortest_dists) # (B, 1)
  hausdorff_dists = tf.squeeze(hausdorff_dists)

  return hausdorff_dists

if __name__=='__main__':
  u = np.array([
                [
                  [1,0],
                  [0,1],
                  [-1,0],
                  [0,-1]
                ], 
                [
                  [1,0],
                  [0,1],
                  [-1,0],
                  [0,-1]
                ]
              ])

  v = np.array([
                [
                  [2,0],
                  [0,2],
                  [-2,0],
                  [0,-4]
                ], 
                [
                  [2,0],
                  [0,2],
                  [-2,0],
                  [0,-4]
                ]
              ])
  u_tensor = tf.constant(u, dtype=tf.float32)
  u_tensor = tf.tile(u_tensor, (1,500,1))
  v_tensor = tf.constant(v, dtype=tf.float32)
  v_tensor = tf.tile(v_tensor, (1,500,1))
  distances = directed_hausdorff(u_tensor, v_tensor)
  distances1 = directed_hausdorff(v_tensor, u_tensor)

  with tf.Session() as sess:
    # Init variables
    init = tf.global_variables_initializer()
    sess.run(init)

    d_val = sess.run(distances)
    print(d_val)
    print(d_val.shape)

    d_val1 = sess.run(distances1)
    print(d_val1)
    print(d_val1.shape)
