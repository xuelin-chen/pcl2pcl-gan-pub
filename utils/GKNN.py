import collections
import os
import time
import sys

import numpy as np
import pygraph.algorithms
import pygraph.algorithms.minmax
import pygraph.classes.graph
import tensorflow as tf
#from matplotlib import pyplot as plt
from scipy import spatial
from tqdm import tqdm



import pc_util
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TF_SAMPLING_DIR = os.path.join(BASE_DIR, '../tf_ops/sampling')
sys.path.append(TF_SAMPLING_DIR)
from tf_sampling import farthest_point_sample

class GKNN():
    def __init__(self, point_path, patch_size=2048, patch_num=100, normalization=False, add_noise=False):
        print(point_path)
        self.name = point_path.split('/')[-1][:-4]
        self.data = pc_util.read_ply(point_path)
        self.data = self.data[:,0:3]

        #####
        # angles = np.asarray([0.25 * np.pi, 0.25 * np.pi, 0.25 * np.pi])
        # Rx = np.array([[1, 0, 0],
        #                [0, np.cos(angles[0]), -np.sin(angles[0])],
        #                [0, np.sin(angles[0]), np.cos(angles[0])]])
        # Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
        #                [0, 1, 0],
        #                [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        # Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
        #                [np.sin(angles[2]), np.cos(angles[2]), 0],
        #                [0, 0, 1]])
        # rotation_matrix = np.dot(Rz, np.dot(Ry, Rx))
        # self.data = np.dot(self.data, rotation_matrix)
        #####
        
        self.clean_data = self.data

        # self.data = self.data[np.random.permutation(len(self.data))[:100000]]
        self.centroid = np.mean(self.data, axis=0, keepdims=True)
        self.furthest_distance = np.amax(np.sqrt(np.sum((self.data - self.centroid) ** 2, axis=-1)), keepdims=True)

        if normalization:
            print("Normalize the point data")
            self.data = (self.data-self.centroid)/self.furthest_distance
            self.clean_data = self.data
        if add_noise:
            print("Add gaussian noise into the point")
            #self.data = jitter_perturbation_point_cloud(np.expand_dims(self.data,axis=0), sigma=self.furthest_distance * 0.004, clip=self.furthest_distance * 0.01)
            self.data = self.data[0]

        print("Total %d points" % len(self.data))

        self.patch_size = patch_size
        self.patch_num = patch_num

        start = time.time()
        self.nbrs = spatial.cKDTree(self.clean_data)
        # dists,idxs = self.nbrs.query(self.clean_data,k=6,distance_upper_bound=0.2)
        dists,idxs = self.nbrs.query(self.clean_data,k=16)
        self.graph=[] # xl: not used??
        for item,dist in zip(idxs,dists):
            item = item[dist<0.07] #use 0.03 for chair7 model; otherwise use 0.05
            self.graph.append(set(item))
        print("Build the graph cost %f second" % (time.time() - start))

        self.graph2 = pygraph.classes.graph.graph()
        self.graph2.add_nodes(range(len(self.clean_data)))
        sid = 0
        for idx, dist in zip(idxs, dists):
            for eid, d in zip(idx, dist):
                if not self.graph2.has_edge((sid, eid)) and eid < len(self.clean_data):
                    self.graph2.add_edge((sid, eid), d)
            sid = sid + 1
        print("Build the graph cost %f second" % (time.time() - start))

        return

    def bfs_knn(self, seed=0, patch_size=1024):
        q = collections.deque()
        visited = set()
        result = []
        q.append(seed)
        while len(visited)<patch_size and q:
            vertex = q.popleft()
            if vertex not in visited:
                visited.add(vertex)
                result.append(vertex)
                if len(q)<patch_size*5:
                    q.extend(self.graph[vertex] - visited)
        return result

    def geodesic_knn(self, seed=0, patch_size=1024):
        _, dist = pygraph.algorithms.minmax.shortest_path(self.graph2, seed)
        dist_list = np.asarray([dist[item] if item in dist else 10000 for item in range(len(self.data))])
        idx = np.argsort(dist_list)
        return idx[:patch_size]


    def estimate_single_density(self,id=0,patch_size=128):
        query_point = self.data[id]
        try:
            point = self.bfs_knn(id, patch_size=patch_size)
        except:
            return np.asarray([0])
        dist = np.sum((query_point-point)**2,axis=-1)
        avg_dist = np.sum(dist)/patch_size
        return avg_dist

    def estimate_density(self):
        self.density=[]
        for id in tqdm(range(len(self.data))):
            dist = self.estimate_single_density(id)
            self.density.append(dist)
        self.density = np.asarray(self.density)
        #plt.hist(self.density)
        #plt.show()

    def get_seed_fromdensity(self,seed_num,idx=None):
        if idx is None:
            candidata_num = min(len(self.data), seed_num * 50)
            print("Total %d candidata random points" % candidata_num)
            idx = np.random.permutation(len(self.data))[:candidata_num]
        density = []
        for item in tqdm(idx):
            dist = self.estimate_single_density(item)
            density.append(dist)
        density = np.asarray(density)
        density = density*density
        density = density/np.sum(density)
        idx = np.random.choice(idx,size=seed_num,replace=False,p=density)
        return idx


    def get_idx(self, num):
        idx1 = np.asarray([])

        # randomly select seed
        idx2 = np.random.randint(0, len(self.clean_data)-20, [num-len(idx1), 1])
        idx2 = idx2 + np.arange(0,20).reshape((1,20))
        point = self.clean_data[idx2]
        point = np.mean(point, axis=1)
        _,idx2 = self.nbrs.query(point,k=10)

        all_samples = tf.convert_to_tensor(self.clean_data, dtype=np.float32)
        all_samples = tf.expand_dims(all_samples, axis=0)
        sample_indices = farthest_point_sample(num, all_samples)
        idxxxx = self.sess.run(sample_indices)
        idxxxx_np = np.squeeze(np.array(idxxxx))
        return idxxxx_np       

    def crop_patch(self, save_root_path, use_dijkstra=True, scale_ratio=1, gpu_id=0):
        if save_root_path[-1]=='/':
            save_root_path = save_root_path[:-1]
        if not os.path.exists(save_root_path):
            os.makedirs(save_root_path)

        print('Creating TF session...')
        config = tf.ConfigProto()
        config.gpu_options.visible_device_list=str(gpu_id)
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.sess = tf.Session(config=config)
        
        seeds = self.get_idx(self.patch_num)
        print(seeds)

        self.sess.close()

        i = -1
        for seed in (seeds):
            i = i+1
            # patch_size = self.patch_size*np.random.randint(1,scale_ratio+1)
            assert scale_ratio>=1.0
            patch_size = int(self.patch_size*np.random.uniform(1.0, scale_ratio))
            try:
                if use_dijkstra:
                    idx = self.geodesic_knn(seed, patch_size)
                else:
                    idx = self.bfs_knn(seed, patch_size)
            except Exception as e:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno)
            idxx = np.random.permutation(patch_size)[:self.patch_size]
            idxx.sort()
            idx = idx[idxx]
            point = self.data[idx]

            #print("patch:%d  point:%d" % (i, patch_size))

            output_filename = '%s/%s_%d.ply' % (save_root_path, self.name, i)
            pc_util.write_ply(point, output_filename)
            #np.savetxt('%s/%s_%d.xyz' % (save_root_path, self.name, i), point, fmt='%0.6f')


    def crop_patch_boxes(self, save_root_path, use_dijkstra=True, id=0, boxes = None, scale_ratio=1,random_ratio=0.7):
        if save_root_path[-1]=='/':
            save_root_path = save_root_path[:-1]
        if not os.path.exists(save_root_path):
            os.makedirs(save_root_path)
            os.makedirs(save_root_path + "_dist")
            os.makedirs(save_root_path+"_edge")
            os.makedirs(save_root_path + "_edgepoint")
            os.makedirs(save_root_path + "_face")
            os.makedirs(save_root_path + "_facepoint")

        ## the first part
        if boxes is not None:
            centroid = []
            for box in boxes:
                cen = np.mean(np.reshape(box,[-1,3]),axis=0)
                centroid.append(cen)
            centroid = np.asarray(centroid)
            idx = self.nbrs.query_ball_point(centroid, r=0.03)

            total_idx = []
            for item in idx:
                total_idx.extend(item)
            total_idx = np.asarray(total_idx)

            corner_num = int(self.patch_num*random_ratio)
            idx = np.random.permutation(len(total_idx))[:corner_num]
            seeds1 = total_idx[idx]
            config = tf.ConfigProto()
            config.gpu_options.visible_device_list = str(id)
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            self.sess = tf.Session(config=config)
            tf_point = tf.placeholder(tf.float32, [1, self.patch_size, 3])
            tf_edge = tf.placeholder(tf.float32, [1, self.edge.shape[0], 6])
            tf_face = tf.placeholder(tf.float32, [1, self.face.shape[0], 9])
            tf_edge_dist = model_utils.distance_point2edge(tf_point, tf_edge)
            tf_edge_dist = tf.sqrt(tf.squeeze(tf_edge_dist, axis=0))
            tf_face_dist = model_utils.distance_point2mesh(tf_point, tf_face)
            tf_face_dist = tf.sqrt(tf.squeeze(tf_face_dist, axis=0))

            i = -1
            for seed in tqdm(seeds1):
                t0 = time.time()
                i = i + 1
                assert scale_ratio >= 1.0
                patch_size = int(self.patch_size * np.random.uniform(1.0, scale_ratio))
                try:
                    if use_dijkstra:
                        idx = self.geodesic_knn(seed, patch_size)
                    else:
                        idx = self.bfs_knn(seed, patch_size)
                except:
                    print("has exception")
                    continue
                idxx = np.random.permutation(patch_size)[:self.patch_size]
                idxx.sort()
                idx = idx[idxx]
                point = self.data[idx]
                clean_point = self.clean_data[idx]
                t1 = time.time()
                edge_dist, face_dist = self.sess.run([tf_edge_dist, tf_face_dist],
                                                     feed_dict={tf_point: np.expand_dims(clean_point, axis=0),
                                                                tf_edge: np.expand_dims(self.edge, axis=0),
                                                                tf_face: np.expand_dims(self.face, axis=0)})
                subedge = self.get_subedge(edge_dist)
                subface = self.get_subface(face_dist)
                t2 = time.time()
                # print t1-t0, t2-t1
                print("patch:%d  point:%d  subedge:%d  subface:%d" % (i, patch_size, len(subedge), len(subface)))

                dist_min = np.reshape(np.min(edge_dist, axis=-1), [-1, 1])
                np.savetxt('%s/%s_%d.xyz' % (save_root_path, self.name, i), point, fmt='%0.6f')
                np.savetxt('%s_dist/%s_%d.xyz' % (save_root_path, self.name, i), np.concatenate([point, dist_min], axis=-1),
                           fmt='%0.6f')
                np.savetxt('%s_edge/%s_%d.xyz' % (save_root_path, self.name, i), subedge, fmt='%0.6f')
                np.savetxt('%s_edgepoint/%s_%d.xyz' % (save_root_path, self.name, i), sampling_from_edge(subedge),
                           fmt='%0.6f')
                np.savetxt('%s_face/%s_%d.xyz' % (save_root_path, self.name, i), subface, fmt='%0.6f')
                np.savetxt('%s_facepoint/%s_%d.xyz' % (save_root_path, self.name, i), sampling_from_face(subface),
                           fmt='%0.6f')
            self.sess.close()
        else:
            corner_num = 0

        ## second part
        seeds2 = self.get_idx((self.patch_num-corner_num) * 4, edge_patch_ratio=0.0)
        seeds2 = np.asarray(seeds2).astype(np.int32)
        ##remove those seed near the corner
        if boxes is not None:
            d1 = self.data[seeds2]
            d2 = centroid
            dist = spatial.distance_matrix(d1,d2)
            min_distance  = np.amin(dist,axis=1)
            new_seed = []
            for item1, item2 in zip(seeds2, min_distance):
                if item2 >0.05:
                    new_seed.append(item1)
            new_seed = np.asarray(new_seed)
            seeds2 = new_seed[:(self.patch_num-corner_num)]
        else:
            seeds2 = seeds2[:(self.patch_num-corner_num)]

        self.edge = np.asarray([[10, 10, 10, 20, 20, 20]])
        config = tf.ConfigProto()
        config.gpu_options.visible_device_list=str(id)
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.sess = tf.Session(config=config)
        tf_point = tf.placeholder(tf.float32, [1, self.patch_size, 3])
        tf_edge = tf.placeholder(tf.float32, [1, self.edge.shape[0], 6])
        tf_face = tf.placeholder(tf.float32, [1, self.face.shape[0], 9])
        tf_edge_dist = model_utils.distance_point2edge(tf_point, tf_edge)
        tf_edge_dist = tf.sqrt(tf.squeeze(tf_edge_dist, axis=0))
        tf_face_dist = model_utils.distance_point2mesh(tf_point, tf_face)
        tf_face_dist = tf.sqrt(tf.squeeze(tf_face_dist, axis=0))

        i = corner_num-1
        for seed in tqdm(seeds2):
            t0 = time.time()
            i = i+1
            # patch_size = self.patch_size*np.random.randint(1,scale_ratio+1)
            assert scale_ratio>=1.0
            patch_size = int(self.patch_size*np.random.uniform(1.0, scale_ratio))
            try:
                if use_dijkstra:
                    idx = self.geodesic_knn(seed, patch_size)
                else:
                    idx = self.bfs_knn(seed, patch_size)
            except:
                print("has exception")
                continue
            idxx = np.random.permutation(patch_size)[:self.patch_size]
            idxx.sort()
            idx = idx[idxx]
            point = self.data[idx]
            clean_point = self.clean_data[idx]
            t1 = time.time()
            edge_dist,face_dist = self.sess.run([tf_edge_dist,tf_face_dist],feed_dict={tf_point: np.expand_dims(clean_point, axis=0),
                                                                                       tf_edge: np.expand_dims(self.edge, axis=0),
                                                                                       tf_face: np.expand_dims(self.face, axis=0)})
            subedge = self.get_subedge(edge_dist)
            subface = self.get_subface(face_dist)
            t2 = time.time()
            # print t1-t0, t2-t1
            print("patch:%d  point:%d  subedge:%d  subface:%d" % (i, patch_size, len(subedge),len(subface)))

            dist_min = np.reshape(np.min(edge_dist, axis=-1),[-1,1])
            np.savetxt('%s/%s_%d.xyz' % (save_root_path, self.name, i), point, fmt='%0.6f')
            np.savetxt('%s_dist/%s_%d.xyz' % (save_root_path,self.name, i), np.concatenate([point,dist_min],axis=-1), fmt='%0.6f')
            np.savetxt('%s_edge/%s_%d.xyz' % (save_root_path, self.name, i), subedge,fmt='%0.6f')
            np.savetxt('%s_edgepoint/%s_%d.xyz' % (save_root_path, self.name, i), sampling_from_edge(subedge),fmt='%0.6f')
            np.savetxt('%s_face/%s_%d.xyz' % (save_root_path, self.name, i), subface, fmt='%0.6f')
            np.savetxt('%s_facepoint/%s_%d.xyz' % (save_root_path, self.name, i), sampling_from_face(subface), fmt='%0.6f')
        self.sess.close()

if __name__ == '__main__':
    gm = GKNN(point_path='aae036d8ebdc472535836c728d324152_clean.ply',
              patch_size=2048, patch_num=100)
    gm.crop_patch('./test_GKNN',use_dijkstra=True,scale_ratio=1, gpu_id=0)

def query_neighbor(pred_pts, sample_pts, radius=None):
    if np.isscalar(radius):
        radius = np.asarray([radius])
    radius = np.asarray(radius)
    pred_tree = spatial.cKDTree(pred_pts)
    sample_tree = spatial.cKDTree(sample_pts)
    counts = []
    for radi in radius:
        idx = sample_tree.query_ball_tree(pred_tree, r=radi)
        number = [len(item) for item in idx]
        counts.append(number)
    counts = np.asarray(counts)
    return counts


