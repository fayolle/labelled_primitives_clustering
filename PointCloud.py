import numpy as np
from sklearn.cluster import DBSCAN


def color_map(index, norm=True):
    colors = [
        [0.,0.,0.],
        [0.5000, 0.5400, 0.5300], 
        [0.8900, 0.1500, 0.2100], 
        [0.6400, 0.5800, 0.5000],
        [1.0000, 0.3800, 0.0100], 
        [1.0000, 0.6600, 0.1400], 
        [0.4980, 1.0000, 0.0000],
        [0.4980, 1.0000, 0.8314], 
        [0.9412, 0.9725, 1.0000], 
        [0.5412, 0.1686, 0.8863],
        [0.5765, 0.4392, 0.8588], 
        [0.3600, 0.1400, 0.4300], 
        [0.5600, 0.3700, 0.6000],
    ]

    color = colors[index % len(colors)]

    if not norm:
        color[0] *= 255
        color[1] *= 255
        color[2] *= 255

    return color


def cluster2Color(cluster, cluster_idx):
    colors = np.zeros(shape=(len(cluster), 3))
    point_idx = 0
    for _ in cluster:
        colors[point_idx, :] = color_map(cluster_idx)
        point_idx += 1

    return colors


def label2color(labels):
    num = labels.shape[0]
    colors = np.zeros((num, 3))

    print(labels)

    minl, maxl = np.min(labels), np.max(labels)
    for l in range(minl, maxl + 1):
        colors[labels == l, :] = color_map(l)

    return colors


def read_pointcloud(path, delimiter=' ', hasHeader=True):
    with open(path, 'r') as f:
        if hasHeader:
            _ = f.readline()
        pc = [[float(x) for x in line.rstrip().split(delimiter)] for line in f if line != '']

    return np.asarray(pc)[:, :6]


def read_segps(path, delimiter=' ', hasHeader=True):
    with open(path, 'r') as f:
        if hasHeader:
            _ = f.readline()

        pc = [[float(x) for x in line.rstrip().split(delimiter)] for line in f if line != '']

    return np.asarray(pc)
    

def write_pointcloud(file, pc, numCols=6):
    np.savetxt(file, pc[:, :numCols], header=str(len(pc)) + ' ' + str(numCols), comments='')


def farthest_point_sampling(pts, K):
    if K > 0:
        if pts.shape[0] < K:
            return pts
    else:
        return pts

    def calc_distances(p0, points):
        return ((p0[:3] - points[:, :3]) ** 2).sum(axis=1)

    farthest_pts = np.zeros((K, pts.shape[1]))
    farthest_pts[0] = pts[np.random.randint(len(pts))]
    distances = calc_distances(farthest_pts[0], pts)
    for i in range(1, K):
        farthest_pts[i] = pts[np.argmax(distances)]
        distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))

    return farthest_pts


def cluster_dbscan(data, selected_indices, eps, min_samples=5, metric='euclidean', algo='auto'):
    db_res = DBSCAN(eps=eps, metric=metric, n_jobs=-1, min_samples=min_samples, algorithm=algo).fit(data[:, selected_indices])

    labels = db_res.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    clusters = {}
    for idx, l in enumerate(labels):
        if l == -1:
            continue
        clusters.setdefault(str(l), []).append(data[idx, :])

    npClusters = []
    for cluster in clusters.values():
        npClusters.append(np.array(cluster))

    return npClusters


# Plane -> 0
# Sphere -> 1
# Cylinder -> 2
# Cone -> 3
# Torus -> 4 
def write_clusters(path, clusters, type_column=6):
    file = open(path, "w")
    file.write(str(len(clusters)) + "\n")

    for cluster in clusters:
        types = np.unique(cluster[:, type_column], axis=0).astype(int)

        def type_mapping(t):
            return t

        types = np.array([type_mapping(t) for t in types])
        print("Types: {}, Points: {}".format(types, cluster.shape[0]))

        np.savetxt(file, types.reshape(1, types.shape[0]), delimiter=';', header='', comments='', fmt='%i')
        np.savetxt(file, cluster[:, :6], header=str(len(cluster)) + ' ' + str(6), comments='')


def normalize_pointcloud(pc, factor=1.0):
    max = pc.max(axis=0)
    max += max * 0.01
    min = pc.min(axis=0)
    min -= min * 0.01

    f = np.max([abs(max[0] - min[0]), abs(max[1] - min[1]), abs(max[2] - min[2])])

    pc[:, 0:3] /= (f * factor)
    pc[:, 3:6] /= (np.linalg.norm(pc[:, 3:6], ord=2, axis=1, keepdims=True))

    return pc


def filter_clusters(clusters, filter):
    filtered_clusters = []

    for c in clusters:
        if filter(c):
            filtered_clusters.append(c)

    return filtered_clusters


def append_onehotencoded_type(data, factor=1.0):
    types = data[:, 6].astype(int)
    res = np.zeros((len(types), 5))
    res[np.arange(len(types)), types] = factor

    return np.column_stack((data, res))


def cluster_cubes(data, cluster_dims, max_points_per_cluster=-1, min_points_per_cluster=-1):
    if cluster_dims[0] == 1 and cluster_dims[1] == 1 and cluster_dims[2] == 1:
        print("No need to cluster")
        return [farthest_point_sampling(data, max_points_per_cluster)]

    max = data[:, :3].max(axis=0)
    max += max * 0.01

    min = data[:, :3].min(axis=0)
    min -= min * 0.01

    size = (max - min)

    clusters = {}

    cluster_size = size / np.array(cluster_dims, dtype=np.float32)

    print('Min: ' + str(min) + ' Max: ' + str(max))
    print('Cluster Size: ' + str(cluster_size))

    for row in data:
        cluster_pos = ((row[:3] - min) / cluster_size).astype(int)
        cluster_idx = cluster_dims[0] * cluster_dims[2] * cluster_pos[1] + cluster_dims[0] * cluster_pos[2] + \
                      cluster_pos[0]
        clusters.setdefault(cluster_idx, []).append(row)

    # Apply farthest point sampling to each cluster
    final_clusters = []
    for key, cluster in clusters.items():
        c = np.vstack(cluster)
        if c.shape[0] < min_points_per_cluster and -1 != min_points_per_cluster:
            continue

        if max_points_per_cluster is not -1:
            final_clusters.append(farthest_point_sampling(c, max_points_per_cluster))
        else:
            final_clusters.append(c)

    return final_clusters
