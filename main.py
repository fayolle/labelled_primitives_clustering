import polyscope as ps
import numpy as np
import sys


from PointCloud import append_onehotencoded_type
from PointCloud import write_clusters
from PointCloud import cluster2Color
from PointCloud import cluster_dbscan
from PointCloud import read_segps
from PointCloud import normalize_pointcloud


def main(input_filename, output_filename):     
    # If we need to simplify the input point-cloud, call cluster_cubes 
    # grid_cluster_max_pts = 3000 # 8192
    # grid_clusters = [3,3,3]
    
    # TODO These should be options 
    type_cluster_eps = 0.1
    type_cluster_min_pts = 50

    # Load input point-cloud with coordinates, normals, primitive type and initial cluster id
    pc_seg = read_segps(input_filename)
    pc_seg = normalize_pointcloud(pc_seg)
    final_pc = pc_seg[:, :7] # ignore the label

    final_pc = append_onehotencoded_type(final_pc)

    print("DB Scan on input point cloud " + str(final_pc.shape))
    total_clusters = []

    clusters = cluster_dbscan(final_pc, [0, 1, 2, 3, 4, 5], eps=type_cluster_eps, min_samples=type_cluster_min_pts)
    print("Pre-clustering done. Clusters: ", len(clusters))

    for cluster in clusters:
        print("Second level clustering")

        prim_types_in_cluster = len(np.unique(cluster[:, 6], axis=0))
        if prim_types_in_cluster == 1:
            print("No need for second level clustering since there is only a single primitive type in the cluster.")
            total_clusters.append(cluster)
        else:
            sub_clusters = cluster_dbscan(cluster, [0, 1, 2, 7, 8, 9, 10, 11], eps=type_cluster_eps, min_samples=type_cluster_min_pts)
            print("Sub clusters: ", len(sub_clusters))
            total_clusters.extend(sub_clusters)

    result_clusters = list(filter(lambda c: c.shape[0] > type_cluster_min_pts, total_clusters))

    for cluster in result_clusters:
        print("Cluster: ", cluster.shape[0])

    print('Saving clusters')
    write_clusters(output_filename, result_clusters, 6)

    print('Done')
    
    return result_clusters


def usage(cmdline):
    print('Command line used: ' + ' '.join(cmdline))
    print('------')
    print('Usage: python ' + cmdline[0] + ' input_filename.segps output_filename.txt')
    print('where input_filename.segps is the input point-cloud (with normals, primitive types and labels')
    print('and output_filename.txt is the output file with the list of clusters.')
    

if __name__ == '__main__':
    if len(sys.argv) != 3: 
        usage(sys.argv)
        sys.exit(1)
    
    ps.init()

    result_clusters = main(sys.argv[1], sys.argv[2])

    for i, result_cluster in enumerate(result_clusters):
        pc = ps.register_point_cloud("points_" + str(i), result_cluster[:, :3], radius=0.01)
        pc.add_color_quantity("prim types", cluster2Color(result_cluster,i), True)

    ps.show()
