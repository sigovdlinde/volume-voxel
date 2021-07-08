# This is a script to convert the results of Bram Dekkers Pointcloud to true voxels
# and being able to make cross-sections in every dimention.
#
# Made by Sigo van der Linde (11052759) of the Univeristy of Amsterdam as a bachelor
# thesis.

import numpy as np
import open3d as o3d
import math
import time

def export_ply(grids):
    clouds = grids_to_clouds(grids)
    for i, cloud in enumerate(clouds):
        o3d.io.write_point_cloud("cloud_layer_" + str(i) + ".ply" , cloud)

# transform voxelgrids back to pointclouds
def grids_to_clouds(grids):
    clouds = []
    for grid in grids:
        new_cloud = o3d.geometry.PointCloud()
        for point in grid.get_voxels():
            new_cloud.points.append(grid.origin + point.grid_index * grid.voxel_size)
        clouds.append(new_cloud)
    return clouds


#############################################################################
# Print volume.
def print_volumes(grids, dimension):
    total_volume = 0
    for i, grid in enumerate(grids):
        volume = len(grid.get_voxels()) * dimension**3
        total_volume += volume
        print("The volume of layer %d is %.2f m3 or %d cm3." %
        (i, volume, volume * 10**6))

    print("The total volume is %.2f m3 or %d cm3." %
    (total_volume, total_volume * 10**6))

# plot all voxel layers.
def plot_voxels(grids):
    o3d.visualization.draw_geometries(grids)

# Read PLY files.
def read_pointclouds(names):
    pointclouds = []
    for name in names:
        pointclouds.append(o3d.io.read_point_cloud(name))
    return pointclouds

# Convert pointclouds to voxelgrids.
def clouds_to_grids(clouds, voxel_size = 0.05):
    grids = []
    for cloud in clouds:
        grids.append(o3d.geometry.VoxelGrid.create_from_point_cloud(cloud, voxel_size))
    return grids

# Rotate a pointcloud in its x, y or z axis.
def rotate_pointclouds(clouds, rotate_d, rotate_deg):
    s = math.sin(math.radians(rotate_deg))
    c = math.cos(math.radians(rotate_deg))

    if rotate_d == 0:
        R = [[1, 0, 0], [0, c, -s], [0, s, c]]
    elif rotate_d == 1:
        R = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    elif rotate_d == 2:
        R = [[c, -s, 0], [s, c, 0], [0, 0, 1]]

    for cloud in clouds:
        cloud.rotate(np.asarray(R, dtype=np.float64), True)
    return clouds

# Find the minimum and maximum value in different point clouds in a certain axis.
def find_max_min(clouds, slice_d):
    max = -np.Inf
    min = np.Inf
    for cloud in clouds:
        if max <= cloud.get_max_bound()[slice_d]:
            max = cloud.get_max_bound()[slice_d]
        if min >= cloud.get_min_bound()[slice_d]:
            min = cloud.get_min_bound()[slice_d]
    return max, min

# Based on given percentage give the length on the axis where the cross section should be made.
def find_cutoff(percentage, max, min):
    length = abs(max) + abs(min)
    return percentage/100 * length + min

# Cross-section of all layers in any dimention.
def cross_section(clouds, slice_d, percentage):
    new_clouds = []
    max, min = find_max_min(clouds, slice_d)
    cutoff = find_cutoff(percentage, max, min)

    for cloud in clouds:
        new_cloud = o3d.geometry.PointCloud()
        points = np.asarray(cloud.points)

        for point in points:
                if point[slice_d] > cutoff:
                    new_cloud.points.append(point)
                    new_cloud.colors.append(cloud.colors[0])

        new_clouds.append(new_cloud)
    return new_clouds

# Color different layers.
def color_clouds(clouds, colors):
    new_clouds = []

    for i, cloud in enumerate(clouds):
        color = np.asarray([float(x/255) for x in colors[i]])
        cloud.paint_uniform_color(color)

    return clouds

# slice_d: determine on which axis to slice, 0=x-axis, 1=y-axis, 2=z-axis.
# cross: True if you want to slice.
# rotate_cloud: True if you want to rotate.
# percentage: percentage of the slice (1-99).
# colors: color per layer in rgb.
def pointcloud_algo(names, dimension):
    cross = False
    rotate_cloud = False
    slice_d = 0
    percentage = 55
    rotate_d = 2
    rotate_deg = -10
    dimension = 0.05

    # just supporting 2 layers now, need to add mroe colors for more layers.
    colors = [[50,50,50], [150,150,150]]

    clouds = read_pointclouds(names)

    if rotate_cloud:
        clouds = rotate_pointclouds(clouds, rotate_d, rotate_deg)

    if cross:
        clouds = cross_section(clouds, slice_d, percentage)

    clouds = color_clouds(clouds, colors)
    grids = clouds_to_grids(clouds, dimension)

    return grids

#############################################################################
# Combine different point clouds into 1 cloud.
def combine_pointclouds(clouds):
    new_cloud = o3d.geometry.PointCloud()
    for cloud in clouds:
        for point in np.asarray(cloud.points):
            new_cloud.points.append(point)
    return new_cloud

# trim point cloud based on given treshold.
def cut_pointcloud(cloud1, cloud2, threshold):
    distances = cloud1.compute_point_cloud_distance(cloud2)
    cloud = o3d.geometry.PointCloud()
    points = np.asarray(cloud1.points)

    for i, point in enumerate(points):
        if threshold < distances[i]:
            cloud.points.append(point)

    return cloud

# trim both clouds.
def trim_2_pointclouds(cloud1, cloud2, threshold):
    clouds = []
    clouds.append(cut_pointcloud(cloud1, cloud2, threshold))
    clouds.append(cut_pointcloud(cloud2, cloud1, threshold))
    return clouds

# treshold: Value in meters for the distance between the 2 clouds which is allowed as minimum.
def pivot_algo(names, dimension):
    threshold = 0.05
    clouds = read_pointclouds(names)
    clouds = trim_2_pointclouds(clouds[0], clouds[1], threshold)
    cloud = combine_pointclouds(clouds)
    cloud.estimate_normals()

    # get all nearest neighbour distances and compute the mean.
    distances = cloud.compute_nearest_neighbor_distance()
    r = 3 * np.mean(distances)
    dv = o3d.utility.DoubleVector([r, r * 2])
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(cloud, dv)

    mesh.paint_uniform_color([0, 100/255, 0])

    return mesh

############################################################################

def meshes_to_grids(meshes, voxel_size=0.05):
        grids = []
        for mesh in meshes:
            grids.append(o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size))
        return grids

def plot_mesh(mesh):
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True, mesh_show_wireframe=False)

# names: give the filenames of the layers which need to be used in the visualization, can be obj or ply files.
# dimension: the X*X*X dimensions of the primitive for which the pointcloud was created in meters.
def main():

    # Files used for pointcloud to voxel.
    names = ["pointcloud.ply", "pointcloud2.ply"]

    # Files used for ball-pivoting.
    names_pivot = ['HALOS_trenches_10072019.ply','HALOS_trenches_15072019.ply']

    # True for pointcloud to voxelgrid, False for ball-pivoting
    pointcloud = False

    # For exporting voxelgrid to ply file.
    export = False

    # Dimension of a voxel in meters.
    dimension = 0.05

    # When ball-pivoting works, this can be set to true to transform to voxelgrid.
    mesh_full = False

    if pointcloud:
        grids = pointcloud_algo(names, dimension)
        print_volumes(grids, dimension)
        plot_voxels(grids)

        if export:
            export_ply(grids)

    else:
        mesh = pivot_algo(names_pivot, dimension)
        plot_mesh(mesh)

        if mesh_full:
            grids = meshes_to_grids([mesh], dimension)
            print_volumes(grids, dimension)

            if export:
                export_ply(grids)

if __name__ == "__main__":
    main()
