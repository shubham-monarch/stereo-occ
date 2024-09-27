#! /usr/bin/env python3
import open3d as o3d    
import open3d.core as o3c
import shutil
import os
import logging, coloredlogs
from pathlib import Path
from tqdm import tqdm
import random
import numpy as np  
from collections import defaultdict

# custom modules
from utils import io_utils

# TO-DO
# - priority based collapsing
# - removal of below-ground points
# - testing over 50 pointclouds
# - custom voxel downsampling

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger, force=True)

def get_random_segmented_pcd(src_foler: Path) -> Path: 
    '''
    Get a random ply file from the source folder
    '''    
    ply_files = [f for f in os.listdir(src_folder) if f.endswith('.ply')]
    if ply_files:
        random_ply_file = random.choice(ply_files)
        random_ply_path = os.path.join(src_folder, random_ply_file)
    
    return random_ply_path

def calculate_angles(normal_vector):
    x_axis = np.array([1, 0, 0])
    y_axis = np.array([0, 1, 0])
    z_axis = np.array([0, 0, 1])
    
    angle_x = np.arccos(np.dot(normal_vector, x_axis) / np.linalg.norm(normal_vector))
    angle_y = np.arccos(np.dot(normal_vector, y_axis) / np.linalg.norm(normal_vector))
    angle_z = np.arccos(np.dot(normal_vector, z_axis) / np.linalg.norm(normal_vector))
    
    return np.degrees(angle_x), np.degrees(angle_y), np.degrees(angle_z)

def get_class_pointcloud(pcd, class_label):
    '''
    Get the point cloud of a specific class
    '''
    mask = pcd.point["label"] == class_label
    pcd_labels = pcd.select_by_index(mask.nonzero()[0])
    return pcd_labels

def get_class_plane(pcd, class_label):
    '''
    Get the inliers / normal vector for the labelled pointcloud
    '''
    pcd_class = get_class_pointcloud(pcd, class_label)
    plane_model, inliers = pcd_class.segment_plane(distance_threshold=0.01,
                                                    ransac_n=3,
                                                    num_iterations=1000)
    [a, b, c, d] = plane_model.numpy()
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal) 
    return normal, inliers

def align_normal_to_y_axis(normal_):
    y_axis = np.array([0, 1, 0])
    v = np.cross(normal_, y_axis)
    s = np.linalg.norm(v)
    c = np.dot(normal_, y_axis)
    I = np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = I + vx + np.dot(vx, vx) * ((1 - c) / (s ** 2))
    return R

def compute_tilt_matrix(pcd):
    '''
    Compute the tilt of the point cloud
    '''
    normal, _ = get_class_plane(pcd, 2)
    R = align_normal_to_y_axis(normal)
    
    # check normal
    normal_ = np.dot(normal, R.T)
    angles_transformed = calculate_angles(normal_)
    logger.info(f"Ground plane makes {angles_transformed[0]} degrees with axes!")
    return R

def remove_points_by_labels(pcd, labels: np.array):
    pcd_ = pcd.clone()

    pcd_labels = pcd_.point["label"].numpy()
    pcd_positions = pcd_.point["positions"].numpy()
    pcd_colors = pcd_.point["colors"].numpy()
    
    mask = np.isin(pcd_labels, labels, invert=True)

    pcd_positions = pcd_positions[mask.flatten()]
    pcd_colors = pcd_colors[mask.flatten()]
    pcd_labels = pcd_labels[mask.flatten()]

    pcd_.point["positions"] = o3d.core.Tensor(pcd_positions, dtype=o3d.core.Dtype.Float32)
    pcd_.point["colors"] = o3d.core.Tensor(pcd_colors, dtype=o3d.core.Dtype.Float32)
    pcd_.point["label"] = o3d.core.Tensor(pcd_labels, dtype=o3d.core.Dtype.Int32)
    
    
    logger.warning(f"len(pcd_.point.positions): {len(pcd_.point.positions)}")
    logger.warning(f"len(pcd.point.positions): {len(pcd.point.positions)}")
    return pcd_


def extract_voxel_centers(voxel_grid):
    return np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])

def combine_voxel_grids(voxel_grid1, voxel_grid2):
    voxel_centers1 = extract_voxel_centers(voxel_grid1)
    voxel_centers2 = extract_voxel_centers(voxel_grid2)
    
    combined_voxel_centers = np.vstack((voxel_centers1, voxel_centers2))
    
    combined_pcd = o3d.geometry.PointCloud()
    combined_pcd.points = o3d.utility.Vector3dVector(combined_voxel_centers)
    
    combined_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        combined_pcd, 
        voxel_size=voxel_grid1.voxel_size
    )
    
    return combined_voxel_grid

def display_inlier_outlier(cloud : o3d.t.geometry.PointCloud, mask : o3c.Tensor):
    inlier_cloud = cloud.select_by_mask(mask)
    outlier_cloud = cloud.select_by_mask(mask, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud = outlier_cloud.paint_uniform_color([1.0, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    inlier_cloud = o3d.visualization.draw_geometries([inlier_cloud.to_legacy(), outlier_cloud.to_legacy()],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])
LABEL_COLOR_MAP = { 
    0: [0, 0, 0],        # black
    1: [246, 4, 228],    # purple
    2: [173, 94, 48],    # blue
    3: [68, 171, 117],   # brown
    4: [162, 122, 174],  # gray
    5: [121, 119, 148],  # pink
    6: [253, 75, 40],    # orange
    7: [170, 60, 100],   # dark pink
    8: [60, 100, 179]    # green
}

LABELS = {    
    "VINE_POLE": 5,  
    "VINE_CANOPY": 3,
    "VINE_STEM": 4,  
    "NAVIGABLE_SPACE": 2,  
    "OBSTACLE": 1
}

if __name__ == "__main__":
    
    src_folder = "../ply/segmented-1056_to_1198/"
    src_path = os.path.join(src_folder, "1.ply")
    pcd_input = o3d.t.io.read_point_cloud(src_path)
    
    # pcd correction
    R = compute_tilt_matrix(pcd_input)
    
    pcd_corrected = pcd_input.clone()
    pcd_corrected.rotate(R, center=(0, 0, 0))

    # GLOBAL POINTCLOUD FILTERING

    # removing unwanted labels => [vegetation, tractor-hood, void, sky]
    valid_labels = np.array(list(LABELS.values()))
    valid_mask = np.isin(pcd_corrected.point['label'].numpy(), valid_labels)
    
    pcd_filtered = pcd_corrected.select_by_mask(valid_mask.flatten())
    original_points = len(pcd_corrected.point['positions'])
    filtered_points = len(pcd_filtered.point['positions'])
    reduction_percentage = ((original_points - filtered_points) / original_points) * 100
    
    unique_labels = np.unique(pcd_filtered.point['label'].numpy())
    
    logger.info(f"=================================")    
    logger.info(f"Before filtering: {original_points}")
    logger.info(f"After filtering: {filtered_points}")
    logger.info(f"Reduction percentage: {reduction_percentage:.2f}%")
    logger.info(f"Unique labels in pcd_filtered: {unique_labels}")
    logger.info(f"=================================\n")
    
    # class-wise point cloud extraction
    pcd_canopy = get_class_pointcloud(pcd_filtered, LABELS["VINE_CANOPY"])
    pcd_pole = get_class_pointcloud(pcd_filtered, LABELS["VINE_POLE"])
    pcd_stem = get_class_pointcloud(pcd_filtered, LABELS["VINE_STEM"])
    pcd_obstacle = get_class_pointcloud(pcd_filtered, LABELS["OBSTACLE"])
    pcd_navigable = get_class_pointcloud(pcd_filtered, LABELS["NAVIGABLE_SPACE"])

    # num-points for each class
    total_points = len(pcd_filtered.point['positions'])
    canopy_points = len(pcd_canopy.point['positions'])
    pole_points = len(pcd_pole.point['positions'])
    stem_points = len(pcd_stem.point['positions'])
    obstacle_points = len(pcd_obstacle.point['positions'])
    navigable_points = len(pcd_navigable.point['positions'])

    # % points for each class
    canopy_percentage = (canopy_points / total_points) * 100
    pole_percentage = (pole_points / total_points) * 100
    stem_percentage = (stem_points / total_points) * 100
    obstacle_percentage = (obstacle_points / total_points) * 100
    navigable_percentage = (navigable_points / total_points) * 100

    logger.info(f"=================================")    
    logger.info(f"Total points: {total_points}")
    logger.info(f"Canopy points: {canopy_points} ({canopy_percentage:.2f}%)")
    logger.info(f"Pole points: {pole_points} ({pole_percentage:.2f}%)")
    logger.info(f"Stem points: {stem_points} ({stem_percentage:.2f}%)")
    logger.info(f"Obstacle points: {obstacle_points} ({obstacle_percentage:.2f}%)")
    logger.info(f"Navigable points: {navigable_points} ({navigable_percentage:.2f}%)")
    logger.info(f"=================================\n")

    # DOWNSAMPLING LABEL-WISE POINTCLOUD
    down_pcd = pcd_filtered.voxel_down_sample(voxel_size=0.1)
    down_canopy = pcd_canopy.voxel_down_sample(voxel_size=0.1)
    down_pole = pcd_pole.voxel_down_sample(voxel_size=0.01)
    down_stem = pcd_stem.voxel_down_sample(voxel_size=0.1)
    down_obstacle = pcd_obstacle.voxel_down_sample(voxel_size=0.1)
    down_navigable = pcd_navigable.voxel_down_sample(voxel_size=0.1)

    down_total_points = len(down_pcd.point['positions'])
    down_canopy_points = len(down_canopy.point['positions'])
    down_pole_points = len(down_pole.point['positions'])
    down_stem_points = len(down_stem.point['positions'])
    down_obstacle_points = len(down_obstacle.point['positions'])
    down_navigable_points = len(down_navigable.point['positions'])
    
    down_canopy_percentage = (down_canopy_points / down_total_points) * 100
    down_pole_percentage = (down_pole_points / down_total_points) * 100
    down_stem_percentage = (down_stem_points / down_total_points) * 100
    down_obstacle_percentage = (down_obstacle_points / down_total_points) * 100
    down_navigable_percentage = (down_navigable_points / down_total_points) * 100
    
    logger.info(f"=================================")    
    logger.info(f"Downsampled Total points: {down_total_points}")
    logger.info(f"Downsampled Canopy points: {down_canopy_points} => ({down_canopy_percentage:.2f}%)")
    logger.info(f"Downsampled Pole points: {down_pole_points} => ({down_pole_percentage:.2f}%)")
    logger.info(f"Downsampled Stem points: {down_stem_points} => ({down_stem_percentage:.2f}%)")
    logger.info(f"Downsampled Obstacle points: {down_obstacle_points} => ({down_obstacle_percentage:.2f}%)")
    logger.info(f"Downsampled Navigable points: {down_navigable_points} => ({down_navigable_percentage:.2f}%)")
    logger.info(f"=================================\n")
    
    # # Compute the number of points with the same (x,z) value in down_stem and down_pole
    # down_pcd_positions = down_pcd.point['positions'].numpy()
    # down_pole_positions = down_pole.point['positions'].numpy()

    # # Extract x and z coordinates
    # down_pcd_xz = down_pcd_positions[:, [0, 2]]
    # down_pole_xz = down_pole_positions[:, [0, 2]]

    # # Find unique (x,z) pairs in both point clouds
    # unique_down_pcd_xz = np.unique(down_pcd_xz, axis=0)
    # unique_down_pole_xz = np.unique(down_pole_xz, axis=0)

    # logger.info(f"Number of unique (x,z) pairs in down_pcd: {len(unique_down_pcd_xz)}")
    # logger.info(f"Number of unique (x,z) pairs in down_pole: {len(unique_down_pole_xz)}")

    # #  View the 2D arrays as 1D structured arrays
    # dtype = np.dtype((np.void, unique_down_pcd_xz.dtype.itemsize * unique_down_pcd_xz.shape[1]))
    # unique_down_pcd_xz_flat = unique_down_pcd_xz.view(dtype).flatten()
    # unique_down_pole_xz_flat = unique_down_pole_xz.view(dtype).flatten()

    # # Find common (x,z) pairs between the two point clouds
    # common_xz_flat = np.intersect1d(unique_down_pcd_xz_flat, unique_down_pole_xz_flat)

    # # Convert back to the original shape
    # common_xz = common_xz_flat.view(unique_down_pcd_xz.dtype).reshape(-1, unique_down_pcd_xz.shape[1])

    # logger.info(f"Number of common (x,z) pairs: {len(common_xz)}")

    # # Collapse all points for down_pole along the y-axis using tensor operations
    # down_pole_positions = down_pole.point['positions'].numpy()
    # collapsed_down_pole_positions = np.zeros_like(down_pole_positions)
    # collapsed_down_pole_positions[:, [0, 2]] = down_pole_positions[:, [0, 2]]
    # collapsed_down_pole = o3d.geometry.PointCloud()
    # collapsed_down_pole.points = o3d.utility.Vector3dVector(collapsed_down_pole_positions)

    # down_pole_tensor = down_pole.point['positions']
    # down_pole_tensor[1] = 0
    # collapsed_pole = o3d.t.geometry.PointCloud(down_pole_tensor)

    # print("Statistical oulier removal")
    # cl, ind = down_canopy.remove_statistical_outliers(nb_neighbors=20,
    #                                                 std_ratio=0.1)
    # display_inlier_outlier(down_canopy, ind)

    # cl, ind = down_canopy.remove_radius_outliers(nb_points=16, search_radius=0.05)
    # display_inlier_outlier(down_canopy, ind)

    # cl_in = down_pole.select_by_mask(ind)
    # cl_out = down_pole.select_by_mask(ind, invert=True)

    # down_pole.paint_uniform_color([0.0, 1.0, 0.0])

    # visualization wind
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # co-ordinate frame for vis window    
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)

    # adding point clouds to visualizer
    vis.add_geometry(pcd_filtered.to_legacy())
    # vis.add_geometry(collapsed_pole.to_legacy())
    # vis.add_geometry(down_pole.to_legacy())
    # vis.add_geometry(down_stem.to_legacy())
    # vis.add_geometry(cl_in.to_legacy())
    view_ctr = vis.get_view_control()
    view_ctr.set_front(np.array([0, 0, -1]))
    view_ctr.set_up(np.array([0, -1, 0]))
    # view_ctr.set_up(np.array([0, 1, 0]))
    
    vis.run()
    vis.destroy_window()

    
