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
# - crop pointcloud to bounding box
# - hidden point removal 
# - removal of below-ground points
# - testing over 50 pointclouds
# - custom voxel downsampling
# - farthest point sampling
# - checkout bev-former voxelizer
# - statistical outlier removal

# LOGGING SETUP
logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(lineno)d')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
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

def axis_angles(vec):
    '''
    Calculate the angles between input vector and the coordinate axes
    '''
    x_axis = np.array([1, 0, 0])
    y_axis = np.array([0, 1, 0])
    z_axis = np.array([0, 0, 1])
    
    angle_x = np.arccos(np.dot(vec, x_axis) / np.linalg.norm(vec))
    angle_y = np.arccos(np.dot(vec, y_axis) / np.linalg.norm(vec))
    angle_z = np.arccos(np.dot(vec, z_axis) / np.linalg.norm(vec))
    
    return np.degrees(angle_x), np.degrees(angle_y), np.degrees(angle_z)

def get_class_pointcloud(pcd, class_label):
    '''
    Returns class-specific point cloud
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
    '''
    Rotation matrix to align the normal vector to the y-axis
    '''
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
    Compute navigation-space tilt w.r.t y-axis
    '''
    normal, _ = get_class_plane(pcd, LABELS["NAVIGABLE_SPACE"]["id"])
    R = align_normal_to_y_axis(normal)

    return R

def filter_radius_outliers(pcd, nb_points, search_radius):
    '''
    Filter radius-based outliers from the point cloud
    '''
    _, ind = pcd.remove_radius_outliers(nb_points=nb_points, search_radius=search_radius)
    inliers = pcd.select_by_mask(ind)
    outliers = pcd.select_by_mask(ind, invert=True)
    return inliers, outliers

def collapse_along_y_axis(pcd):
    pcd.point['positions'][:, 1] = 0
    return pcd


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
    "OBSTACLE": {"id": 1, "priority": 1},
    "VINE_POLE": {"id": 5, "priority": 2},  
    "VINE_CANOPY": {"id": 3, "priority": 3},
    "VINE_STEM": {"id": 4, "priority": 4},  
    "NAVIGABLE_SPACE": {"id": 2, "priority": 5},  
}

if __name__ == "__main__":
    
    src_folder = "../ply/segmented-1056_to_1198/"
    src_path = os.path.join(src_folder, "1.ply")
    pcd_input = o3d.t.io.read_point_cloud(src_path)

    # logger.warning(f"type(pcd_input.point['positions']): {type(pcd_input.point['positions'])}") 
    
    # pcd correction
    R = compute_tilt_matrix(pcd_input)
    
    # sanity check
    # normal, _ = get_class_plane(pcd_input, 2)
    normal, _ = get_class_plane(pcd_input, LABELS["NAVIGABLE_SPACE"]["id"])
    normal_ = np.dot(normal, R.T)
    angles = axis_angles(normal_)
    logger.info(f"=================================")    
    logger.info(f"axis_angles: {angles}")
    logger.info(f"Ground plane makes {angles} degrees with y-axis!")
    logger.info(f"=================================\n")

    # angle between normal and y-axis should be close to 0 degrees
    if not np.isclose(angles[1], 0, atol=1):
        logger.error(f"=================================")    
        logger.error(f"Error: angles_transformed[1] is {angles[1]}, but it should be close to 0 degrees. Please check the tilt correction!")
        logger.error(f"=================================\n")
        exit(1)


    # logger.warning(f"type(pcd_input.point['positions']): {type(pcd_input.point['positions'])}")

    pcd_corrected = pcd_input.clone()
    pcd_corrected.rotate(R, center=(0, 0, 0))

    # logger.warning(f"type(pcd_corrected.point['positions']): {type(pcd_corrected.point['positions'])}")

    # FILTERING UNWANTED LABELS => [VEGETATION, TRACTOR-HOOD, VOID, SKY]
    valid_labels = np.array([label["id"] for label in LABELS.values()])
    valid_mask = np.isin(pcd_corrected.point['label'].numpy(), valid_labels)
    
    pcd_filtered = pcd_corrected.select_by_mask(valid_mask.flatten())
    original_points = len(pcd_corrected.point['positions'])
    filtered_points = len(pcd_filtered.point['positions'])
    reduction_percentage = ((original_points - filtered_points) / original_points) * 100

    # logger.warning(f"type(pcd_filtered.point['positions']: {type(pcd_filtered.point['positions'])}")
    
    unique_labels = np.unique(pcd_filtered.point['label'].numpy())
    
    logger.info(f"=================================")    
    logger.info(f"Before filtering: {original_points}")
    logger.info(f"After filtering: {filtered_points}")
    logger.info(f"Reduction %: {reduction_percentage:.2f}%")
    logger.info(f"Unique labels in pcd_filtered: {unique_labels}")
    logger.info(f"=================================\n")
    
    # class-wise point cloud extraction
    pcd_canopy = get_class_pointcloud(pcd_filtered, LABELS["VINE_CANOPY"]["id"])
    pcd_pole = get_class_pointcloud(pcd_filtered, LABELS["VINE_POLE"]["id"])
    pcd_stem = get_class_pointcloud(pcd_filtered, LABELS["VINE_STEM"]["id"])
    pcd_obstacle = get_class_pointcloud(pcd_filtered, LABELS["OBSTACLE"]["id"])
    pcd_navigable = get_class_pointcloud(pcd_filtered, LABELS["NAVIGABLE_SPACE"]["id"])
    
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
    logger.info(f"Canopy points: {canopy_points} [{canopy_percentage:.2f}%]")
    logger.info(f"Pole points: {pole_points} [{pole_percentage:.2f}%]")
    logger.info(f"Stem points: {stem_points} [{stem_percentage:.2f}%]")
    logger.info(f"Obstacle points: {obstacle_points} [{obstacle_percentage:.2f}%]")
    logger.info(f"Navigable points: {navigable_points} [{navigable_percentage:.2f}%]")
    logger.info(f"=================================\n")

    # downsampling label-wise pointcloud
    down_pcd = pcd_filtered.voxel_down_sample(voxel_size=0.1)
    down_canopy = pcd_canopy.voxel_down_sample(voxel_size=0.1)
    down_pole = pcd_pole.voxel_down_sample(voxel_size=0.01)
    down_stem = pcd_stem.voxel_down_sample(voxel_size=0.1)
    down_obstacle = pcd_obstacle.voxel_down_sample(voxel_size=0.1)
    down_navigable = pcd_navigable.voxel_down_sample(voxel_size=0.1)

    # logger.warning(f"type(down_pcd.point['positions']): {type(down_pcd.point['positions'])}")

    down_total_points = len(down_pcd.point['positions'].numpy())
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
    logger.info(f"[AFTER DOWNSAMPLING]")
    logger.info(f"Total points: {down_total_points} [-{100 - down_total_points / total_points * 100:.2f}%]")
    logger.info(f"Canopy points: {down_canopy_points} [-{100 - down_canopy_points / canopy_points * 100:.2f}%]")
    logger.info(f"Pole points: {down_pole_points} [-{100 - down_pole_points / pole_points * 100:.2f}%]")
    logger.info(f"Stem points: {down_stem_points} [-{100 - down_stem_points / stem_points * 100:.2f}%]")
    logger.info(f"Obstacle points: {down_obstacle_points} [-{100 - down_obstacle_points / obstacle_points * 100:.2f}%]")
    logger.info(f"Navigable points: {down_navigable_points} [-{100 - down_navigable_points / navigable_points * 100:.2f}%]")
    logger.info(f"=================================\n")
    
    # radius-based outlier removal
    filtered_canopy, _ = filter_radius_outliers(down_canopy, nb_points=16, search_radius=0.05)
    filtered_pole, outliers_pole = filter_radius_outliers(down_pole, nb_points=16, search_radius=0.05)
    filtered_stem, _ = filter_radius_outliers(down_stem, nb_points=16, search_radius=0.05)
    filtered_obstacle, _ = filter_radius_outliers(down_obstacle, nb_points=16, search_radius=0.05)
    filtered_navigable, _ = filter_radius_outliers(down_navigable, nb_points=16, search_radius=0.05)

    
    logger.info(f"=================================")    
    logger.info(f"[AFTER RADIUS-BASED OUTLIER REMOVAL]")
    logger.info(f"Canopy points: {len(filtered_canopy.point['positions'])} [-{100 - (down_canopy_points - len(filtered_canopy.point['positions'])) / down_canopy_points * 100:.2f}%]")
    logger.info(f"Pole points: {len(filtered_pole.point['positions'])} [-{100 - (down_pole_points - len(filtered_pole.point['positions'])) / down_pole_points * 100:.2f}%]")
    logger.info(f"Stem points: {len(filtered_stem.point['positions'])} [-{100 - (down_stem_points - len(filtered_stem.point['positions'])) / down_stem_points * 100:.2f}%]")
    logger.info(f"Obstacle points: {len(filtered_obstacle.point['positions'])} [-{100 - (down_obstacle_points - len(filtered_obstacle.point['positions'])) / down_obstacle_points * 100:.2f}%]")
    logger.info(f"Navigable points: {len(filtered_navigable.point['positions'])} [-{100 - (down_navigable_points - len(filtered_navigable.point['positions'])) / down_navigable_points * 100:.2f}%]")
    logger.info(f"=================================\n")

    
    # bev_pole = collapse_along_y_axis(down_pole)
    # bev_stem = collapse_along_y_axis(down_stem)
    # bev_obstacle = collapse_along_y_axis(down_obstacle)
    # bev_navigable = collapse_along_y_axis(down_navigable)

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
    # vis.add_geometry(pcd_filtered.to_legacy())
    # vis.add_geometry(filtered_pole.to_legacy())
    # vis.add_geometry(outliers_pole.to_legacy())
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

    
