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
# - different downsampling ratios for different classes
# - removal of below-ground points
# - testing over 50 pointclouds
# - custom voxel downsampling
# - farthest point sampling
# - checkout bev-former voxelizer
# - remove below-ground points
# - statistical outlier removal
# - refactor compute_tilt_matrix()
# - make project_to_ground_plane more robust
# - remove non-inliers ground points

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

def get_class_plane(pcd: o3d.t.geometry.PointCloud, class_label: int):
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

def get_plane_model(pcd: o3d.t.geometry.PointCloud, class_label: int):
    '''
    returns [a,b,c,d]
    '''
    pcd_class = get_class_pointcloud(pcd, class_label)
    plane_model, inliers = pcd_class.segment_plane(distance_threshold=0.01,
                                                    ransac_n=3,
                                                    num_iterations=1000)
    
    return plane_model.numpy()

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


# def project_points_to_plane_model(plane_model, pcd):
#     '''
#     Projects all the points of the point cloud along the plane defined by the plane model (a, b, c, d)
#     '''
#     logger.warning(f"Inside project_points_to_plane_model()")
#     logger.warning(f"plane_model: {plane_model}")

#     a, b, c, d = plane_model
#     points = np.asarray(pcd.point['positions'])
    
#     # Normalize the plane normal vector
#     normal = np.array([a, b, c])
#     normal = normal / np.linalg.norm(normal)
    
#     # Calculate the distance from each point to the plane
#     distances = (np.dot(points, normal) + d) / np.linalg.norm(normal)
    
#     # Project the points onto the plane
#     projected_points = points - np.outer(distances, normal)
    
#     # Update the point cloud with the projected points
#     pcd.point['positions'] = o3d.core.Tensor(projected_points, dtype=o3c.float32)
    
#     return pcd



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

LABEL_ID_TO_PRIORITY = {
    1: 1,
    5: 2,
    4: 3,
    3: 4,
    2: 5,
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

    # angle between normal and y-axis should be close to 0 / 180 degrees
    if not np.isclose(angles[1], 0, atol=1) and np.isclose(angles[1], 180, atol=1):
        logger.error(f"=================================")    
        logger.error(f"Error: angles_transformed[1] is {angles[1]}, but it should be close to 0 degrees. Please check the tilt correction!")
        logger.error(f"=================================\n")
        exit(1)


    # logger.warning(f"type(pcd_input.point['positions']): {type(pcd_input.point['positions'])}")

    pcd_corrected = pcd_input.clone()
    pcd_corrected.rotate(R, center=(0, 0, 0))

    # logger.warning(f"type(pcd_corrected.point['positions']): {type(pcd_corrected.point['positions'])}")

    # filtering unwanted labels => [vegetation, tractor-hood, void, sky]
    valid_labels = np.array([label["id"] for label in LABELS.values()])
    valid_mask = np.isin(pcd_corrected.point['label'].numpy(), valid_labels)
    
    pcd_filtered = pcd_corrected.select_by_mask(valid_mask.flatten())
    original_points = len(pcd_corrected.point['positions'])
    filtered_points = len(pcd_filtered.point['positions'])
    reduction_percentage = ((original_points - filtered_points) / original_points) * 100    
    
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
    down_pcd = pcd_filtered.voxel_down_sample(voxel_size=0.01)
    down_canopy = pcd_canopy.voxel_down_sample(voxel_size=0.01)
    down_pole = pcd_pole.voxel_down_sample(voxel_size=0.01)
    down_navigable = pcd_navigable.voxel_down_sample(voxel_size=0.01)
    # down_stem = pcd_stem.voxel_down_sample(voxel_size=0.01)
    # down_obstacle = pcd_obstacle.voxel_down_sample(voxel_size=0.01)
    down_obstacle = pcd_obstacle.clone()
    down_stem = pcd_stem.clone()
    down_pole = pcd_pole.clone()

    down_total_points = len(down_pcd.point['positions'].numpy())
    down_canopy_points = len(down_canopy.point['positions'])
    down_pole_points = len(down_pole.point['positions'])
    down_stem_points = len(down_stem.point['positions'])
    down_obstacle_points = len(down_obstacle.point['positions'])
    down_navigable_points = len(down_navigable.point['positions'])

    total_reduction_pct = (total_points - down_total_points) / total_points * 100
    canopy_reduction_pct = (canopy_points - down_canopy_points) / canopy_points * 100
    pole_reduction_pct = (pole_points - down_pole_points) / pole_points * 100
    stem_reduction_pct = (stem_points - down_stem_points) / stem_points * 100
    obstacle_reduction_pct = (obstacle_points - down_obstacle_points) / obstacle_points * 100
    navigable_reduction_pct = (navigable_points - down_navigable_points) / navigable_points * 100
    
    logger.info(f"=================================")    
    logger.info(f"[AFTER DOWNSAMPLING]")
    logger.info(f"Total points: {down_total_points} [-{total_reduction_pct:.2f}%]")
    logger.info(f"Canopy points: {down_canopy_points} [-{canopy_reduction_pct:.2f}%]")
    logger.info(f"Pole points: {down_pole_points} [-{pole_reduction_pct:.2f}%]")
    logger.info(f"Stem points: {down_stem_points} [-{stem_reduction_pct:.2f}%]")
    logger.info(f"Obstacle points: {down_obstacle_points} [-{obstacle_reduction_pct:.2f}%]")
    logger.info(f"Navigable points: {down_navigable_points} [-{navigable_reduction_pct:.2f}%]")
    logger.info(f"=================================\n")
    
    # radius-based outlier removal
    # rad_filt_canopy, outliers_canopy = filter_radius_outliers(down_canopy, nb_points=1, search_radius=0.1)
    rad_filt_pole, _ = filter_radius_outliers(down_pole, nb_points=16, search_radius=0.05)
    rad_filt_stem, _ = filter_radius_outliers(down_stem, nb_points=16, search_radius=0.05)
    # rad_filt_obstacle, _ = filter_radius_outliers(down_obstacle, nb_points=16, search_radius=0.05)
    # rad_filt_navigable, _ = filter_radius_outliers(down_navigable, nb_points=16, search_radius=0.05)

    # rad_filt_canopy_points = len(rad_filt_canopy.point['positions'].numpy())
    rad_filt_pole_points = len(rad_filt_pole.point['positions'].numpy())
    rad_filt_stem_points = len(rad_filt_stem.point['positions'].numpy())
    # rad_filt_obstacle_points = len(rad_filt_obstacle.point['positions'].numpy())
    # rad_filt_navigable_points = len(rad_filt_navigable.point['positions'].numpy())

    # canopy_reduction_pct = (down_canopy_points - rad_filt_canopy_points) / down_canopy_points * 100
    pole_reduction_pct = (down_pole_points - rad_filt_pole_points) / down_pole_points * 100
    stem_reduction_pct = (down_stem_points - rad_filt_stem_points) / down_stem_points * 100
    # obstacle_reduction_pct = (down_obstacle_points - rad_filt_obstacle_points) / down_obstacle_points * 100
    # navigable_reduction_pct = (down_navigable_points - rad_filt_navigable_points) / down_navigable_points * 100
    
    logger.info(f"=================================")    
    logger.info(f"[AFTER RADIUS-BASED OUTLIER REMOVAL]")
    # logger.info(f"Canopy points: {rad_filt_canopy_points} [-{canopy_reduction_pct:.2f}%]")
    logger.info(f"Pole points: {rad_filt_pole_points} [-{pole_reduction_pct:.2f}%]")
    logger.info(f"Stem points: {rad_filt_stem_points} [-{stem_reduction_pct:.2f}%]")
    # logger.info(f"Obstacle points: {rad_filt_obstacle_points} [-{obstacle_reduction_pct:.2f}%]")
    # logger.info(f"Navigable points: {rad_filt_navigable_points} [-{navigable_reduction_pct:.2f}%]")
    logger.info(f"=================================\n")

    # logger.info(f"Number of points in rad_filt_canopy: {len(rad_filt_canopy.point['positions'])}")
    # logger.info(f"Number of points in outliers_canopy: {len(outliers_canopy.point['positions'])}")

    
    
    # projecting points to the navigable plane
    normal, inliers = get_class_plane(pcd_navigable, LABELS["NAVIGABLE_SPACE"]["id"])
    normal = normal / np.linalg.norm(normal)

    inliers_navigable = pcd_navigable.select_by_index(inliers)

    import matplotlib.pyplot as plt

    # # Extract x, y, z values from inliers_navigable
    # inliers_positions = inliers_navigable.point['positions'].numpy()
    # x_values = inliers_positions[:, 0]
    # y_values = inliers_positions[:, 1]
    # z_values = inliers_positions[:, 2]

    # # Plot histograms for x, y, z values
    # fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # axs[0].hist(x_values, bins=50, color='r', alpha=0.7)
    # axs[0].set_title('Histogram of X values')
    # axs[0].set_xlabel('X')
    # axs[0].set_ylabel('Frequency')

    # axs[1].hist(y_values, bins=50, color='g', alpha=0.7)
    # axs[1].set_title('Histogram of Y values')
    # axs[1].set_xlabel('Y')
    # axs[1].set_ylabel('Frequency')

    # axs[2].hist(z_values, bins=50, color='b', alpha=0.7)
    # axs[2].set_title('Histogram of Z values')
    # axs[2].set_xlabel('Z')
    # axs[2].set_ylabel('Frequency')

    # plt.tight_layout()
    # plt.show()

    
    # compute angle with y-axis
    angle_y = axis_angles(normal)[1]
    logger.info(f"Angle between normal and y-axis: {angle_y:.2f} degrees")

    # align normal with +y-axis if angle with y-axis is negative
    if angle_y < 0:
        normal = -normal

    # logger.info(f"rad_filt_canopy.point['positions'].shape: {rad_filt_canopy.point['positions'].shape}")
    # logger.info(f"normal.shape: {normal.shape}")
    
    # normal_tensor = o3d.core.Tensor(normal, dtype=o3c.float32)

    # navigable_plane_model = get_plane_model(pcd_navigable, LABELS["NAVIGABLE_SPACE"]["id"])
    # logger.warning(f"navigable_plane_model: {navigable_plane_model}")

    # label-wise BEV generation
    projected_canopy = down_canopy.clone()
    projected_canopy.point['positions'][:, 1] = 2.0

    projected_pole = rad_filt_pole.clone()
    projected_pole.point['positions'][:, 1] = 2.0

    projected_stem = rad_filt_stem.clone()
    projected_stem.point['positions'][:, 1] = 2.0

    projected_obstacle = down_obstacle.clone()
    projected_obstacle.point['positions'][:, 1] = 2.0


    bev_collection = [inliers_navigable, projected_canopy, projected_pole, projected_stem, projected_obstacle]
    bev_collection = [inliers_navigable, projected_pole, projected_stem, projected_obstacle]
    
    # Vertically stack the point positions of the bev_collection pointclouds
    position_tensors = [pcd.point['positions'].numpy() for pcd in bev_collection]
    stacked_positions = o3c.Tensor(np.vstack(position_tensors), dtype=o3c.Dtype.Float32)
    
    # # Vertically stack the point labels of the bev_collection pointclouds
    label_tensors = [pcd.point['label'].numpy() for pcd in bev_collection]
    stacked_labels = o3c.Tensor(np.vstack(label_tensors), dtype=o3c.Dtype.Int32)

    # # Vertically stack the point colors of the bev_collection pointclouds
    color_tensors = [pcd.point['colors'].numpy() for pcd in bev_collection]
    stacked_colors = o3c.Tensor(np.vstack(color_tensors), dtype=o3c.Dtype.UInt8)
    
    # Create a color tensor of red color
    # num_points = stacked_positions.shape[0]
    # red_color = np.array([255, 0, 0], dtype=np.uint8)
    # color_tensor = o3c.Tensor(np.tile(red_color, (num_points, 1)), dtype=o3c.Dtype.UInt8)

    # Create a unified point cloud
    map_to_tensors = {}
    map_to_tensors['positions'] = stacked_positions
    map_to_tensors['label'] = stacked_labels
    map_to_tensors['colors'] = stacked_colors
    # map_to_tensors['colors'] = color_tensor
    combined_pcd = o3d.t.geometry.PointCloud(map_to_tensors)
    
    logger.info(f"type(inliers_navigable): {type(inliers_navigable)}")
    logger.info(f"type(projected_pole): {type(projected_pole)}")
    logger.info(f"type(combined_pcd): {type(combined_pcd)}")

    # logger.info(f"=================================")    
    # logger.info(f"combined_pcd.point['positions'].shape: {combined_pcd.point['positions'].shape} type: {type(combined_pcd.point['positions'])}")
    # logger.info(f"combined_pcd.point['label'].shape: {combined_pcd.point['label'].shape} type: {type(combined_pcd.point['label'])}")
    # logger.info(f"combined_pcd.point['colors'].shape: {combined_pcd.point['colors'].shape} type: {type(combined_pcd.point['colors'])}")
    # logger.info(f"=================================\n")
    
    # for pcd in bev_collection:
    #     position_tensor = pcd.point['positions'].numpy()
    #     color_tensor = pcd.point['colors'].numpy()
    #     label_tensor = pcd.point['label'].numpy()
        
    #     logger.info(f"position_tensor.shape: {position_tensor.shape}")
    #     logger.info(f"color_tensor.shape: {color_tensor.shape}")
    #     logger.info(f"label_tensor.shape: {label_tensor.shape}")
    

    # # Vertically stack the point positions of the bev_collection pointclouds
    # position_tensors = [pcd.point['positions'].numpy() for pcd in bev_collection]
    # stacked_positions = o3c.Tensor(np.vstack(position_tensors), dtype=o3c.float32)
    
    # # Vertically stack the point labels of the bev_collection pointclouds
    # label_tensors = [pcd.point['label'].numpy() for pcd in bev_collection]
    # stacked_labels = o3c.Tensor(np.vstack(label_tensors), dtype=o3c.int32)


    # # Replace the 'label' field with LABEL_ID_TO_PRIORITY[label]
    # priority_labels = [LABEL_ID_TO_PRIORITY[label] for label in stacked_labels.numpy().flatten()]
    # stacked_priority_labels = o3c.Tensor(np.array(priority_labels).reshape(stacked_labels.shape), dtype=o3c.int32)
    
    # logger.info(f"stacked_priority_labels[10,10,10]: {stacked_priority_labels[10,10,10]}")
    # logger.info(f"stacked_priority_labels[10,10,100]: {stacked_priority_labels[10,10,10]}")
    

    # # Vertically stack the point colors of the bev_collection pointclouds
    # color_tensors = [pcd.point['colors'].numpy() for pcd in bev_collection]
    # stacked_colors = o3c.Tensor(np.vstack(color_tensors), dtype=o3c.float32)
    
    # logger.warning(f"stacked_positions.shape: {stacked_positions.shape}")
    # logger.warning(f"stacked_labels.shape: {stacked_labels.shape}")
    # logger.warning(f"stacked_colors.shape: {stacked_colors.shape}")

    
    # logger.warning(f"stacked_positions.shape: {stacked_positions.shape}")

    # exit(1)
    # bev_pcd = o3d.t.geometry.PointCloud()
    
    # map_to_tensors = {}
    # position_tensors = []
    # color_tensors = []
    # label_tensors = []



        

        
    # logger.warning(f"=================================")    
    # logger.warning(f"position_tensors.shape: {position_tensors.shape}")
    # logger.warning(f"color_tensors.shape: {color_tensors.shape}")
    # logger.warning(f"label_tensors.shape: {label_tensors.shape}")   
    # logger.warning(f"=================================\n")

    # exit(1)

    # Extract x, y, z values from projected_canopy
    # projected_positions = projected_canopy.point['positions'].numpy()
    # x_values = projected_positions[:, 0]
    # y_values = projected_positions[:, 1]
    # z_values = projected_positions[:, 2]

    # # Plot histograms for x, y, z values
    # fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # axs[0].hist(x_values, bins=50, color='r', alpha=0.7)
    # axs[0].set_title('Histogram of X values')
    # axs[0].set_xlabel('X')
    # axs[0].set_ylabel('Frequency')

    # axs[1].hist(y_values, bins=50, color='g', alpha=0.7)
    # axs[1].set_title('Histogram of Y values')
    # axs[1].set_xlabel('Y')
    # axs[1].set_ylabel('Frequency')

    # axs[2].hist(z_values, bins=50, color='b', alpha=0.7)
    # axs[2].set_title('Histogram of Z values')
    # axs[2].set_xlabel('Z')
    # axs[2].set_ylabel('Frequency')

    # plt.tight_layout()
    # plt.show()

    # logger.info(f"projected_canopy.point['positions'].shape: {projected_canopy.point['positions'].shape}")

    # rad_filt_canopy.paint_uniform_color([1.0, 0.0, 0.0])

    # # bev generation
    # bev_pole = collapse_along_y_axis(rad_filt_pole)
    
    # visualization wind
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # co-ordinate frame for vis window    
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)

    # adding point clouds to visualizer
    vis.add_geometry(combined_pcd.to_legacy())
    # vis.add_geometry(projected_canopy.to_legacy())
    # vis.add_geometry(projected_pole.to_legacy())
    # vis.add_geometry(projected_stem.to_legacy())
    # vis.add_geometry(projected_obstacle.to_legacy())
    # vis.add_geometry(inliers_navigable.to_legacy())
    # vis.add_geometry(combined_pcd.to_legacy())
    # view control
    view_ctr = vis.get_view_control()
    view_ctr.set_front(np.array([0, 0, -1]))
    view_ctr.set_up(np.array([0, -1, 0]))
    # view_ctr.set_up(np.array([0, 1, 0]))
    
    vis.run()
    vis.destroy_window()

    
