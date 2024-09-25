#! /usr/bin/env python3
import open3d as o3d    
import os
import shutil
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

def compute_pcd_tilt(pcd):
    '''
    Compute the tilt of the point cloud
    '''
    normal, _ = get_class_plane(pcd, 2)
    R = align_normal_to_y_axis(normal)
    
    # check normal
    normal_ = np.dot(normal, R.T)
    angles_transformed = calculate_angles(normal_)
    logger.info(f"Angles of normal_ with x, y, z axes: {angles_transformed}")

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

LABELS = {    
    "VINE_POLE": 5,  
    "VINE_CANOPY": 3,
    "VINE_STEM": 4,  
    "NAVIGABLE_SPACE": 2,  
}

if __name__ == "__main__":
    
    src_folder = "ply/segmented-1056_to_1198/"
    # random_pointcloud_path = get_random_segmented_pcd(src_folder)
    src_path = os.path.join(src_folder, "1.ply")
    pcd = o3d.t.io.read_point_cloud(src_path)
    
    
    # tilt correction
    R = compute_pcd_tilt(pcd)
    pcd_ground = get_class_pointcloud(pcd, 2)
    
    # pcd_ground correction
    pcd_ground_ = pcd_ground.clone()
    pcd_ground_.rotate(R, center=(0, 0, 0))
    
    # paint black
    pcd_ground_.paint_uniform_color([0.0, 0.0, 0.0])  # RGB values for black
    
    # label priority order
  
    
    # pcd correction
    pcd_ = pcd.clone()
    pcd_.rotate(R, center=(0, 0, 0))

    PCD = pcd_.clone()
    pcd_canopy = get_class_pointcloud(PCD, LABELS["VINE_CANOPY"])
    pcd_pole = get_class_pointcloud(PCD, LABELS["VINE_POLE"])
    pcd_stem = get_class_pointcloud(PCD, LABELS["VINE_STEM"])

    logger.info(f"type(pcd_canopy.point): {type(pcd_canopy.point)}")
    
    # updating primary key
    logger.info(f"primary_key: {pcd_canopy.point.primary_key}")
    pcd_canopy.point.primary_key = "colors"
    logger.info(f"primary_key: {pcd_canopy.point.primary_key}")
    
    logger.info(f"pcd_canopy.point['positions'].shape: {pcd_canopy.point['positions'].shape} "
                f"type: {type(pcd_canopy.point['positions'])} "
                f"type: {pcd_canopy.point['positions'].dtype}")
    logger.info(f"pcd_canopy.point['colors'].shape: {pcd_canopy.point['colors'].shape} "
                f"type: {pcd_canopy.point['colors'].dtype} " 
                f"type: {pcd_canopy.point['colors'].dtype}") 
    logger.info(f"pcd_canopy.point['label'].shape: {pcd_canopy.point['label'].shape} "
                f"type: {pcd_canopy.point['label'].dtype} "
                f"type: {pcd_canopy.point['label'].dtype}")


    # DS_VINE_CANOPY = PCD_VINE_CANOPY.voxel_down_sample(voxel_size=0.1)
    # DS_VINE_POLE = PCD_VINE_POLE.voxel_down_sample(voxel_size=0.1)
    # DOWNSAMPLED_VINE_STEM = PCD_VINE_STEM.voxel_down_sample(voxel_size=0.1)

    # logger.info(f"num_points(DOWNSAMPLED_VINE_CANOPY): {DS_VINE_CANOPY.point.positions.shape}")

    # logger.info(f"type(DS_VINE_CANOPY): {type(DS_VINE_CANOPY)}")
    # logger.info(f"type(DS_VINE_CANOPY.points.positions): {type(DS_VINE_CANOPY.point.positions)}")
    # # visualize

    # new_tensor = o3d.core.append(DS_VINE_CANOPY.point.positions, 
    #                              o3d.core.Tensor(np.random.rand(3), dtype=o3d.core.Dtype.Float32))


    # VOXELS_CANOPY = o3d.geometry.VoxelGrid.create_from_point_cloud(
    #     PCD_VINE_CANOPY.to_legacy(), 
    #     voxel_size=0.1
    # )

    # VOXELS_POLE = o3d.geometry.VoxelGrid.create_from_point_cloud(
    #     PCD_VINE_POLE.to_legacy(), 
    #     voxel_size=0.1
    # )

    # VOXELS_STEM = o3d.geometry.VoxelGrid.create_from_point_cloud(
    #     PCD_VINE_STEM.to_legacy(), 
    #     voxel_size=0.1
    # )

    # from open3d.ml.torch.layers import VoxelPooling

    # # Merge VOXELS_CANOPY and VOXELS_POLE using VoxelPooling
    # merged_voxels = VoxelPooling(voxel_size=0.1, point_cloud_range=[-10, -10, -10, 10, 10, 10], max_num_points=5)
    # merged_voxels.add_voxels(VOXELS_CANOPY)
    # merged_voxels.add_voxels(VOXELS_POLE)
    # merged_voxels.pool()

    # VOXELS_POLE_SHIFTED = o3d.geometry.VoxelGrid(VOXELS_POLE)
    # VOXELS_POLE_SHIFTED.origin = VOXELS_POLE.origin + np.array([0, 10, 0])


    # # VOXELS_COMBINED = combine_voxel_grids(VOXELS_CANOPY, VOXELS_POLE)
    # T_origin = VOXELS_POLE.origin - VOXELS_CANOPY.origin
    
    # for voxel in tqdm(VOXELS_POLE.get_voxels(), desc="Processing voxels"):
    #     voxel_idx_pole = voxel.grid_index
    #     point_pole = VOXELS_POLE.get_voxel_center_coordinate(voxel_idx_pole)
    #     point_pole = point_pole - VOXELS_CANOPY.origin
        
    #     # point_canopy = point_pole - T_origin
    #     voxel_idx_canopy = VOXELS_CANOPY.get_voxel(point_pole)

    #     voxel_canopy = o3d.geometry.Voxel(grid_index=voxel_idx_canopy)
    #     VOXELS_CANOPY.remove_voxel(voxel_idx_canopy)
    #     VOXELS_CANOPY.add_voxel(voxel_canopy)

        

    # visualization wind
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # co-ordinate frame for vis window    
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)

    # vis.add_geometry(VOXELS_CANOPY)
    # vis.add_geometry(VOXELS_POLE_SHIFTED)
    # vis.add_geometry(VOXELS_POLE)
    # # adjust camera view
    view_ctr = vis.get_view_control()
    view_ctr.set_front(np.array([0, 0, -1]))
    view_ctr.set_up(np.array([0, -1, 0]))
    # view_ctr.set_up(np.array([0, 1, 0]))
    
    vis.run()
    vis.destroy_window()

    
