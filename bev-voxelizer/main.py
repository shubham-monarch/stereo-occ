#! /usr/bin/env python3
import open3d as o3d    
import os
import shutil
import logging, coloredlogs
from pathlib import Path
from tqdm import tqdm
import random
import numpy as np  

# custom modules
from utils import io_utils

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


# def key_callback(vis):
#     logger.warning(f"===================")
#     logger.warning(f"KEYBACK TRIGGERED!")
#     logger.warning(f"===================")
#     print('key')

#     return False


def update_camera_view(vis, point_cloud):
    # Create coordinate frame
    
    # Set the camera view to be at the pointcloud origin
    ctr = vis.get_view_control()

    bbox = point_cloud.to_legacy().get_axis_aligned_bounding_box()
    bbox_min = bbox.get_min_bound()
    bbox_max = bbox.get_max_bound()
    dimensions = bbox_max - bbox_min

    center = bbox.get_center()
    ctr.set_lookat(center)
    
    # Set the camera position 
    ctr.set_front([0, 0, -1])  # Looking along negative z-axis
    ctr.set_up([0, -1, 0])     # Up direction is negative y-axis (Open3D convention)
    ctr.set_zoom(0.8)

    return vis
    


if __name__ == "__main__":
    
    src_folder = "ply/segmented-1056_to_1198/"
    random_pointcloud_path = get_random_segmented_pcd(src_folder)
    
    pcd = o3d.t.io.read_point_cloud(random_pointcloud_path)
    pcd_original = pcd.clone()
    pcd_corrected = pcd.clone()
    
    # Paint the entire point cloud yellow
    yellow_color = np.array([1.0, 1.0, 0.0])  # RGB values for yellow
    num_points = pcd_original.point.positions.shape[0]
    pcd_original.point.colors = o3d.core.Tensor(np.tile(yellow_color, (num_points, 1)))

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # add segmented-pointcloud to vis windows
    # vis.add_geometry(point_cloud.to_legacy())

    # add coordinate frame to vis window
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)

    # adjusting default camera view for better visualization
    vis = update_camera_view(vis, pcd)
    
    
    # pointcloud_legacy = point_cloud.to_legacy()
    
    num_points = pcd.point.positions.shape[0]
    logger.info(f"num_points: {num_points}")

    for i in range(min(1, num_points)):
        logger.info(f"Point {i + 1}:")
        x, y, z = pcd.point.positions[i].numpy()
        r, g, b = pcd.point.colors[i].numpy()
        label = pcd.point.label[i].item()
        logger.info(f"  x: {x:.6f}, y: {y:.6f}, z: {z:.6f}")
        logger.info(f"  r: {r:.6f}, g: {g:.6f}, b: {b:.6f}")
        logger.info(f"  label: {label}")
        logger.info("---")

    # Filter points with label value 2 (ground)
    ground_mask = pcd.point["label"] == 2
    pcd_ground = pcd.select_by_index(ground_mask.nonzero()[0])

    # Fit a plane to the ground points
    ground_plane_model, inliers = pcd_ground.segment_plane(distance_threshold=0.01,
                                                    ransac_n=3,
                                                    num_iterations=1000)
    [a, b, c, d] = ground_plane_model.numpy()

   
    # Calculate the normal vector of the ground plane
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)  # Normalize the vector

    # Calculate the rotation matrix to align the normal with the y-axis
    y_axis = np.array([0, -1, 0])  # Changed to negative y-axis
    rotation_axis = np.cross(normal, y_axis)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    
    cos_theta = np.dot(normal, y_axis)
    sin_theta = np.sqrt(1 - cos_theta**2)
    
    # Rodrigues' rotation formula
    K = np.array([
        [0, -rotation_axis[2], rotation_axis[1]],
        [rotation_axis[2], 0, -rotation_axis[0]],
        [-rotation_axis[1], rotation_axis[0], 0]
    ])
    rotation_matrix = np.eye(3) + sin_theta * K + (1 - cos_theta) * np.dot(K, K)

    # Apply the rotation to the entire point cloud
    pcd_corrected.point.positions = o3d.core.Tensor(np.dot(pcd_original.point.positions.numpy(), rotation_matrix.T))
    # Paint pcd_corrected to red
    # pcd_corrected.paint_uniform_color([1.0, 0.0, 0.0])  # Red color in RGB
    
    vis.add_geometry(pcd_corrected.to_legacy())   
    vis.add_geometry(pcd_original.to_legacy())

    # Save pcd_corrected to disk as .ply file
    output_path = "corrected_pointcloud.ply"
    o3d.io.write_point_cloud(output_path, pcd_corrected.to_legacy())
    logger.info(f"Corrected point cloud saved to {output_path}")

    # # Update the ground points
    # pcd_ground = pcd.select_by_index(ground_mask.nonzero()[0])

    # # Recalculate inliers and outliers with the rotated point cloud
    # _, inliers = pcd_ground.segment_plane(distance_threshold=0.01,
    #                                       ransac_n=3,
    #                                       num_iterations=1000)

    # inlier_cloud = pcd_ground.select_by_index(inliers)
    # inlier_cloud.paint_uniform_color([1.0, 1.0, 0.0])
    
    # # outlier_cloud = pcd_ground.select_by_index(inliers, invert=True)
    # # outlier_cloud.paint_uniform_color([0.0, 0.0, 0.0])

    # vis.add_geometry(inlier_cloud.to_legacy())
    # # vis.add_geometry(pcd_ground.to_legacy())

    # # Update the camera view for the rotated point cloud
    # # vis = update_camera_view(vis, pcd)

    vis.run()
    vis.destroy_window()

    
