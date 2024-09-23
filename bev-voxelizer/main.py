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


# def update_camera_view(vis, point_cloud):
#     # Create coordinate frame
    
#     # Set the camera view to be at the pointcloud origin
#     ctr = vis.get_view_control()

#     bbox = point_cloud.to_legacy().get_axis_aligned_bounding_box()
#     bbox_min = bbox.get_min_bound()
#     bbox_max = bbox.get_max_bound()
#     dimensions = bbox_max - bbox_min

#     center = bbox.get_center()
#     # ctr.set_lookat(center)
    
#     # Set the camera position 
#     # ctr.set_front([0, 0, -1])  # Looking along negative z-axis
#     # ctr.set_up([0, 1, 0])     # Up direction is negative y-axis (Open3D convention)

#     # ctr.set_zoom(0.8)

#     return vis


def calculate_angles(normal_vector):
        x_axis = np.array([1, 0, 0])
        y_axis = np.array([0, 1, 0])
        z_axis = np.array([0, 0, 1])
        
        angle_x = np.arccos(np.dot(normal_vector, x_axis) / np.linalg.norm(normal_vector))
        angle_y = np.arccos(np.dot(normal_vector, y_axis) / np.linalg.norm(normal_vector))
        angle_z = np.arccos(np.dot(normal_vector, z_axis) / np.linalg.norm(normal_vector))
        
        return np.degrees(angle_x), np.degrees(angle_y), np.degrees(angle_z)
     


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
    # vis = update_camera_view(vis, pcd)
    
    
    
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
    y_axis = np.array([0, 1, 0])
    v = np.cross(normal, y_axis)
    s = np.linalg.norm(v)
    c = np.dot(normal, y_axis)
    I = np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = I + vx + np.dot(vx, vx) * ((1 - c) / (s ** 2))

    # Apply the rotation to the ground normal
    transformed_normal = np.dot(normal, R.T)

    # Calculate angles for pcd_ground
    angles_ground = calculate_angles(normal)
    logger.info(f"Angles of pcd_ground with x, y, z axes: {angles_ground}")

    # Calculate angles for pcd_ground_transformed
    angles_transformed = calculate_angles(transformed_normal)
    logger.info(f"Angles of pcd_ground_transformed with x, y, z axes: {angles_transformed}")


    # Transform all the points in pcd_ground using the rotation matrix R
    pcd_ground_transformed = pcd_ground.translate((0, 0, 0), relative=False)
    pcd_ground_transformed.rotate(R, center=(0, 0, 0))
    
    num_points_ground = len(pcd_ground.point.positions)
    num_points_ground_transformed = len(pcd_ground_transformed.point.positions)
    print(f"Number of points in pcd_ground: {num_points_ground}")
    print(f"Number of points in pcd_ground_transformed: {num_points_ground_transformed}")

    # Paint pcd_ground_transformed to yellow
    pcd_ground_transformed.paint_uniform_color([0.0, 1.0, 0.0])  # Green color

    # Paint pcd_ground to blue
    pcd_ground.paint_uniform_color([0.0, 0.0, 1.0])  # Blue color

    # Add both geometries to the visualizer
    # vis.add_geometry(pcd_ground_transformed.to_legacy())
    vis.add_geometry(pcd_ground.to_legacy())

    # # vis.add_geometry(pcd.to_legacy())
    # vis.add_geometry(pcd_ground_transformed.to_legacy())

    # update camera view
    view_ctr = vis.get_view_control()
    
    # view_ctr.rotate(180, 0)
    view_ctr.set_front(np.array([0, 0, -1]))
    view_ctr.set_up(np.array([0, -1, 0]))
    
    vis.run()
    vis.destroy_window()

    
