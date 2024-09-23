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
    

if __name__ == "__main__":
    
    src_folder = "ply/segmented-1056_to_1198/"
    random_pointcloud_path = get_random_segmented_pcd(src_folder)
    
    pcd = o3d.t.io.read_point_cloud(random_pointcloud_path)
    
    # visualization wind
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # add co-ordinate frame to vis window    
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)

    # tilt correction
    R = compute_pcd_tilt(pcd)
    pcd_ground = get_class_pointcloud(pcd, 2)
    
    # rotate pcd_ground_
    pcd_ground_ = pcd_ground.clone()
    pcd_ground_.rotate(R, center=(0, 0, 0))
    
    # paint yellow
    pcd_ground_.paint_uniform_color([1.0, 1.0, 0.0])  # RGB values for yellow
    vis.add_geometry(pcd_ground_.to_legacy())


    # adjust camera view
    view_ctr = vis.get_view_control()
    view_ctr.set_front(np.array([0, 0, -1]))
    view_ctr.set_up(np.array([0, -1, 0]))
    
    vis.run()
    vis.destroy_window()

    
