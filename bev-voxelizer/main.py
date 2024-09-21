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
coloredlogs.install(level='DEBUG', logger=logger, force=True)

def get_random_segmented_pcd(src_foler: Path) -> Path: 
    '''
    Get a random ply file from the source folder
    '''    
    ply_files = [f for f in os.listdir(src_folder) if f.endswith('.ply')]
    if ply_files:
        random_ply_file = random.choice(ply_files)
        random_ply_path = os.path.join(src_folder, random_ply_file)
    
    return random_ply_path


def key_callback(vis):
    logger.warning(f"===================")
    logger.warning(f"KEYBACK TRIGGERED!")
    logger.warning(f"===================")
    print('key')

    return False

def update_camera_view(vis, point_cloud):
    # Create coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)

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
    logging.basicConfig(level=logging.INFO) 

    src_folder = "ply/segmented-1056_to_1198/"
    random_pointcloud_path = get_random_segmented_pcd(src_folder)
    
    point_cloud = o3d.t.io.read_point_cloud(random_pointcloud_path)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(point_cloud.to_legacy())
    vis = update_camera_view(vis, point_cloud)
    
    # Print attributes of the first 3 points in point_cloud
    # num_points = min(3, point_cloud.point.positions.shape[0])
    # for i in range(num_points):s
    #     logger.info(f"Point {i + 1} attributes:")
    #     for attr_name in point_cloud.point:
    #         attr_value = point_cloud.point[attr_name][i]
    #         logger.info(f"  {attr_name}: {attr_value}")
    #     logger.info("---")





    vis.run()
    vis.destroy_window()
    

    