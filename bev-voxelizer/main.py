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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO) 

    src_folder = "ply/segmented-1056_to_1198/"
    random_segmented_pcd = get_random_segmented_pcd(src_folder)
    
    point_cloud = o3d.io.read_point_cloud(random_segmented_pcd)
    
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.add_geometry(point_cloud)
    
    # Create coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)

    # Set the camera view to be at the pointcloud origin
    ctr = vis.get_view_control()
    
    # Get the bounding box of the point cloud
    bbox = point_cloud.get_axis_aligned_bounding_box()
    bbox_min = bbox.get_min_bound()
    bbox_max = bbox.get_max_bound()
    dimensions = bbox_max - bbox_min

    center = bbox.get_center()
    ctr.set_lookat(center)
     
    logger.info(f"center: {center}")
    
    # Set the camera position 
    ctr.set_front([0, 0, -1])  # Looking along negative z-axis
    ctr.set_up([0, -1, 0])     # Up direction is negative y-axis (Open3D convention)
    ctr.set_zoom(0.8)

    



    vis.run()
    vis.destroy_window()
    

    