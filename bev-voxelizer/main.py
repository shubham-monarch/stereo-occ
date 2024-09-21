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

    # vis.register_key_callback(120, key_callback)
    vis.register_key_callback(ord("X"), key_callback)

    vis.run()
    vis.destroy_window()
    

    