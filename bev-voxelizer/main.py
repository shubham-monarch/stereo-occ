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
import matplotlib.pyplot as plt
import time

# custom modules
from utils import io_utils
from bev_voxelizer import BevVoxelizer

# TO-DO
# - priority based collapsing
# - crop pointcloud to bounding boxs
# - hidden point removal 
# - farthest point sampling
# - checkout bev-former voxelizer
# - statistical outlier removal
# - refactor compute_tilt_matrix()
# - make project_to_ground_plane more robust

# LOGGING SETUP
logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(lineno)d')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
coloredlogs.install(level='INFO', logger=logger, force=True)


if __name__ == "__main__":
    
    src_folder = "pcd-files/vineyards/RJM"
    
    vis = o3d.visualization.Visualizer()
    
    
    segmented_pcd_folder = "pcd-files/vineyards/RJM"
    segmented_pcd_folder_files = os.listdir(segmented_pcd_folder)
    
    # random.seed(0)
    # file = random.choice(segmented_pcd_folder_files)
    file  = "vineyards_RJM_15.ply"
    # file = "vineyards_RJM_38.ply"
    
    try:
        logger.warning(f"=================================")        
        logger.warning(f"Processing {file}")
        logger.warning(f"=================================\n")
                
        pcd_input = o3d.t.io.read_point_cloud(os.path.join(segmented_pcd_folder, file))
        
        bev_voxelizer = BevVoxelizer()
        combined_pcd = bev_voxelizer.generate_bev_voxels(pcd_input)

        # output_file_path = os.path.join(segmented_pcd_folder, "combined_output.ply")
        o3d.t.io.write_point_cloud("combined_output.ply", combined_pcd)
        logger.info(f"Saved combined point cloud to combined_output.ply")
        
        
        vis.create_window()
        
        # Co-ordinate frame for vis window      
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3, origin=[0, 0, 0])
        vis.add_geometry(coordinate_frame)
        
        # Adding point clouds to visualizer
        vis.add_geometry(combined_pcd.to_legacy())
        
        view_ctr = vis.get_view_control()
        view_ctr.set_front(np.array([0, -1, 0]))
        view_ctr.set_up(np.array([0, 0, 1]))
        view_ctr.set_zoom(0.9)
        
        vis.run()
        vis.destroy_window()
    except Exception as e:
        logger.error(f"Error processing {file}: {e}")