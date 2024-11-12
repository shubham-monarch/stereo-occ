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
    
    input_path_HOOD = "input/with-hood.ply"
    input_path_NO_HOOD = "input/no-hood.ply"
    
    pcd_HOOD = o3d.t.io.read_point_cloud(input_path_HOOD)
    pcd_NO_HOOD = o3d.t.io.read_point_cloud(input_path_NO_HOOD)
    
    # Visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Co-ordinate frame for vis window    
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=4, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)
    
    # PCL utilities class
    pcd_utils = BevVoxelizer()
    
    # HOOD [blue] vs NO HOOD [red]
    
    # ground-plane comparison before tilting
    ground_pcd_HOOD = pcd_utils.get_class_pointcloud(pcd_HOOD, pcd_utils.LABELS["GROUND"]["id"])
    ground_pcd_NO_HOOD = pcd_utils.get_class_pointcloud(pcd_NO_HOOD, pcd_utils.LABELS["GROUND"]["id"])

    # visualization
    ground_pcd_HOOD.paint_uniform_color([0.0, 0.0, 1.0])  # Blue
    ground_pcd_NO_HOOD.paint_uniform_color([1.0, 0.0, 0.0])  # Red
    
    vis.add_geometry(ground_pcd_HOOD.to_legacy())
    vis.add_geometry(ground_pcd_NO_HOOD.to_legacy())

    # ground-plane comparison after tilting

    # tractor-hood points comparison
    
    
    # src_path = os.path.join(src_folder, file_name)

    
    
    


    # VISUALIZER
    
    # Adding point clouds to visualizer
    
    view_ctr = vis.get_view_control()
    # view_ctr.set_front(np.array([0, -1, 0]))
    # view_ctr.set_up(np.array([0, 0, 1]))
    view_ctr.set_zoom(0.5)
    
    vis.run()
    vis.destroy_window()

    
