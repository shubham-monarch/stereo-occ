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
    
    # visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # adding co-ordinate frame to visualizer
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=4, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)
    
    # PCL utilities class
    pcd_utils = BevVoxelizer()

    # HOOD [blue] vs NO HOOD [red]
    
    # ================================================
    # ground-plane comparison [before tilt-correction]
    # ================================================
    
    # ground_pcd_HOOD = pcd_utils.get_class_pointcloud(pcd_HOOD, pcd_utils.LABELS["NAVIGABLE_SPACE"]["id"])
    # ground_pcd_NO_HOOD = pcd_utils.get_class_pointcloud(pcd_NO_HOOD, pcd_utils.LABELS["NAVIGABLE_SPACE"]["id"])

    # [add to visualizer]
    # ground_pcd_HOOD.paint_uniform_color([0.0, 0.0, 1.0])  # Blue
    # ground_pcd_NO_HOOD.paint_uniform_color([1.0, 0.0, 0.0])  # Red
    
    # vis.add_geometry(ground_pcd_HOOD.to_legacy())
    # vis.add_geometry(ground_pcd_NO_HOOD.to_legacy())

    # ================================================
    # ground-plane comparison [after tilt-correction]
    # ================================================
    
    # R_HOOD = pcd_utils.compute_tilt_matrix(pcd_HOOD)
    # yaw_HOOD, pitch_HOOD, roll_HOOD = pcd_utils.rotation_matrix_to_ypr(R_HOOD)

    # logger.info(f"=================================")    
    # logger.info(f"Yaw: {yaw_HOOD:.2f} degrees, Pitch: {pitch_HOOD:.2f} degrees, Roll: {roll_HOOD:.2f} degrees")
    # logger.info(f"=================================\n")

    # R_NO_HOOD = pcd_utils.compute_tilt_matrix(pcd_NO_HOOD)
    # yaw_NO_HOOD, pitch_NO_HOOD, roll_NO_HOOD = pcd_utils.rotation_matrix_to_ypr(R_NO_HOOD)

    # logger.info(f"=================================")    
    # logger.info(f"Yaw: {yaw_NO_HOOD:.2f} degrees, Pitch: {pitch_NO_HOOD:.2f} degrees, Roll: {roll_NO_HOOD:.2f} degrees")
    # logger.info(f"=================================\n")

    # pcd_HOOD.rotate(R_HOOD, center=(0, 0, 0))
    # pcd_NO_HOOD.rotate(R_NO_HOOD, center=(0, 0, 0))

    # ground_pcd_HOOD = pcd_utils.get_class_pointcloud(pcd_HOOD, pcd_utils.LABELS["NAVIGABLE_SPACE"]["id"])
    # ground_pcd_NO_HOOD = pcd_utils.get_class_pointcloud(pcd_NO_HOOD, pcd_utils.LABELS["NAVIGABLE_SPACE"]["id"])

    # # [add to visualizer]
    # ground_pcd_HOOD.paint_uniform_color([0.0, 0.0, 1.0])  # Blue
    # ground_pcd_NO_HOOD.paint_uniform_color([1.0, 0.0, 0.0])  # Red
    
    # vis.add_geometry(ground_pcd_HOOD.to_legacy())
    # vis.add_geometry(ground_pcd_NO_HOOD.to_legacy())

    # ================================================
    # tractor-hood points comparison
    # ================================================

    # hood_pcd_HOOD = pcd_utils.get_class_pointcloud(pcd_HOOD, pcd_utils.LABELS["TRACTOR_HOOD"]["id"])
    # hood_pcd_NO_HOOD = pcd_utils.get_class_pointcloud(pcd_NO_HOOD, pcd_utils.LABELS["TRACTOR_HOOD"]["id"])

    # # [add to visualizer]
    # hood_pcd_HOOD.paint_uniform_color([0.0, 0.0, 1.0])  # Blue
    # hood_pcd_NO_HOOD.paint_uniform_color([1.0, 0.0, 0.0])  # Red
    
    # vis.add_geometry(hood_pcd_HOOD.to_legacy())
    # vis.add_geometry(hood_pcd_NO_HOOD.to_legacy())

    
    
    


    # VISUALIZER
    
    # Adding point clouds to visualizer
    view_ctr = vis.get_view_control()
    view_ctr.set_front(np.array([0, -1, 0]))
    view_ctr.set_up(np.array([0, 0, 1]))
    view_ctr.set_zoom(0.5)


   

    
    vis.run()
    vis.destroy_window()

    
