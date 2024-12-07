#! /usr/bin/env python3

import open3d as o3d
import matplotlib.pyplot as plt

from helpers import crop_pointcloud
from bev_voxelizer import BevVoxelizer
from logger import get_logger

logger = get_logger("debug")

if __name__ == "__main__":  


    # # CASE 2: Crop point clouds and save them to disk
    # src_pcd_path = "train-data/144/left-segmented-labelled.ply"
    src_pcd_path = "debug/left-segmented-labelled.ply"
    dst_pcd_path = "debug/cropped_pointcloud.ply"
    
    bev_voxelizer = BevVoxelizer()
    
    src_pcd = o3d.t.io.read_point_cloud(src_pcd_path)
    src_pcd = bev_voxelizer.tilt_rectification(src_pcd)
    
    crop_pointcloud(src_pcd, dst_pcd_path)
   