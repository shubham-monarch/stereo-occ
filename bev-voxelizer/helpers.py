#!/usr/bin/env python3

import os
import shutil
import open3d as o3d
from tqdm import tqdm
import numpy as np
import cv2


from bev_voxelizer import BevVoxelizer
from utils.data_generator import pcd_to_segmentation_mask_mono, mono_to_rgb_mask
from utils.log_utils import get_logger

logger = get_logger("helpers")

def crop_pointcloud(
    pcd_path: str,
    output_path: str = None
) -> None:
    """Crop a point cloud to a specified area and save it to disk."""
    
    assert output_path is not None, "output_path must be provided"

    # Read the point cloud from the given path
    pcd = o3d.t.io.read_point_cloud(pcd_path)

    # Extract point coordinates and labels
    x_coords = pcd.point['positions'][:, 0].numpy()
    z_coords = pcd.point['positions'][:, 2].numpy()
    labels = pcd.point['label'].numpy()
    colors = pcd.point['colors'].numpy()

    # Crop points to 20m x 10m area
    valid_indices = np.where(
        (x_coords >= -10) & (x_coords <= 10) & 
        (z_coords >= 0) & (z_coords <= 20)
    )[0]

    x_coords = x_coords[valid_indices]
    z_coords = z_coords[valid_indices]
    labels = labels[valid_indices]
   
    # Create a new point cloud with the cropped points
    cropped_pcd = o3d.t.geometry.PointCloud()
    cropped_pcd.point['positions'] = o3d.core.Tensor(np.column_stack((x_coords, np.zeros_like(x_coords), z_coords)), dtype=o3d.core.Dtype.Float32)
    cropped_pcd.point['labels'] = o3d.core.Tensor(labels, dtype=o3d.core.Dtype.Int32)
    cropped_pcd.point['colors'] = o3d.core.Tensor(colors, dtype=o3d.core.Dtype.Float32)
    
    o3d.t.io.write_point_cloud(output_path, cropped_pcd)
    
    logger.info(f"=================================")
    logger.info(f"Cropped point cloud saved to {output_path}")
    logger.info(f"=================================\n")



def find_left_segmented_labelled_files(folder_path):
    """Recursively find all 'left-segmented-labelled.ply' files in the given folder path.   """
    
    
    left_segmented_labelled_files = []

    total_files = sum(len(files) for _, _, files in os.walk(folder_path) if 'left-segmented-labelled.ply' in files)
    with tqdm(total=total_files, desc="Processing files", ncols=100) as pbar:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file == 'left-segmented-labelled.ply':
                    file_path = os.path.join(root, file)

                    logger.warning(f"=================================")
                    logger.warning(f"Processing {file_path}")
                    logger.warning(f"=================================\n")

                    left_segmented_labelled_files.append(file_path)
                    
                    pcd_input = o3d.t.io.read_point_cloud(file_path)    
                    bev_voxelizer = BevVoxelizer()
                    combined_pcd = bev_voxelizer.generate_bev_voxels(pcd_input)

                    seg_mask_mono = pcd_to_segmentation_mask_mono(combined_pcd)
                    seg_mask_rgb = mono_to_rgb_mask(seg_mask_mono, "Mavis.yaml")
                    
                    seg_mask_mono_path = os.path.join(root, 'seg-mask-mono.png')
                    seg_mask_rgb_path = os.path.join(root, 'seg-mask-rgb.png')
                    
                    cv2.imwrite(seg_mask_mono_path, seg_mask_mono)
                    cv2.imwrite(seg_mask_rgb_path, seg_mask_rgb)
                    
                    pbar.update(1)

                





if __name__ == "__main__":
    train_folder = "train-data"
    # occ_data_folder = "occ-data"
    
    # CASE 1: 
    find_left_segmented_labelled_files(train_folder)

    # CASE 2: Crop point clouds and save them to disk
    src_pcd_path = "train-data/2024_06_06_utc/2024_06_06_utc_000000.ply"
    dst_pcd_path = "cropped_pointcloud.ply"
    crop_pointcloud(src_pcd_path, dst_pcd_path)
