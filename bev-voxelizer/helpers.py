#!/usr/bin/env python3

import os
import shutil
import open3d as o3d
from tqdm import tqdm
import numpy as np
import cv2
import os
import fnmatch
import torch
import logging
import sys
import coloredlogs
import yaml
import matplotlib.pyplot as plt

from logger import get_logger
from bev_voxelizer import BevVoxelizer


logger = get_logger("helpers")

def list_base_folders(folder_path):
    base_folders = []
    for root, dirs, files in os.walk(folder_path):
        for dir_name in dirs:
            base_folders.append(os.path.join(root, dir_name))
    return base_folders

def plot_plane_histogram(self,pcd: o3d.t.geometry.PointCloud):
    
        
        positions = pcd.point['positions'].numpy()
        x_values = positions[:, 0]
        y_values = positions[:, 1]
        z_values = positions[:, 2]

        fig, axs = plt.subplots(3, 1, figsize=(10, 15))

        axs[0].hist(x_values, bins=50, color='r', alpha=0.7)
        axs[0].set_title('Histogram of X values')
        axs[0].set_xlabel('X')
        axs[0].set_ylabel('Frequency')

        axs[1].hist(y_values, bins=50, color='g', alpha=0.7)
        axs[1].set_title('Histogram of Y values')
        axs[1].set_xlabel('Y')
        axs[1].set_ylabel('Frequency')

        axs[2].hist(z_values, bins=50, color='b', alpha=0.7)
        axs[2].set_title('Histogram of Z values')
        axs[2].set_xlabel('Z')
        axs[2].set_ylabel('Frequency')

        plt.tight_layout()
        plt.show()


def get_label_colors_from_yaml(yaml_path=None):
    """Read label colors from Mavis.yaml config file."""
    
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Get BGR colors directly from yaml color_map
    label_colors_bgr = config['color_map']
    
    # Convert BGR to RGB by reversing color channels
    label_colors_rgb = {
        label: color[::-1] 
        for label, color in label_colors_bgr.items()
    }
    
    return label_colors_bgr, label_colors_rgb
        

def mono_to_rgb_mask(mono_mask: np.ndarray, yaml_path: str = "Mavis.yaml") -> np.ndarray:
    """Convert single channel segmentation mask to RGB using label mapping from a YAML file."""
    
    label_colors_bgr, _ = get_label_colors_from_yaml(yaml_path)
    
    H, W = mono_mask.shape
    rgb_mask = np.zeros((H, W, 3), dtype=np.uint8)
    
    for label_id, rgb_value in label_colors_bgr.items():
        mask = mono_mask == label_id
        rgb_mask[mask] = rgb_value
        
    return rgb_mask


def count_unique_labels(mask_img: np.ndarray):
    """Count and return the number of unique labels in a segmented mask image."""
    
    if mask_img.ndim == 3:
        # Convert RGB to single integer for each unique color
        mask_flat = mask_img.reshape(-1, 3)
    elif mask_img.ndim == 2:
        # For 2D array, flatten directly
        mask_flat = mask_img.flatten()
    else:
        raise ValueError("mask_img must be either a 2D or 3D array")
    
    unique_colors = np.unique(mask_flat, axis=0)
    
    return len(unique_colors), unique_colors

def pcd_to_segmentation_mask_mono(pcd: o3d.t.geometry.PointCloud, H: int = 480, W: int = 640) -> np.ndarray:
    """Generate a 2D segmentation mask from a labeled pointcloud.    """
    
    # Extract point coordinates and labels
    x_coords = pcd.point['positions'][:, 0].numpy()
    z_coords = pcd.point['positions'][:, 2].numpy()
    labels = pcd.point['label'].numpy()

    # Crop points to 20m x 10m area
    valid_indices = np.where(
        (x_coords >= -10) & (x_coords <= 10) & 
        (z_coords >= 0) & (z_coords <= 20)
    )[0]

    x_coords = x_coords[valid_indices]
    z_coords = z_coords[valid_indices]
    labels = labels[valid_indices]

    # Scale coordinates to image dimensions
    x_min, x_max = x_coords.min(), x_coords.max()
    z_min, z_max = z_coords.min(), z_coords.max()

    x_scaled = ((x_coords - x_min) / (x_max - x_min) * (W - 1)).astype(np.int32)
    z_scaled = ((z_coords - z_min) / (z_max - z_min) * (H - 1)).astype(np.int32)

    # Create empty mask
    mask = np.zeros((H, W), dtype=np.uint8)

    # Label mapping (using original label values directly)
    # 1: Obstacle
    # 2: Navigable Space
    # 3: Vine Canopy  
    # 4: Vine Stem
    # 5: Vine Pole

    # # Fill mask with label values
    # for x, z, label in zip(x_scaled, z_scaled, labels):
    #     if 1 <= label <= 5:  # Only use valid label values
    #         mask[z, x] = label

    # Fill mask with label values
    for x, z, label in zip(x_scaled, z_scaled, labels):
        mask[H - z - 1, x] = label  # Invert z to match image coordinates

    return mask


def crop_pointcloud(
    src_pcd: o3d.t.geometry.PointCloud,
    output_path: str
) -> None:
    """Crop a point cloud to a specified area and save it to disk."""
    
    logger.info(f"=================================")
    logger.info(f"Cropping point cloud to {output_path}")
    logger.info(f"=================================\n")

    # Extract point coordinates and labels
    x_coords = src_pcd.point['positions'][:, 0].numpy()
    y_coords = src_pcd.point['positions'][:, 1].numpy()
    z_coords = src_pcd.point['positions'][:, 2].numpy()
    labels = src_pcd.point['label'].numpy()
    colors = src_pcd.point['colors'].numpy()

    # Crop points to 20m x 10m area
    valid_indices = np.where(
        (z_coords >= 0) & (z_coords <= 5)
    )[0]

    x_coords = x_coords[valid_indices]
    y_coords = y_coords[valid_indices]
    z_coords = z_coords[valid_indices]
    labels = labels[valid_indices]
    colors = colors[valid_indices]

    # Create a new point cloud with the cropped points
    cropped_pcd = o3d.t.geometry.PointCloud()
    cropped_pcd.point['positions'] = o3d.core.Tensor(np.column_stack((x_coords, y_coords, z_coords)), dtype=o3d.core.Dtype.Float32)
    cropped_pcd.point['label'] = o3d.core.Tensor(labels, dtype=o3d.core.Dtype.UInt8)
    cropped_pcd.point['colors'] = o3d.core.Tensor(colors, dtype=o3d.core.Dtype.UInt8)
    
    o3d.t.io.write_point_cloud(output_path, cropped_pcd)
    logger.info(f"Cropped point cloud saved to {output_path}")


# def add_seg_masks_to_dataset(folder_path):
#     """Recursively find all 'left-segmented-labelled.ply' files in the given folder path."""
    
    
#     left_segmented_labelled_files = []

#     total_files = sum(len(files) for _, _, files in os.walk(folder_path) if 'left-segmented-labelled.ply' in files)
#     with tqdm(total=total_files, desc="Processing files", ncols=100) as pbar:
#         for root, dirs, files in os.walk(folder_path):
#             for file in files:
#                 if file == 'left-segmented-labelled.ply':
#                     file_path = os.path.join(root, file)

#                     logger.warning(f"=================================")
#                     logger.warning(f"Processing {file_path}")
#                     logger.warning(f"=================================\n")

#                     left_segmented_labelled_files.append(file_path)
                    
#                     pcd_input = o3d.t.io.read_point_cloud(file_path)    
#                     bev_voxelizer = BevVoxelizer()
#                     combined_pcd = bev_voxelizer.generate_bev_voxels(pcd_input)

#                     seg_mask_mono = pcd_to_segmentation_mask_mono(combined_pcd)
#                     seg_mask_rgb = mono_to_rgb_mask(seg_mask_mono, "Mavis.yaml")
                    
#                     seg_mask_mono_path = os.path.join(root, 'seg-mask-mono.png')
#                     seg_mask_rgb_path = os.path.join(root, 'seg-mask-rgb.png')
                    
#                     cv2.imwrite(seg_mask_mono_path, seg_mask_mono)
#                     cv2.imwrite(seg_mask_rgb_path, seg_mask_rgb)
                    
#                     pbar.update(1)




if __name__ == "__main__":
    # train_folder = "train-data"
    # occ_data_folder = "occ-data"
    
    # CASE 1: 
    # add_seg_masks_to_dataset(train_folder)

    pass
    