#! /usr/bin/env python3

import open3d as o3d
import os
import random
import numpy as np
import cv2
import shutil
from tqdm import tqdm

from bev_voxelizer import BevVoxelizer
from utils.log_utils import get_logger

logger = get_logger("data_generator")

def list_base_folders(folder_path):
    base_folders = []
    for root, dirs, files in os.walk(folder_path):
        for dir_name in dirs:
            base_folders.append(os.path.join(root, dir_name))
    return base_folders


# def mono_to_rgb_mask(mono_mask: np.ndarray, label_colors: dict) -> np.ndarray:
#     """Convert single channel segmentation mask to RGB using provided label mapping.
    
#     Args:
#         mono_mask: Single channel segmentation mask with integer labels
#         label_colors: Dictionary mapping label IDs to RGB values (e.g. {1: [255,0,0]})
        
#     Returns:
#         RGB segmentation mask with shape (H,W,3)
#     """
#     H, W = mono_mask.shape
#     rgb_mask = np.zeros((H, W, 3), dtype=np.uint8)
    
#     for label_id, rgb_value in label_colors.items():
#         mask = mono_mask == label_id
#         rgb_mask[mask] = rgb_value
        
#     return rgb_mask


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



def pcd_to_segmentation_mask_mono(
    pcd: o3d.t.geometry.PointCloud,
    H: int = 480,
    W: int = 640
) -> np.ndarray:
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

    # Fill mask with label values
    for x, z, label in zip(x_scaled, z_scaled, labels):
        if 1 <= label <= 5:  # Only use valid label values
            mask[z, x] = label

    return mask



# def pcd_to_bev(
#     pcd: o3d.t.geometry.PointCloud,  
#     H: int = 480, 
#     W: int = 640
# ) -> np.ndarray:

#     '''Convert a 3D PCD => BEV => (H, W, 3) numpy array'''
    
#     # bev_voxelizer = BevVoxelizer()
#     # pcd_BEV = bev_voxelizer.generate_bev_voxels(pcd)

#     x_coords = pcd.point['positions'][:, 0].numpy()
#     z_coords = pcd.point['positions'][:, 2].numpy()
#     colors = pcd.point['colors'].numpy()  # RGB values in [0, 255] range

#     # # Print sample color values
#     # logger.info(f"=================================")
#     # logger.info(f"Sample color values:")
#     # sample_size = min(5, len(colors))  # Take up to 5 samples
#     # for i in range(sample_size):
#     #     logger.info(f"Color {i}: RGB={colors[i]}")
#     # logger.info(f"Color value range: [{colors.min()}, {colors.max()}]")
#     # logger.info(f"Color shape: {colors.shape}")
#     # logger.info(f"=================================\n")

#     # cropping bev points to a 20m x 10m area
#     valid_indices = np.where(
#         (x_coords >= -10) & (x_coords <= 10) & 
#         (z_coords >= 0) & (z_coords <= 20)
#     )[0]

#     x_coords = x_coords[valid_indices]
#     z_coords = z_coords[valid_indices]
#     colors = colors[valid_indices]

#     x_min, x_max = x_coords.min(), x_coords.max()
#     z_min, z_max = z_coords.min(), z_coords.max()

#     x_scaled = ((x_coords - x_min) / (x_max - x_min) * (W - 1)).astype(np.int32)
#     z_scaled = ((z_coords - z_min) / (z_max - z_min) * (H - 1)).astype(np.int32)

#     # rgb to bgr
#     colors_bgr = colors[:, [2, 1, 0]]

#     # Print number of unique colors in colors_bgr
#     unique_colors_bgr = np.unique(colors_bgr, axis=0)
#     logger.info(f"=================================")
#     logger.info(f"Number of unique colors in BGR: {len(unique_colors_bgr)}")
#     logger.info(f"=================================\n")

#     # Print number of unique labels
#     unique_labels = np.unique(pcd.point['label'].numpy())
#     logger.info(f"=================================")
#     logger.info(f"Number of unique labels: {len(unique_labels)}")
#     logger.info(f"Unique labels: {unique_labels}")
#     logger.info(f"=================================\n")

#     # Create a blank color image
#     image_BEV = np.zeros((H, W, 3), dtype=np.uint8)

#     # Plot the points on the image with their original colors
#     for x, z, color in zip(x_scaled, z_scaled, colors_bgr):
#         image_BEV[H - z - 1, x] = color  # Invert z to match image coordinates

#     return image_BEV

# def pcd_to_bev(
#     pcd_path: str,  
#     H: int = 480, 
#     W: int = 640
# ) -> np.ndarray:

#     '''Convert a 3D PCD => BEV => (H, W, 3) numpy array'''

#     pcd_3D = o3d.t.io.read_point_cloud(pcd_path)
    
#     bev_voxelizer = BevVoxelizer()
#     pcd_BEV = bev_voxelizer.generate_bev_voxels(pcd_3D)

#     x_coords = pcd_BEV.point['positions'][:, 0].numpy()
#     z_coords = pcd_BEV.point['positions'][:, 2].numpy()
#     colors = pcd_BEV.point['colors'].numpy()  # Assuming colors are in [0, 1] range

    

#     # cropping bev points to a 20m x 10m area
#     valid_indices = np.where(
#         (x_coords >= -10) & (x_coords <= 10) & 
#         (z_coords >= 0) & (z_coords <= 20)
#     )[0]

#     x_coords = x_coords[valid_indices]
#     z_coords = z_coords[valid_indices]
#     colors = colors[valid_indices]

#     x_min, x_max = x_coords.min(), x_coords.max()
#     z_min, z_max = z_coords.min(), z_coords.max()

#     x_scaled = ((x_coords - x_min) / (x_max - x_min) * (W - 1)).astype(np.int32)
#     z_scaled = ((z_coords - z_min) / (z_max - z_min) * (H - 1)).astype(np.int32)

#     # rgb to bgr
#     colors_bgr = colors[:, [2, 1, 0]]

#     # Create a blank color image
#     image_BEV = np.zeros((H, W, 3), dtype=np.uint8)

#     # Plot the points on the image with their original colors
#     for x, z, color in zip(x_scaled, z_scaled, colors_bgr):
#         image_BEV[H - z - 1, x] = color  # Invert z to match image coordinates

#     return image_BEV


if __name__ == "__main__":
    logger = get_logger()

    segmented_pcd_dir = "/home/skumar/ssd/2024_06_06_utc"
    segmented_bev_dir = "train-data"
    
    segmented_pcd_folders = list_base_folders(segmented_pcd_dir)
    

    for i in tqdm(range(200), desc="Processing folders"):
        try:
            random_folder = random.choice(segmented_pcd_folders)
            
            with open("logs/processing_log.txt", "a") as log_file:
                log_file.write("=================================\n")
                log_file.write(f"Processing {random_folder}\n")
                log_file.write("=================================\n")
            
            random_pcd_file = os.path.join(random_folder, "left-segmented-labelled.ply")

            bev_image = pcd_to_segmentation_mask_mono(random_pcd_file)
            
            output_path = os.path.join(random_folder, "bev_image_mono.png")
            cv2.imwrite(output_path, bev_image)

            destination_folder = os.path.join(segmented_bev_dir, str(i))
            shutil.copytree(random_folder, destination_folder)

            with open("logs/processing_log.txt", "a") as log_file:
                log_file.write(f"Saved BEV image to {output_path}\n")
                log_file.write(f"Copied {random_folder} to {destination_folder}\n")
                log_file.write("=================================\n")

        except Exception as e:
            logger.error(f"Error processing folder {random_folder}: {e}")

   