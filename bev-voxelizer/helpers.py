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
from torch.utils.data import Dataset, DataLoader


from bev_voxelizer import BevVoxelizer
from utils.data_generator import pcd_to_segmentation_mask_mono, mono_to_rgb_mask
from utils.log_utils import get_logger

logger = get_logger("helpers")



class PointCloudDataset(Dataset):
    """Custom dataset for loading point clouds."""
    
    def __init__(self, data_dir):
        self.logger = get_logger("PointCloudDataset")
        self.data_dir = data_dir
        self.pointcloud_files = self.find_files()
        
    def find_files(self):
        """Recursively find files named 'left-segmented-labelled.ply'."""
        matches = []
        for root, _, files in os.walk(self.data_dir):
            for filename in fnmatch.filter(files, "left-segmented-labelled.ply"):
                matches.append(os.path.join(root, filename))
        
        
        self.logger.info(f"=================================")
        self.logger.info(f"Found {len(matches)} point cloud files in {self.data_dir}")
        self.logger.info(f"=================================\n")
        
        return matches

    def load_pointcloud(self, file):
        """Load a point cloud from a file."""
        return o3d.t.io.read_point_cloud(file)

    def __len__(self):
        """Return the total number of point clouds."""
        return len(self.pointcloud_files)

    def __getitem__(self, idx):
        """Return a single point cloud as a tensor."""
        file = self.pointcloud_files[idx]
        pointcloud = self.load_pointcloud(file)

        # Convert the point cloud to a numpy array and then to a tensor
        positions = pointcloud.point['positions'].numpy()
        labels = pointcloud.point['label'].numpy()
        colors = pointcloud.point['colors'].numpy()
        
        self.logger.info(f"=================================")
        self.logger.info(f"positions.shape: {positions.shape}")
        self.logger.info(f"labels.shape: {labels.shape}")
        self.logger.info(f"colors.shape: {colors.shape}")
        self.logger.info(f"=================================\n")

        # Return as a dictionary or a tuple
        return {
            'positions': torch.tensor(positions, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int32),
            'colors': torch.tensor(colors, dtype=torch.float32)
        }

def crop_pointcloud(
    pcd_path: str,
    output_path: str = None
) -> None:
    """Crop a point cloud to a specified area and save it to disk."""
    
    assert output_path is not None, "output_path must be provided"
    
    logger.info(f"=================================")
    logger.info(f"Cropping point cloud from {pcd_path} to {output_path}")
    logger.info(f"=================================\n")

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
    colors = colors[valid_indices]

    # Create a new point cloud with the cropped points
    cropped_pcd = o3d.t.geometry.PointCloud()
    cropped_pcd.point['positions'] = o3d.core.Tensor(np.column_stack((x_coords, np.zeros_like(x_coords), z_coords)), dtype=o3d.core.Dtype.Float32)
    cropped_pcd.point['labels'] = o3d.core.Tensor(labels, dtype=o3d.core.Dtype.Int32)
    cropped_pcd.point['colors'] = o3d.core.Tensor(colors, dtype=o3d.core.Dtype.Float32)
    
    o3d.t.io.write_point_cloud(output_path, cropped_pcd)
    
    logger.info(f"=================================")
    logger.info(f"Cropped point cloud saved to {output_path}")
    logger.info(f"=================================\n")

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

    # # CASE 2: Crop point clouds and save them to disk
    # src_pcd_path = "train-data/144/left-segmented-labelled.ply"
    # dst_pcd_path = "debug/cropped_pointcloud.ply"
    # crop_pointcloud(src_pcd_path, dst_pcd_path)


    # CASE 3: Create a dataloader for point clouds
    dataset = PointCloudDataset(data_dir="train-data")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    logger.warning(f"=================================")
    logger.warning(f"Dataloader created with {len(dataloader)} batches")
    logger.warning(f"=================================\n")

    for idx, batch in enumerate(dataloader):
        logger.info(f"=================================")
        logger.info(f"batch {idx}  ==> {batch}")
        logger.info(f"=================================\n")