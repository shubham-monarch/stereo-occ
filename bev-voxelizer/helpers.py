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

def find_left_segmented_labelled_files(folder_path):
    """
    Recursively find all 'left-segmented-labelled.ply' files in the given folder path.

    Args:
        folder_path (str): The path to the folder to search in.

    Returns:
        list: A list of paths to 'left-segmented-labelled.ply' files.
    """
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
    find_left_segmented_labelled_files(train_folder)
