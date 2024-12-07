#! /usr/bin/env python3

import open3d as o3d
import os
import random
import numpy as np
import cv2
import shutil
from tqdm import tqdm
import yaml

from bev_voxelizer import BevVoxelizer
from helpers import get_logger




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

   