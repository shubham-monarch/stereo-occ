#! /usr/bin/env python3

import open3d as o3d    
import os
import shutil
import logging, coloredlogs
from pathlib import Path

# custom imports
from utils import io_utils

logger = logging.getLogger(__name__)


def extract_and_rename_ply_files(
    source_folder: Path,
    dest_folder: Path = None
) -> None:
    
    base_folder = os.path.dirname(source_folder)
    src_folder_name = os.path.basename(source_folder)

    if dest_folder is None: 
        dest_folder = os.path.join(base_folder, f"segmented-{src_folder_name}")

    io_utils.create_folders([dest_folder])
    
    # Walk through all directories and subdirectories
    for i, (root, dirs, files) in enumerate(os.walk(source_folder)):
        if "left-segmented-labelled.ply" in files:
            source_file = os.path.join(root, "left-segmented-labelled.ply")
            
            if os.path.exists(source_file):
                # Create a unique filename based on the subfolder structure
                destination_file = os.path.join(dest_folder, f"{i}.ply")

                logger.warning(f"source_file: {source_file}")   
                logger.warning(f"destination_file: {destination_file}")
                shutil.copy(source_file, destination_file)
            
            else:
                logger.warning(f"Source file not found: {source_file}")
                continue
            # Copy and rename the file
            # shutil.copy(source_file, destination_file)
            # logging.info(f"Copied and renamed: {source_file} -> {destination_file}")



# extracts left-segmented.ply files 
# from the sfm folder
def collect_segmented_ply_files(sfm_folder):
    pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO) 

    src_folder = Path("ply/1056_to_1198/")
    extract_and_rename_ply_files(src_folder)

    # pcd = o3d.t.io.read_point_cloud("ply_segmented/1056_to_1198/")
    # print(pcd)