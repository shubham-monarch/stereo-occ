#! /usr/bin/env python3

import cv2
import numpy as np
import logging
import matplotlib.pyplot as plt
import os
import shutil
import time
import math
from typing import List
import random
import json
import sys
from pathlib import Path
from tqdm import tqdm
import coloredlogs


logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger, force=True)

def collect_s3_pointclouds(
    source_folder: Path,
    dest_folder: Path = None
) -> str:
    
    '''
    aws utils to collect pointclouds from s3
    '''
    
    base_folder = os.path.dirname(source_folder)
    src_folder_name = os.path.basename(source_folder)

    if dest_folder is None: 
        dest_folder = os.path.join(base_folder, f"segmented-{src_folder_name}")

    create_folders([dest_folder])
    
    # Walk through all directories and subdirectories
    total_items = sum([len(files) for _, _, files in os.walk(source_folder)])
    for i, (root, dirs, files) in enumerate(tqdm(os.walk(source_folder), total=total_items, desc="Processing files")):
        if "left-segmented-labelled.ply" in files:
            source_file = os.path.join(root, "left-segmented-labelled.ply")
            
            if os.path.exists(source_file):
                # Create a unique filename based on the subfolder structure
                destination_file = os.path.join(dest_folder, f"{i}.ply")
                shutil.copy(source_file, destination_file)
            
            else:
                logger.warning(f"Source file not found: {source_file}")
                continue
            
    return dest_folder  

def delete_files(file_paths):
    for file_path in file_paths:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                logging.error(f"Error {e} while deleting {file_path}")
        else:
            logging.info(f"The file {file_path} does not exist.")

def delete_folders(folders):
	for folder_path in folders:
			# logging.debug(f"Deleting the old files in {folder_path}")
			if os.path.exists(folder_path):
				try: 
					shutil.rmtree(folder_path)
				except Exception as e :
					logging.error(f"Error {e} while deleting {folder_path}")
					# time.sleep(1)  # wait for 1 second before retrying
			
def create_folders(folders):
	for path in folders:
		os.makedirs(path, exist_ok=True)
		# logging.warning(f"Created the {path} folder!")