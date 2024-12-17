#! /usr/bin/env python3

import boto3
import os
import random
from logger import get_logger
from typing import List

class LeafFolder:
    def __init__(self, src_uri: str, target_uri: str):
        self.src_uri = src_uri
        self.target_uri = target_uri


class DataGeneratorS3:
    def __init__(self, src_uris: List[str] = None, dest_folder: str = None, index_json: str = None):
        
        pass
    
    
    
    def get_target_folder_uri(self, src_uri: str, dest_folder:str = "bev-dataset"):
        ''' Make leaf-folder path relative to the bev-dataset folder '''
        return src_uri.replace("occ-dataset", dest_folder, 1)
        
   

    
    def get_leaf_folders(self, s3_uri: str) -> list:
        """Get all leaf folders from the given S3 URI"""
        
        # Parse S3 URI to get bucket and prefix
        s3_parts = s3_uri.replace("s3://", "").split("/", 1)
        bucket_name = s3_parts[0]
        prefix = s3_parts[1] if len(s3_parts) > 1 else ""
        
        # Initialize S3 client
        s3 = boto3.client('s3')
        
        # Get all objects with the given prefix
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
        
        # Keep track of all folders and their parent folders
        all_folders = set()
        parent_folders = set()
        
        # Process all objects
        for page in pages:
            if 'Contents' not in page:
                continue
                
            for obj in page['Contents']:
                # Get the full path
                path = obj['Key']
                
                # Skip the prefix itself
                if path == prefix:
                    continue
                    
                # Get all folder paths in this object's path
                parts = path.split('/')
                for i in range(len(parts)-1):
                    folder = '/'.join(parts[:i+1])
                    if folder:
                        all_folders.add(folder)
                        
                        # If this isn't the immediate parent, it's a parent folder
                        if i < len(parts)-2:
                            parent_folders.add(folder)
        
        # Leaf folders are those that aren't parents of other folders
        leaf_folders = all_folders - parent_folders
        
        # Convert back to S3 URIs
        leaf_folder_uris = [f"s3://{bucket_name}/{folder}" for folder in sorted(leaf_folders)]
        
        return leaf_folder_uris

if __name__ == "__main__":
    # uri_list = [
    #     "s3://occupancy-dataset/occ-dataset/vineyards/gallo/",
    # ]

    logger = get_logger("__main__")
    data_generator_s3 = DataGeneratorS3()

    # src_uri = "s3://occupancy-dataset/occ-dataset/vineyards/gallo/2024_06_07_utc/svo_files/front_2024-06-04-10-24-57.svo/1398_to_1540/frame-1400/"
    src_uri = "s3://occupancy-dataset/occ-dataset/"
    # target_uri = data_generator_s3.get_target_uri(src_uri, "bev-dataset")

    # logger.info(f"===================")
    # logger.info(f"target_uri: {target_uri}")
    # logger.info(f"===================\n")

    leaf_folders = data_generator_s3.get_leaf_folders(src_uri)
    logger.info(f"===================")
    logger.info(f"len(leaf_folders): {len(leaf_folders)}")
    logger.info(f"leaf_folders[:20]: {leaf_folders[:20]}")
    logger.info(f"===================\n")

    # leaf_folders = data_generator_s3.get_leaf_folders("s3://occupancy-dataset/occ-dataset/vineyards/gallo/")
    
    # logger = get_logger("data-generator-s3")
    
    # logger.info(f"===================")
    # logger.info(f"len(leaf_folders): {len(leaf_folders)}")
    # logger.info(f"===================")

    
    # random_values = random.sample(leaf_folders, min(20, len(leaf_folders)))
    
    # logger.info(f"===================")
    # logger.info(f"Random values from leaf folders: {random_values}")
    # logger.info(f"===================\n")
