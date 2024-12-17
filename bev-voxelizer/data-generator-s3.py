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
        
        # # Validate that all s3 URIs have occ-dataset as root directory
        # for uri in src_uris:
        #     bucket_name, prefix = uri.replace("s3://", "").split("/", 1)
        #     root_dir = prefix.split("/")[0]
        #     assert root_dir == "occ-dataset", f"Root directory must be 'occ-dataset', got '{root_dir}' in {uri}"
        
        
        # self.logger = get_logger("data-generator-s3")
        # self.leaf_folders = self.get_leaf_folders(src_uris)
        # self.index_json = index_json
        pass
    
    
    
    def get_target_uri(self, src_uri: str, dest_folder:str = "bev-dataset"):
        ''' make leaf-folder path relative to the bev-dataset folder '''
        return src_uri.replace("occ-dataset", dest_folder, 1)
        



    
if __name__ == "__main__":
    # uri_list = [
    #     "s3://occupancy-dataset/occ-dataset/vineyards/gallo/",
    # ]

    logger = get_logger("__main__")
    data_generator_s3 = DataGeneratorS3()

    src_uri = "s3://occupancy-dataset/occ-dataset/vineyards/gallo/2024_06_07_utc/svo_files/front_2024-06-04-10-24-57.svo/1398_to_1540/frame-1400/"
    target_uri = data_generator_s3.get_target_uri(src_uri, "bev-dataset")

    logger.info(f"===================")
    logger.info(f"target_uri: {target_uri}")
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
