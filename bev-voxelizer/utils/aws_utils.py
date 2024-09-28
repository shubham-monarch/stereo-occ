#! /usr/bin/env python3

import boto3
import logging
import coloredlogs
from tqdm import tqdm
import os

# LOGGING SETUP
logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(lineno)d')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
coloredlogs.install(level='INFO', logger=logger, force=True)

def list_unique_segmented_pcds(s3_uri: str):
    """
    Takes an S3 URI for a folder and returns the S3 URIs of all 'left-segmented-labelled.ply' files inside 'frame-' sub-folders recursively.
    Ensures that all the selected 'frame-' folders have a unique parent folder.
    """
    s3 = boto3.client('s3')
    
    # Parse the S3 URI
    if not s3_uri.startswith("s3://"):
        raise ValueError("Invalid S3 URI")
    
    s3_uri = s3_uri[5:]
    bucket_name, prefix = s3_uri.split('/', 1)
    
    def list_frame_subfolders(bucket_name, prefix, pbar, parent_folders):
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix, Delimiter='/')
        frame_files_list = []
        
        if 'CommonPrefixes' in response:
            subfolders = [common_prefix['Prefix'] for common_prefix in response['CommonPrefixes']]
            for subfolder in subfolders:
                pbar.update(1)
                parent_folder = subfolder.split('/')[-3]  # Get the parent folder name
                if subfolder.startswith(prefix + 'frame-') and parent_folder not in parent_folders:
                    file_key = subfolder + 'left-segmented-labelled.ply'
                    try:
                        s3.head_object(Bucket=bucket_name, Key=file_key)
                        frame_files_list.append(f"s3://{bucket_name}/{file_key}")
                        parent_folders.add(parent_folder)
                    except:
                        pass
                frame_files_list.extend(list_frame_subfolders(bucket_name, subfolder, pbar, parent_folders))
        
        return frame_files_list
    
    # Get the total number of subfolders for the progress bar
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix, Delimiter='/')
    total_subfolders = sum(1 for _ in response.get('CommonPrefixes', []))
    
    with tqdm(total=total_subfolders, desc="Processing subfolders") as pbar:
        parent_folders = set()
        return list_frame_subfolders(bucket_name, prefix, pbar, parent_folders)
    
def list_s3_subfolders(s3_uri: str):
    """
    Takes an S3 URI for a folder and returns the S3 URIs of all the sub-folders ending with .svo inside that folder recursively.
    """
    s3 = boto3.client('s3')
    
    # Parse the S3 URI
    if not s3_uri.startswith("s3://"):
        raise ValueError("Invalid S3 URI")
    
    s3_uri = s3_uri[5:]
    bucket_name, prefix = s3_uri.split('/', 1)
    
    def list_subfolders(bucket_name, prefix, pbar):
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix, Delimiter='/')
        subfolders_list = []
        
        if 'CommonPrefixes' in response:
            subfolders = [common_prefix['Prefix'] for common_prefix in response['CommonPrefixes']]
            for subfolder in subfolders:
                pbar.update(1)
                if subfolder.endswith('.svo/'):
                    subfolders_list.append(f"s3://{bucket_name}/{subfolder}")
                subfolders_list.extend(list_subfolders(bucket_name, subfolder, pbar))
        
        return subfolders_list
    
    # Get the total number of subfolders for the progress bar
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix, Delimiter='/')
    total_subfolders = sum(1 for _ in response.get('CommonPrefixes', []))
    
    with tqdm(total=total_subfolders, desc="Processing subfolders") as pbar:
        return list_subfolders(bucket_name, prefix, pbar)


def download_s3_file(s3_uri, local_path):
        s3 = boto3.client('s3')
        bucket_name, key = s3_uri.replace("s3://", "").split("/", 1)
        s3.download_file(bucket_name, key, local_path)

def save_segmented_pcds(segmented_pcds, base_local_path):
    if not os.path.exists(base_local_path):
        os.makedirs(base_local_path)
    
    with tqdm(total=len(segmented_pcds), desc="Downloading segmented PCDs") as pbar:
        for idx, s3_uri in enumerate(segmented_pcds, start=1):
            subfolder_name = s3_uri.split('/')[4] + "_" + s3_uri.split('/')[5]  # Concatenate the immediate and next to immediate sub-folder name
            local_filename = f"{subfolder_name}_{idx}.ply"
            local_path = os.path.join(base_local_path, local_filename)
            download_s3_file(s3_uri, local_path)
            # logger.info(f"Downloaded {s3_uri} to {local_path}")
            pbar.update(1)


if __name__ == "__main__":
    
    s3_uri = "s3://occupancy-dataset/occ-dataset/vineyards/treasury/"
    svo_folders = list_s3_subfolders(s3_uri)

    logger.info(f"=================================")    
    logger.info(svo_folders)
    logger.info(f"=================================\n")

    folder = svo_folders
    segmented_pcds = []
    for folder in svo_folders:
        segmented_pcds.extend(list_unique_segmented_pcds(folder))
    
    logger.info(f"=================================")    
    logger.info(f"Total segmented PCDs: {len(segmented_pcds)}")
    for idx, pcd in enumerate(segmented_pcds, start=1):
        sub_folders = "/".join(pcd.split('/')[-3:])
        logger.info(f"{idx} => {sub_folders}")
    logger.info(f"=================================\n")
    
    
    base_local_path = "downloaded_segmented_pcds"
    
    logger.info(f"=================================")    
    save_segmented_pcds(segmented_pcds, base_local_path)
    logger.info(f"=================================\n")