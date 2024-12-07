import os
import cv2
import numpy as np
import open3d as o3d
import json
from tqdm import tqdm

from logger import get_logger

logger = get_logger("generate-train-data")

class SBEVDataset:
    def __init__(self):
        pass
    

if __name__ == "__main__":
    
    # generate training data for sbevnet -> 
    # [left.jpg, right.jpg, seg-mask-mono.png, seg-mask-rgb.png, camera-extrinsics.json]
    # transform to sbevnet co-ordinate frame
    
    pass