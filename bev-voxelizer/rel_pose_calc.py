#!/usr/bin/env python3

import tqdm, tqdm.notebook
tqdm.tqdm = tqdm.notebook.tqdm  # notebook-friendly progress bars
import os
import time
import sys
import numpy as np
from hloc import extract_features, match_features, reconstruction, pairs_from_exhaustive, visualization
from hloc.visualization import plot_images, read_image
from hloc.utils.viz_3d import init_figure, plot_points, plot_reconstruction, plot_camera_colmap
from pixsfm.util.visualize import init_image, plot_points2D
from pixsfm.refine_hloc import PixSfM
from pixsfm import ostream_redirect
from PIL import Image, ImageDraw
import pycolmap
from pathlib import Path
import pandas as pd
from read_write_model import qvec2rotmat, read_cameras_binary, read_images_binary
from tabulate import tabulate

    
# redirect the C++ outputs to notebook cells
cpp_out = ostream_redirect(stderr=True, stdout=True)
cpp_out.__enter__()

#3outputs = Path('../pixsfm_outputs/')
#ef_dir_locked = outputs / "ref_locked"


'''
e_lw => left camera pose in world frame (4 * 4)
e_rw => right camera pose in world frame (4 * 4)
'''
def calculate_relative_pose(e_lw: np.ndarray, e_rw: np.ndarray):
    from scipy.spatial.transform import Rotation
    e_wl = np.linalg.inv(e_lw)
    e_rl = np.dot(e_rw,np.linalg.inv(e_lw))
    R = e_rl[:3,:3] #extracting the rotation matrix
    dx = e_rl[0,3]
    dy = e_rl[1,3]
    dz = e_rl[2,3]
    dquat = Rotation.from_matrix(R).as_quat()
    rel_pose = [dx,dy,dz]
    for q in dquat: 
        rel_pose.append(q)
    return rel_pose

def cam_extrinsics(img):
    R = qvec2rotmat(img.qvec)
    t = img.tvec.reshape(3,-1)
    #print(f"R: {R} t: {t}")
    R_t = np.concatenate((R,t), axis = 1)
    #R_t = np.vstack([np.array([0,0,0,1]), R_t])
    R_t = np.vstack([R_t, np.array([0,0,0,1])])
    return R_t    #  4 * 4 matrix


def display_rel_poses(sparse_dir: Path):
    print("inside display_rel_poses!")
    #sparse_dir = ref_dir_locked 
    print(f"sparse_dir: {sparse_dir.as_posix()}")
    sparse_images = sparse_dir / "images.bin"
    sparse_points3D = sparse_dir / "points3D.bin"
    sparse_cameras = sparse_dir / "cameras.bin"

    sparse_img_dict = read_images_binary(sparse_images)
    print(f"{len(sparse_img_dict.keys())} ==> {sparse_img_dict.keys()}")
    print(f"min_key: {min(sparse_img_dict.keys())} mx_key: {max(sparse_img_dict.keys())}")

    sparse_cam_dict = read_cameras_binary(sparse_cameras)
    for id,cam in sparse_cam_dict.items():
        print(f"cam_params: {cam.params}")

    sorted_keys = sorted(sparse_img_dict.keys())
    for key in sorted_keys: 
        print(f"{key}: {sparse_img_dict[key].name}")

    rel_poses = []
    num_images = len(sparse_img_dict.keys())
    for idx in range(1, num_images // 2 + 1):
        left_img = sparse_img_dict[idx]
        right_img = sparse_img_dict[idx + num_images//2]
        #print(f"left_img_name: {left_img.name} right_img_name: {right_img.name}")
        e_lw = cam_extrinsics(left_img)  #left camera pose w.r.t. world
        e_rw = cam_extrinsics(right_img) #right camera pose w.r.t world
        e_rl = calculate_relative_pose(e_lw, e_rw)
        rel_poses.append(e_rl)


    import pandas as pd
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.notebook_repr_html', True)
    df = pd.DataFrame(rel_poses, columns=['dx', 'dy', 'dz', 'qx' , 'qy', 'qz' , 'qw'])
    #df.style
    print(tabulate(df, headers='keys', tablefmt='psql'))


if __name__ == "__main__":

    #print("Inside main funciton!")
    sparse_without_rba= Path("../../sparse-reconstruction/output/ref_locked")
    sparse_with_rba= Path("../output/")

    display_rel_poses(sparse_without_rba)
    display_rel_poses(sparse_with_rba)