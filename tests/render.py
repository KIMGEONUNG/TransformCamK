import numpy as np
from glob import glob
import os
from PIL import Image
import open3d as o3d
import pickle
import sys

sys.path.append('.')
from utils import resize_intrinsic_param, crop_intrinsic_param

MAX = 65504.0
LEN = 512


def sort_by_d(
    d: np.ndarray,
    coords: np.ndarray,
    colors: np.ndarray,
    indices: np.ndarray,
):
    # Calculate the indices that would sort the 'd' array
    sort_indices = np.argsort(
        d[0], axis=0)[::-1]  # This reverses the array for descending order

    # Use these indices to sort 'coords' and 'colors'
    sorted_coords = coords[:, sort_indices]
    sorted_colors = colors[:, sort_indices]

    sorted_indices = indices[sort_indices]

    return sorted_coords, sorted_colors, sorted_indices


def mk_image(coords, colors, W, H):

    # print(coords.shape) --> (3,N)
    indices = np.arange(coords.shape[1])

    if coords.shape[0] != 3:
        coords = coords.T
        # assert coords.shape[0] == 3

    if colors.shape[0] != 3:
        colors = colors.T
        # assert colors.shape[0] == 3

    d = coords[2:, :]
    coords = coords[:2, :] / d
    coords = np.round(coords).astype(np.int32)  # (2, N)
    coords, colors, indices = sort_by_d(d, coords, colors, indices)

    image = np.zeros((H, W, 3))
    visualized_indices = np.full((H, W), -1, dtype=int)

    image[coords[1, :], coords[0, :]] = colors.T  # image use Y,X order
    visualized_indices[coords[1, :], coords[0, :]] = indices

    # image = image[::-1, :, :]
    # visualized_indices = visualized_indices[::-1, :]

    return image, visualized_indices


def cal_valid(coords, W, H):
    d = coords[2]
    valid_idx = np.where(
        np.logical_and.reduce((
            d > 0,
            coords[0] / d >= 0,
            coords[1] / d >= 0,
            coords[0] / d <= W - 1,
            coords[1] / d <= H - 1,
        )))[0]
    return valid_idx


def main():
    pcd = o3d.io.read_point_cloud('tests/pcd_color.ply')

    revision = np.eye(3)
    revision[2, 2] = -1  # z-flip

    path = 'tests/pose.npy'
    with open(path, 'rb') as f:
        camera = pickle.load(f)
    points = np.array(pcd.points).transpose()
    colors = np.array(pcd.colors).transpose()

    W, H = 512, 512
    K = camera['intrinsic']
    fx, fy = 800, 800
    cx, cy = 256, 256


    K = np.array([
        [fx, 0., cx],
        [0., fy, cy],
        [0., 0., 1.],
    ])

    RT = camera['extrinsic']
    R = RT[:3, :3]
    T = RT[:3, 3:]

    # new_pts_coord_world2 = R_inv @ (revision @ K_inv @ homogen - T)
    # pixel_coord = K @ np.linalg.inv(revision) @ (R @ points + T)
    pixel_coord = K @ revision @ (R @ points + T)

    # FILTER BY FRUSTUM
    valid_idx = cal_valid(pixel_coord, W, H)
    points = points[:, valid_idx]
    pixel_coord = pixel_coord[:, valid_idx]
    colors = colors[:, valid_idx]  # (3, N)

    image, vis_indices = mk_image(pixel_coord, colors, W, H)
    image = (np.clip(image, 0, 1) * 255).astype('uint8')
    image = Image.fromarray(image)
    image.show()


if __name__ == "__main__":
    main()
