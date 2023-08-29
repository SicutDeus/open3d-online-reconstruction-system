import open3d as o3d
import numpy as np
from config import *
import matplotlib.pyplot as plt


def plot_timestamps(timestamps, name):
    plt.figure(num=name)
    xstamps = [i for i in range(0, len(timestamps))]
    plt.plot(xstamps, timestamps, 'b', label='line one', linewidth=5)
    plt.show()


def get_current_frame_from_dir(current_index):
    color_frame_name = f'{current_index:05}.jpg'
    depth_frame_name = f'{current_index:05}.png'
    color_img = o3d.io.read_image(color_frames_path + '\\' + color_frame_name)
    depth_img = o3d.io.read_image(depth_frames_path + '\\' + depth_frame_name)
    return color_img, depth_img

def make_rgbd_image_from_color_and_depth_frames(current_index):
    color_raw, depth_raw = get_current_frame_from_dir(current_index)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)
    return rgbd_image

def add_grid_on_vis(vis, pcd):
    bbox = pcd.get_axis_aligned_bounding_box()
    min_bound = bbox.min_bound
    max_bound = bbox.max_bound
    voxel_size = 0.5
    for y in np.arange(min_bound[1], max_bound[1], voxel_size):
        for z in np.arange(min_bound[2], max_bound[2], voxel_size):
            points = np.array([[min_bound[0], y, z], [max_bound[0], y, z]])
            lines = o3d.geometry.LineSet()
            lines.points = o3d.utility.Vector3dVector(points)
            lines.lines = o3d.utility.Vector2iVector([[0, 1]])
            lines.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # Red color
            vis.add_geometry(lines)

    # Create grid lines along the Y-axis
    for x in np.arange(min_bound[0], max_bound[0], voxel_size):
        for z in np.arange(min_bound[2], max_bound[2], voxel_size):
            points = np.array([[x, min_bound[1], z], [x, max_bound[1], z]])
            lines = o3d.geometry.LineSet()
            lines.points = o3d.utility.Vector3dVector(points)
            lines.lines = o3d.utility.Vector2iVector([[0, 1]])
            lines.colors = o3d.utility.Vector3dVector([[0, 1, 0]])  # Green color
            vis.add_geometry(lines)

    # Create grid lines along the Z-axis
    for x in np.arange(min_bound[0], max_bound[0], voxel_size):
        for y in np.arange(min_bound[1], max_bound[1], voxel_size):
            points = np.array([[x, y, min_bound[2]], [x, y, max_bound[2]]])
            lines = o3d.geometry.LineSet()
            lines.points = o3d.utility.Vector3dVector(points)
            lines.lines = o3d.utility.Vector2iVector([[0, 1]])
            lines.colors = o3d.utility.Vector3dVector([[0, 0, 1]])  # Blue color
            vis.add_geometry(lines)

def rotate_pcd(pcd):
    axis = (1, 0, 0)
    angle = np.pi
    axis = np.array(axis)
    R = pcd.get_rotation_matrix_from_axis_angle(axis * angle)
    pcd.rotate(R, center=(0, 0, 0))
