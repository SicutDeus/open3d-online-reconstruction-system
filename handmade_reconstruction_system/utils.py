import open3d as o3d
import numpy as np
import os
import shutil
from handmade_reconstruction_system.config import config
import re


def get_depth_and_color_frames_from_dir(dir, index):
    image_index = f'{index:05}'
    rgb = o3d.io.read_image(f'{dir}\\rgb\\{image_index}.jpg')
    depth = o3d.io.read_image(f'{dir}\\depth\\{image_index}.png')
    return rgb, depth


def rotate_pcd(pcd):
    axis = (1, 0, 0)
    angle = np.pi
    axis = np.array(axis)
    R = pcd.get_rotation_matrix_from_axis_angle(axis * angle)
    pcd.rotate(R, center=(0, 0, 0))


def make_clean_folder(path_folder):
    if not os.path.exists(path_folder):
        os.mkdir(path_folder)
    else:
        shutil.rmtree(path_folder)
        os.mkdir(path_folder)


def clear_previous_temp_folder(image_index):
    shutil.rmtree(f'{config["temp_dir"]}\\{image_index - 2}')


def make_path_into_temp_dir(inner_path, image_index = -1):
    if inner_path.count(config['default_filling']) > 0:
        return os.path.join(config['temp_dir'] + f'\\{image_index}',
                            inner_path % image_index)
    return os.path.join(config['temp_dir'] + f'\\{image_index}', inner_path)


def write_info_about_frame(image_index, pcd, rgb_frame, depth_frame):
    o3d.io.write_point_cloud(make_path_into_temp_dir(config['default_frame_point_cloud_name'], image_index), pcd)
    o3d.io.write_image(make_path_into_temp_dir(config['default_rgb_frame_name'], image_index), rgb_frame)
    o3d.io.write_image(make_path_into_temp_dir(config['default_depth_frame_name'], image_index), depth_frame)


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
            lines.colors = o3d.utility.Vector3dVector(config['red_color'])  # Red color
            vis.add_geometry(lines)

    # Create grid lines along the Y-axis
    for x in np.arange(min_bound[0], max_bound[0], voxel_size):
        for z in np.arange(min_bound[2], max_bound[2], voxel_size):
            points = np.array([[x, min_bound[1], z], [x, max_bound[1], z]])
            lines = o3d.geometry.LineSet()
            lines.points = o3d.utility.Vector3dVector(points)
            lines.lines = o3d.utility.Vector2iVector([[0, 1]])
            lines.colors = o3d.utility.Vector3dVector(config['green_color'])
            vis.add_geometry(lines)

    # Create grid lines along the Z-axis
    for x in np.arange(min_bound[0], max_bound[0], voxel_size):
        for y in np.arange(min_bound[1], max_bound[1], voxel_size):
            points = np.array([[x, y, min_bound[2]], [x, y, max_bound[2]]])
            lines = o3d.geometry.LineSet()
            lines.points = o3d.utility.Vector3dVector(points)
            lines.lines = o3d.utility.Vector2iVector([[0, 1]])
            lines.colors = o3d.utility.Vector3dVector(config['blue_color'])
            vis.add_geometry(lines)


def visualize_point_cloud(pcd):
    rotate_pcd(pcd)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
