import os

from handmade_reconstruction_system.config import config
import open3d as o3d
from handmade_reconstruction_system.utils import *
from handmade_reconstruction_system.config import config

def visualize_point_cloud_test(pcd):
    vis = o3d.visualization.Visualizer()
    rotate_pcd(pcd)
    vis.create_window()
    vis.add_geometry(pcd)
    add_grid_on_vis(vis, pcd)
    vis.run()
    vis.destroy_window()



def make_point_cloud_for_rgbd_image(rgbd_image):
    return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, config['intrinsic'], config['extrinsic'])

def run(rgb_frame, depth_frame, image_index):
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_frame, depth_frame)
    pcd = make_point_cloud_for_rgbd_image(rgbd_image)
    os.mkdir(os.path.join(config['temp_dir'], f'{image_index}'))
    write_info_about_frame(image_index, pcd, rgb_frame, depth_frame)
