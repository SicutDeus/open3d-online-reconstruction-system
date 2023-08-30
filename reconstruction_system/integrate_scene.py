import os
import open3d as o3d
from reconstruction_system.utils import calculate_execution_time, write_pcd_size
from reconstruction_system.config import config
from reconstruction_system.utils import make_path_into_temp_dir

def add_pcd_to_scene():
    '''
    Добавление поинт клауда в сцену
    :return: None
    '''
    if len(os.listdir(config['temp_dir'])) > 1:
        dir = os.listdir(config['temp_dir'])[-1]
        pose_graph = o3d.io.read_pose_graph(make_path_into_temp_dir(config['scene_path'], dir) +
                                            f'\\{config["default_optimized_pose_graph_name"] % int(dir)}')

        rgb = o3d.io.read_image(os.path.join(config['temp_dir'], dir, config["default_rgb_frame_name"] % int(dir)))
        depth = o3d.io.read_image(os.path.join(config['temp_dir'], dir, config["default_depth_frame_name"] % int(dir)))

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, config['intrinsic'],
                                                             extrinsic=config['extrinsic'])
        if len(pose_graph.nodes) > 0:
            pcd.transform(pose_graph.nodes[int(dir)].pose)
        config['optimized_result_pcd'] += pcd.voxel_down_sample(voxel_size=config['final_optimization_voxel_size'])
        o3d.io.write_point_cloud(make_path_into_temp_dir(config['scene_path'], dir) +
                                 f'\\{config["default_optimized_result_pcd_name"] % int(dir)}', config['optimized_result_pcd'])
        write_pcd_size(make_path_into_temp_dir(config['scene_path'], dir) +
                                 f'\\{config["default_optimized_result_pcd_name"] % int(dir)}')

@calculate_execution_time
def run():
    add_pcd_to_scene()

