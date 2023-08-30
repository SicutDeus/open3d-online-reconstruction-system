import os
import open3d as o3d
from reconstruction_system.utils import write_info_about_frame, calculate_execution_time
from reconstruction_system.config import config


def make_point_cloud_for_rgbd_image(rgbd_image):
    '''
    Создание поинт клауда для ргбд изображения
    :param rgbd_image: ргбд изображение
    :return: поинт клауд, полученный из ргб изображения
    '''
    return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, config['intrinsic'], config['extrinsic'])


@calculate_execution_time
def run(rgb_frame, depth_frame, image_index):
    '''
    Создает директорию и поинт клауд для текущего изображения
    :param rgb_frame: ргб кадр
    :param depth_frame: кадр глубины
    :param image_index: текущий индекс изображения
    :return: None
    '''
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_frame, depth_frame)
    pcd = make_point_cloud_for_rgbd_image(rgbd_image)
    os.mkdir(os.path.join(config['temp_dir'], f'{image_index}'))
    write_info_about_frame(image_index, pcd, rgb_frame, depth_frame)
