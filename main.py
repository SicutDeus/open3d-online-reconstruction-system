import numpy as np
from PIL import Image
import os
import cv2
import shutil
import random
import open3d as o3d
from open3d.cpu.pybind.camera import PinholeCameraIntrinsic

import config
from config import *
import matplotlib.pyplot as plt
from utils import *
import time
import datetime

def visualize_rgbd_image(rgbd_image):
    plt.subplot(1, 2, 1)
    plt.title('Grayscale image')
    plt.imshow(rgbd_image.color)
    plt.subplot(1, 2, 2)
    plt.title('Depth image')
    plt.imshow(rgbd_image.depth)
    plt.grid(axis='x')
    plt.show()

def make_video_from_frames(path, shape, video_name, should_contains = ''):
    out = cv2.VideoWriter(video_name + '.avi', cv2.VideoWriter_fourcc(*'DIVX'), 60, shape)
    for file in os.listdir(path):
        img = cv2.imread(path + '\\' + file)
        out.write(img)


def make_point_cloud_from_rgbd_image(rgbd_image):
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return pcd

def get_pcd_from_color_and_depth_frames(frame_index):
    initial_rgbd_image = make_rgbd_image_from_color_and_depth_frames(frame_index)
    return make_point_cloud_from_rgbd_image(initial_rgbd_image)


def transform_point_cloud_to_fit_another(source_pcd, target_pcd):
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, max_correspondence_distance, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    source_pcd.transform(reg_p2p.transformation)
    return source_pcd


def downsample_pcd(pcd):
    return pcd.voxel_down_sample(voxel_size=default_voxel_size)


def filter_points_in_pcd(pcd):
    labels = np.array(pcd.cluster_dbscan(eps=0.02, min_points=10))
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[labels != -1])
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return pcd


def make_scene_v2():
    times = [0, 0, 0, 0]

    from reconstruction_system.initialize_config import initialize_config, dataset_loader

    config = dataset_loader('')
    config['debug_mode'] = False

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=config["tsdf_cubic_size"] / 512.0,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    data_dirs = 'E:\\all_data_dirs'
    ind = 0
    for _ in os.listdir(data_dirs):
        config['path_dataset'] = f'{data_dirs}\\{ind}'
        config['color_images'] = f'{data_dirs}\\{ind}\\rgb'
        config['depth_images'] = f'{data_dirs}\\{ind}\\depth'

        ind += 1
        start_time = time.time()

        from reconstruction_system import make_fragments
        make_fragments.run(config)
        times[0] = time.time() - start_time
        start_time = time.time()

        from reconstruction_system import register_fragments
        register_fragments.run(config)
        times[1] = time.time() - start_time
        start_time = time.time()
        from reconstruction_system import refine_registration
        refine_registration.run(config)
        times[2] = time.time() - start_time
        start_time = time.time()
        from reconstruction_system import integrate_scene
        volume = integrate_scene.run(config, volume)
        times[3] = time.time() - start_time

        print("====================================")
        print("Elapsed time (in h:m:s)")
        print("====================================")
        print("- Making fragments    %s" % datetime.timedelta(seconds=times[0]))
        print("- Register fragments  %s" % datetime.timedelta(seconds=times[1]))
        print("- Refine registration %s" % datetime.timedelta(seconds=times[2]))
        print("- Integrate frames    %s" % datetime.timedelta(seconds=times[3]))
        print("- Total               %s" % datetime.timedelta(seconds=sum(times)))
    print(config['path_dataset'])

def visualize_point_cloud(pcd):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    add_grid_on_vis(vis, pcd)
    vis.run()

def rotate_pcd(pcd):
    axis = (1, 0, 0)
    angle = np.pi
    axis = np.array(axis)
    R = pcd.get_rotation_matrix_from_axis_angle(axis * angle)
    pcd.rotate(R, center=(0, 0, 0))


def check():
    pcd = o3d.io.read_point_cloud('E:\\all_data_dirs\\9\\scene\\integrated.ply')
    rotate_pcd(pcd)
    visualize_point_cloud(pcd)

def pcd_to_jpg(pcd):
    rotate_pcd(pcd)
    pcd.transform(trans_init)
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image('kek.png')
    vis.destroy_window()


def diff_data():
    import shutil
    initial_color_path = 'E:\\open3dData\\rgb\\'
    initial_depth_path = 'E:\\open3dData\\depth\\'
    data_main_dir = 'E:\\all_data_dirs'
    index = 0
    for i in range(0, 50, 5):
        os.mkdir(data_main_dir+f'\\{index}')
        os.mkdir(data_main_dir+f'\\{index}\\rgb')
        os.mkdir(data_main_dir+f'\\{index}\\depth')
        color_files = [initial_color_path + f'{i+j:05}.jpg' for j in range(5)]
        depth_files = [initial_depth_path + f'{i+j:05}.png' for j in range(5)]
        for color_file in color_files:
            shutil.copy(color_file, data_main_dir + f'\\{index}\\rgb')
        for depth_file in depth_files:
            shutil.copy(depth_file, data_main_dir + f'\\{index}\\depth')
        index += 1

def get_random_color_and_depth_files_from_dir(dir):
    total_image_count = len(os.listdir(f'{dir}\\rgb'))
    rnd = random.randint(0,total_image_count)
    image_index = f'{rnd:05}'
    rgb1 = o3d.io.read_image(f'{dir}\\rgb\\{image_index}.jpg')
    depth1 = o3d.io.read_image(f'{dir}\\depth\\{image_index}.png')
    return rgb1, depth1


from handmade_reconstruction_system import process_single_rgbd_image
from handmade_reconstruction_system import utils
from handmade_reconstruction_system.config import config
from handmade_reconstruction_system import register_frames
from handmade_reconstruction_system import integrate_scene
def check_size(path):
    import win32com.client as com

    fso = com.Dispatch("Scripting.FileSystemObject")
    folder = fso.GetFolder(path)
    mb = 1024 * 1024.0
    print ("Общий размер папки temp {} МБ".format(folder.Size / mb))

def test_handmade():
    image_index = 0
    utils.make_clean_folder(config['temp_dir'])
    open('weights.txt', 'w').close()
    open('times.txt', 'w').close()
    open('keypoints.txt', 'w').close()
    for _ in os.listdir(f'{config["images_dir"]}\\rgb'):
        rgb_frame, depth_frame = utils.get_depth_and_color_frames_from_dir(config['images_dir'], image_index)

        start_time = time.time()
        process_single_rgbd_image.run(rgb_frame, depth_frame, image_index)
        time_process = round(time.time()-start_time, 3)

        start_time = time.time()
        register_frames.run(image_index)
        time_register = round(time.time()-start_time, 3)

        start_time = time.time()
        integrate_scene.test()
        time_integrate = round(time.time()-start_time, 3)

        with open('times.txt', 'a') as f:
            f.write(f'{time_process} {time_register} {time_integrate}\n')

        if image_index > 1:
            utils.clear_previous_temp_folder(image_index)

        image_index += 1
        print(image_index)
        if image_index == 250:
            break


    from handmade_reconstruction_system.utils import visualize_point_cloud
    downsample_pcd(config['optimized_result_pcd'])
    visualize_point_cloud(config['optimized_result_pcd'])

def vis_pcd():
    path = 'E:\\Open3DProject\\temp\\165\\scene\\optimized_result_pcd_00165.ply'
    pcd = o3d.io.read_point_cloud(path)
    visualize_point_cloud(pcd)

def plot_results():
    weigths_path = 'weights.txt'
    times_path = 'times.txt'
    keypoints_path = 'keypoints.txt'
    weights_stamps = []
    proccess_timestamps = []
    register_timestamps = []
    integrate_timestamps = []
    keypoints_stamps = []
    with open(weigths_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            weights_stamps.append(line.split(' ')[0])
        plot_timestamps(weights_stamps, 'weight of point cloud')
    with open(keypoints_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            keypoints_stamps.append(line)
        plot_timestamps(keypoints_stamps, 'Number of keypoints')
    with open(times_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            splited = line.split(' ')
            proccess_timestamps.append(splited[0])
            register_timestamps.append(splited[1])
            integrate_timestamps.append(splited[2])
        plot_timestamps(proccess_timestamps, 'proccess frame')
        plot_timestamps(register_timestamps, 'register frames')
        plot_timestamps(integrate_timestamps, 'integrate frames')

if __name__ == '__main__':
    plot_results()
    #pcd = make_scene_from_frames()
    #visualize_point_cloud(pcd)
    #make_scene_v2()
    #test_handmade()
    #vis_pcd()
    #test_handmade()
    #test_rgbd_to_jpg()
    #diff_data()
    #diff_data()
    #check()
    #make_video_from_frames(color_frames_path,(640, 480),'rgb_video')
    #rgbd_image = make_rgbd_image_from_color_and_depth_frames(color_frames_path,depth_frames_path, 2426)
    #visualize_rgbd_image(rgbd_image)
    #pcd = make_point_cloud_from_rgbd_image(rgbd_image)
    #visualize_point_cloud(pcd)
    #pcd = make_scene_from_frames()
    #visualize_point_clo