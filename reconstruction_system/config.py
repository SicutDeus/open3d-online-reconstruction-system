import open3d as o3d
import numpy as np

pose_graph = o3d.pipelines.registration.PoseGraph()
pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.identity(4)))
default_filling = '%05d'
config = {
    'intrinsic': o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault),

    'extrinsic': np.eye(4),
    'odometry': np.identity(4),
    'init_transformation': np.identity(4),

    'voxel_size': 0.05,
    'final_optimization_voxel_size': 0.1,
    'depth_diff_max': 0.07,
    'max_correspondence_distance': 0.05 * 1.4,
    'preference_loop_closure': 5.0,
    'edge_prune_threshold': 0.25,
    'tsdf_cubic_size': 3.0,
    'max_iter': 50,
    'reference_node': 0,

    'default_filling': default_filling,
    'scene_path': 'scene\\',
    'images_dir': 'E:\\open3dData',
    'temp_dir': 'reconstruction_system\\temp\\',
    'default_frame_point_cloud_name': f'frame_point_cloud_{default_filling}.ply',
    'default_rgb_frame_name': f'rgb_frame_{default_filling}.jpg',
    'default_depth_frame_name': f'depth_frame_{default_filling}.png',
    'default_pose_graph_name': f'pose_graph_{default_filling}.json',
    'default_optimized_pose_graph_name': f'optimized_pose_graph_{default_filling}.json',
    'default_result_pcd_name': f'result_pcd_{default_filling}.ply',
    'default_optimized_result_pcd_name': f'optimized_result_pcd_{default_filling}.ply',

    'result_pcd': o3d.geometry.PointCloud(),
    'optimized_result_pcd': o3d.geometry.PointCloud(),
    'pose_graph': pose_graph,

    'weights_log_filename': 'reconstruction_system\\_weights.txt',
    'times_log_filename': 'reconstruction_system\\_times.txt',
    'keypoints_log_filename': 'reconstruction_system\\_keypoints.txt',

    'icp_method': 'color',
    'odometry_case': True,

    'red_color' : [[1, 0, 0]],
    'green_color': [[0, 1, 0]],
    'blue_color': [[0, 0, 1]],
}