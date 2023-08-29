import open3d as o3d
import os
from handmade_reconstruction_system.utils import *
from handmade_reconstruction_system.refine_registration import multiscale_icp
from handmade_reconstruction_system.optimize_pose_graph import *

def preprocess_point_cloud(pcd):
    voxel_size = config["voxel_size"]
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0,
                                             max_nn=30))
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0,
                                             max_nn=100))
    return pcd_down, pcd_fpfh


def register_point_cloud_fpfh(source, target, source_fpfh, target_fpfh):
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    distance_threshold = config["voxel_size"] * 1.4
    if config["global_registration"] == "fgr":
        result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            source, target, source_fpfh, target_fpfh,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=distance_threshold))
    if config["global_registration"] == "ransac":
        # Fallback to preset parameters that works better
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source, target, source_fpfh, target_fpfh, False, distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(
                False), 4,
            [
                o3d.pipelines.registration.
                CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(
                1000000, 0.999))
    if result.transformation.trace() == 4.0:
        return False, np.identity(4), np.zeros((6, 6))
    information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, distance_threshold, result.transformation)
    if information[5, 5] / min(len(source.points), len(target.points)) < 0.3:
        return False, np.identity(4), np.zeros((6, 6))
    return True, result.transformation, information


def compute_initial_registration(source_down, target_down, source_fpfh, target_fpfh):
    success = True
    if config['odometry_case']:  # odometry case
        (transformation, information) = multiscale_icp(source_down, target_down)
    else:  # loop closure case
        (success, transformation,
         information) = register_point_cloud_fpfh(source_down, target_down,
                                                  source_fpfh, target_fpfh)
    return success, transformation, information


def update_posegraph_for_scene(image_index, transformation, information, pose_graph):
    odometry = np.dot(transformation, config['odometry'])
    if config['odometry_case']:  # odometry case
        odometry_inv = np.linalg.inv(odometry)
        pose_graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(odometry_inv))
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(image_index-1,
                                                     image_index,
                                                     transformation,
                                                     information,
                                                     uncertain=False))
    else:  # loop closure case
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(image_index-1,
                                                     image_index,
                                                     transformation,
                                                     information,
                                                     uncertain=True))

    return odometry, pose_graph

def make_posegraph_for_scene(source_pcd, target_pcd, image_index):
    success, transformation, information = register_point_cloud_pair(source_pcd, target_pcd)
    odometry, pose_graph = update_posegraph_for_scene(image_index, transformation, information, config['pose_graph'])
    o3d.io.write_pose_graph(make_path_into_temp_dir(config['scene_path'], image_index) + config['default_pose_graph_name'] % image_index,pose_graph)

def register_point_cloud_pair(source_pcd, target_pcd):
    (source_down, source_fpfh) = preprocess_point_cloud(source_pcd)
    (target_down, target_fpfh) = preprocess_point_cloud(target_pcd)
    (success, transformation, information) = \
        compute_initial_registration(source_down, target_down,
                                     source_fpfh, target_fpfh)
    if not success:
        return False, np.identity(4), np.identity(6)
    return True, transformation, information


def run(image_index):
    make_clean_folder(make_path_into_temp_dir(config['scene_path'], image_index))
    if image_index > 0:
        source_pcd = o3d.io.read_point_cloud(
            make_path_into_temp_dir(config['default_frame_point_cloud_name'], image_index - 1))
        target_pcd = o3d.io.read_point_cloud(
            make_path_into_temp_dir(config['default_frame_point_cloud_name'], image_index))
        make_posegraph_for_scene(source_pcd, target_pcd, image_index)
        optimize_posegraph_for_scene(image_index)
