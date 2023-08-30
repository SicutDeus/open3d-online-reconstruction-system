import open3d as o3d
from reconstruction_system.config import config
from reconstruction_system.utils import make_path_into_temp_dir

def run_posegraph_optimization(pose_graph_name, pose_graph_optimized_name):
    '''
    Запуск оптимизации графа поз
    :param pose_graph_name: граф поз
    :param pose_graph_optimized_name: название для оптимизированного графа пощ
    :return: None
    '''
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    method = o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt()
    criteria = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(
    )
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        config['max_correspondence_distance'],
        config['edge_prune_threshold'],
        config['preference_loop_closure'],
        config['reference_node']
    )
    pose_graph = o3d.io.read_pose_graph(pose_graph_name)
    o3d.pipelines.registration.global_optimization(pose_graph, method, criteria,
                                                   option)
    o3d.io.write_pose_graph(pose_graph_optimized_name, pose_graph)
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

def optimize_posegraph_for_scene(image_index):
    '''
    Оптимизация графа поз для сцены
    :param image_index: индекс изображения
    :return: None
    '''
    pose_graph_path = make_path_into_temp_dir(config['scene_path'], image_index) + config['default_pose_graph_name'] % image_index
    pose_graph_optimized_name = make_path_into_temp_dir(config['scene_path'], image_index) + config['default_optimized_pose_graph_name'] % image_index
    run_posegraph_optimization(pose_graph_path, pose_graph_optimized_name)
