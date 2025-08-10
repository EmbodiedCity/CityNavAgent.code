import json
import torch
import time
import airsim
import pickle
import numpy as np
import cv2
import os

from PIL import Image
from tqdm import tqdm
from typing import List

from src.llm.query_llm import OpenAI_LLM_v2
from src.llm.prompt_builder import landmark_caption_prompt_builder, \
    route_planning_prompt_builder, parse_viewpoint_response_v2

from airsim_plugin.airsim_settings import ObservationDirections

from utils.env_utils import getPoseAfterMakeActions, get_pano_observations, get_front_observations
from utils.maps import build_semantic_map, visualize_semantic_point_cloud, update_camera_pose,\
    convert_global_pc, statistical_filter, find_closest_node, compute_shortest_path
from utils.utils import calculate_movement_steps, calculate_movement_steps_mem, append_text_to_image

from external.Grounded_Sam_Lite.groundingdino.util.inference import load_model, predict
from external.Grounded_Sam_Lite.grounded_sam_api import GroundedSam
import external.Grounded_Sam_Lite.groundingdino.datasets.transforms as T

from external.lm_nav.navigation_graph import NavigationGraph
from external.lm_nav import pipeline

from scipy.spatial.transform import Rotation as R
from evaluator.nav_evaluator import CityNavEvaluator

from airsim_plugin.AirVLNSimulatorClientTool import AirVLNSimulatorClientTool


def convert_airsim_pose(pose):
    assert len(pose) == 7, "The length of input pose must be 7"
    formatted_airsim_pose = airsim.Pose(
        position_val=airsim.Vector3r(
            pose[0],
            pose[1],
            pose[2]
        ),
        orientation_val=airsim.Quaternionr(
            x_val=pose[3],
            y_val=pose[4],
            z_val=pose[5],
            w_val=pose[6],
        )
    )
    return formatted_airsim_pose


def semantic_map_grounding(
        vlm,
        rgb_imgs: List[np.ndarray],
        dep_imgs: List[np.ndarray],
        cur_pose: np.ndarray,
        caption: str,
        visulization=False
) -> (np.ndarray, np.ndarray):
    transform = T.Compose(
        [
            # T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    pcs = []
    lms = []
    lds = []

    merged_pc = None
    merged_lm = None
    merged_ld = {"None": 0}

    for i in range(len(rgb_imgs)):
        image = rgb_imgs[i]
        depth = dep_imgs[i].squeeze()

        h, w, _ = image.shape
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image, _ = transform(image, None)

        boxes, logits, phrases = predict(
            model=vlm,
            image=image,
            caption=caption,
            box_threshold=0.35,
            text_threshold=0.3
        )

        bboxes = boxes * torch.Tensor([w, h, w, h])
        bboxes = box_convert(bboxes, in_fmt='cxcywh', out_fmt='xyxy').numpy()

        rot_rad = (i-2)*np.pi/4
        new_cam_pose = update_camera_pose(cur_pose, rot_rad)
        pc, lm, ld = build_semantic_map(depth, 90, new_cam_pose, bboxes, phrases)
        # visualize_semantic_point_cloud(pc, lm)

        pcs.append(pc)
        lms.append(lm)
        lds.append(ld)

        # uniform class
        for cls in ld:
            if cls not in merged_ld:
                new_l = len(merged_ld)
                merged_ld[cls] = new_l

    # uniform label
    for i in range(len(lms)):
        lm = lms[i]
        ld = lds[i]

        rep = {}
        for cls in ld:
            if cls in merged_ld:
                rep[ld[cls]] = merged_ld[cls]

        v_rep = np.vectorize(rep.get)
        lm = v_rep(lm)

        lms[i] = lm

    merged_pc = np.concatenate(pcs, axis=0)
    merged_lm = np.concatenate(lms, axis=0)

    if visulization:
        visualize_semantic_point_cloud(merged_pc, merged_lm)

    return merged_pc, merged_lm, merged_ld


def explore_pipeline_by_dino(
        curr_pose,
        llm, vlm,
        image_path: List[str],
        rgb_imgs: List[np.ndarray],
        dep_imgs: List[np.ndarray],
        navigation_instruction: str,
        scene_objects: List[str], landmarks_route: List[str]
):
    # image caption
    time1 = time.time()
    observed_obj = set()
    caption_prompt = landmark_caption_prompt_builder(scene_objects)
    for img_p in image_path:
        caption_res_str = llm.query_api(caption_prompt, image_path=img_p, show_response=False)
        obs_strs = caption_res_str.split(".")
        for o in obs_strs:
            if o.strip(" ") not in observed_obj:
                observed_obj.add(o)
    obs_obj_str = ".".join(list(observed_obj))
    time1 = time.time()
    route_predict_prompt = route_planning_prompt_builder(obs_obj_str, navigation_instruction, landmarks_route[0])
    route_predicted = llm.query_api(route_predict_prompt, show_response=False)

    # print(f"query time: {time.time()-time1}")
    print("route_predict_prompt: ", route_predict_prompt)
    print("route_predicted: ", route_predicted)

    # route point prediction
    cur_pos = np.array(list(curr_pose.position))
    cur_ori = np.array([curr_pose.orientation.x_val, curr_pose.orientation.y_val, curr_pose.orientation.z_val, curr_pose.orientation.w_val])
    cur_pose= np.concatenate([cur_pos, cur_ori], axis=0)

    # image grounding
    time1 = time.time()
    semantic_map, semantic_label, semantic_cls = \
        semantic_map_grounding(vlm, rgb_imgs, dep_imgs, cur_pose, route_predicted, visulization=False)

    # convert semantic map to airsim coordinate
    cam2ego_rot = np.array([[0, 0, 1.0],
                        [1.0, 0, 0],
                        [0, 1.0, 0]])
    ego2world_rot = R.from_quat(list(curr_pose.orientation)).as_matrix()
    coord_rot   = ego2world_rot.dot(cam2ego_rot)
    coord_trans = np.array(list(curr_pose.position)).reshape(-1, 1)
    semantic_map = (coord_rot.dot(semantic_map.T) + coord_trans).T      # n*3 in world coord system
    # print(f"semantic map construction time: {time.time() - time1}")

    time1 = time.time()

    routes = route_predicted.split(".")
    if routes[0].strip(" ") not in semantic_cls:
        route_coords = cur_pos
        # todo: ramdom walk
        pass
    else:
        next_route_label = semantic_cls[routes[0].strip(" ")]
        route_semantic_map = semantic_map[semantic_label.ravel()==next_route_label]
        route_coords = np.mean(route_semantic_map, axis=0)     # (3,)
        z_coord = cur_pos[2]
        alpha = 0.6
        route_coords = alpha * route_coords + (1-alpha) * cur_pos
        # route_coords[2] = z_coord

        if np.any(np.isnan(route_coords)):
            route_coords = cur_pos

        dir_vec_2d = route_coords[:2] - cur_pos[:2]
        if route_coords[2] > -2:
            route_coords[2] = 2

    time1 = time.time()
    # low level path
    rel_trans = route_coords - cur_pos
    yaw = np.arctan2(rel_trans[1], rel_trans[0])
    new_quat = R.from_euler('z', yaw, degrees=False).as_quat()

    new_pos = route_coords
    new_pose = convert_airsim_pose(list(new_pos)+list(new_quat))

    # calculate step size
    dist = np.abs(rel_trans)
    step_size = np.abs(np.rad2deg(yaw)) // 15 + dist[2] // 2 + np.sqrt(dist[0]**2+dist[1]**2) // 5

    print(f"low level planning time: {time.time()-time1}")
    print(f"curr pose: {curr_pose}, new pose: {new_pose}, object point: {route_coords}")
    return int(step_size), new_pose


def explore_pipeline_by_sam(
        curr_pose,
        llm, vlm,
        image_path: List[str],
        rgb_imgs: List[np.ndarray],
        dep_imgs: List[np.ndarray],
        obs_poses: List[np.ndarray],
        navigation_instruction: str,
        scene_objects: List[str],
        landmarks_route: List[str],
        next_landmark_idx: int,
):
    obs_viewpoint = ["left", "slightly_left", "front", "slightly_right", "right"]
    viewpoint_img_path = {}
    viewpoint_rgb_imgs = {}
    viewpoint_dep_imgs = {}
    viewpoint_poses = {}
    next_subgoal_found = False
    for k in range(len(obs_viewpoint)):
        viewpoint = obs_viewpoint[k]
        viewpoint_img_path[viewpoint] = image_path[k]
        viewpoint_rgb_imgs[viewpoint] = rgb_imgs[k]
        viewpoint_dep_imgs[viewpoint] = dep_imgs[k]
        viewpoint_poses[viewpoint] = obs_poses[k]

    traversed_landmarks = landmarks_route[:next_landmark_idx]
    route_predict_prompt = route_planning_prompt_builder(navigation_instruction, landmarks_route, traversed_landmarks, landmarks_route[next_landmark_idx])
    route_predicted = llm.query_viewpoint_api(route_predict_prompt, viewpoint_img_path, show_response=False)
    route_predicted_dict = parse_viewpoint_response_v2(route_predicted)

    if route_predicted_dict["is_found"]:
        next_subgoal_found = True

    # build semantic point cloud
    semantic_pc = []
    seg_succ_all = False
    for vp, obj in route_predicted_dict.items():
        if vp == "is_found":
            continue
        rgb_img = viewpoint_rgb_imgs[vp]
        dep_img = viewpoint_dep_imgs[vp].squeeze()
        pose = viewpoint_poses[vp]

        route_mask, seg_succ = vlm.greedy_mask_predict(rgb_img, obj, visualize=False)
        seg_succ_all = seg_succ_all or seg_succ

        if seg_succ:
            print(f"Selected viewpoint, object: {vp}, {obj}")
            part_pc, filter_idx = convert_global_pc(dep_img, 90, pose, route_mask)
            semantic_part_pc = part_pc[filter_idx]
            if len(semantic_part_pc) > 30:
                semantic_part_pc, _ = statistical_filter(semantic_part_pc)
            if len(semantic_part_pc > 0):
                semantic_pc.append(semantic_part_pc)

    if len(semantic_pc) > 0:
        semantic_pc = semantic_pc[0]
    else:
        semantic_pc = np.zeros((1, 3))

    # route point prediction
    cur_pos = np.array([curr_pose.position.x_val, curr_pose.position.y_val, curr_pose.position.z_val])
    cur_ori = np.array([curr_pose.orientation.x_val, curr_pose.orientation.y_val, curr_pose.orientation.z_val, curr_pose.orientation.w_val])
    cur_pose= np.concatenate([cur_pos, cur_ori], axis=0)

    if not seg_succ_all:
        route_coords = cur_pos
    else:
        route_coords = np.mean(semantic_pc, axis=0)
        if not np.any(route_coords):        # if all zeros
            route_coords = cur_pos
        alpha = 0.6
        route_coords = alpha * route_coords + (1-alpha) * cur_pos
        if np.any(np.isnan(route_coords)):
            route_coords = cur_pos

        dir_vec_2d = route_coords[:2] - cur_pos[:2]
        if route_coords[2] > -5:
            route_coords[2] = 5

    time1 = time.time()
    # low level path
    rel_trans = route_coords - cur_pos
    yaw = np.arctan2(rel_trans[1], rel_trans[0])
    new_quat = R.from_euler('z', yaw, degrees=False).as_quat()
    new_pos = route_coords
    new_pose = convert_airsim_pose(list(new_pos)+list(new_quat))

    # calculate step size
    dist = np.abs(rel_trans)
    step_size = np.abs(np.rad2deg(yaw)) // 15 + dist[2] // 2 + np.sqrt(dist[0]**2+dist[1]**2) // 5

    # print(f"low level planning time: {time.time()-time1}")

    return int(step_size), new_pose, next_subgoal_found


def CityNavAgent(scene_id, split, data_dir="./data", max_step_size=200, vlm_name="dino", record=False):
    data_root = os.path.join(data_dir, f"gt_by_env/{env_id}/{split}_landmk.json")
    graph_root = os.path.join(data_dir, f"mem_graphs_pruned/{env_id}/{split}")
    graph_act_root = os.path.join(data_dir, f'mem_graphs/{env_id}.pkl')

    os.makedirs("obs_imgs", exist_ok=True)

    predict_routes = []
    with open(data_root, 'r') as f:
        navi_tasks = json.load(f)['episodes']

    nav_evaluator = CityNavEvaluator()

    # load LLM
    llm = OpenAI_LLM_v2(
        max_tokens=10000,
        model_name="gpt-4o",
        api_key="your api key",
        client_type="openai",
        cache_name="navigation",
        finish_reasons=["stop", "length"],
    )

    if vlm_name == "dino":
        vlm = load_model(
            "external/Grounded_Sam_Lite/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            "external/Grounded_Sam_Lite/weights/groundingdino_swint_ogc.pth"
        )
    elif vlm_name == "sam":
        vlm = GroundedSam(
            dino_checkpoint_path="external/Grounded_Sam_Lite/weights/groundingdino_swint_ogc.pth",
            sam_checkpoint_path="external/Grounded_Sam_Lite/weights/sam_vit_h_4b8939.pth"
        )

    # load env
    machines_info_xxx = [
        {
            'MACHINE_IP': '127.0.0.1',
            'SOCKET_PORT': 30000,
            'MAX_SCENE_NUM': 8,
            'open_scenes': [scene_id],
        },
    ]

    tool = AirVLNSimulatorClientTool(machines_info=machines_info_xxx)
    tool.run_call()

    # navigation pipeline
    for i in tqdm(range(len(navi_tasks))):
        navi_task = navi_tasks[i]
        # load scene info
        episode_id = navi_task['episode_id']

        print(f"================================ Start episode {episode_id} ==================================")
        # load graph
        mem_graph = NavigationGraph(os.path.join(graph_root, f"{episode_id}.pkl"))
        with open(graph_act_root, 'rb') as f:
            mem_act_graph = pickle.load(f)

        landmarks = navi_task["instruction"]["landmarks"]
        if len(landmarks) == 0:
            continue

        next_landmark_idx = 0
        object_info = []
        instruction = navi_task["instruction"]['instruction_text']
        reference_path = navi_task['reference_path']
        start_pos = reference_path[0][:3]
        end_pos = reference_path[-1][:3]

        step_size = 0
        hist_step_size = []

        curr_pose = convert_airsim_pose(navi_task["start_position"]+navi_task["start_rotation"][1:]+[navi_task["start_rotation"][0]])
        target_pose = convert_airsim_pose(navi_task["goals"][0]['position']+[0, 0, 0, 1])

        # set env
        tool.setPoses([[curr_pose]])

        data_dict = {
            "episode_id": episode_id,
            "instruction": instruction,
            "gt_traj": [pose[:3] for pose in reference_path],
            "pred_traj": [],        # todo: pred_traj should be depreciated
            "pred_traj_explore": [list(curr_pose.position)+list(airsim.to_eularian_angles(curr_pose.orientation))],
            "pred_traj_memory": []
        }

        # take off
        for _ in range(5):
            new_pose = getPoseAfterMakeActions(curr_pose, [4])
            curr_pose = new_pose
            tool.setPoses([[curr_pose]])

        while step_size < max_step_size:

            time_s = time.time()
            # get observation
            try:
                pano_obs, pano_pose = get_pano_observations(curr_pose, tool, scene_id=scene_id)
                pano_obs_imgs = [pano_obs[6][0], pano_obs[7][0], pano_obs[0][0], pano_obs[1][0], pano_obs[2][0], pano_obs[4][0]]
                pano_obs_deps = [pano_obs[6][1], pano_obs[7][1], pano_obs[0][1], pano_obs[1][1], pano_obs[2][1], pano_obs[4][1]]
                pano_obs_poses = [pano_pose[6], pano_pose[7], pano_pose[0], pano_pose[1], pano_pose[2], pano_pose[4]]

                pano_obs_imgs_path = ["obs_imgs/rgb_obs_{}.png".format(view_drc.replace(" ", "_")) for view_drc in
                                      ObservationDirections+["back"]]
                pano_obs_deps_path = ["obs_imgs/dep_obs_{}.npy".format(view_drc.replace(" ", "_")) for view_drc in
                                      ObservationDirections+["back"]]
                pano_pose_path = ["obs_imgs/pose_{}.npy".format(view_drc.replace(" ", "_")) for view_drc in
                                      ObservationDirections+["back"]]

                for j in range(len(pano_obs_imgs_path)):
                    cv2.imwrite(pano_obs_imgs_path[j], pano_obs_imgs[j])
                    np.save(pano_obs_deps_path[j], pano_obs_deps[j])
                    np.save(pano_pose_path[j], pano_obs_poses[j])

                    pano_obs_depvis = (pano_obs_deps[j].squeeze() * 255).astype(np.uint8)
                    pano_obs_depvis = np.stack([pano_obs_depvis for _ in range(3)], axis=2)

                    cv2.imwrite(pano_obs_deps_path[j].replace("npy", "png"), pano_obs_depvis)

            except Exception as e:
                data_dict['pred_traj'].append(list(curr_pose.position))
                print(f"Task idx: {i}. Step size: {step_size}. Success: False. Failed to get images. Exception: {e}")
                break
            # print(f"observation time: {time.time()-time_s}")

            # calculate current position to the graph
            cls_node = find_closest_node(mem_graph._graph, list(curr_pose.position), thresh=20)

            # explore or exploit
            # exploit
            if cls_node is not None:
                print("Find the memory graph node!!!")
                with open(pano_obs_imgs_path[0], "rb") as file:
                    imgf = file.read()
                with open(pano_obs_imgs_path[-1], "rb") as file:
                    imgb = file.read()

                obs = {
                    "pos": np.array(list(curr_pose.position)),
                    "image": [imgf, imgb]
                }
                new_node = mem_graph.add_vertix(obs)
                mem_graph.add_edge(new_node, cls_node)

                rest_landmarks = landmarks[next_landmark_idx:]
                result = pipeline.full_pipeline(mem_graph, start_node=new_node, landmarks=rest_landmarks, alpha=0.0001)

                # evaluate
                walk = [a[0] for a in result["walk"]]

                node_traj = [mem_graph.get_node_data(node)["position"].tolist() for node in walk]
                sz, action_traj = calculate_movement_steps_mem(mem_act_graph, node_traj)

                rest_steps = int(min(max_step_size-step_size, sz))

                rest_walks = action_traj[:rest_steps]

                data_dict['pred_traj'].extend(rest_walks)
                data_dict['pred_traj_memory'].extend(rest_walks)

                stop_pos = rest_walks[-1][:3]
                curr_pose = convert_airsim_pose(list(stop_pos) + list(curr_pose.orientation))
                tool.setPoses([[curr_pose]])

                step_size += rest_steps

                break
            # explore
            else:
                print("No memory graph reached, keep exploring ...")
                time1 = time.time()

                if vlm_name == "dino":
                    _, new_pose = explore_pipeline_by_dino(
                        curr_pose, llm, vlm,
                        pano_obs_imgs_path[:5],
                        pano_obs_imgs[:5],
                        pano_obs_deps[:5],
                        instruction, object_info, landmarks)
                elif vlm_name == "sam":
                    _, new_pose, next_landmark_found = explore_pipeline_by_sam(
                        curr_pose, llm, vlm,
                        pano_obs_imgs_path[:5],
                        pano_obs_imgs[:5],
                        pano_obs_deps[:5],
                        pano_obs_poses[:5],
                        instruction, object_info, landmarks, next_landmark_idx)

                # print(f"explore pipeline time: {time.time()-time1}")

                sz, mid_coords = calculate_movement_steps(curr_pose, new_pose)
                data_dict['pred_traj'].extend([mid_coord[:3] for mid_coord in mid_coords])
                data_dict['pred_traj_explore'].extend(mid_coords)

                tool.setPoses([[new_pose]])
                curr_pose = new_pose

                step_size += sz
                hist_step_size.append(sz)
                if next_landmark_found:
                    next_landmark_idx += 1

                # print(f"total reference time: {time.time() - time_s}")

                if next_landmark_idx >= len(landmarks):
                    print(f"Task idx: {i}. Total steps: {step_size}. Exploration finished.")
                    break

                if len(hist_step_size)>=4 and sum(hist_step_size[-4:-1]) == 0.0:
                    print(f"Task idx: {i}. Total steps: {step_size}. Success: False. Stuck!!")
                    break

        stop_pos = np.array(list(curr_pose.position))
        target_pos = np.array(list(target_pose.position))
        ne = np.linalg.norm(np.array(target_pos) - np.array(stop_pos))

        if ne < 20:
            data_dict.update({"success": True})
            print(f"############## Episode {episode_id}: success, NE: {ne}. Step size: {step_size}")
        else:
            data_dict.update({"success": False})
            print(f"############## Episode {episode_id}: failed. NE: {ne}")

        nav_evaluator.update(data_dict)
        nav_evaluator.log_metrics()

        predict_routes.append(data_dict)

        if record:
            for pr in predict_routes:
                final_traj = []
                final_traj.extend(pr['pred_traj_explore'])

                mem_traj = pr['pred_traj_memory']
                if len(mem_traj) == 0:
                    continue

                _, mid_coords = calculate_movement_steps_mem(mem_act_graph, mem_traj)
                final_traj.extend(mid_coords)

                pr.update({'final_pred_traj': final_traj})
            with open(f'output/output_data_{env_id}.json', 'w') as f:
                json.dump(predict_routes, f, indent=4)

    nav_evaluator.log_metrics()


def replay_path(trajectory_files, scene_id, img_type='rgb'):

    # load env
    machines_info_xxx = [
        {
            'MACHINE_IP': '127.0.0.1',
            'SOCKET_PORT': 30000,
            'MAX_SCENE_NUM': 8,
            'open_scenes': [scene_id],
        },
    ]

    tool = AirVLNSimulatorClientTool(machines_info=machines_info_xxx)
    tool.run_call()

    with open(trajectory_files, 'r') as f:
        meta_data = json.load(f)

    for i, traj_info in enumerate(meta_data):
        # text_instruction = traj_info['instruction']
        episode_id = traj_info['episode_id']
        try:
            pred_traj = traj_info['final_pred_traj']
        except Exception as e:
            print(e)
            continue
        if not traj_info['success']:
            continue
        if len(pred_traj) > 2000:
            continue

        save_dir_rgb = os.path.join(f"./output/video/{scene_id}", episode_id, 'rgb')
        os.makedirs(save_dir_rgb, exist_ok=True)
        print(f"image saved in :{save_dir_rgb}")

        save_dir_dep = os.path.join(f"./output/video/{scene_id}", episode_id, 'dep')
        os.makedirs(save_dir_dep, exist_ok=True)
        print(f"depth saved in :{save_dir_dep}")

        for j in tqdm(range(len(pred_traj))):
            pose = pred_traj[j]
            pos = pose[:3]
            p, r, y = pose[3:]
            ori = airsim.to_quaternion(p, r, y)

            curr_pose = convert_airsim_pose(pos+[ori.x_val, ori.y_val, ori.z_val, ori.w_val])
            tool.setPoses([[curr_pose]])

            try:
                pano_obs, pano_pose = get_front_observations(curr_pose, tool, scene_id=scene_id)
                pano_obs_imgs = pano_obs[0][0]
                pano_obs_deps = pano_obs[0][1] * 300

                if img_type == 'rgb':
                    pano_obs_imgs_path = os.path.join(save_dir_rgb, f"rgb_obs_front_{j}.png")
                    cv2.imwrite(pano_obs_imgs_path, pano_obs_imgs)
                elif img_type == 'dep':
                    pano_obs_imgs_path = os.path.join(save_dir_dep, f"dep_obs_front_{j}.npy")
                    np.save(pano_obs_imgs_path, pano_obs_deps)
                elif img_type == 'all':
                    pano_obs_imgs_path = os.path.join(save_dir_rgb, f"rgb_obs_front_{j}.png")
                    cv2.imwrite(pano_obs_imgs_path, pano_obs_imgs)

                    pano_obs_imgs_path = os.path.join(save_dir_dep, f"dep_obs_front_{j}.npy")
                    np.save(pano_obs_imgs_path, pano_obs_deps)

            except Exception as e:
                print(f"{e}, skip {episode_id}")


def make_demo_video(data_root, env_id, episode_id):
    data_dir = f"{data_root}/{env_id}/{episode_id}/rgb"
    save_dir = f"{data_root}/{env_id}/{episode_id}"
    traj_data_path = os.path.join('output', f'output_data_{env_id}.json')

    tgt_traj = None
    with open(traj_data_path, 'r') as f:
        output_trajs = json.load(f)
    for out_traj in output_trajs:
        if out_traj['episode_id'] == episode_id:
            tgt_traj = out_traj
            break

    instruction = tgt_traj['instruction']
    img_files = os.listdir(data_dir)
    sorted_img_files = sorted(img_files, key=lambda name: int(name.split('_')[-1].split('.')[0]))

    frames = []
    for img_f in sorted_img_files:
        img = cv2.imread(os.path.join(data_dir, img_f))

        frame = append_text_to_image(img, instruction)
        frames.append(frame)

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(
        os.path.join(save_dir, 'demo.avi'), fourcc, 10, (w, h))

    for frame in frames:
        out.write(frame)

    out.release()
    print("Video processing complete.")


if __name__ == '__main__':
    env_id = 3
    split = "val_seen"
    save_demo = False

    # 1. record path; 2. replay the path; 3. make demo video
    CityNavAgent(env_id, split, max_step_size=60, vlm_name="sam", record=save_demo)
    if save_demo:
        replay_path(f"./output/output_data_{env_id}.json", env_id, img_type='rgb')
        make_demo_video('./output/video', env_id=env_id, episode_id='3IRIK4HM3JIZ640FRHTYZU0EJ9Y6CH')
