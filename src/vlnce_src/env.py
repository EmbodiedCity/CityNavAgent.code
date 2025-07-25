import copy
import random
import time

import msgpack_numpy
import numpy as np
import math
from gym import spaces
import lmdb
import os
import json
from pathlib import Path
import airsim
import threading
from fastdtw import fastdtw
import tqdm

from typing import Dict, List, Optional
import sys
sys.path.append("../..")

from src.common.param import args
from utils.logger import logger
from airsim_plugin.AirVLNSimulatorClientTool import AirVLNSimulatorClientTool
from airsim_plugin.airsim_settings import AirsimActions, AirsimActionSettings
from utils.env_utils import SimState, getPoseAfterMakeAction, getPoseAfterMakeActions
from utils.env_vector import VectorEnvUtil
from utils.shorest_path_sensor import EuclideanDistance3


def load_my_datasets(splits):
    import random
    data = []
    vocab = {}
    old_state = random.getstate()
    for split in splits:
        components = split.split("@")
        number = -1
        if len(components) > 1:
            split, number = components[0], int(components[1])

        # Load Json
        with open(str(Path(args.project_prefix) / 'DATA/data/aerialvln/{}.json'.format(split)), 'r', encoding='utf-8') as f:
        # with open(str("/home/vincent/py-pro/AirVLN-main/" + 'DATA/data/aerialvln/{}.json'.format(split)), 'r', encoding='utf-8') as f:
            new_data = json.load(f)
            # vocab = new_data['instruction_vocab']
            new_data = new_data['episodes']

        # Partition
        if number > 0:
            random.seed(1)              # Make the data deterministic, additive
            random.shuffle(new_data)
            new_data = new_data[:number]

        # Join
        data += new_data
    random.setstate(old_state)      # Recover the state of the random generator
    return data, vocab


class AirVLNENV:
    def __init__(self, batch_size=8, split='train',
                 seed=1, tokenizer=None,
                 dataset_group_by_scene=True,
                 ):
        self.batch_size = batch_size
        self.split = split
        self.seed = seed
        if tokenizer:
            self.tok = tokenizer
        self.dataset_group_by_scene = dataset_group_by_scene

        load_data, vocab = load_my_datasets([split])
        self.ori_raw_data = load_data.copy()
        self.vocab = vocab.copy()
        # args.vocab_size = self.vocab['num_vocab']
        logger.info('Loaded with {} instructions, using split: {}'.format(len(load_data), split))

        self.index_data = 0
        self.data = []
        pbar = tqdm.tqdm(total=len(self.ori_raw_data))

        # tokenize the instruction in the raw data and output tokenized data
        for i_item, item in enumerate(self.ori_raw_data):
            if args.collect_type in ['TF']:
                if len(list(args.TF_mode_load_scene)) > 0 and str(item['scene_id']) not in list(args.TF_mode_load_scene):
                    pbar.update()
                    continue

            if args.collect_type in ['dagger', 'SF']:
                if len(list(args.dagger_mode_load_scene)) > 0 and str(item['scene_id']) not in list(args.dagger_mode_load_scene):
                    pbar.update()
                    continue

            new_item = dict(item).copy()
            if args.tokenizer_use_bert:
                text = item['instruction']['instruction_text']
                instruction_tokens = tokenizer(
                    text,
                    truncation=True,
                    max_length=args.maxInput,
                    padding='max_length',
                    return_tensors="pt"
                )['input_ids'][0]
            else:
                instruction_tokens = tokenizer.encode_sentence(item['instruction']['instruction_text'])
            new_item['instruction']['instruction_tokens'] = instruction_tokens
            self.data.append(new_item)
            pbar.update()
        pbar.close()

        # create dictionary from traj id to token or episode
        self.trajectory_id_2_instruction_tokens = {}
        self.trajectory_id_2_episode_ids = {}
        for i_item, item in enumerate(self.data):
            if item['trajectory_id'] not in self.trajectory_id_2_instruction_tokens.keys():
                self.trajectory_id_2_instruction_tokens[item['trajectory_id']] = []
                self.trajectory_id_2_instruction_tokens[item['trajectory_id']].append(
                    item['instruction']['instruction_tokens']
                )
            else:
                self.trajectory_id_2_instruction_tokens[item['trajectory_id']].append(
                    item['instruction']['instruction_tokens']
                )

            if item['trajectory_id'] not in self.trajectory_id_2_episode_ids.keys():
                self.trajectory_id_2_episode_ids[item['trajectory_id']] = []
                self.trajectory_id_2_episode_ids[item['trajectory_id']].append(
                    item['episode_id']
                )
            else:
                self.trajectory_id_2_episode_ids[item['trajectory_id']].append(
                    item['episode_id']
                )

        random.shuffle(self.data)
        if args.EVAL_NUM != -1 and int(args.EVAL_NUM) > 0:
            [random.shuffle(self.data) for i in range(10)]
            self.data = self.data[:int(args.EVAL_NUM)].copy()

        # cluster the data with same scenes
        # eg. [scene_id1, ..., scene_id1, scene_id2, ..., scene_id2, scene_id3, ..., scene_id3, ...]
        if dataset_group_by_scene:
            self.data = self._group_scenes()
            logger.warning('dataset grouped by scene')

        scenes = [item['scene_id'] for item in self.data]
        self.scenes = set(scenes)

        self.observation_space = spaces.Dict({
            "rgb": spaces.Box(low=0, high=255, shape=(args.Image_Height_RGB, args.Image_Width_RGB, 3), dtype=np.uint8),
            "depth": spaces.Box(low=0, high=1, shape=(args.Image_Height_DEPTH, args.Image_Width_DEPTH, 1), dtype=np.float32),
            "instruction": spaces.Discrete(0),
            "progress": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "teacher_action": spaces.Box(low=0, high=100, shape=(1,)),
        })
        self.action_space = spaces.Discrete(int(len(AirsimActions)))

        self.sim_states: Optional[List[SimState], List[None]] = [None for _ in range(batch_size)]
        self.last_scene_id_list = []
        self.one_scene_could_use_num = 5000
        self.this_scene_used_cnt = 0

        if args.collect_type in ['TF']:

            if args.run_type in ['collect']:
                self.lmdb_features_dir = str(Path(args.project_prefix) / 'DATA' / 'img_features' / str(args.run_type) / str(args.name) / str(split))
                self.lmdb_rgb_dir = str(Path(args.project_prefix) / 'DATA' / 'img_features' / str(args.run_type) / str(args.name) / (str(split)+'_rgb'))
                self.lmdb_depth_dir = str(Path(args.project_prefix) / 'DATA' / 'img_features' / str(args.run_type) / str(args.name) / (str(split)+'_depth'))

                if not os.path.exists(str(self.lmdb_features_dir)):
                    os.makedirs(str(self.lmdb_features_dir), exist_ok=True)
                if not os.path.exists(str(self.lmdb_rgb_dir)):
                    os.makedirs(str(self.lmdb_rgb_dir), exist_ok=True)
                if not os.path.exists(str(self.lmdb_depth_dir)):
                    os.makedirs(str(self.lmdb_depth_dir), exist_ok=True)

                lmdb_features_map_size = 5.0e12  # 1.0e11  100GB
                lmdb_rgb_map_size = 5.0e12  # 1.0e11  100GB
                lmdb_depth_map_size = 5.0e12  # 1.0e11  100GB

                try:
                    self.lmdb_features_env = lmdb.open(self.lmdb_features_dir, map_size=int(lmdb_features_map_size), readahead=False,)
                    self.lmdb_features_start_id = self.lmdb_features_env.stat()["entries"]
                    self.lmdb_features_txn = self.lmdb_features_env.begin(write=True)
                    self.threading_lock_lmdb_features_txn = threading.Lock()
                    logger.info('init lmdb of {}, {}, lmdb_start_id: {}'.format(split, 'features', self.lmdb_features_start_id))

                    self.lmdb_collected_keys = set()
                    with tqdm.tqdm(
                        total=int(self.lmdb_features_start_id), dynamic_ncols=True
                    ) as pbar:
                        for key in self.lmdb_features_txn.cursor().iternext(keys=True, values=False):
                            pbar.update()
                            self.lmdb_collected_keys.add(key.decode())

                    self.lmdb_rgb_env = lmdb.open(self.lmdb_rgb_dir, map_size=int(lmdb_rgb_map_size), readahead=False,)
                    self.lmdb_rgb_start_id = self.lmdb_rgb_env.stat()["entries"]
                    self.lmdb_rgb_txn = self.lmdb_rgb_env.begin(write=True)
                    self.threading_lock_lmdb_rgb_txn = threading.Lock()
                    logger.info('init lmdb of {}, {}, lmdb_start_id: {}'.format(split, 'rgb', self.lmdb_rgb_start_id))

                    self.lmdb_depth_env = lmdb.open(self.lmdb_depth_dir, map_size=int(lmdb_depth_map_size), readahead=False,)
                    self.lmdb_depth_start_id = self.lmdb_depth_env.stat()["entries"]
                    self.lmdb_depth_txn = self.lmdb_depth_env.begin(write=True)
                    self.threading_lock_lmdb_depth_txn = threading.Lock()
                    logger.info('init lmdb of {}, {}, lmdb_start_id: {}'.format(split, 'depth', self.lmdb_depth_start_id))
                except lmdb.Error as err:
                    logger.error(err)
                    raise err

            if args.run_type in ['eval']:
                self.lmdb_features_dir = str(Path(args.project_prefix) / 'DATA' / 'img_features' / str(args.run_type) / str(args.name) / '{}_{}'.format(str(split), args.make_dir_time))

                if not os.path.exists(str(self.lmdb_features_dir)):
                    os.makedirs(str(self.lmdb_features_dir), exist_ok=True)

                lmdb_features_map_size = 1.0e11  # 1.0e6  1M

                try:
                    self.lmdb_features_env = lmdb.open(self.lmdb_features_dir, map_size=int(lmdb_features_map_size), readahead=False,)
                    self.lmdb_features_start_id = self.lmdb_features_env.stat()["entries"]
                    self.lmdb_features_txn = self.lmdb_features_env.begin(write=True)
                    self.threading_lock_lmdb_features_txn = threading.Lock()
                    logger.info('init lmdb of {}, {}, lmdb_start_id: {}'.format(split, 'features', self.lmdb_features_start_id))

                    self.lmdb_collected_keys = set()
                    with tqdm.tqdm(
                        total=int(self.lmdb_features_start_id), dynamic_ncols=True
                    ) as pbar:
                        for key in self.lmdb_features_txn.cursor().iternext(keys=True, values=False):
                            pbar.update()
                            self.lmdb_collected_keys.add(key.decode())

                except lmdb.Error as err:
                    logger.error(err)
                    raise err

        if args.collect_type in ['dagger', 'SF']:
            self.lmdb_features_dir = str(Path(args.project_prefix) / 'DATA' / 'img_features' / str(args.run_type) / str(args.name) / str(split))

            if not os.path.exists(str(self.lmdb_features_dir)):
                os.makedirs(str(self.lmdb_features_dir), exist_ok=True)

            lmdb_features_map_size = 20.0e12  # 1.0e11  100GB

            try:
                self.lmdb_features_env = lmdb.open(self.lmdb_features_dir, map_size=int(lmdb_features_map_size), readahead=False,)
                self.lmdb_features_start_id = self.lmdb_features_env.stat()["entries"]
                self.lmdb_features_txn = self.lmdb_features_env.begin(write=True)
                self.threading_lock_lmdb_features_txn = threading.Lock()
                logger.info('init lmdb of {}, {}, lmdb_start_id: {}'.format(split, 'features', self.lmdb_features_start_id))

                self.lmdb_collected_keys = set()
                with tqdm.tqdm(
                    total=int(self.lmdb_features_start_id), dynamic_ncols=True
                ) as pbar:
                    for key in self.lmdb_features_txn.cursor().iternext(keys=True, values=False):
                        pbar.update()
                        if len(str(key.decode()).split('_')) <= 1:
                            self.lmdb_collected_keys.add(
                                '{}_0'.format(key.decode())
                            )
                        else:
                            self.lmdb_collected_keys.add(key.decode())

            except lmdb.Error as err:
                logger.error(err)
                raise err

        self.init_VectorEnvUtil()

    def _group_scenes(self):
        assert self.dataset_group_by_scene, 'error args param'

        scene_sort_keys: Dict[str, int] = {}
        for item in self.data:
            if str(item['scene_id']) not in scene_sort_keys:
                scene_sort_keys[str(item['scene_id'])] = len(scene_sort_keys)

        return sorted(self.data, key=lambda e: scene_sort_keys[str(e['scene_id'])])

    def init_VectorEnvUtil(self):
        self.delete_VectorEnvUtil()

        self.load_scenes = [int(_scene) for _scene in list(self.scenes)]
        self.VectorEnvUtil = VectorEnvUtil(self.load_scenes, self.batch_size)

    def delete_VectorEnvUtil(self):
        if hasattr(self, 'VectorEnvUtil'):
            del self.VectorEnvUtil

        import gc
        gc.collect()

    #
    def next_minibatch(self, skip_scenes=[], data_it=0):
        batch = []

        while True:
            if self.index_data >= len(self.data)-1:
                random.shuffle(self.data)
                logger.warning('random shuffle data')
                if self.dataset_group_by_scene:
                    self.data = self._group_scenes()
                    logger.warning('dataset grouped by scene')

                if len(batch) == 0:
                    self.index_data = 0
                    self.batch = None
                    return

                self.index_data = self.batch_size - len(batch)
                batch += self.data[:self.index_data]
                break

            new_episode = self.data[self.index_data]

            #
            if new_episode['scene_id'] in skip_scenes:
                self.index_data += 1
                continue

            if args.run_type in ['collect', 'train'] and args.collect_type in ['TF']:
                lmdb_key = '{}'.format(new_episode['episode_id'])
                if lmdb_key in self.lmdb_collected_keys:
                    self.index_data += 1
                    continue
                else:
                    batch.append(new_episode)
                    self.index_data += 1
            elif args.run_type in ['collect', 'train'] and args.collect_type in ['dagger', 'SF']:
                lmdb_key = '{}_{}'.format(new_episode['episode_id'], data_it)
                if lmdb_key in self.lmdb_collected_keys:
                    self.index_data += 1
                    continue
                else:
                    batch.append(new_episode)
                    self.index_data += 1
            else:
                batch.append(new_episode)
                self.index_data += 1

            if len(batch) == self.batch_size:
                break

        self.batch = copy.deepcopy(batch)
        assert len(self.batch) == self.batch_size, 'next_minibatch error'

        self.VectorEnvUtil.set_batch(self.batch)


    #
    def changeToNewEpisodes(self):
        self._changeEnv(need_change=False)

        self._setEpisodes()

        self.update_measurements()

    def _changeEnv(self, need_change: bool = True):
        scene_id_list = [item['scene_id'] for item in self.batch]
        assert len(scene_id_list) == self.batch_size, '错误'

        machines_info_template = copy.deepcopy(args.machines_info)
        total_max_scene_num = 0
        for item in machines_info_template:
            total_max_scene_num += item['MAX_SCENE_NUM']
        assert self.batch_size <= total_max_scene_num, 'error args param: batch_size'

        # 构造机器信息 TODO
        machines_info = []
        ix = 0
        for index, item in enumerate(machines_info_template):
            machines_info.append(item)
            delta = min(self.batch_size, item['MAX_SCENE_NUM'], len(scene_id_list)-ix)
            machines_info[index]['open_scenes'] = scene_id_list[ix : ix + delta]
            ix += delta

        #
        cnt = 0
        for item in machines_info:
            cnt += len(item['open_scenes'])
        assert self.batch_size == cnt, 'error create machines_info'

        #
        if self.this_scene_used_cnt < self.one_scene_could_use_num and \
                len(set(scene_id_list)) == 1 and len(set(self.last_scene_id_list)) == 1 and \
                scene_id_list[0] is not None and self.last_scene_id_list[0] is not None and scene_id_list[0] == self.last_scene_id_list[0] and \
                need_change == False:
            self.this_scene_used_cnt += 1
            logger.warning('no need to change env: {}'.format(scene_id_list))
            return
        else:
            logger.warning('to change env: {}'.format(scene_id_list))

        #
        while True:
            try:
                self.machines_info = copy.deepcopy(machines_info)
                if (not args.ablate_rgb or not args.ablate_depth):
                    self.simulator_tool = AirVLNSimulatorClientTool(machines_info=self.machines_info)
                    self.simulator_tool.run_call()
                break
            except Exception as e:
                logger.error("启动场景失败 {}".format(e))
                time.sleep(3)
            except:
                logger.error('启动场景失败')
                time.sleep(3)

        self.last_scene_id_list = scene_id_list.copy()
        self.this_scene_used_cnt = 1

    # set initial poses in virtual environment and record in python by SimState
    def _setEpisodes(self):
        start_position_list = [item['start_position'] for item in self.batch]
        start_rotation_list = [item['start_rotation'] for item in self.batch]

        #
        poses = []
        cnt = 0
        for index_1, item in enumerate(self.machines_info):
            poses.append([])
            for index_2, _ in enumerate(item['open_scenes']):
                pose = airsim.Pose(
                    position_val=airsim.Vector3r(
                        x_val=start_position_list[cnt][0],
                        y_val=start_position_list[cnt][1],
                        z_val=start_position_list[cnt][2],
                    ),
                    orientation_val=airsim.Quaternionr(
                        x_val=start_rotation_list[cnt][1],
                        y_val=start_rotation_list[cnt][2],
                        z_val=start_rotation_list[cnt][3],
                        w_val=start_rotation_list[cnt][0],
                    ),
                )
                poses[index_1].append(pose)
                cnt += 1

        #
        if (not args.ablate_rgb or not args.ablate_depth):
            result = self.simulator_tool.setPoses(poses=poses)
            if not result:
                logger.error('设置位置失败')
                self.reset_to_this_pose(poses)

        #
        cnt = 0
        for index_1, item in enumerate(self.machines_info):
            for index_2, _ in enumerate(item['open_scenes']):
                pose = airsim.Pose(
                    position_val=airsim.Vector3r(
                        x_val=start_position_list[cnt][0],
                        y_val=start_position_list[cnt][1],
                        z_val=start_position_list[cnt][2],
                    ),
                    orientation_val=airsim.Quaternionr(
                        x_val=start_rotation_list[cnt][1],
                        y_val=start_rotation_list[cnt][2],
                        z_val=start_rotation_list[cnt][3],
                        w_val=start_rotation_list[cnt][0],
                    ),
                )
                self.sim_states[cnt] = SimState(index=cnt, step=0, episode_info=self.batch[cnt], pose=pose)
                self.sim_states[cnt].trajectory = [[
                    pose.position.x_val, pose.position.y_val, pose.position.z_val, # xyz
                    pose.orientation.x_val, pose.orientation.y_val, pose.orientation.z_val, pose.orientation.w_val, # xyzw
                ]]
                cnt += 1


    #
    def get_obs(self, camera_id='front_0'):
        obs_states = self._getStates(camera_id)

        obs, states = self.VectorEnvUtil.get_obs(obs_states)
        self.sim_states = states

        return obs

    def _getStates(self, camera_id='front_0'):
        while True:
            if (not args.ablate_rgb or not args.ablate_depth):
                responses = self.simulator_tool.getImageResponses(get_rgb=not bool(args.ablate_rgb), get_depth=not bool(args.ablate_depth), camera_id=camera_id)
            else:
                responses = [[(None, None) for j in range(self.batch_size)] for i in range(len(self.machines_info))]
            if responses is None:
                poses = self._get_current_pose()
                self.reset_to_this_pose(poses)
                time.sleep(3)
            else:
                break

        #
        cnt = 0
        for item in responses:
            cnt += len(item)
        assert len(responses) == len(self.machines_info), 'error'
        assert cnt == self.batch_size, 'error'

        #
        if args.run_type in ['eval'] or \
            (args.run_type in ['collect'] and args.collect_type in ['dagger']):
            cnt = 0
            for index_1, item in enumerate(self.machines_info):
                for index_2 in range(len(item['open_scenes'])):
                    depth_image = responses[index_1][index_2][1]
                    collision_sensor_result = (np.array(depth_image) < 0.004).sum() / np.array(depth_image).flatten().shape[0]
                    if collision_sensor_result > 1:
                        self.sim_states[cnt].is_collisioned = True
                        self.sim_states[cnt].is_end = True
                        logger.warning('collisioned: {}'.format(cnt))

                    cnt += 1

        #
        states = [None for _ in range(self.batch_size)]
        cnt = 0
        for index_1, item in enumerate(self.machines_info):
            for index_2 in range(len(item['open_scenes'])):
                rgb_image = responses[index_1][index_2][0]
                if rgb_image is not None:
                    _rgb_image = np.array(rgb_image)
                else:
                    _rgb_image = None

                depth_image = responses[index_1][index_2][1]
                if depth_image is not None:
                    _depth_image = np.array(depth_image)
                else:
                    _depth_image = None

                state = self.sim_states[cnt]

                states[cnt] = (_rgb_image, _depth_image, state)
                cnt += 1

                #
                if self.split in ['train'] and args.run_type in ['collect'] and args.collect_type in ['TF']:
                    trajectory_id = state.episode_info['trajectory_id']
                    step = state.step
                    lmdb_rgb_key = '{}_{}_rgb'.format(trajectory_id, step)
                    lmdb_depth_key = '{}_{}_depth'.format(trajectory_id, step)

                    if rgb_image is not None:
                        self.threading_lock_lmdb_rgb_txn.acquire()
                        self.lmdb_rgb_txn.put(
                            lmdb_rgb_key.encode(),
                            msgpack_numpy.packb(
                                rgb_image, use_bin_type=True
                            ),
                        )
                        self.threading_lock_lmdb_rgb_txn.release()

                    if depth_image is not None:
                        self.threading_lock_lmdb_depth_txn.acquire()
                        self.lmdb_depth_txn.put(
                            lmdb_depth_key.encode(),
                            msgpack_numpy.packb(
                                depth_image, use_bin_type=True
                            ),
                        )
                        self.threading_lock_lmdb_depth_txn.release()

        return states

    def _get_current_pose(self) -> list:
        poses = []

        cnt = 0
        for index_1, item in enumerate(self.machines_info):
            poses.append([])
            for index_2, _ in enumerate(item['open_scenes']):
                poses[index_1].append(
                    self.sim_states[cnt].pose
                )
                cnt += 1

        return poses


    #
    def reset(self):
        self.changeToNewEpisodes()
        return self.get_obs()


    def reset_to_this_pose(self, poses, need_change=True):
        #
        self._changeEnv(need_change=need_change)

        #
        if (not args.ablate_rgb or not args.ablate_depth):
            result = self.simulator_tool.setPoses(poses=poses)
            if not result:
                logger.error('重置到此位置失败')
                self.reset_to_this_pose(poses)


    def makeActions(
            self,
            action_list: List[int],
            update_statue=True):
        #
        poses = []
        for index, action in enumerate(action_list):
            if self.sim_states[index].is_end == True:
                action = AirsimActions.STOP
                # continue

            if action == AirsimActions.STOP or self.sim_states[index].step >= int(args.maxAction):
                self.sim_states[index].is_end = True


            state = self.sim_states[index]

            pose = copy.deepcopy(state.pose)
            new_pose = getPoseAfterMakeAction(pose, action)
            poses.append(new_pose)

        poses_formatted = []
        cnt = 0
        for index_1, item in enumerate(self.machines_info):
            poses_formatted.append([])
            for index_2, _ in enumerate(item['open_scenes']):
                poses_formatted[index_1].append(poses[cnt])
                cnt += 1

        #
        if (not args.ablate_rgb or not args.ablate_depth):
            result = self.simulator_tool.setPoses(poses=poses_formatted)
            if not result:
                logger.error('设置位置失败')
                self.reset_to_this_pose(poses_formatted)

        if update_statue:
            for index, action in enumerate(action_list):
                if self.sim_states[index].is_end == True:
                    continue

                if action == AirsimActions.STOP or self.sim_states[index].step >= int(args.maxAction):
                    self.sim_states[index].is_end = True

                self.sim_states[index].step += 1
                self.sim_states[index].pose = poses[index]
                self.sim_states[index].trajectory.append([
                    poses[index].position.x_val, poses[index].position.y_val, poses[index].position.z_val, # xyz
                    poses[index].orientation.x_val, poses[index].orientation.y_val, poses[index].orientation.z_val, poses[index].orientation.w_val, # xyzw
                ])
                self.sim_states[index].pre_action = action

            # update measurement
            if args.run_type not in ['collect']:
                self.update_measurements()


    # support make a squence of action at once
    def makeVirtualActions(
            self,
            action_list: List[List],
            update_statue=True):
        #
        poses = []
        for index, actions in enumerate(action_list):
            if self.sim_states[index].is_end == True:
                actions = []
                # continue

            if AirsimActions.STOP in actions:
                self.sim_states[index].is_end = True
                act_idx = actions.index(AirsimActions.STOP)
                actions = actions[:act_idx]

            state = self.sim_states[index]

            pose = copy.deepcopy(state.pose)
            new_pose = getPoseAfterMakeActions(pose, actions)
            poses.append(new_pose)

        poses_formatted = []
        cnt = 0
        for index_1, item in enumerate(self.machines_info):
            poses_formatted.append([])
            for index_2, _ in enumerate(item['open_scenes']):
                poses_formatted[index_1].append(poses[cnt])
                cnt += 1

        if (not args.ablate_rgb or not args.ablate_depth):
            result = self.simulator_tool.setPoses(poses=poses_formatted)
            if not result:
                logger.error('设置位置失败')
                self.reset_to_this_pose(poses_formatted)

        for index, actions in enumerate(action_list):
            self.sim_states[index].pose = poses[index]

        # # todo: update is not supported so far
        # if update_statue:
        #     for index, action in enumerate(action_list):
        #         self.sim_states[index].step += len(actions)
        #         self.sim_states[index].pose = poses[index]
        #         self.sim_states[index].trajectory.append([
        #             poses[index].position.x_val, poses[index].position.y_val, poses[index].position.z_val, # xyz
        #             poses[index].orientation.x_val, poses[index].orientation.y_val, poses[index].orientation.z_val, poses[index].orientation.w_val, # xyzw
        #         ])
        #         self.sim_states[index].pre_action = action
        #
        #     # update measurement
        #     if args.run_type not in ['collect']:
        #         self.update_measurements()

    # metrics related to success
    def update_measurements(self):
        self._update_DistanceToGoal()
        self._updata_Success()
        self._updata_NDTW()
        self._updata_SDTW()
        self._update_PathLength()
        self._update_OracleSuccess()
        self._update_StepsTaken()

    def _update_DistanceToGoal(self):
        for i, state in enumerate(self.sim_states):

            current_position = np.array([
                state.pose.position.x_val,
                state.pose.position.y_val,
                state.pose.position.z_val
            ])

            if self.sim_states[i].DistanceToGoal['_previous_position'] is None or \
                not np.allclose(self.sim_states[i].DistanceToGoal['_previous_position'], current_position, atol=1):
                distance_to_target = EuclideanDistance3(
                    np.array(current_position)[0:2],
                    np.array(state.episode_info['goals'][0]['position'])[0:2]
                )
                self.sim_states[i].DistanceToGoal['_previous_position'] = current_position
                self.sim_states[i].DistanceToGoal['_metric'] = distance_to_target

    # update if success
    def _updata_Success(self):
        for i, state in enumerate(self.sim_states):
            distance_to_target = self.sim_states[i].DistanceToGoal['_metric']
            if (
                self.sim_states[i].is_end
                and distance_to_target <= self.sim_states[i].SUCCESS_DISTANCE
            ):
                self.sim_states[i].Success['_metric'] = 1.0
            else:
                self.sim_states[i].Success['_metric'] = 0.0

    def _updata_NDTW(self):
        def euclidean_distance(
                position_a,
                position_b,
        ) -> float:
            return np.linalg.norm(
                np.array(position_b) - np.array(position_a), ord=2
            )

        for i, state in enumerate(self.sim_states):

            current_position = np.array([
                state.pose.position.x_val,
                state.pose.position.y_val,
                state.pose.position.z_val
            ])

            if len(state.NDTW['locations']) == 0:
                self.sim_states[i].NDTW['locations'].append(current_position)
            else:
                if current_position.tolist() == state.NDTW['locations'][-1].tolist():
                    continue
                self.sim_states[i].NDTW['locations'].append(current_position)

            dtw_distance = fastdtw(
                self.sim_states[i].NDTW['locations'], self.sim_states[i].NDTW['gt_locations'], dist=euclidean_distance
            )[0]

            nDTW = np.exp(
                -dtw_distance / (len(self.sim_states[i].NDTW['gt_locations']) * self.sim_states[i].SUCCESS_DISTANCE)
            )
            self.sim_states[i].NDTW['_metric'] = nDTW

    def _updata_SDTW(self):
        for i, state in enumerate(self.sim_states):
            ep_success = self.sim_states[i].Success['_metric']
            nDTW = self.sim_states[i].NDTW['_metric']
            self.sim_states[i].SDTW['_metric'] = ep_success * nDTW

    def _update_PathLength(self):
        for i, state in enumerate(self.sim_states):

            current_position = np.array([
                state.pose.position.x_val,
                state.pose.position.y_val,
                state.pose.position.z_val
            ])

            if state.PathLength['_previous_position'] is None:
                self.sim_states[i].PathLength['_previous_position'] = current_position

            self.sim_states[i].PathLength['_metric'] += EuclideanDistance3(
                current_position, self.sim_states[i].PathLength['_previous_position']
            )
            self.sim_states[i].PathLength['_previous_position'] = current_position

    def _update_OracleSuccess(self):
        for i, state in enumerate(self.sim_states):
            d = self.sim_states[i].DistanceToGoal['_metric']
            self.sim_states[i].OracleSuccess['_metric'] = float(
                self.sim_states[i].OracleSuccess['_metric'] or d <= self.sim_states[i].SUCCESS_DISTANCE
            )

    def _update_StepsTaken(self):
        for i, state in enumerate(self.sim_states):
            self.sim_states[i].StepsTaken['_metric'] = self.sim_states[i].step

    def navi_task_preprocessing(self, item):
        instruction_text = item["instruction"]["instruction_text"]
        reference_path = item["reference_path"]
        actions = item["actions"]

        instruction_text = "take off and turn left fly over the trees. now turn left and cross over the pond and turn right and move towards the football court ground. now slow down to the floor and stay there."

        subtasks = [t.strip(" ") for t in instruction_text.split(".") if t]
        subtask_checkpoints = [[] for _ in range(len(subtasks))]
        subtask_checkpoints[0] = ["take off", "turn left", "fly over the trees"]
        subtask_checkpoints[1] = ["turn left", "cross over the pond", "turn right",
                                  "move towards the football court ground"]
        subtask_checkpoints[2] = [2]

        subtask_checkpoint_nodes_idx = [[] for _ in range(len(subtasks))]
        # all index is +1, since new pose is get after the action
        subtask_checkpoint_nodes_idx[0] = [{"action_code": 4, "action_idx": [10]},
                                           {"action_code": 2, "action_idx": [16]},
                                           {"action_code": 1, "action_idx": [21, 26, 31, 36, 41, 46, 51, 56, 61, 66, 71, 76, 80, 85]}]     # the idx that pose should be recorded
        subtask_checkpoint_nodes_idx[1] = [{"action_code": 2, "action_idx": [90]},
                                           {"action_code": 1, "action_idx": [95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 149]},
                                           {"action_code": 3, "action_idx": [154]},
                                           {"action_code": 1, "action_idx": [159, 164, 169, 174, 179, 184, 189, 194, 199, 204, 209, 214, 219, 224, 229, 234]}]
        subtask_checkpoint_nodes_idx[2] = [{"action_code": 5, "action_idx": [239, 244, 249, 252]},
                                           {"action_code": 0, "action_idx": [252]}]

        # layer 1: subtask
        subtask_checkpoint_node_poses = []
        for i in range(len(subtask_checkpoint_nodes_idx)):
            # layer 2: checkpoints
            checkpoint_poses = []
            for j in range(len(subtask_checkpoint_nodes_idx[i])):
                # layer 3: nodes
                pose_node = {}
                pose_node["action_code"] = subtask_checkpoint_nodes_idx[i][j]["action_code"]
                pose_node["pose"] = []
                for k in range(len(subtask_checkpoint_nodes_idx[i][j]["action_idx"])):
                    # print(subtask_checkpoint_nodes_idx[i][j]["action_idx"][k])
                    pose = reference_path[subtask_checkpoint_nodes_idx[i][j]["action_idx"][k]]
                    pose_node["pose"].append(pose)
                checkpoint_poses.append(pose_node)
            subtask_checkpoint_node_poses.append(checkpoint_poses)

        subtask_checkpoint_node_actions = []
        head = 0
        for i in range(len(subtask_checkpoint_nodes_idx)):
            # layer 2: checkpoints
            checkpoint_actions = []
            for j in range(len(subtask_checkpoint_nodes_idx[i])):
                action_node = {}
                action_node["action_code"] = subtask_checkpoint_nodes_idx[i][j]["action_code"]
                action_node["actions"] = []
                for k in range(len(subtask_checkpoint_nodes_idx[i][j]["action_idx"])):
                    if action_node["action_code"] == 0:
                        action_node["actions"].append([0])
                        continue
                    tail = subtask_checkpoint_nodes_idx[i][j]["action_idx"][k]
                    action_node["actions"].append(actions[head:tail])
                    head = tail

                checkpoint_actions.append(action_node)
            subtask_checkpoint_node_actions.append(checkpoint_actions)

        return subtasks, subtask_checkpoints, subtask_checkpoint_node_poses, subtask_checkpoint_node_actions



if __name__ == "__main__":
    os.environ["http_proxy"] = "http://127.0.0.1:7890"
    os.environ["https_proxy"] = "http://127.0.0.1:7890"

    from transformers import BertTokenizer
    tok = BertTokenizer.from_pretrained('bert-base-uncased')
    test_env = AirVLNENV(batch_size=args.batchSize, split='val_seen', tokenizer=tok)
    # test.scenes = set()
    test_env.next_minibatch(skip_scenes=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26])
    if test_env.batch is None:
        logger.warning('train_env.batch is None, going to break and stop collect')
    print(test_env.scenes)
    # print(len(test_env.data))
    # print(test_env.data[0])
    # print(test.observation_space)
    obs = test_env.reset()
    # obs = test_env.get_obs()
    print(obs)
    print(test_env.machines_info)
