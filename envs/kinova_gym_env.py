import gym
import torch
from gym import spaces
from gym.utils import seeding
import os, time
import numpy as np
import pybullet as p
import pybullet_data

from envs.kinova_env import KinovaEnv, BLOCK_MODEL_DIR

SITE_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'site_description')


class PhysClientWrapper:
    """
    This is used to make sure each BulletRobotEnv has its own physicsClient and
    they do not cross-communicate.
    """
    def __init__(self, other, physics_client_id):
        self.other = other
        self.physicsClientId = physics_client_id

    def __getattr__(self, name):
        if hasattr(self.other, name):
            attr = getattr(self.other, name)
            if callable(attr):
                return lambda *args, **kwargs: self._wrap(attr, args, kwargs)
            return attr
        raise AttributeError(name)

    def _wrap(self, func, args, kwargs):
        kwargs["physicsClientId"] = self.physicsClientId
        return func(*args, **kwargs)


class KinovaGymBaseEnv(gym.Env):
    def __init__(self, actionRepeat=10, timestep=1./240, render=False, init_qpos=None,
                 init_end_effector_pos=(0.5, 0.0, -0.1), init_end_effector_orn=(0, -np.pi, np.pi / 2), reward_type="dense", double_agent = False,
                 use_orn=False, use_roll=False, include_v=False, need_GUI = True):
        if init_qpos is None:
            init_qpos = [0., 0., -0.127, 4.234, 1.597, -0.150, 0.585, 2.860, 0.0,
                         0.7505, 0.0, 0.7505, 0.0]
        self.actionRepeat = actionRepeat
        self.timestep = timestep
        self.init_qpos = init_qpos
        self.init_end_effector_pos = init_end_effector_pos
        self.init_end_effector_orn = init_end_effector_orn
        self.reward_type = reward_type
        self.use_orn = use_orn
        self.use_roll = use_roll
        self.include_v = include_v
        self._render = render
        self._double_agent = double_agent
        self._setup_env()
        self._need_GUI = need_GUI

        obs = self.reset()
        # observation_space, action_space
        self.observation_space = spaces.Dict(dict(
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype=np.float32),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype=np.float32),
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['desired_goal'].shape, dtype=np.float32)
        ))
        if self._double_agent:
            if self.use_orn:
                self.action_space = spaces.Box(-1., 1., shape=(14, ), dtype=np.float32)
            elif self.use_roll:
                self.action_space = spaces.Box(-1., 1., shape=(10, ), dtype=np.float32)
            else:
                self.action_space = spaces.Box(-1., 1., shape=(8, ), dtype=np.float32)
        else:
            if self.use_orn:
                self.action_space = spaces.Box(-1., 1., shape=(7,), dtype=np.float32)
            elif self.use_roll:
                self.action_space = spaces.Box(-1., 1., shape=(5,), dtype=np.float32)
            else:
                self.action_space = spaces.Box(-1., 1., shape=(4,), dtype=np.float32)

    def _setup_env(self):
        if self._render:
            physics_client = p.connect(p.GUI)
            if not self._need_GUI:
                p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            physics_client = p.connect(p.DIRECT)
        self.p = PhysClientWrapper(p, physics_client)
        self._bullet_dataRoot = pybullet_data.getDataPath()

        self.seed()

        self.p.resetSimulation()
        self.p.setTimeStep(self.timestep)
        self.p.setGravity(0, 0, -10)
        self.p.loadURDF(os.path.join(self._bullet_dataRoot, "plane.urdf"), [0, 0, -1])

        self.p.loadURDF(os.path.join(self._bullet_dataRoot, "table/table.urdf"), [0.500000, 0.00000, -.820000],
                        [0.000000, 0.000000, 0.0, 1.0])
        self.robot = KinovaEnv(self.p, init_qpos=self.init_qpos, init_end_effector_pos=self.init_end_effector_pos,
                               init_end_effector_orn=self.init_end_effector_orn)

        if self._double_agent:
            self.robot2 = KinovaEnv(self.p, start_pos=(1,0,-0.1),init_qpos=self.init_qpos, init_end_effector_pos=self.init_end_effector_pos,
                                    init_end_effector_orn=self.init_end_effector_orn)
        self.p.stepSimulation()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_observation(self):
        pass

    def compute_reward(self, achieved_goal, desired_goal, info=None):
        r, _ = self.reward_and_success(achieved_goal, desired_goal, info)
        return r

    def reward_and_success(self, achieved_goal, desired_goal, info):
        pass

    def step(self, action):
        if self._double_agent:
            action = np.clip(action, self.action_space.low, self.action_space.high)
            action = np.reshape(action, [2,-1])
            # action[0] = np.clip(action[0], self.action_space.low, self.action_space.high)
            # action[1]= np.clip(action[1], self.action_space.low, self.action_space.high)
            dpos = np.asarray(action[0][0:3]) * 0.05  # Relative shift
            dpos2 = np.asarray(action[1][0:3]) * 0.05  # Relative shift
            action_origin = action
            action = action_origin[0][3:]
            action2 = action_origin[1][3:]
            if self.use_orn:
                orn = np.asarray(action[0:3]) * np.pi  # (-pi, pi)
                orn2 = np.asarray(action2[0:3]) * np.pi
                action = action[3:]
                action2 = action2[3:]
            elif self.use_roll:
                orn = np.asarray([0, -np.pi, action[0] * np.pi])
                orn2 = np.asarray([0, -np.pi, action2[0] * np.pi])
                action = action[1:]
                action = action2[1:]
            else:
                orn = np.array([0, -np.pi, np.pi / 2])
                orn2 = orn
            assert len(action) == 1 and len(action2) == 1
            # cur_finger1, cur_finger2 = self.robot.get_finger_state()
            fingers = np.asarray([0.7505 + action[0] * 0.7505, 0.7505 + action[0] * 0.7505])
            fingers2 = np.asarray([0.7505 + action2[0] * 0.7505, 0.7505 + action[0] * 0.7505])
            fingers = np.clip(fingers, 0., 1.51)
            fingers2 = np.clip(fingers2, 0., 1.51)
            tips = np.asarray([0., 0.])
            tips2 = tips
            current_pos = self.robot.get_end_effector_pos()
            current_pos2 = self.robot2.get_end_effector_pos()
            target_pos = current_pos + dpos
            target_pos2 = current_pos2 + dpos2
            if target_pos[2] < -0.15:  # Force the end effector to not flattening to the table
                target_pos[2] = -0.15
            self.robot.apply_action(np.concatenate([target_pos, orn, fingers, tips]))

            if target_pos2[2] < -0.15:  # Force the end effector to not flattening to the table
                target_pos2[2] = -0.15
            self.robot2.apply_action(np.concatenate([target_pos2, orn2, fingers2, tips2]))
            for i in range(self.actionRepeat):
                self.p.stepSimulation()
                if self._render:
                    time.sleep(self.timestep)
            next_obs = self.get_observation()
            info = dict()
            if hasattr(self, 'previous_distance'):
                info = dict(previous_distance=self.previous_distance)
            reward, is_success = self.reward_and_success(next_obs['achieved_goal'], next_obs['desired_goal'], info)
            done = False
            if hasattr(self, 'unhealthy'):
                info['unhealthy'] = self.unhealthy
                if self.unhealthy:
                    done = True
            info['is_success'] = is_success
        else:
            action = np.clip(action, self.action_space.low, self.action_space.high)
            # TODO: scale action range
            dpos = np.asarray(action[0:3]) * 0.05  # Relative shift
            action = action[3:]
            if self.use_orn:
                orn = np.asarray(action[0:3]) * np.pi  # (-pi, pi)
                action = action[3:]
            elif self.use_roll:
                orn = np.asarray([0, -np.pi, action[0] * np.pi])
                action = action[1:]
            else:
                orn = np.array([0, -np.pi, np.pi / 2])
            assert len(action) == 1
            # cur_finger1, cur_finger2 = self.robot.get_finger_state()
            fingers = np.asarray([0.7505 + action[0] * 0.7505, 0.7505 + action[0] * 0.7505])
            fingers = np.clip(fingers, 0., 1.51)
            tips = np.asarray([0., 0.])
            current_pos = self.robot.get_end_effector_pos()
            target_pos = current_pos + dpos
            if target_pos[2] < -0.15:  # Force the end effector to not flattening to the table
                target_pos[2] = -0.15
            self.robot.apply_action(np.concatenate([target_pos, orn, fingers, tips]))
            for i in range(self.actionRepeat):
                self.p.stepSimulation()
                if self._render:
                    time.sleep(self.timestep)
            next_obs = self.get_observation()
            info = dict()
            if hasattr(self, 'previous_distance'):
                info = dict(previous_distance=self.previous_distance)
            reward, is_success = self.reward_and_success(next_obs['achieved_goal'], next_obs['desired_goal'], info)
            done = False
            if hasattr(self, 'unhealthy'):
                info['unhealthy'] = self.unhealthy
                if self.unhealthy:
                    done = True
            info['is_success'] = is_success
        return next_obs, reward, done, info

    def get_state(self):
        stateId = self.p.saveState()
        return stateId

    def set_state(self, state):
        self.p.restoreState(stateId=state)

    def render(self, mode='human'):
        if mode == 'rgb_array':
            view_matrix = self.p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.5, 0.0, 0.0],
                                                                   distance=1.2,
                                                                   yaw=0,
                                                                   pitch=-30,
                                                                   roll=0,
                                                                   upAxisIndex=2)
            proj_matrix = self.p.computeProjectionMatrixFOV(fov=60,
                                                            aspect=1.0,
                                                            nearVal=0.1,
                                                            farVal=100.0)
            (_, _, px, _, _) = self.p.getCameraImage(width=1000,
                                                     height=1000,
                                                     viewMatrix=view_matrix,
                                                     projectionMatrix=proj_matrix,
                                                     # renderer=p.ER_BULLET_HARDWARE_OPENGL,
                                                     )
            rgb_array = np.array(px, dtype=np.uint8)
            rgb_array = np.reshape(rgb_array, (1000, 1000, 4))

            rgb_array = rgb_array[:, :, :3]
            return rgb_array

    def __del__(self):
        self.p.disconnect()


class KinovaGymEnv(KinovaGymBaseEnv):
    def __init__(self, actionRepeat=40, timestep=1./240, render=False,
                 init_qpos=None, reward_type="dense", use_orn=False, use_roll=False, include_v=True, need_GUI = True,
                 obj_range=0.15, goal_range=0.15, init_block_xy=(0.5, 0.0), goal_in_the_air=False, double_agent = False
                 ):
        self.init_block_xy = init_block_xy
        self.distance_threshold = 0.03
        self.obj_range = obj_range
        self.goal_range = goal_range
        self.goal_in_the_air = goal_in_the_air
        self.block_height = None
        self.goal_height = None
        self.goal = None
        self.previous_distance = None
        self._double_agent = double_agent
        self._need_GUI = need_GUI
        # self.start_from_success = None
        super(KinovaGymEnv, self).__init__(actionRepeat=actionRepeat, timestep=timestep, render=render,
                                           init_qpos=init_qpos, reward_type=reward_type, use_orn=use_orn, need_GUI = need_GUI,
                                           use_roll=use_roll, include_v=include_v, double_agent=double_agent)

        # self.render("rgb_array")

    def _setup_env(self):
        super(KinovaGymEnv, self)._setup_env()
        self.blockId = self.p.loadURDF(os.path.join(BLOCK_MODEL_DIR, "block.urdf"),
                                       [self.init_block_xy[0], self.init_block_xy[1], -0.16],
                                       [0.0, 0.0, 0.0, 1.0])
        # self.goalId = self.p.loadURDF(os.path.join(SITE_MODEL_DIR, "site.urdf"), [0.3, -0.1, 0.0])

        self.goalId = self.p.loadURDF(os.path.join(BLOCK_MODEL_DIR, "block_rectangle.urdf"), [0.3, -0.1, 0.0], [0.0, 0.0, 0.0, 1.0])
        self.p.changeVisualShape(self.goalId, -1, rgbaColor=[1, 0, 0, 1])
        for _ in range(50):
            self.p.stepSimulation()
        blockPos, *_ = self.p.getBasePositionAndOrientation(self.blockId)
        self.block_height = blockPos[2]


    def reset(self):
        if self._double_agent:
            self.robot.reset()
            self.robot2.reset()
            # random block position
            block_xy = np.asarray(self.init_block_xy) + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            while np.linalg.norm(block_xy - self.robot.endEffectorPos[:2]) < 0.1 or np.linalg.norm(block_xy - self.robot2.endEffectorPos[:2]) < 0.1 :
                block_xy = np.asarray(self.init_block_xy) + self.np_random.uniform(-self.obj_range, self.obj_range,
                                                                                   size=2)
            self.p.resetBasePositionAndOrientation(self.blockId, (block_xy[0], block_xy[1], self.block_height),
                                                   (0.0, 0.0, 0.0, 1.0))
            self.p.stepSimulation()

            # Sample goal.
            self.goal = self._sample_goal()
            # # TODO: avoid too much random data getting reward
            # if np.linalg.norm(self.goal - np.concatenate([block_xy, [self.block_height]])) < self.distance_threshold:
            #     self.start_from_success = True
            # else:
            #     self.start_from_success = False
            # while np.linalg.norm(self.goal - np.concatenate([block_xy, [self.block_height]])) < self.distance_threshold:
            #     self.goal = self._sample_goal()
            self.p.resetBasePositionAndOrientation(self.goalId, tuple(self.goal), (0.0, 0.0, 0.0, 1.0))
            obs = self.get_observation()
            # print(obs)
            self.previous_distance = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])

        else:

            self.robot.reset()
            # random block position
            block_xy = np.asarray(self.init_block_xy) + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            while np.linalg.norm(block_xy - self.robot.endEffectorPos[:2]) < 0.1:
                block_xy = np.asarray(self.init_block_xy) + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            self.p.resetBasePositionAndOrientation(self.blockId, (block_xy[0], block_xy[1], self.block_height), (0.0, 0.0, 0.0, 1.0))
            self.p.stepSimulation()

            # Sample goal.
            self.goal = self._sample_goal()
            # # TODO: avoid too much random data getting reward
            # if np.linalg.norm(self.goal - np.concatenate([block_xy, [self.block_height]])) < self.distance_threshold:
            #     self.start_from_success = True
            # else:
            #     self.start_from_success = False
            # while np.linalg.norm(self.goal - np.concatenate([block_xy, [self.block_height]])) < self.distance_threshold:
            #     self.goal = self._sample_goal()
            self.p.resetBasePositionAndOrientation(self.goalId, tuple(self.goal), (0.0, 0.0, 0.0, 1.0))
            obs = self.get_observation()
            # print(obs)
            self.previous_distance = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
        return obs

    def _sample_goal(self):
        goal_xy = self.np_random.uniform(-self.goal_range, self.goal_range, size=2) + np.asarray(self.init_block_xy)
        goal_z = self.block_height
        if self.goal_in_the_air:
            if self.np_random.uniform() < 0.5:
                goal_z += self.np_random.uniform(0, 0.45)
        goal = np.concatenate([goal_xy, [goal_z]])
        return goal

    def get_site_position(self):
        sitePos, _ = self.p.getBasePositionAndOrientation(self.goalId)
        return sitePos

    def get_observation(self):
        if self._double_agent:
            end_effector_pos, end_effector_orn, end_effector_vel, finger_joints, tip_joints, finger_vel, tip_vel = self.robot.get_observation()
            end_effector_pos2, end_effector_orn2, end_effector_vel2, finger_joints2, tip_joints2, finger_vel2, tip_vel2 = self.robot2.get_observation()
            blockPos, blockOrn = self.p.getBasePositionAndOrientation(self.blockId)
            blockOrn = self.p.getEulerFromQuaternion(blockOrn)
            blockVel, blockVela = self.p.getBaseVelocity(self.blockId)
            end_effector_pos, end_effector_orn, end_effector_vel, finger_joints, tip_joints, finger_vel, tip_vel, blockPos, blockOrn, blockVel = \
                map(np.asarray,
                    [end_effector_pos, end_effector_orn, end_effector_vel, finger_joints, tip_joints, finger_vel,
                     tip_vel, blockPos, blockOrn, blockVel])
            end_effector_pos2, end_effector_orn2, end_effector_vel2, finger_joints2, tip_joints2, finger_vel2, tip_vel2 = \
                map(np.asarray,
                    [end_effector_pos2, end_effector_orn2, end_effector_vel2, finger_joints2, tip_joints2, finger_vel2,
                     tip_vel2])
            # print('block orn', blockOrn)
            relative_blockVel1 = blockVel - end_effector_vel
            relative_blockVel2 = blockVel - end_effector_vel2
            relativePos = blockPos - end_effector_pos
            relativePos2 = blockPos - end_effector_pos2
            if self.include_v:
                state = np.concatenate(
                    [end_effector_pos, end_effector_orn, blockPos, blockOrn, relativePos, finger_joints, tip_joints,
                     end_effector_vel, blockVel, relative_blockVel1, finger_vel, tip_vel])
                state2 = np.concatenate(
                    [end_effector_pos2, end_effector_orn2, blockPos, blockOrn, relativePos2, finger_joints2, tip_joints2,
                     end_effector_vel2, blockVel, relative_blockVel2, finger_vel2, tip_vel2])


            else:
                state = np.concatenate(
                    [end_effector_pos, end_effector_orn, blockPos, blockOrn, relativePos, finger_joints, tip_joints])
                state2 = np.concatenate(
                    [end_effector_pos2, end_effector_orn2, blockPos, blockOrn, relativePos2, finger_joints2, tip_joints2])

            state = np.concatenate([state, state2], axis=0)
            achieved_goal = blockPos
            # achieved_goal = np.concatenate([end_effector_pos, end_effector_pos2], axis = 0)
            # print(achieved_goal)
            desired_goal = self.goal.copy()
            # desired_goal2 = self.goal.copy()
            # desired_goal = end_effector_pos.copy()
            # desired_goal2 = end_effector_pos2.copy()
            desired_goal[2] += 0.05
            # desired_goal2 +=0.2
            # desired_goal = np.concatenate([desired_goal, desired_goal2], axis = 0)
            # print(desired_goal)
        else:
            end_effector_pos, end_effector_orn, end_effector_vel, finger_joints, tip_joints, finger_vel, tip_vel = self.robot.get_observation()
            blockPos, blockOrn = self.p.getBasePositionAndOrientation(self.blockId)
            blockOrn = self.p.getEulerFromQuaternion(blockOrn)
            blockVel, blockVela = self.p.getBaseVelocity(self.blockId)
            end_effector_pos, end_effector_orn, end_effector_vel, finger_joints, tip_joints, finger_vel, tip_vel, blockPos, blockOrn, blockVel = \
                map(np.asarray, [end_effector_pos, end_effector_orn, end_effector_vel, finger_joints, tip_joints, finger_vel, tip_vel, blockPos, blockOrn, blockVel])
            # print('block orn', blockOrn)
            blockVel -= end_effector_vel
            relativePos = blockPos - end_effector_pos
            if self.include_v:
                # state = np.concatenate([end_effector_pos, end_effector_orn, blockPos, blockOrn, relativePos, finger_joints, tip_joints, end_effector_vel, blockVel, finger_vel, tip_vel])
                state = np.concatenate(
                    [end_effector_pos, end_effector_orn, finger_joints, tip_joints, end_effector_vel, finger_vel,
                     tip_vel])
            else:
                state = np.concatenate([end_effector_pos, end_effector_orn, blockPos, blockOrn, relativePos, finger_joints, tip_joints])
            # achieved_goal = blockPos
            achieved_goal = end_effector_pos
            desired_goal = self.goal.copy()
            # desired_goal[2] += 0.05
            # desired_goal += 0.1

        return dict(observation=state, achieved_goal=achieved_goal, desired_goal=desired_goal)

    def step(self, action):
        next_obs, reward, done, info = super().step(action)
        if  info['is_success']:
            done = True
        return next_obs, reward, done, info

    def reward_and_success(self, achieved_goal, desired_goal, info=None, epsilon = 0.001):
        if self._double_agent:
            # print("achieved_goal is {},  disired_goal is {}".format(achieved_goal, desired_goal))
            distance = np.linalg.norm(achieved_goal - desired_goal)
            # print("distance is ", distance)
            stable = False
            if hasattr(self, "previous_distance"):
                if self.previous_distance and abs(self.previous_distance - distance) < epsilon:
                    stable = True
            self.previous_distance = distance
            if self.reward_type == "dense":
                # TODO: tweak
                reward = (-distance)
                # reward = info['previous_distance'] - distance
                return reward, distance < self.distance_threshold and stable
            elif self.reward_type == "shaped":
                end_effector_pos, *_ = self.robot.get_observation()
                end_effector_pos = np.asarray(end_effector_pos)
                dist_near = np.linalg.norm(end_effector_pos - achieved_goal)
                self.previous_distance = distance + 0.5 * dist_near
                reward = info['previous_distance'] - distance - 0.5 * dist_near
                return reward, distance < self.distance_threshold and stable
            elif self.reward_type == "incremental":
                reward = info['previous_distance'] - distance
                return reward, distance < self.distance_threshold and stable
            elif self.reward_type == "sparse":
                # Negative reward
                return (distance < self.distance_threshold).astype(np.float32) - 1, distance < self.distance_threshold and stable
            else:
                raise NotImplementedError

        else:
            distance = np.linalg.norm(achieved_goal - desired_goal)
            self.previous_distance = distance
            if self.reward_type == "dense":
                # TODO: tweak
                reward = (-distance)
                # reward = info['previous_distance'] - distance
                return reward, distance < self.distance_threshold
            elif self.reward_type == "shaped":
                end_effector_pos, *_ = self.robot.get_observation()
                end_effector_pos = np.asarray(end_effector_pos)
                dist_near = np.linalg.norm(end_effector_pos - achieved_goal)
                self.previous_distance = distance + 0.5 * dist_near
                reward = info['previous_distance'] - distance - 0.5 * dist_near
                return reward, distance < self.distance_threshold
            elif self.reward_type == "incremental":
                reward = info['previous_distance'] - distance
                return reward, distance < self.distance_threshold
            elif self.reward_type == "sparse":
                # Negative reward
                return (distance < self.distance_threshold).astype(np.float32) - 1, distance < self.distance_threshold
            else:
                raise NotImplementedError


class KinovaReach(KinovaGymEnv):
    def _sample_goal(self):
        return self.np_random.uniform(-self.goal_range, self.goal_range, size=3) + np.asarray([self.init_block_xy[0], self.init_block_xy[1], 0.0])

    def get_observation(self):
        end_effector_pos, end_effector_orn, end_effector_vel, finger_joints, tip_joints, finger_vel, tip_vel = self.robot.get_observation()
        end_effector_pos, end_effector_orn, end_effector_vel, finger_joints, tip_joints, finger_vel, tip_vel = \
            map(np.asarray, [end_effector_pos, end_effector_orn, end_effector_vel, finger_joints, tip_joints, finger_vel, tip_vel])
        if self.include_v:
            state = np.concatenate([end_effector_pos, end_effector_orn, finger_joints, tip_joints, end_effector_vel, finger_vel, tip_vel])
        else:
            state = np.concatenate([end_effector_pos, end_effector_orn, finger_joints, tip_joints])
        achieved_goal = end_effector_pos

        desired_goal = self.goal.copy()
        return dict(observation=state, achieved_goal=achieved_goal, desired_goal=desired_goal)

    def reward_and_success(self, achieved_goal, desired_goal, info=None):
        distance = np.linalg.norm(achieved_goal - desired_goal)
        self.previous_distance = distance
        if self.reward_type == "dense":
            # TODO: tweak
            reward = (- distance)
            return reward, distance < self.distance_threshold
        elif self.reward_type == "incremental":
            reward = info['previous_distance'] - distance
            return reward, distance < self.distance_threshold
        elif self.reward_type == "sparse":
            # Negative reward
            return (distance < self.distance_threshold).astype(np.float32) - 1, distance < self.distance_threshold
            # return (distance < self.distance_threshold).astype(np.float32), distance < self.distance_threshold
        else:
            raise NotImplementedError


class KinovaRealObjEnv(KinovaGymBaseEnv):
    def __init__(self, actionRepeat=10, timestep=1. / 240, render=False, init_qpos=None, obj_name="plate",
                 init_end_effecor_pos=(0.6, 0.0, -0.1), init_end_effector_orn=(np.pi / 2, np.pi / 2, np.pi / 2), reward_type="dense", use_orn=False, use_roll=True, include_v=False,
                 obj_range=0.15, goal_range=0.15, init_obj_xy=(0.7, 0.0), goal_in_the_air=False):
        # TODO: initial qpos needs to be tweaked
        if init_qpos is None:
            init_qpos = [0., 0., 0.179, 4.664, 1.317, -3.745, 2.215, 3.964, 0.0,
                         0.750, 0.0, 0.750, 0.0]
        self.obj_name = obj_name
        self.init_obj_xy = init_obj_xy
        self.distance_threshold = 0.05
        self.obj_range = obj_range
        self.goal_range = goal_range
        self.goal_in_the_air = goal_in_the_air
        self.obj_height = None
        self.goal = None
        self.previous_distance = None
        super(KinovaRealObjEnv, self).__init__(actionRepeat=actionRepeat, timestep=timestep, render=render,
                                               init_qpos=init_qpos, init_end_effector_pos=init_end_effecor_pos,
                                               init_end_effector_orn=init_end_effector_orn, reward_type=reward_type,
                                               use_orn=use_orn, use_roll=use_roll, include_v=include_v)

    def _setup_env(self):
        super(KinovaRealObjEnv, self)._setup_env()
        self.objId = self.p.loadURDF(os.path.join(BLOCK_MODEL_DIR, "%s.urdf" % self.obj_name),
                                     [self.init_obj_xy[0], self.init_obj_xy[1], -0.15],
                                     [0.0, 0.0, 0.0, 1.0])
        self.p.changeVisualShape(self.objId, -1, textureUniqueId=self.p.loadTexture(
            os.path.join(BLOCK_MODEL_DIR, "%s.png" % self.obj_name)))
        self.goalId = self.p.loadURDF(os.path.join(SITE_MODEL_DIR, "site.urdf"), [0.3, -0.1, 0.0])
        self.p.stepSimulation()
        objPos, *_ = self.p.getBasePositionAndOrientation(self.objId)
        self.obj_height = objPos[2]

    def reset(self):
        self.robot.reset()
        # random block position
        obj_xy = np.asarray(self.init_obj_xy) + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
        while np.linalg.norm(obj_xy - self.robot.endEffectorPos[:2]) < 0.1:
            obj_xy = np.asarray(self.init_obj_xy) + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
        self.p.resetBasePositionAndOrientation(self.objId, (obj_xy[0], obj_xy[1], self.obj_height), (0.0, 0.0, 0.0, 1.0))
        self.p.stepSimulation()

        # Sample goal.
        self.goal = self._sample_goal()
        self.p.resetBasePositionAndOrientation(self.goalId, tuple(self.goal[:3]), (0.0, 0.0, 0.0, 1.0))
        obs = self.get_observation()
        # print(obs)
        self.previous_distance = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
        return obs

    def _sample_goal(self):
        goal_xy = self.np_random.uniform(-self.goal_range, self.goal_range, size=2) + np.asarray(self.init_obj_xy)
        goal_z = self.obj_height
        if self.goal_in_the_air:
            if self.np_random.uniform() < 0.5:
                goal_z += self.np_random.uniform(0, 0.45)
        goal_nvec = np.array([0., 0., 1.])
        goal = np.concatenate([goal_xy, [goal_z], goal_nvec])
        return goal

    def get_site_position(self):
        sitePos, _ = self.p.getBasePositionAndOrientation(self.goalId)
        return sitePos

    def get_observation(self):
        end_effector_pos, end_effector_orn, end_effector_vel, finger_joints, tip_joints, finger_vel, tip_vel = self.robot.get_observation()
        objPos, objOrn = self.p.getBasePositionAndOrientation(self.objId)
        objNvec = np.asarray(self.p.getMatrixFromQuaternion(objOrn))[2: 9: 3]
        objOrn = self.p.getEulerFromQuaternion(objOrn)
        objVel, objVela = self.p.getBaseVelocity(self.objId)
        end_effector_pos, end_effector_orn, end_effector_vel, finger_joints, tip_joints, finger_vel, tip_vel, objPos, objOrn, objVel = \
            map(np.asarray, [end_effector_pos, end_effector_orn, end_effector_vel, finger_joints, tip_joints, finger_vel, tip_vel, objPos, objOrn, objVel])
        # print('block orn', blockOrn)
        objVel -= end_effector_vel
        relativePos = objPos - end_effector_pos
        if self.include_v:
            state = np.concatenate([end_effector_pos, end_effector_orn, objPos, objOrn, relativePos, finger_joints, tip_joints, end_effector_vel, objVel, finger_vel, tip_vel])
        else:
            state = np.concatenate([end_effector_pos, end_effector_orn, objPos, objOrn, relativePos, finger_joints, tip_joints])
        achieved_goal = np.concatenate([objPos, objNvec])
        desired_goal = self.goal.copy()
        return dict(observation=state, achieved_goal=achieved_goal, desired_goal=desired_goal)

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # TODO: scale action range
        dpos = np.asarray(action[0:3]) * 0.05  # Relative shift
        action = action[3:]
        if self.use_orn:
            orn = np.asarray(self.init_end_effector_orn) + np.asarray(action[0:3]) * 0.1 * np.pi  # (-pi, pi)
            for i in range(3):
                if orn[i] > np.pi:
                    orn[i] -= 2 * np.pi
                if orn[i] < -np.pi:
                    orn[i] += 2 * np.pi
            action = action[3:]
        elif self.use_roll:
            orn = np.asarray([np.pi / 2, action[0] * np.pi, np.pi / 2])
            action = action[1:]
        else:
            orn = np.array(self.init_end_effector_orn)
        assert len(action) == 1
        cur_finger1, cur_finger2 = self.robot.get_finger_state()
        fingers = np.asarray([cur_finger1 + action[0], cur_finger2 + action[0]])
        fingers = np.clip(fingers, 0., 1.51)
        tips = np.asarray([0.5, 0.5])
        current_pos = self.robot.get_end_effector_pos()
        target_pos = current_pos + dpos
        self.robot.apply_action(np.concatenate([target_pos, orn, fingers, tips]))
        for i in range(self.actionRepeat):
            self.p.stepSimulation()
            if self._render:
                time.sleep(self.timestep)
        next_obs = self.get_observation()
        info = dict(previous_distance=self.previous_distance)
        reward, is_success = self.reward_and_success(next_obs['achieved_goal'], next_obs['desired_goal'], info)
        done = False
        info['is_success'] = is_success
        return next_obs, reward, done, info

    def reward_and_success(self, achieved_goal, desired_goal, info=None):
        distance = np.linalg.norm(achieved_goal[:3] - desired_goal[:3])
        dotprod = np.dot(achieved_goal[3:], desired_goal[3:])
        self.previous_distance = distance
        success = np.logical_and(distance < self.distance_threshold, dotprod > 0.8)
        if self.reward_type == "dense":
            # TODO: tweak
            reward = info['previous_distance'] - distance + (dotprod - 1)
            return reward, success
        elif self.reward_type == "sparse":
            success = np.logical_and(distance < self.distance_threshold, dotprod > 0.8)
            return success.astype(np.float32) - 1, success
        else:
            raise NotImplementedError
