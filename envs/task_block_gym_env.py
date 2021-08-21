import numpy as np

from envs.kinova_gym_env import *
from envs.task_block_env import TaskblockEnv

class TaskblockgymEnv(KinovaGymEnv):
    def __init__(self, actionRepeat=40, timestep=1./240, render=False,
                 init_qpos=None, reward_type="dense", use_orn=False, use_roll=False, include_v=True, need_GUI = True,
                 obj_range=0.15, goal_range=0.15, init_block_xy=(0.5, 0.0), goal_in_the_air=False, double_agent = False,
                 using_binary_task = False, task_family = ("rotate", "lift", "pick")):

        self.using_bin = using_binary_task
        self.task_family = task_family
        super(TaskblockgymEnv, self).__init__(actionRepeat=actionRepeat, timestep=timestep, render=render,
                 init_qpos=init_qpos, reward_type=reward_type, use_orn=use_orn, use_roll=use_roll, include_v=include_v, need_GUI = need_GUI,
                 obj_range=obj_range, goal_range=goal_range, init_block_xy=init_block_xy, goal_in_the_air=goal_in_the_air, double_agent = double_agent
                 )


        if self.using_bin:
            if self._double_agent:
                if self.use_orn:
                    self.action_space = spaces.Tuple((spaces.Discrete(len(task_family)), spaces.Discrete(len(task_family)),spaces.Box(-1., 1., shape=(14,), dtype=np.float32)))
                elif self.use_roll:
                    self.action_space = spaces.Tuple((spaces.Discrete(len(task_family)), spaces.Discrete(len(task_family)),spaces.Box(-1., 1., shape=(10,), dtype=np.float32)))
                else:
                    self.action_space = spaces.Tuple((spaces.Discrete(len(task_family)), spaces.Discrete(len(task_family)), spaces.Box(-1., 1., shape=(8,), dtype=np.float32)))
            else:
                if self.use_orn:
                    self.action_space = spaces.Tuple((spaces.Discrete(len(task_family)), spaces.Box(-1., 1., shape=(7,), dtype=np.float32)))
                elif self.use_roll:
                    self.action_space = spaces.Tuple((spaces.Discrete(len(task_family)), spaces.Box(-1., 1., shape=(5,), dtype=np.float32)))
                else:
                    self.action_space = spaces.Tuple((spaces.Discrete(len(task_family)), spaces.Box(-1., 1., shape=(4,), dtype=np.float32)))

    def _setup_env(self):

        if self._render:
            physics_client = p.connect(p.GUI)
            if not self._need_GUI:
                p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            else:
                self.p = PhysClientWrapper(p, physics_client)
                self.render("rgb_array")
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
        self.robot = TaskblockEnv(self.p, init_qpos=self.init_qpos, init_end_effector_pos=self.init_end_effector_pos,
                                  init_end_effector_orn=self.init_end_effector_orn, using_binary_task=self.using_bin, task_family=self.task_family)

        if self._double_agent:

            self.robot2 = TaskblockEnv(self.p, start_pos=(1, 0, -0.2),init_qpos=self.init_qpos, init_end_effector_pos=self.init_end_effector_pos,
                                       init_end_effector_orn=self.init_end_effector_orn, using_binary_task=self.using_bin, task_family=self.task_family)
        self.p.stepSimulation()

        self.blockId = self.p.loadURDF(os.path.join(BLOCK_MODEL_DIR, "block.urdf"),
                                       [self.init_block_xy[0], self.init_block_xy[1], -0.16],
                                       [0.0, 0.0, 0.0, 1.0])
        self.goalId = self.p.loadURDF(os.path.join(SITE_MODEL_DIR, "site.urdf"), [0.3, -0.1, 0.0])
        # self.goalId = self.p.loadURDF(os.path.join(r"C:\Users\xintao201712\Documents\haihua\xarm_soft-main\pybullet_xarm_envs\site_description", "site.urdf"), [0.3, -0.1, 0.0])
        # self.goalId = self.p.loadURDF(os.path.join(BLOCK_MODEL_DIR, "block_retctangle.urdf"), [0.3, -0.1, 0.0],
        #                               [0.0, 0.0, 0.0, 1.0])
        self.p.changeVisualShape(self.goalId, -1, rgbaColor=[1, 0, 0, 1])
        for _ in range(50):
            self.p.stepSimulation()
        blockPos, *_ = self.p.getBasePositionAndOrientation(self.blockId)
        self.block_height = blockPos[2]

    def _step(self, action : tuple):
        if self._need_GUI:
            self.render("rgb_array")
        if self._double_agent:
            if self.using_bin:
                task_mask = [int(action[0])]
                task_mask2 = [int(action[1])]
                # action[0] = np.clip(action[0], 0, len(self.task_family) -0.001)

                action = np.clip(action[2], self.action_space[2].low, self.action_space[2].high)
            else:
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
                action2 = action2[1:]
            else:
                orn = np.array([0, -np.pi, np.pi / 2])
                orn2 = orn
            assert len(action) == 1 and len(action2) == 1
            # cur_finger1, cur_finger2 = self.robot.get_finger_state()
            fingers = np.asarray([0.7505 + action[0] * 0.7505, 0.7505 + action[0] * 0.7505])
            fingers2 = np.asarray([0.7505 + action2[0] * 0.7505, 0.7505 + action2[0] * 0.7505])
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
            if self.using_bin:
                self.robot.apply_action(np.concatenate([task_mask, target_pos, orn, fingers, tips]))
            else:
                self.robot.apply_action(np.concatenate([target_pos, orn, fingers, tips]))
            if target_pos2[2] < -0.15:  # Force the end effector to not flattening to the table
                target_pos2[2] = -0.15

            if self.using_bin:
                self.robot2.apply_action(np.concatenate([task_mask2, target_pos2, orn2, fingers2, tips2]))
            else:
                self.robot2.apply_action(np.concatenate([target_pos, orn2, fingers2, tips2]))
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
            if self.using_bin:

                # action[0] = np.clip(action[0], 0, len(self.task_family) - 0.001)
                task_mask = [int(action[0])]
                action = np.clip(action[1 ], self.action_space[1].low, self.action_space[1].high)
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
            if self.using_bin:
                self.robot.apply_action(np.concatenate([task_mask, target_pos, orn, fingers, tips]))
            else:
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

    def step(self, action):
        next_obs, reward, done, info = self._step(action)
        if  info['is_success']:
            done = True
        return next_obs, reward, done, info

