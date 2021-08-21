from envs.kinova_env import *
import pybullet as p
from envs.kinova_gym_env import *
class TaskblockEnv(KinovaEnv):
    def __init__(self, physics_client, urdfrootpath=BLOCK_MODEL_DIR, start_pos=(0., 0., -0.2), init_qpos=None,
                 init_end_effector_pos=(0.5, 0., -0.1), init_end_effector_orn=(0, -math.pi, math.pi / 2),
                 lock_finger=False, task_family = None, using_binary_task = False):
        super(TaskblockEnv, self).__init__(physics_client = physics_client, urdfrootpath=urdfrootpath, start_pos = start_pos,init_qpos=init_qpos,
                 init_end_effector_pos=init_end_effector_pos, init_end_effector_orn=init_end_effector_orn, lock_finger=lock_finger)
        self.task_family = task_family
        self.using_bin = using_binary_task


    def choose_task(self, action):
        # print("action is ", action)
        position = self.get_end_effector_pos()
        orn = self.get_end_effector_orn()
        fingers = self.get_finger_state()[:2]
        tips = self.get_finger_state()[2:]

        if self.task_family[int(action[0])] == "lift":
            position = action[1:4]
            fingers = [0.0, 0.0]
            tips = [0.0, 0.0]
            orn = [0.0, 0.0, 1.0]



        if self.task_family[int(action[0])] == "rotate":
            orn = self.p.getQuaternionFromEuler(action[4:7])
            fingers = [0.0, 0.0]
            tips = [0.0, 0.0]


        if self.task_family[int(action[0])] == "pick":
            fingers = action[7:9]
            tips = action[9:11]

        self._move_end_effector(position, orn, fingers, tips)
        self._step_callback()

    def apply_action(self, action):
        if self.using_bin:
            self.choose_task(action)
        else:
                position = action[0:3]  # Now absolute position
                orn = self.p.getQuaternionFromEuler(action[3:6])
                # fingers
                fingers = action[6:8]
                tips = action[8:10]
                self._move_end_effector(position, orn, fingers, tips)
                self._step_callback()


if __name__ == "__main__":
    task_family = ["lift", "pick", "rotate"]
    init_qpos = [0., 0., -0.127, 4.234, 1.597, -0.150, 0.585, 2.860, 0.0,
                 0.7505, 0.0, 0.7505, 0.0]
    BLOCK_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'kinova_description', 'urdf')
    physics_client =p.connect(p.GUI)
    physics_client = PhysClientWrapper(p, physics_client)
    env = TaskblockEnv(physics_client, BLOCK_MODEL_DIR, init_qpos = init_qpos, task_family=task_family)