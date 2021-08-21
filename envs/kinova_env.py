import os, time
import math
import numpy as np


BLOCK_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'kinova_description', 'urdf')


class KinovaEnv(object):
    def __init__(self, physics_client, urdfrootpath=BLOCK_MODEL_DIR, start_pos = (0., 0., -0.1) ,init_qpos=None,
                 init_end_effector_pos=(0.5, 0., -0.1), init_end_effector_orn=(0, -math.pi, math.pi/2), lock_finger=False):
        self.p = physics_client
        self.urdfrootpath = urdfrootpath
        self.init_qpos = init_qpos
        self.endEffectorPos = init_end_effector_pos
        self.endEffectorOrn = init_end_effector_orn
        self.lock_finger = lock_finger
        self._kinova = None
        self.motorNames = []
        self.motorIndices = []
        self.end_effector_index = 8
        self.finger_index = [9, 11]
        self.finger_tip_index = [10, 12]
        # self.maxVelocity = .35  # TODO
        self.maxVelocity = []
        # self.maxForce = 2000.  # TODO
        self.maxForce = []
        self.jointDamping = []
        self.IKInfo = None


        start_orientation = [0., 0., 1., 0.]
        self._kinova = self.p.loadURDF(os.path.join(self.urdfrootpath, 'j2n6s200.urdf'), start_pos, start_orientation, useFixedBase=1)
        self.num_joints = self.p.getNumJoints(self._kinova)
        for j_idx in range(self.p.getNumJoints(self._kinova)):
            joint_info = self.p.getJointInfo(self._kinova, j_idx)
            qIndex = joint_info[3]
            if qIndex > -1:
                self.motorNames.append(str(joint_info[1]))
                self.motorIndices.append(j_idx)
            self.maxVelocity.append(joint_info[11])
            self.maxForce.append(joint_info[10])
            self.jointDamping.append(joint_info[6])
        self.compute_ik_information()
        self.reset()

    def reset(self):
        for j_idx in range(self.p.getNumJoints(self._kinova)):
            self.p.resetJointState(self._kinova, j_idx, self.init_qpos[j_idx])
        # print(self.motorNames, self.motorIndices)
        # print('before:', p.getLinkState(self._kinova, self.end_effector_index)[0], p.getLinkState(self._kinova, self.end_effector_index)[1])
        endEffectorOrn = self.p.getQuaternionFromEuler(self.endEffectorOrn)
        fingers_state = np.array([0.7505, 0.7505])
        tips_state = np.array([0., 0.])
        target_endpos = np.asarray(self.endEffectorPos) + np.concatenate([np.random.uniform(-0.15, 0.15, size=2), [0.]])
        jointPoses = self.p.calculateInverseKinematics(self._kinova,
                                                       self.end_effector_index,
                                                       target_endpos,
                                                       endEffectorOrn,
                                                       **self.IKInfo)
        for i in range(len(self.motorIndices)):
            self.p.resetJointState(self._kinova, self.motorIndices[i], jointPoses[i])
        self.p.stepSimulation()

        # self._move_end_effector(target_endpos, endEffectorOrn, fingers_state, tips_state)
        # for i in range(500):
        #     # self._move_end_effector(self.endEffectorPos, endEffectorOrn, fingers_state, tips_state)
        #     if i % 25 == 0:
        #         self._move_end_effector(target_endpos, endEffectorOrn, fingers_state, tips_state)
        #     self.p.stepSimulation()
        #     if np.linalg.norm(self.get_end_effector_pos() - target_endpos) < 0.05:
        #         # print('achieve early')
        #         break
        # print('initialization', 'target', target_endpos, 'actual', self.get_end_effector_pos())
        # print('after', self.p.getLinkState(self._kinova, self.end_effector_index)[0], self.p.getEulerFromQuaternion(self.p.getLinkState(self._kinova, self.end_effector_index)[1]))
        # for i in range(self.p.getNumJoints(self._kinova)):
        #     print(self.p.getJointState(self._kinova, i))
        # exit()

    def compute_ik_information(self):
        """ Finds the values for the IK solver. """
        joint_information = list(
            map(lambda i: self.p.getJointInfo(self._kinova, i),
                range(self.num_joints)))
        self.IKInfo = {}
        assert all([len(joint_information[i]) == 17 for i in range(self.num_joints)])
        self.IKInfo["solver"] = 0
        self.IKInfo["jointDamping"] = [0.005] * self.num_joints

    def get_observation(self):
        end_effector_state = self.p.getLinkState(self._kinova, self.end_effector_index, computeLinkVelocity=1)
        end_effector_pos, end_effector_orn, _, _, _, _, end_effector_vl, end_effector_va = end_effector_state
        end_effector_orn = self.p.getEulerFromQuaternion(end_effector_orn)
        finger1_state, finger2_state = self.p.getJointStates(self._kinova, self.finger_index)
        tip1_state, tip2_state = self.p.getJointStates(self._kinova, self.finger_tip_index)
        finger1_pos, finger1_vel, *_ = finger1_state
        finger2_pos, finger2_vel, *_ = finger2_state
        tip1_pos, tip1_vel, *_ = tip1_state
        tip2_pos, tip2_vel, *_ = tip2_state
        return end_effector_pos, end_effector_orn, end_effector_vl, (finger1_pos, finger2_pos), (tip1_pos, tip2_pos), \
               (finger1_vel, finger2_vel), (tip1_vel, tip2_vel)

    def apply_action(self, action):
        position = action[0:3]  # Now absolute position
        orn = self.p.getQuaternionFromEuler(action[3:6])
        # fingers
        fingers = action[6:8]
        tips = action[8:10]
        self._move_end_effector(position, orn, fingers, tips)
        self._step_callback()

    def _move_end_effector(self, pos, orn, fingers, finger_tips):
        jointPoses = self.p.calculateInverseKinematics(self._kinova,
                                                       self.end_effector_index,
                                                       pos,
                                                       orn,
                                                       **self.IKInfo)
        '''
        # For debugging
        for i in range(len(self.motorIndices)):
            self.p.resetJointState(self._kinova, self.motorIndices[i], jointPoses[i])
        self.p.stepSimulation()
        print('debugging', 'target', pos, 'actual', self.get_end_effector_pos())
        # exit()
        '''
        current_jointPoses = self.p.getJointStates(self._kinova, self.motorIndices)
        current_jointPos = [current_jointPose[0] for current_jointPose in current_jointPoses]
        # for i in range(len(self.motorIndices)):
        #     self.p.setJointMotorControl2(bodyUniqueId=self._kinova,
        #                                  jointIndex=self.motorIndices[i],
        #                                  controlMode=self.p.POSITION_CONTROL,
        #                                  targetPosition=jointPoses[i],
        #                                  targetVelocity=0,
        #                                  force=self.maxForce[self.motorIndices[i]],
        #                                  maxVelocity=self.maxVelocity[self.motorIndices[i]],
        #                                  # positionGain=0.03,
        #                                  positionGain=0.03,
        #                                  # velocityGain=1,
        #                                  velocityGain=0.3,)
        for i in range(len(self.motorIndices)):
            targetVelocity = 12 * (jointPoses[i] - current_jointPos[i])
            targetVelocity = np.clip(targetVelocity, -self.maxVelocity[self.motorIndices[i]], self.maxVelocity[self.motorIndices[i]])
            # print(targetVelocity)
            targetVelocity = np.clip(targetVelocity, -30, 30)
            self.p.setJointMotorControl2(bodyUniqueId=self._kinova,
                                         jointIndex=self.motorIndices[i],
                                         controlMode=self.p.VELOCITY_CONTROL,
                                         targetVelocity=targetVelocity,
                                         force=self.maxForce[self.motorIndices[i]],
                                         )
        for i in range(len(self.finger_index)):
            self.p.setJointMotorControl2(bodyUniqueId=self._kinova,
                                         jointIndex=self.finger_index[i],
                                         controlMode=self.p.POSITION_CONTROL,
                                         targetPosition=fingers[i],
                                         targetVelocity=0,
                                         # force=self.maxForce,
                                         force=self.maxForce[self.finger_index[i]],
                                         maxVelocity=self.maxVelocity[self.finger_index[i]],
                                         positionGain=3,
                                         velocityGain=0,
                                         # velocityGain=1
                                         )
        for i in range(len(self.finger_tip_index)):
            self.p.setJointMotorControl2(bodyUniqueId=self._kinova,
                                         jointIndex=self.finger_tip_index[i],
                                         controlMode=self.p.POSITION_CONTROL,
                                         targetPosition=finger_tips[i],
                                         targetVelocity=0,
                                         # force=self.maxForce,
                                         force=self.maxForce[self.finger_tip_index[i]],
                                         maxVelocity=self.maxVelocity[self.finger_tip_index[i]],
                                         positionGain=3,
                                         velocityGain=0,
                                         # velocityGain=1,
                                         )

    def get_end_effector_pos(self):
        state = self.p.getLinkState(self._kinova, self.end_effector_index)
        return np.asarray(state[0])

    def get_end_effector_orn(self):
        state = self.p.getLinkState(self._kinova, self.end_effector_index)
        return np.asarray(self.p.getEulerFromQuaternion(state[1]))

    def get_finger_state(self):
        finger1_state, finger2_state = self.p.getJointStates(self._kinova, self.finger_index)
        finger1_tip, finger2_tip = self.p.getJointStates(self._kinova, self.finger_tip_index)
        finger1_pos, *_ = finger1_state
        finger2_pos, *_ = finger2_state
        finger1_tip_pos, *_ = finger1_tip
        finger2_tip_pos, *_ = finger2_tip
        return finger1_pos, finger2_pos, finger1_tip_pos, finger2_tip_pos

    def _step_callback(self):
        if self.lock_finger:
            for i in self.finger_index:
                self.p.resetJointState(self._kinova, i, 1.51)
            for i in self.finger_tip_index:
                self.p.resetJointState(self._kinova, i, 0.)
