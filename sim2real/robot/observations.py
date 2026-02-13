import numpy as np
from sim2real.utils.rot_helper import get_gravity_orientation, transform_imu_data
from sim2real.utils.remote_controller import RemoteController


class Observations:
    def __init__(self, config):
        self.config = config
        self.remote = RemoteController()

        self.joint_pos = np.zeros(config.num_actions, dtype=np.float32)
        self.joint_vel = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)

        self.current_obs = np.zeros(config.num_obs, dtype=np.float32)
        self.history = np.zeros((config.history_length, config.num_obs), dtype=np.float32)
        self.first_run = True

        self.clip_min_command = np.array(
            [
                self.config.command_range["lin_vel_x"][0],
                self.config.command_range["lin_vel_y"][0],
                self.config.command_range["ang_vel_z"][0],
            ],
            dtype=np.float32,
        )
        self.clip_max_command = np.array(
            [
                self.config.command_range["lin_vel_x"][1],
                self.config.command_range["lin_vel_y"][1],
                self.config.command_range["ang_vel_z"][1],
            ],
            dtype=np.float32,
        )

    def set_remote_from_wireless(self, wireless_remote):
        self.remote.set(wireless_remote)

    def _read_joints(self, low_state):
        for i, motor_idx in enumerate(self.config.joint2motor_idx):
            self.joint_pos[i] = low_state.motor_state[motor_idx].q
            self.joint_vel[i] = low_state.motor_state[motor_idx].dq

    def _read_imu(self, low_state):
        quat = low_state.imu_state.quaternion
        ang_vel = np.array([low_state.imu_state.gyroscope], dtype=np.float32)

        if self.config.imu_type == "torso":
            waist_yaw = low_state.motor_state[self.config.torso_idx].q
            waist_yaw_omega = low_state.motor_state[self.config.torso_idx].dq
            quat, ang_vel = transform_imu_data(
                waist_yaw=waist_yaw,
                waist_yaw_omega=waist_yaw_omega,
                imu_quat=quat,
                imu_omega=ang_vel,
            )

        return quat, ang_vel

    def compute_obs(self, low_state) -> np.ndarray:
        self._read_joints(low_state)
        quat, ang_vel = self._read_imu(low_state)

        gravity_orientation = get_gravity_orientation(quat)

        joint_pos = (self.joint_pos - self.config.default_joint_pos) * self.config.dof_pos_scale
        joint_vel = self.joint_vel * self.config.dof_vel_scale
        ang_vel = ang_vel * self.config.ang_vel_scale

        command = np.array(
            [self.remote.ly, -self.remote.lx, -self.remote.rx],
            dtype=np.float32,
        )
        command *= self.config.command_scale
        command = np.clip(command, self.clip_min_command, self.clip_max_command)

        num_actions = self.config.num_actions
        self.current_obs[:3] = ang_vel
        self.current_obs[3:6] = gravity_orientation
        self.current_obs[6:9] = command
        self.current_obs[9 : 9 + num_actions] = joint_pos
        self.current_obs[9 + num_actions : 9 + num_actions * 2] = joint_vel
        self.current_obs[9 + num_actions * 2 : 9 + num_actions * 3] = self.action

        if self.first_run:
            self.history[:] = self.current_obs.reshape(1, -1)
            self.first_run = False
        else:
            self.history = np.concatenate((self.history[1:], self.current_obs.reshape(1, -1)), axis=0)

        return self.history.reshape(1, -1).astype(np.float32)

    def set_last_action(self, action: np.ndarray):
        self.action = action.astype(np.float32, copy=False)