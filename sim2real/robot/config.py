import numpy as np
import yaml


class Config:
    def __init__(self, file_path) -> None:
        with open(file_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

            self.control_dt = config["control_dt"]

            self.msg_type = config["msg_type"]
            self.imu_type = config["imu_type"]

            self.weak_motor = []
            if "weak_motor" in config:
                self.weak_motor = config["weak_motor"]

            self.lowcmd_topic = config["lowcmd_topic"]
            self.lowstate_topic = config["lowstate_topic"]

            self.policy_path = config["policy_path"]

            self.joint2motor_idx = config["joint2motor_idx"]
            self.kps = config["kps"]
            self.kds = config["kds"]
            self.default_joint_pos = np.array(config["default_joint_pos"], dtype=np.float32)

            if "torso_idx" in config:
                self.torso_idx = config["torso_idx"]

            self.ang_vel_scale = config["ang_vel_scale"]
            self.dof_pos_scale = config["dof_pos_scale"]
            self.dof_vel_scale = config["dof_vel_scale"]
            self.action_scale = config["action_scale"]
            self.command_scale = np.array(config["command_scale"], dtype=np.float32)

            self.num_actions = config["num_actions"]
            self.num_obs = config["num_obs"]

            self.history_length = config["history_length"]
            self.command_range = config["command_range"]
