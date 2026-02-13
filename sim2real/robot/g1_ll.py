import sys
import time
from threading import Lock

import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import (
    unitree_go_msg_dds__LowCmd_,
    unitree_go_msg_dds__LowState_,
    unitree_hg_msg_dds__LowCmd_,
    unitree_hg_msg_dds__LowState_,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

from sim2real.utils.command_helper import MotorMode, create_damping_cmd, init_cmd_go, init_cmd_hg
from sim2real.utils.remote_controller import KeyMap


class G1LowLevelController:
    def __init__(self, config, net: str, observations):
        self.config = config
        self.obs = observations 
        self.cmd_lock = Lock()

        ChannelFactoryInitialize(0, net)

        if config.msg_type == "hg":
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
            self.lowstate_subscriber.Init(self._low_state_handler, 10)

        elif config.msg_type == "go":
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
            self.low_state = unitree_go_msg_dds__LowState_()

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
            self.lowstate_subscriber.Init(self._low_state_handler, 10)
        else:
            raise ValueError("Invalid msg_type")

        self.publish_thread = RecurrentThread(interval=1 / 500, target=self._publish)

        self._wait_for_low_state()

        if config.msg_type == "hg":
            self.low_cmd = init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        else:
            self.low_cmd = init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

        self.publish_thread.Start()

    def _low_state_handler(self, msg):
        self.low_state = msg
        self.obs.set_remote_from_wireless(self.low_state.wireless_remote)

    def _publish(self):
        with self.cmd_lock:
            self.low_cmd.crc = CRC().Crc(self.low_cmd)
            self.lowcmd_publisher_.Write(self.low_cmd)

    def _wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        self.mode_machine_ = self.low_state.mode_machine
        print("Successfully connected to the robot.")

    def stop_and_exit(self):
        print("Select Button detected, Exit!")
        self.publish_thread.Wait()
        with self.cmd_lock:
            self.low_cmd = create_damping_cmd(self.low_cmd)
            self.low_cmd.crc = CRC().Crc(self.low_cmd)
            self.lowcmd_publisher_.Write(self.low_cmd)
        time.sleep(0.2)
        sys.exit(0)

    def shutdown_to_damping(self):
        with self.cmd_lock:
            self.low_cmd = create_damping_cmd(self.low_cmd)
            self.low_cmd.crc = CRC().Crc(self.low_cmd)
            self.lowcmd_publisher_.Write(self.low_cmd)
        time.sleep(0.2)

    def wait_for_start_signal(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal to move to default pos...")
        while self.obs.remote.button[KeyMap.start] != 1:
            if self.obs.remote.button[KeyMap.select] == 1:
                self.stop_and_exit()
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
        print("Moving to default pos.")
        total_time = 2.0
        num_step = int(total_time / self.config.control_dt)

        dof_idx = self.config.joint2motor_idx
        dof_size = len(dof_idx)

        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q

        for i in range(num_step):
            if self.obs.remote.button[KeyMap.select] == 1:
                self.stop_and_exit()

            alpha = i / num_step
            with self.cmd_lock:
                for j in range(dof_size):
                    motor_idx = dof_idx[j]
                    target_pos = self.config.default_joint_pos[j]
                    self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                    self.low_cmd.motor_cmd[motor_idx].dq = 0
                    self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[j]
                    self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[j]
                    self.low_cmd.motor_cmd[motor_idx].tau = 0

            time.sleep(self.config.control_dt)

    def wait_for_control_signal(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal to Start Control...")
        while self.obs.remote.button[KeyMap.A] != 1:
            if self.obs.remote.button[KeyMap.select] == 1:
                self.stop_and_exit()
            time.sleep(self.config.control_dt)

    def set_target_positions(self, target_dof_pos: np.ndarray):
        with self.cmd_lock:
            for i, motor_idx in enumerate(self.config.joint2motor_idx):
                self.low_cmd.motor_cmd[motor_idx].q = float(target_dof_pos[i])