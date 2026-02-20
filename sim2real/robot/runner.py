import time
import numpy as np
import torch
from unitree_sdk2py.utils.thread import RecurrentThread

from sim2real.utils.remote_controller import KeyMap
from sim2real.robot.observations import Observations
from sim2real.robot.g1_ll import G1LowLevelController
from sim2real.robot.config import Config
import argparse


class PolicyRunner:

    def __init__(self, config: Config, net: str):
        self.config = config

        self.obs = Observations(config)
        self.ll = G1LowLevelController(config=config, net=net, observations=self.obs)

        self.policy = torch.jit.load(config.policy_path).eval()

        #Just to filter out dummy values from the policy
        for _ in range(50):
            with torch.inference_mode():
                dummy = self.obs.history.reshape(1, -1).astype(np.float32)
                self.policy(torch.from_numpy(dummy))

        self.run_thread = RecurrentThread(interval=self.config.control_dt, target=self._step)

    def start(self):
        self.ll.wait_for_start_signal()
        self.ll.move_to_default_pos()
        self.ll.wait_for_control_signal()

        print("Start Control!")
        self.run_thread.Start()

    def _step(self):
        low_state = self.ll.low_state

        obs_vec = self.obs.compute_obs(low_state)

        with torch.inference_mode():
            action_t = self.policy(torch.from_numpy(obs_vec).clip(-100, 100)).clip(-100, 100)

        action = action_t.detach().cpu().numpy().squeeze().astype(np.float32)
        self.obs.set_last_action(action)

        target_dof_pos = self.config.default_joint_pos + action * self.config.action_scale
        # Uncomment this to write to the robot
        self.ll.set_target_positions(target_dof_pos)

        if self.obs.remote.button[KeyMap.select] == 1:
            self.ll.stop_and_exit()

    def spin(self):
        try:
            while True:
                if self.obs.remote.button[KeyMap.select] == 1:
                    print("Select Button detected, Exit!")
                    break
                time.sleep(0.01)
        finally:
            self.run_thread.Wait()
            self.ll.publish_thread.Wait()
            self.ll.shutdown_to_damping()
            print("Exit")


if __name__ == "__main__":
    
    config_path = "sim2real/configs/g1.yaml"
    net = "enp8s0"
    config = Config(config_path)
    runner = PolicyRunner(config, net)
    runner.start()
    runner.spin()