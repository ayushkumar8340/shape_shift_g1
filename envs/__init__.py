# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).


from envs.base.base_env import BaseEnv
from envs.base.base_env_config import BaseAgentCfg, BaseEnvCfg
from envs.base.crouch_env import G1CrouchEnv
from envs.g1.g1_config import (
    G1FlatAgentCfg,
    G1FlatEnvCfg,
    G1RoughAgentCfg,
    G1RoughEnvCfg,
)

from envs.g1.g1_crouch_walk_config import(
    G1CrouchFlatEnvCfg,
    G1CrouchFlatAgentCfg,
    G1CrouchRoughEnvCfg,
    G1CrouchRoughAgentCfg
)

from utils.task_registry import task_registry

task_registry.register("g1_flat", BaseEnv, G1FlatEnvCfg(), G1FlatAgentCfg())
task_registry.register("g1_rough", BaseEnv, G1RoughEnvCfg(), G1RoughAgentCfg())


task_registry.register("g1_crouch_flat",G1CrouchEnv,G1CrouchFlatEnvCfg(),G1CrouchFlatAgentCfg())
