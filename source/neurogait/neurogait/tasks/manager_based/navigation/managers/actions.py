"""Action configuration for the NeuroGait navigation task."""

from isaaclab.envs import mdp as env_mdp
from isaaclab.utils import configclass


@configclass
class ActionsCfg:
    """Joint position actions for legged locomotion."""

    joint_pos = env_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.5,
        use_default_offset=True,
    )
