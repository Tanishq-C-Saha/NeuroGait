"""Observation configuration for the NeuroGait navigation task."""

from isaaclab.envs import mdp as env_mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise


@configclass
class ObservationsCfg:
    """Proprioceptive observations for the policy.

    Camera-based terms (occupancy_grid) are added per-CP in navigation_env_cfg.py
    so the base config stays robot-agnostic.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Standard legged-locomotion observations."""

        base_lin_vel      = ObsTerm(func=env_mdp.base_lin_vel,      noise=Unoise(n_min=-0.1,  n_max=0.1))
        base_ang_vel      = ObsTerm(func=env_mdp.base_ang_vel,      noise=Unoise(n_min=-0.2,  n_max=0.2))
        projected_gravity = ObsTerm(func=env_mdp.projected_gravity,  noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=env_mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos         = ObsTerm(func=env_mdp.joint_pos_rel,     noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel         = ObsTerm(func=env_mdp.joint_vel_rel,     noise=Unoise(n_min=-1.5,  n_max=1.5))
        actions           = ObsTerm(func=env_mdp.last_action)
        height_scan       = ObsTerm(
            func=env_mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption  = True
            self.concatenate_terms  = True

    policy: PolicyCfg = PolicyCfg()
