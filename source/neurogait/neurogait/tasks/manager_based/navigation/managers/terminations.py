"""Termination configuration for the NeuroGait navigation task."""

from isaaclab.envs import mdp as env_mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass


@configclass
class TerminationsCfg:
    """Termination conditions."""

    time_out = DoneTerm(func=env_mdp.time_out, time_out=True)

    base_contact = DoneTerm(
        func=env_mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
            "threshold":  1.0,
        },
    )
