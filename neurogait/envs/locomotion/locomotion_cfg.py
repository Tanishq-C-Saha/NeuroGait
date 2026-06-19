# d:/CAPSTONE/Navigation/neurogait/envs/locomotion/locomotion_cfg.py

class LocomotionConfig:
    """
    Locomotion parameters (limits, default stance offsets) for Unitree Go2.
    """
    default_joint_angles = {
        "FL_hip": 0.0, "FL_thigh": 0.9, "FL_calf": -1.8,
        "FR_hip": 0.0, "FR_thigh": 0.9, "FR_calf": -1.8,
        "RL_hip": 0.0, "RL_thigh": 0.9, "RL_calf": -1.8,
        "RR_hip": 0.0, "RR_thigh": 0.9, "RR_calf": -1.8
    }
    
    stiffness = 20.0
    damping = 0.5
    action_scale = 0.25
    decimation = 4
