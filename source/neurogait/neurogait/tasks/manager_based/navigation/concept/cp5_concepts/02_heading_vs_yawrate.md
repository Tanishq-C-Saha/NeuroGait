# 02: Heading Target vs Yaw Rate

## The confusion
What does `action[2]` from the navigation policy represent?
Does the locomotion policy want a heading angle or a yaw rate?

## The answer
`action[2]` is a **heading target angle** in radians, range **[-π, π]**.
It is NOT a yaw rate. Sending yaw_rate as heading causes uncontrolled spinning.

## How it works

When `heading_command=True` is set in the locomotion's `UniformVelocityCommandCfg`, the command
generator samples a target heading angle instead of a yaw rate. The locomotion policy's internal
PD controller then computes the required yaw_rate to turn toward that heading.

The locomotion policy's observation always has 3 velocity_command dims: `[vx, vy, third]`.
The `third` slot's interpretation depends on how the policy was trained:
- Trained with `heading_command=False` → `third` = yaw_rate (rad/s)
- Trained with `heading_command=True` → `third` = heading angle (rad)

The NeuroGait navigation locomotion was trained with `heading_command=True` (see
`managers/commands.py` line 19: `heading_command=True`).

Therefore: navigation policy must output heading angles, not yaw rates.

## CP5 Attempt 1 bug

```
# WRONG — treated heading slot as yaw_rate
vel_term.vel_command_b[0, 2] = yaw_rate   # sent ≈ +2.749 rad/s
# Robot spun erratically and never reached goal
```

```
# CORRECT — output heading target from navigation policy
action[2] ∈ [-π, π]   # locomotion PD controller handles the turn
```

## What breaks if you get this wrong

Robot spins erratically. High yaw_rate values (e.g., 2.749 rad/s) exceed the locomotion
policy's training distribution, causing instability. The robot cannot track the path.

## Code reference

- `neurogait/.../locomotion/velocity/managers/commands.py` line 19: `heading_command=True`
- `IsaacLab/.../mdp/pre_trained_policy_action.py` line 63: velocity_commands remapping
- CP5 model: `models/navigation_policy.py` compute() applies `π * tanh(action[2])`
