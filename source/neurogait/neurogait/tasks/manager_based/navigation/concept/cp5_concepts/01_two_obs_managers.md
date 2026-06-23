# 01: Two Observation Managers

## The confusion
How does the navigation policy see 1615 dims when the locomotion policy needs 235 dims?
Why can't both use the same obs group?

## The answer
`PreTrainedPolicyAction` creates a **second, hidden** `ObservationManager` inside itself.
Both managers read from the same `env.scene["robot"]` but produce different observation vectors
for different consumers.

## How it works

```
Navigation env.step()
  │
  ├── ObservationManager.compute("policy")   ← navigation obs manager
  │     └── occupancy_grid (1600)
  │     └── future_waypoints (9)
  │     └── robot_velocity (3)
  │     └── projected_gravity (3)
  │     = (num_envs, 1615) tensor  →  navigation_policy(obs)  →  [vx, vy, heading]
  │
  └── ActionManager.apply("pre_trained_policy_action")
        └── PreTrainedPolicyAction.apply_actions()
              └── self._low_level_obs_manager.compute("ll_policy")   ← hidden loco obs manager
                    └── base_lin_vel (3)
                    └── base_ang_vel (3)
                    └── projected_gravity (3)
                    └── velocity_commands → self._raw_actions  (3)  ← REMAPPED
                    └── joint_pos (12)
                    └── joint_vel (12)
                    └── actions → last_low_level_action  (12)       ← REMAPPED
                    └── height_scan (187)
                    = (num_envs, 235) tensor  →  locomotion_policy(obs)  →  joint_targets
```

### The remapping (critical detail)

In `PreTrainedPolicyAction.__init__()`:
```python
cfg.low_level_observations.velocity_commands.func = lambda dummy_env: self._raw_actions
cfg.low_level_observations.actions.func           = lambda dummy_env: last_action()
```

Both `velocity_commands` and `actions` in `LOW_LEVEL_ENV_CFG.observations.policy` are
overwritten **by name** at construction time. The locomotion policy then sees the navigation
policy's output as its velocity command.

## What breaks if you get this wrong

1. If `LOW_LEVEL_ENV_CFG.observations.policy` doesn't have an attribute named `velocity_commands`
   or `actions`, the remapping silently does nothing → locomotion policy gets stale/zero commands.
2. If you use `UnitreeGo2FlatEnvCfg` (48 dims) for a model trained on rough (235 dims),
   the TorchScript forward() will crash with a dimension mismatch.

## Code reference

- `IsaacLab/.../navigation/mdp/pre_trained_policy_action.py` lines 61-67 (remapping)
- `neurogait/.../locomotion/velocity/managers/observaions.py` lines 27,30 (the two renamed terms)
