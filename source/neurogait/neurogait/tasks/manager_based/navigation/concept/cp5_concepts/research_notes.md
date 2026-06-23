# CP5 Research Notes

## Survey date: 2026-06-23

---

## Research A — PreTrainedPolicyAction

**File**: `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/navigation/mdp/pre_trained_policy_action.py`

### Key findings

- `action_dim` is **hardcoded to 3** — always [vx, vy, heading/yaw_rate]
- `process_actions()` does `self._raw_actions[:] = actions` — **no scaling at all**
- Action scaling MUST happen upstream (in the model's compute() or skrl wrapper)
- On init, the action term rewires two obs terms by name:
  - `cfg.low_level_observations.velocity_commands.func = lambda: self._raw_actions`
  - `cfg.low_level_observations.actions.func = lambda: last_action()`
- Creates a SECOND internal `ObservationManager({"ll_policy": cfg.low_level_observations}, env)`
- `apply_actions()` runs the locomotion policy every `low_level_decimation` steps

### What breaks if you get this wrong

- If `LOW_LEVEL_ENV_CFG.observations.policy` doesn't have terms named exactly
  `velocity_commands` and `actions`, the remapping is silent no-op → wrong obs → policy fails
- Our loco obs ✓: `velocity_commands = ObsTerm(func=generated_commands, ...)` and `actions = ObsTerm(func=last_action)`

### LOW_LEVEL_ENV_CFG to use

Must match the trained model's observation dimension (235 dims).

Trained model: `logs/rsl_rl/unitree_go2_rough/` → use `UnitreeGo2RoughEnvCfg()`.

Obs breakdown (rough env):
| Term | Dims |
|------|------|
| base_lin_vel | 3 |
| base_ang_vel | 3 |
| projected_gravity | 3 |
| velocity_commands | 3 |
| joint_pos | 12 |
| joint_vel | 12 |
| actions | 12 |
| height_scan (GridPattern 1.6×1.0 @ 0.1m = 17×11) | 187 |
| **Total** | **235** |

---

## Research B — MultiMeshRayCasterCameraCfg

**File**: `IsaacLab/source/isaaclab/isaaclab/sensors/ray_caster/multi_mesh_ray_caster_camera_cfg.py`

### Key findings

- Full class name: `MultiMeshRayCasterCameraCfg` (note: Multi**Mesh**RayCaster, not MultiRayCaster)
- Import: `from isaaclab.sensors import MultiMeshRayCasterCameraCfg` (or `from isaaclab.sensors.ray_caster import ...`)
- Inherits from both `RayCasterCameraCfg` and `MultiMeshRayCasterCfg`
- Key param: `mesh_prim_paths: list[str | RaycastTargetCfg]` — strings auto-converted to `RaycastTargetCfg`
- Uses Warp GPU raycasting against PhysX meshes — no RTX, no GPU texture memory
- Enables 1024+ parallel envs vs ~12 with `CameraCfg`
- Pattern cfg: `PinholeCameraPatternCfg` from `isaaclab.sensors.ray_caster.patterns`
- OffsetCfg: from parent `RayCasterCameraCfg.OffsetCfg`

### Usage for CP5

```python
from isaaclab.sensors import MultiMeshRayCasterCameraCfg
from isaaclab.sensors.ray_caster import patterns

front_cam: MultiMeshRayCasterCameraCfg = MultiMeshRayCasterCameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base",
    mesh_prim_paths=["/World/ground", "{ENV_REGEX_NS}/obstacle_.*"],
    update_period=0.2,  # 5 Hz
    offset=MultiMeshRayCasterCameraCfg.OffsetCfg(
        pos=(0.3, 0.0, 0.1),
        rot=(0.4305, -0.5610, 0.5610, -0.4305),
    ),
    data_types=["distance_to_image_plane"],
    pattern_cfg=patterns.PinholeCameraPatternCfg(
        focal_length=1.88,
        width=80,
        height=60,
    ),
)
```

---

## Research C — Go2 locomotion env config

**File**: `neurogait/tasks/manager_based/locomotion/velocity/config/go2/rough_env_cfg.py`

### Key findings

- `UnitreeGo2RoughEnvCfg` (rough) → 235-dim obs (needed for trained model)
- `UnitreeGo2FlatEnvCfg` (flat) → 48-dim obs (height_scan=None) — WRONG for our model
- Actions: `JointPositionActionCfg(joint_names=[".*"], scale=0.25)` → 12 joint targets
- The navigation env's scene already has `height_scanner` ✓ (required for rough env obs)
- Import: `from neurogait.tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import UnitreeGo2RoughEnvCfg`

---

## Research D — TorchScript export

**File**: `scripts/rsl_rl/play.py`

### Key findings

- rsl_rl's `play.py` auto-exports to JIT: `runner.export_policy_to_jit(path=..., filename="policy.pt")`
- Already exported: `logs/rsl_rl/unitree_go2_rough/2026-06-13_19-33-23/exported/policy.pt`
- Just need to verify input/output shape before using in PreTrainedPolicyActionCfg

---

## Research E — skrl GaussianMixin API

**File**: `isaac-sim/kit/python/lib/python3.11/site-packages/skrl/models/torch/gaussian.py`

### Key findings

- `act()` calls `compute(inputs, role)` and reads `outputs["log_std"]`
- **`compute()` must return `(mean_actions, {"log_std": log_std_tensor})`**
- The prompt's example `(mean_actions, log_std_parameter, {})` is WRONG — log_std must be in dict
- `clip_actions=True` clamps with `torch.clamp(actions, min, max)` using action_space bounds
- `state_preprocessor` and `value_preprocessor` are set in the YAML config, not the model
- Isaac Lab skrl wrapper: `SkrlVecEnvWrapper(env, ml_framework="torch")` from `isaaclab_rl.skrl`
- Training uses `Runner(env, agent_cfg)` where `agent_cfg` is a dict loaded from YAML

---

## Research F — Point cloud utils

**File**: `IsaacLab/source/isaaclab/isaaclab/sensors/camera/utils.py`

### Key findings

- Function: `create_pointcloud_from_depth(intrinsic_matrix, depth, keep_invalid=False, ...)`
- Import: `from isaaclab.sensors.camera.utils import create_pointcloud_from_depth`
- Returns point cloud as (N, 3) or (H, W, 3) depending on `keep_invalid`
- Lower level: `math_utils.unproject_depth(depth, K)` — already used in our existing obs

---

## Research G — Contact sensor

**File**: `IsaacLab/scripts/demos/sensors/contact_sensor.py`

### Key findings

- Already in scene: `MySceneCfg.contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", ...)`
- Data access: `env.scene["contact_forces"].data.net_forces_w` → shape (num_envs, num_bodies, 3)
- For obstacle collision: check if non-foot bodies have significant net force
- `filter_prim_paths_expr` can narrow to specific contact pairs if needed
- history_length=3 means last 3 force readings are available

---

## Summary of critical constants for CP5

| Constant | Value | Source |
|----------|-------|--------|
| Locomotion obs dim | 235 | UnitreeGo2RoughEnvCfg |
| Navigation obs dim | 1615 | grid(1600) + waypoints(9) + vel(3) + gravity(3) |
| Navigation action dim | 3 | PreTrainedPolicyAction.action_dim |
| sim.dt | 0.005 s | locomotion env |
| low_level_decimation | 4 | loco step = 0.02 s |
| nav decimation | 40 | nav step = 0.2 s = 5 Hz |
| render_interval | 40 | must equal nav decimation |
| Grid size | 40×40 cells | 8 m @ 0.2 m/cell |
| Robot row (asymmetric) | 10 | 2 m behind, 6 m ahead |
| Waypoint lookahead | 3 | 9 future dims |
| Exported policy path | logs/rsl_rl/unitree_go2_rough/2026-06-13_19-33-23/exported/policy.pt | |
