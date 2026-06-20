# CP5: First Trained Navigation Policy

## What Was Built

CP5 replaces the rule-based waypoint controller (CP4) with a **learned RL navigation
policy** trained via skrl PPO. The policy outputs 3D velocity commands that drive a
**frozen locomotion backbone** (the RSL-RL Go2 rough policy from CP3) through Isaac
Lab's built-in `PreTrainedPolicyAction` mechanism.

### Files Added / Modified

| File | Change |
|------|--------|
| `mdp/pre_trained_policy_action.py` | Copied `PreTrainedPolicyAction` + `PreTrainedPolicyActionCfg` from IsaacLab |
| `mdp/waypoint_manager.py` | `init_waypoints()` EventTerm — builds A* path once, resets per-env waypoint state |
| `mdp/observations.py` | Added `goal_vector_obs` (3-dim) and `robot_velocity_obs` (3-dim) |
| `mdp/rewards.py` | Added `reward_progress`, `reward_heading`, `penalty_collision`, `penalty_smoothness` |
| `mdp/__init__.py` | Exported all CP5 symbols |
| `config/go2/navigation_env_cfg.py` | Added `NeuroGaitNavigationCP5EnvCfg` + `_PLAY` |
| `config/go2/__init__.py` | Registered `NeuroGait-Navigation-CP5-v0` and `NeuroGait-Navigation-CP5-Play-v0` |
| `config/go2/agents/skrl_navigation_ppo_cfg.yaml` | PPO config [256, 128, 64], RunningStandardScaler |
| `scripts/train_nav.py` | Training script (hydra + skrl Runner) |
| `scripts/play_cp5.py` | Evaluation script (loads checkpoint, runs nav policy) |
| `scripts/test_reward_scales.py` | 100-step random-action reward calibration test |
| `scripts/compare_cp4_cp5.py` | 20-episode CP4 vs. CP5 comparison table |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  NAVIGATION POLICY (skrl PPO, trained)                       │
│  Obs:  1612-dim (base_lin_vel + proj_gravity +               │
│        goal_vector + robot_velocity + occ_grid)              │
│  Act:  [vx, vy, yaw_rate]  (3-dim continuous)               │
└────────────────────────┬────────────────────────────────────┘
                         │ velocity command (3D)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  PreTrainedPolicyAction  (Isaac Lab built-in)                │
│  Loads frozen loco policy (TorchScript).                     │
│  Runs at 50 Hz; nav policy at 5 Hz (decimation=40).         │
│  Injects velocity command → 235-dim loco obs → 12 joints    │
└────────────────────────┬────────────────────────────────────┘
                         │ 12 joint position targets
                         ▼
                  Isaac Sim physics
```

---

## Observation Space (Navigation Policy — 1612-dim)

| Term | Dims | Function |
|------|------|----------|
| `base_lin_vel` | 3 | Linear velocity in base frame (vx, vy, vz) |
| `projected_gravity` | 3 | Gravity vector in base frame (encodes tilt) |
| `goal_vector` | 3 | [dir_x, dir_y, norm_dist] to current A* waypoint in robot frame |
| `robot_velocity` | 3 | [vx, vy, yaw_rate] in base frame |
| `occupancy_grid` | 1600 | 40×40 binary local occupancy grid (GPU path) |
| **Total** | **1612** | |

---

## Action Space (Navigation Policy — 3-dim)

The nav policy outputs `[vx, vy, yaw_rate]` in the frozen locomotion policy's
training range:

| Dimension | Range |
|-----------|-------|
| `vx` (forward) | [−1.0, 1.0] m/s |
| `vy` (lateral) | [−0.5, 0.5] m/s (locomotion range) |
| `yaw_rate` | [−1.0, 1.0] rad/s |

`PreTrainedPolicyAction` passes these directly as velocity commands to the frozen
locomotion policy. The nav policy must learn to keep outputs in these ranges.

---

## Reward Terms (logged separately in TensorBoard)

| Term | Weight | Description |
|------|--------|-------------|
| `termination_penalty` | −200.0 | Episode reset signal |
| `reward_progress` | 1.0 | Distance reduction to current A* waypoint |
| `reward_heading` | 0.3 | cos(heading error) — facing waypoint |
| `penalty_collision` | 2.0 | −1 if base contact force > 1 N |
| `penalty_smoothness` | 0.1 | −‖Δaction‖₂ — discourage jerky commands |

**Waypoint management:** `reward_progress` advances the waypoint index when
the robot comes within 0.3 m. Waypoints come from the same A* planner as CP4
(global occupancy grid with 0.30 m C-space inflation).

---

## PPO Hyperparameters

| Setting | Value | Reason |
|---------|-------|--------|
| Network | [256, 128, 64] ELU | Smaller than loco [512, 256, 128] — simpler task |
| Learning rate | 3e-4 | Standard for PPO |
| Clip ratio | 0.2 | Standard |
| Entropy coeff | 0.01 | Encourage exploration early |
| Rollouts | 24 | Steps per env per update |
| Mini-batches | 4 | Per PPO epoch |
| Epochs | 5 | Per rollout batch |
| `state_preprocessor` | RunningStandardScaler | **CRITICAL** — normalises 1600-dim binary grid |
| `value_preprocessor` | RunningStandardScaler | **CRITICAL** — without this robot splays |

---

## Training Timing

- `sim.dt = 0.005 s`
- Loco policy (PreTrainedPolicyAction): every 4 sim steps → 50 Hz
- Nav policy (`decimation = 40`): every 40 sim steps → **5 Hz**
- Episode length: 30 s → 150 nav steps per episode
- With 1024 envs × 24 rollouts × 2000 updates ≈ 49 M environment steps

---

## Waypoint Management

`init_waypoints()` (EventTerm, mode="reset"):
1. Builds the global occupancy grid from scene rigid objects (cached after first call)
2. Runs A* from (0, 0) to (8, 0) — same as CP4 (cached, shared across all envs)
3. Resets `env._curr_waypoint_idx`, `env._curr_waypoint_pos`, `env._prev_waypoint_dist`,
   `env._prev_nav_action` for env_ids being reset

`reward_progress()` (called every step):
- Computes distance reduction (progress reward)
- Vectorised waypoint advancement: `_curr_waypoint_idx += 1` when `dist < 0.3 m`

---

## Key Design Decisions

### Why PreTrainedPolicyActionCfg (not gym.Wrapper)

Isaac Lab has a built-in `PreTrainedPolicyAction` mechanism that handles the
hierarchical execution natively within the ManagerBasedRLEnv framework. This
avoids the complexity of a gym.Wrapper and integrates cleanly with the
observation/reward/termination managers.

### Why occupancy_grid_obs_gpu

The CPU path (`occupancy_grid_obs`) loops over envs in Python — fine for
1 env (play), but ~6 ms × 1024 envs = 6 s/step for training. The GPU path
(`occupancy_grid_obs_gpu`) scatters in parallel with torch.

### Why RunningStandardScaler is non-negotiable

With 1612-dim obs, 1600 of which are binary (0/1) occupancy values, the
raw obs magnitude is dominated by the grid. Without normalisation, the
policy gradient signal is swamped, and the robot learns to stand still
(locally optimal under the collision penalty) rather than navigate. The
RunningStandardScaler brings all obs terms to mean≈0, std≈1 online.

---

## Passing Bar

- [ ] `PreTrainedPolicyActionCfg` loads TorchScript policy without errors
- [ ] Reward scale test: all weighted terms within 1 order of magnitude
- [ ] 50-iteration smoke test: no crash, reward terms visible in TensorBoard
- [ ] 500-iteration run: total reward trending upward
- [ ] play_cp5.py: robot moves toward goal (even imperfectly)
- [ ] 2000-iteration full run complete
- [ ] CP4 vs CP5 comparison table generated
- [ ] Screen recording captured
- [ ] No shape mismatch errors anywhere

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Robot stands still | Reduce `collision` weight (robot afraid to move) |
| Robot spins in place | Reduce `heading` weight |
| NaN in loss | Verify `RunningStandardScaler` is active |
| Reward flat at 0 | Check `_curr_waypoint_pos` is initialised (run `init_waypoints`) |
| `KeyError: pre_trained_policy_action` | Mismatch between action_manager key and step call |
| Policy not loaded | Check `policy_path` exists (relative to working dir) |
| Base collisions at every step | Contact sensor threshold too low (check Go2 foot forces) |

---

## Commands

```bash
# 0. Verify exported policy loads
~/isaac-sim/kit/python/bin/python3 -c "
import torch
m = torch.jit.load('logs/rsl_rl/unitree_go2_rough/2026-06-13_19-33-23/exported/policy.pt')
print('Policy loaded:', type(m))
"

# 1. Reward scale test (run BEFORE training)
~/isaac-sim/kit/python/bin/python3 scripts/test_reward_scales.py \
  --task NeuroGait-Navigation-CP5-v0 --num_envs 256 --headless --enable_cameras

# 2. Smoke test (50 iterations)
~/isaac-sim/kit/python/bin/python3 scripts/train_nav.py \
  --task NeuroGait-Navigation-CP5-v0 --num_envs 256 --headless --max_iterations 50

# 3. First real run (500 iterations)
~/isaac-sim/kit/python/bin/python3 scripts/train_nav.py \
  --task NeuroGait-Navigation-CP5-v0 --num_envs 1024 --headless --max_iterations 500

# 4. Full training (2000 iterations)
~/isaac-sim/kit/python/bin/python3 scripts/train_nav.py \
  --task NeuroGait-Navigation-CP5-v0 --num_envs 1024 --headless --max_iterations 2000

# 5. Evaluate trained policy
~/isaac-sim/kit/python/bin/python3 scripts/play_cp5.py \
  --task NeuroGait-Navigation-CP5-Play-v0 --num_envs 1 --enable_cameras \
  --checkpoint logs/skrl/neurogait_navigation_cp5/<run>/checkpoints/agent.pt

# 6. Compare CP4 vs CP5
~/isaac-sim/kit/python/bin/python3 scripts/compare_cp4_cp5.py \
  --cp5_checkpoint logs/skrl/neurogait_navigation_cp5/<run>/checkpoints/agent.pt \
  --n_episodes 20 --headless --enable_cameras
```
