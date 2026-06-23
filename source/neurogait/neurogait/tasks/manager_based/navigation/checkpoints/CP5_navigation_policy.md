# CP5: Trained Navigation Policy

## What was built

| File | Purpose |
|------|---------|
| `concept/cp5_concepts/research_notes.md` | API survey findings |
| `concept/cp5_concepts/01_two_obs_managers.md` | PreTrainedPolicyAction internals |
| `concept/cp5_concepts/02_heading_vs_yawrate.md` | heading_command=True |
| `concept/cp5_concepts/03_raycaster_vs_camera.md` | GPU camera choice |
| `concept/cp5_concepts/04_reward_design_citations.md` | 7 reward terms + citations |
| `concept/cp5_concepts/05_cnn_vs_mlp.md` | Architecture rationale |
| `concept/cp5_concepts/06_asymmetric_grid.md` | Robot at row 10 |
| `concept/cp5_concepts/07_pure_pursuit_waypoints.md` | 3-waypoint lookahead |
| `models/navigation_policy.py` | CNN+MLP actor + critic |
| `mdp/observations.py` | CP5 obs terms (grid, waypoints, velocity) |
| `mdp/rewards.py` | 7 CP5 reward terms |
| `mdp/__init__.py` | Updated exports |
| `config/go2/navigation_env_cfg.py` | CP5 env config |
| `config/go2/__init__.py` | CP5 gym registrations |
| `config/go2/agents/skrl_nav_ppo_cfg.yaml` | PPO training config |
| `scripts/cp5/export_locomotion.py` | TorchScript verification |
| `scripts/cp5/test_reward_scales.py` | Pre-training balance check |
| `scripts/cp5/train.py` | skrl PPO training |
| `scripts/cp5/play.py` | Policy evaluation |
| `scripts/cp5/compare_cp4_cp5.py` | Quantitative comparison |
| `scripts/cp5/visualize_perception.py` | 6-panel perception debug |

---

## Architecture

- **PreTrainedPolicyActionCfg**: frozen locomotion policy (TorchScript) inside action manager
- **Two obs managers**: navigation (1615-dim) + locomotion (235-dim, internal to PreTrainedPolicyAction)
- **CNN+MLP policy**: 3-layer conv for 40×40 grid, MLP for scalars, merged head
- **heading_command=True**: action[2] = heading target [-π, π] (NOT yaw rate)
- **MultiMeshRayCasterCameraCfg**: GPU Warp raycasting, enables 512+ envs

---

## Observation (1615 dims)

| Range | Term | Dims | Notes |
|-------|------|------|-------|
| [0:1600] | occupancy_grid_obs_cp5 | 1600 | 40×40 asymmetric, robot at row 10 |
| [1600:1609] | future_waypoints_obs | 9 | 3 × [dir_x, dir_y, dist/10] in robot frame |
| [1609:1612] | robot_velocity_obs | 3 | [vx, vy, yaw_rate] body frame |
| [1612:1615] | projected_gravity | 3 | standard Isaac Lab obs |

**ORDER IS CRITICAL**: CNN in `models/navigation_policy.py` splits at index 1600.

---

## Action (3 dims)

| Index | Meaning | Range | Scaling |
|-------|---------|-------|---------|
| [0] | vx | [-1.0, 1.0] m/s | `tanh(raw)` in model.compute() |
| [1] | vy | [-1.0, 1.0] m/s | `tanh(raw)` in model.compute() |
| [2] | heading target | [-π, π] rad | `π × tanh(raw)` in model.compute() |

---

## Reward terms (7 total, all from published papers)

| Term | Weight | Formula | Source |
|------|--------|---------|--------|
| velocity_toward_goal | +10.0 | `cos(θ_err) × vx × (1 + 1/(1 + 2d²))` | SEA-Nav (Huang et al., 2026) |
| goal_proximity | +3.0 | `(1-tanh(d/5)) + (1-tanh(d/1))` | Li et al. (2025), Eq. 1 |
| goal_reached | +20.0 | `𝟙[d_final < 0.3]` | X-Nav (2025) |
| collision | -5.0 | `-(1 + 4(‖v‖² + ωz²)) × 𝟙[contact]` | SEA-Nav (Huang et al., 2026) |
| stuck | -3.0 | `-𝟙[max_disp < 0.1 over 20 steps & cmd > 0.1]` | SEA-Nav (Huang et al., 2026) |
| heading | +0.5 | `cos(θ_err)` | Standard |
| smoothness | -0.01 | `-‖action_t - action_{t-1}‖` | X-Nav (2025) |

---

## Timing

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| sim.dt | 0.005 s | simulation physics step |
| low_level_decimation | 4 | locomotion at 50 Hz = 0.02 s |
| navigation decimation | 40 | navigation at 5 Hz = 0.20 s |
| render_interval | 40 | must equal decimation |
| nav steps per episode | 150 (30 s) | 30 s / 0.2 s/step |

---

## Bugs fixed from CP5 Attempt 1

| # | Bug | Fix |
|---|-----|-----|
| 1 | `heading_command`: robot spun | action[2] = heading angle, not yaw_rate |
| 2 | CameraCfg: only 12 envs | MultiMeshRayCasterCameraCfg (Warp GPU) |
| 3 | Action not clamped | `tanh(raw)` in model.compute(), `π×tanh` for heading |
| 4 | Reward imbalance | test_reward_scales.py before training |
| 5 | Progress always zero | `_cp5_prev_dist` init'd at `_cp5_init_waypoint_state` (lazy, not per-step) |
| 6 | render_interval | `self.sim.render_interval = self.decimation = 40` |

---

## skrl API fix (critical)

`GaussianMixin.act()` reads `outputs["log_std"]` from what `compute()` returns.

**WRONG** (from prompt template):
```python
return action_mean, log_std_parameter, {}   # KeyError: "log_std"
```

**CORRECT** (implemented):
```python
return mean_actions, {"log_std": self.log_std_parameter}
```

---

## LOW_LEVEL_ENV_CFG

Must be `UnitreeGo2RoughEnvCfg()` (235 dims: 48 proprioceptive + 187 height scan).
`UnitreeGo2FlatEnvCfg()` gives only 48 dims and will crash with dimension mismatch.

---

## Training details

*(Fill after training)*

- Run ID:
- Iterations completed:
- Final reward:
- Training time:

---

## CP4 vs CP5 comparison

*(Fill after compare_cp4_cp5.py)*

| Metric | CP4 (A*) | CP5 (RL) |
|--------|----------|----------|
| Success rate | | |
| Avg time (s) | | |

---

## Passing bar

- [ ] TorchScript export verified: `python scripts/cp5/export_locomotion.py`
- [ ] Reward scales balanced: `python scripts/cp5/test_reward_scales.py --num_envs 256`
- [ ] 50-iter smoke test: no crash, 7 terms in TensorBoard
- [ ] CMD values in range: vx ±1, vy ±1, heading ±π
- [ ] 2000-iter training: reward trends up
- [ ] Robot navigates A→B (play.py shows goal-directed motion)
- [ ] CP4 vs CP5 comparison table (compare_cp4_cp5.py)
- [ ] Concept docs complete (7 files in concept/cp5_concepts/)
- [ ] Screen recording
