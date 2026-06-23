# 04: Reward Design and Literature Citations

## Overview

CP5 uses 7 reward terms, each from a specific published source. This file documents
the formulas, citations, and design rationale for the thesis.

---

## Reward terms and sources

| Term | Weight | Formula | Source |
|------|--------|---------|--------|
| velocity_toward_goal | 10.0 | `cos(θ_err) × vx × (1 + 1/(1 + 2d²))` | SEA-Nav (Huang et al., 2026) |
| goal_proximity | 3.0 | `(1 - tanh(d/5)) + (1 - tanh(d/1))` | Li et al. (2025), Eq. 1 |
| goal_reached | 20.0 | `𝟙[d_final < 0.3]` | X-Nav (2025) |
| collision | -5.0 | `-(1 + 4(‖v‖² + ωz²)) × 𝟙[contact]` | SEA-Nav (Huang et al., 2026) |
| stuck | -3.0 | `-𝟙[Δp_max < 0.1 over 20 steps] × 𝟙[vx_cmd > 0.1]` | SEA-Nav (Huang et al., 2026) |
| heading | 0.5 | `cos(θ_err)` | Standard |
| smoothness | -0.01 | `-‖action_t - action_{t-1}‖` | X-Nav (2025) |

---

## Term-by-term breakdown

### 1. velocity_toward_goal (w=10.0) — SEA-Nav

The dominant learning signal. Rewards approach **speed**, not just distance reduction.

Formula: `r = cos(heading_error) × vx × (1 + 1/(1 + 2d²))`

The proximity bonus `1/(1 + 2d²)` increases as the robot nears the waypoint,
encouraging it to slow slightly as it approaches (better waypoint capture).

> "We reward the projection of linear velocity onto the goal direction,
> scaled by a proximity term that increases near the waypoint."
> — SEA-Nav (Huang et al., 2026), Table III

Why better than naive distance reward: a pure `Δd` reward has gradient
only when the robot moves; it cannot signal speed. This formulation
has a gradient even when the robot is stationary (punishes zero velocity).

### 2. goal_proximity (w=3.0) — Li et al.

Dual-scale tanh shaping for smooth long-range and fine-range guidance.

Formula: `r = (1 - tanh(d/5.0)) + (1 - tanh(d/1.0))`

- σ=5.0: coarse gradient over 0–10 m (always-on directional pull)
- σ=1.0: strong pull within ~2 m (precise waypoint capture)
- tanh avoids gradient discontinuities seen with inverse-distance rewards

> "Dual-scale proximity shaping provides both long-range guidance and
> precise goal capture without reward cliffs."
> — Li et al. (2025), Eq. 1

### 3. goal_reached (w=20.0) — X-Nav

Sparse bonus when final waypoint is within 0.3 m. Signals successful episode completion.

> "A large sparse bonus on goal arrival ensures the policy learns
> the terminal state is qualitatively different from approach."
> — X-Nav (2025)

### 4. collision — velocity-scaled (w=-5.0) — SEA-Nav

High-speed collisions cost up to 5× more than low-speed bumps.

Formula: `r = -(1 + 4(‖v‖² + ωz²)) × 𝟙[contact_on_non_foot_body]`

At max speed (vx=1, vy=1): `(1 + 4(1+1+0)) ≈ 9`
At zero speed: `(1 + 0) = 1`

Teaches the policy to slow down near obstacles, not just avoid them.

> "Velocity-scaled collision penalties encourage speed-aware navigation."
> — SEA-Nav (Huang et al., 2026), Table III

### 5. stuck (w=-3.0) — SEA-Nav

Sliding-window detection: if robot commanded forward but didn't move.

Formula: `-𝟙[max(‖pos_t - pos_{t-k}‖ for k=1..20) < 0.1] × 𝟙[vx_cmd > 0.1]`

Prevents the policy from learning to stand still (which gives zero collision penalty
but also zero velocity reward — a local minimum at w/o this term).

> "Stuck detection prevents the policy from learning immobility as a
> local optimum." — SEA-Nav (Huang et al., 2026), Table III

### 6. heading (w=0.5) — Standard

Formula: `r = cos(yaw - atan2(wp_y - robot_y, wp_x - robot_x))`

Gentle orientation hint. Small weight — should not dominate over forward progress.

### 7. smoothness (w=-0.01) — X-Nav

Formula: `r = -‖action_t - action_{t-1}‖`

Tiny regularizer. Discourages jerky velocity commands that would transmit
to the locomotion policy as rapid velocity changes (causing rough gait).

Track the 3D navigation action, NOT the 12D joint targets.

> "Kinematic smoothing of high-level commands improves locomotion quality."
> — X-Nav (2025)

---

## Thesis citation paragraph

> "Our navigation reward function draws on recent quadruped navigation literature.
> The velocity-toward-goal formulation and velocity-scaled collision penalty are
> adapted from SEA-Nav (Huang et al., 2026), which demonstrated that speed-aware
> rewards are critical for efficient obstacle avoidance. The dual-scale tanh
> proximity shaping follows Li et al. (2025), Eq. 1, providing smooth gradients
> at both long and short ranges. The sparse goal-reached bonus and kinematic
> smoothing regularizer follow X-Nav (2025), which showed that explicit episode
> completion signals improve convergence in long-horizon navigation tasks."

---

## Reward balance verification

Before training, run `scripts/cp5/test_reward_scales.py` with 256 envs and random actions
for 100 steps. All **weighted** reward terms should be within 10× of each other.

Expected rough magnitudes with random actions:
| Term | Raw mean | Weighted |
|------|----------|---------|
| velocity_toward_goal | ~0.15 | ~1.5 |
| goal_proximity | ~0.8 | ~2.4 |
| goal_reached | ~0.0 | ~0.0 |
| collision | ~0.05 | ~-0.25 |
| stuck | ~0.0 | ~0.0 |
| heading | ~0.0 | ~0.0 |
| smoothness | ~0.3 | ~-0.003 |

If any term is >10× the others after weighting, adjust the weight before training.
