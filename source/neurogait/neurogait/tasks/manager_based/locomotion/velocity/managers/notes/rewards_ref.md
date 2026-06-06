# Understanding the Reward System — `RewardsCfg` + `mdp/rewards.py`

*A learn-as-you-clone walkthrough of the two files that define **what the robot is rewarded for**.*
*This is the heart of NeuroGait: reward weights are what you'll tune, sweep, and benchmark. Master this file and you understand the project.*

---

## 1. The big idea: two files, two jobs

The reward system is split across two files, and understanding *why* is the key lesson:

| File | Job | Analogy |
|---|---|---|
| `mdp/rewards.py` | **Defines** reward *functions* — the math that measures a behavior | The measuring instruments |
| `RewardsCfg` (in env cfg) | **Selects** which functions to use, with what **weight** and **params** | The mixing board that sets each one's volume |

A reward *function* (e.g. `feet_air_time`) computes a number per environment — "how good was this behavior right now." The *config* binds that function to a **weight** (how much it counts) and **parameters** (thresholds, which sensor, etc.). The **RewardManager** then every step computes:

```
total_reward = Σ over all terms:  weight_i  ×  function_i(env, **params_i)
```

So the function is reusable and the config is where you *shape* behavior by turning weights up and down. **You will rarely edit the functions; you will constantly edit the weights.** That separation is why you can run dozens of reward experiments without touching code — exactly what a benchmarking project needs.

---

## 2. Part A — `RewardsCfg`: the mixing board

The config groups terms into **task rewards** (positive, drive the goal), **penalties** (negative, shape *how*), and **optional/disabled** terms.

### Task rewards (the "what to achieve")

```python
track_lin_vel_xy_exp = RewTerm(func=env_mdp.track_lin_vel_xy_exp, weight=1.0,
    params={"command_name": "base_velocity", "std": math.sqrt(0.25)})
track_ang_vel_z_exp  = RewTerm(func=env_mdp.track_ang_vel_z_exp,  weight=0.5,
    params={"command_name": "base_velocity", "std": math.sqrt(0.25)})
```

- These two are **the actual task**: follow the commanded linear velocity (weight **1.0**) and yaw rate (weight **0.5**). Everything else just shapes *how* the robot achieves them.
- They reference `"base_velocity"` — the command from `commands.py`. This is the loop closing: command in → track it → reward out.
- `std = sqrt(0.25) = 0.5` is the tolerance of the exponential kernel (explained in Part B). It's the famous `tracking_sigma`.
- Note: linear tracking is **2×** the weight of angular — forward/sideways accuracy matters more than turning accuracy here.

### Penalties (the "how to behave")

```python
lin_vel_z_l2    weight=-2.0      # don't bounce vertically
ang_vel_xy_l2   weight=-0.05     # don't wobble (roll/pitch)
dof_torques_l2  weight=-1.0e-5   # use less energy
dof_acc_l2      weight=-2.5e-7   # move smoothly (low joint acceleration)
action_rate_l2  weight=-0.01     # don't jerk (penalize sudden action changes)
feet_air_time   weight=+0.125    # take real steps, not shuffles  (custom fn)
undesired_contacts weight=-1.0   # don't touch the ground with your THIGHs
```

Each shapes a quality of motion:
- **`lin_vel_z_l2` (−2.0):** the heaviest penalty. Vertical bouncing wastes energy and destabilizes — strongly discouraged.
- **`ang_vel_xy_l2` (−0.05):** mild penalty on roll/pitch wobble; keeps the body level-ish without over-constraining.
- **`dof_torques_l2` (−1e-5) & `dof_acc_l2` (−2.5e-7):** energy and smoothness. Note the *tiny* coefficients — because torque and acceleration are *large* numbers, so a small coefficient still produces a meaningful contribution (this is the trap in §4).
- **`action_rate_l2` (−0.01):** penalizes how much the action changes step-to-step → smooth commands → **this is your jerk metric's reward counterpart.**
- **`feet_air_time` (+0.125):** *positive* — rewards deliberate stepping (uses your custom function). Without it, robots learn to shuffle/skate.
- **`undesired_contacts` (−1.0):** strong penalty if a `THIGH` touches the ground — thighs hitting the floor means a near-fall or bad gait.

### Optional / disabled terms

```python
flat_orientation_l2 = RewTerm(func=env_mdp.flat_orientation_l2, weight=0.0)
dof_pos_limits      = RewTerm(func=env_mdp.joint_pos_limits,     weight=0.0)
```

- **`weight=0.0` means the term is wired up but OFF.** It contributes nothing right now.
- They're left in as **ready knobs**: set `flat_orientation_l2` negative to force a flatter back (useful on rough terrain), or `dof_pos_limits` negative to push joints off their hard stops. You enable them by giving them a weight — no code change needed. (Same pattern as the degenerate ranges in `events.py`: defined-but-disabled.)

### A subtle but important detail: two `mdp` imports

```python
from isaaclab.envs import mdp as env_mdp                                  # built-in functions
from neurogait.tasks.manager_based.locomotion.velocity import mdp         # YOUR custom functions
```

Most terms use `env_mdp.*` (IsaacLab's built-ins), but `feet_air_time` uses `mdp.*` — **your** version from `mdp/rewards.py`. This is how you override or extend the library: define a custom function, import your local `mdp`, and point the term at it. **It also means your custom file contains functions the config doesn't currently use** — they're available for you to wire in (see Part B).

---

## 3. Part B — `mdp/rewards.py`: the measuring instruments

This file defines the reward *functions*. Each takes the `env` and returns a tensor with one reward value per parallel environment. A few patterns recur; learn them once and every function reads easily.

### The exponential kernel (the most important pattern)

Used by the tracking rewards:

```python
return torch.exp(-error / std**2)
```

Why exponential instead of just `-error`?
- **Bounded to (0, 1]:** perfect tracking → 1.0; bad tracking → approaches 0. Never blows up.
- **Shaped gradient:** steep near zero error (strong pull toward perfect tracking) and flat far away (doesn't obsess over hopeless errors). This is *much* easier to optimize than a raw squared error.
- **`std` is the tolerance dial:** small `std` = strict (only near-perfect tracking scores well); large `std` = forgiving. `std=0.5` (so `std²=0.25`) is the standard. **Tuning `std` changes how picky the robot is about tracking — a real knob for your sweep.**

### `feet_air_time` — your custom step-quality reward

```python
first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, body_ids]
last_air_time = contact_sensor.data.last_air_time[:, body_ids]
reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
```

Read it as a sentence: *"At the instant a foot touches down (`first_contact`), reward it by how long it was airborne minus the threshold (0.5 s)."*
- A **long step** (air time > 0.5 s) → positive reward. A **shuffle** (air time < 0.5 s) → negative. This pushes the robot toward clean, deliberate stepping instead of skating.
- The last line is **command masking:** if the commanded speed is below 0.1 m/s (robot told to stand still), the reward is zeroed — you don't want to reward stepping when the robot should be standing. This masking pattern appears in several functions and is essential: *rewards must respect the command.*
- It reads from the **contact sensor** defined in `scenes.py` — this is the scene→reward chain made concrete.

### `feet_slide` — penalize scuffing (available, not yet used)

```python
contacts = (net_forces_w_history... .norm(dim=-1).max(dim=1)[0] > 1.0)   # foot in contact?
body_vel = asset.data.body_lin_vel_w[:, body_ids, :2]                    # foot horizontal speed
reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
```

Penalizes a foot **moving horizontally while it's touching the ground** — i.e. slipping/scuffing. Important for sim-to-real: a real foot that slides loses traction and the policy that relied on sliding fails on hardware. **This is defined but not in your config yet — a strong candidate to add for Go2 robustness.**

### `track_lin_vel_xy_yaw_frame_exp` — a *better* tracking reward (available, not yet used)

```python
vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
```

This computes the velocity in the **yaw-aligned (gravity-aligned) frame** — i.e. relative to the robot's *heading* but ignoring its pitch/roll tilt. The built-in `track_lin_vel_xy_exp` your config currently uses measures in the **base frame**, which tilts with the body. The yaw-frame version is generally **more correct**: "forward" should mean the heading direction, not the direction the tilted chest happens to point. **Newer IsaacLab configs prefer this version — consider swapping to it; it's a small, principled improvement.**

### `stand_still_joint_deviation_l1` & `feet_air_time_positive_biped` (available)

- `stand_still_joint_deviation_l1`: penalizes drifting from the default pose *when commanded to stand still* — keeps the robot from fidgeting at rest. A nice companion to enable for clean standing.
- `feet_air_time_positive_biped`: a two-legged variant (single-stance logic). Not relevant to your quadruped — ignore it.

---

## 4. The trap, restated where it bites: contribution ≠ coefficient

Look at the weights again: `track_lin_vel` is `+1.0` but `dof_torques` is `−1e-5`. It is tempting to think torque "barely matters." **Wrong.** Torque values are large (tens of N·m, squared → hundreds/thousands), so `−1e-5 × (large number)` can contribute as much per step as `+1.0 × (tracking ~1)`. The *coefficient* is small precisely because the *quantity* is large.

**What you must balance is each term's *contribution to total reward per step*, not its raw weight.** So:
- **Log per-term reward contributions** during training (skrl/rsl_rl support this). Watch the actual numbers, not the coefficients.
- A common failure: a penalty silently dominates the task reward, and the robot learns to stand still (minimizing penalties) rather than walk. The per-term log catches this instantly.
- **This is exactly what your Optuna sweep should optimize over** — the weights here are the search space.

---

## 5. The NeuroGait insight: this file IS your benchmark and your tuning

Three project-specific points:

1. **This is where the Week-3 optimization happens.** "Optimize the best algorithm" largely means *tune these weights* (and `std`, and thresholds). The reward config is your primary search space for Optuna.

2. **Off-policy algorithms are reward-shaping-sensitive.** Recall SAC/TD3 are far pickier than PPO. The *same* reward config can make PPO succeed and SAC flounder — not because SAC is worse, but because it's more sensitive to this shaping. So when you benchmark, keep the reward config **identical across all three** (fairness), but be aware that a config tuned for PPO may handicap the off-policy methods. Budget a light per-algorithm reward sanity check before declaring a winner.

3. **Reward design *is* the research, partly.** Your novelty isn't only the architecture — a clean, well-justified, multi-metric reward (energy, jerk, slide, air-time, tracking) that you can defend is itself a contribution. The disabled terms and unused functions (`feet_slide`, yaw-frame tracking, `stand_still`) are your menu for making the policy more robust and more deployable.

---

## 6. Conventions you just learned

| Concept | What it means | Why it matters |
|---|---|---|
| Function vs config split | Math lives in `mdp/`, weights in the cfg | Tune behavior without editing code |
| `weight × function` sum | How total reward is built | The RewardManager's core operation |
| Exponential kernel | `exp(−error/std²)` for tracking | Bounded, well-shaped, easy to optimize; `std` = tolerance |
| L2 penalty | Square a quantity, weight negative | Smoothly discourages large values (energy, jerk, wobble) |
| Command masking | Zero a reward when command is small | Rewards must respect the command (don't reward stepping at rest) |
| `weight=0.0` | Defined but disabled term | A ready knob; enable by setting a weight |
| Custom `mdp` import | Override/extend built-in functions | How you add project-specific rewards |
| Contribution ≠ coefficient | Balance per-step contribution, not raw weight | The #1 reward-tuning mistake |

---

## 7. What you could add / change on your journey

**Locomotion quality & sim-to-real (high value)**
- **Enable `feet_slide`** (negative weight) — reduces foot scuffing, a real sim-to-real win for the Go2.
- **Swap to `track_lin_vel_xy_yaw_frame_exp`** — the more correct yaw-frame tracking reward.
- **Enable `stand_still_joint_deviation_l1`** — cleaner standing when commanded to stop (matters once obstacles make stopping common).
- **Enable `flat_orientation_l2`** (small negative) on rough terrain to keep the back level.
- **Add a base-height term** if you want a consistent stance height.

**Benchmark & research polish**
- **Log per-term contributions** and put the plot in your thesis — it shows you understand reward balance.
- **Pin reward params + seed** across PPO/SAC/TD3 for a fair comparison.
- **Make the weights your Optuna search space** for the optimization phase.
- **For the navigation phase:** the *low-level* reward stays roughly as-is (you freeze it); the *high-level* navigation policy gets its *own* reward (reach goal, avoid collision) — don't mix the two layers' rewards.

---

## 8. Industrial & research conventions to carry forward

- **The reward skeleton here is the ETH/`legged_gym` standard** — tracking (exp kernel) + regularization penalties. Nearly every modern legged-RL paper uses this structure; you're learning the field convention, and your weights are recognizable to any reviewer.
- **Reward shaping is an experiment, not a guess.** Real labs log per-term contributions, sweep weights, and report them. "We used these weights and here's why" is a methods-section requirement.
- **Respect the command in every reward.** Masking rewards by command magnitude is standard and prevents degenerate behaviors (rewarding motion at rest, etc.).
- **Keep functions reusable, configs disposable.** Write the math once, vary the weights freely. This is good software practice that happens to also be good science (controlled experiments).
- **Frame matters.** The yaw-frame vs base-frame tracking distinction is the kind of subtle correctness issue that separates a careful implementation from a sloppy one — worth getting right.

---

## 9. One-paragraph summary

The reward system is split in two: `mdp/rewards.py` **defines** reward functions (the math measuring each behavior) and `RewardsCfg` **selects** them with **weights** and **params**, which the RewardManager sums as `Σ weightᵢ × functionᵢ`. The task is two exponential-kernel **tracking** rewards (linear velocity at weight 1.0, yaw at 0.5) closing the loop with the velocity command; everything else is a **penalty** shaping *how* the robot moves — no vertical bounce (−2.0), low energy/jerk/wobble (tiny coefficients on large quantities), thighs off the ground (−1.0), and a *positive* custom `feet_air_time` (+0.125) rewarding deliberate steps. Two terms are wired but disabled (`weight=0.0`), and the custom file holds extra functions (`feet_slide`, yaw-frame tracking, `stand_still`) you can enable for robustness. The cardinal rule: tune by **per-step contribution, not raw coefficient**, and remember this file is both your **Optuna search space** for the optimization phase and the place where **off-policy SAC/TD3's reward-sensitivity** will show up — so keep it identical and seeded across algorithms for a fair benchmark.