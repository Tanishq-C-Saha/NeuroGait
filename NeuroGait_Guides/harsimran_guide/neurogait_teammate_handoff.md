# NeuroGait — Teammate Handoff: RL Algorithm Track
## Copy the section below as your opening prompt in a new Claude conversation

---

## HANDOFF PROMPT (paste this exactly into a new conversation)

```
I am working on NeuroGait, a masters-level quadruped navigation project
on the Unitree Go2 robot in NVIDIA Isaac Lab 2.3.0 / Isaac Sim 5.1.
I am handling the RL algorithm, reward engineering, policy training,
and hyperparameter tuning side. My teammate is handling the simulation
environment, perception pipeline (depth camera → occupancy grid), and
Isaac Lab scene setup. We are working in parallel and will integrate
at a specific point described below.

PROJECT CONTEXT:
- Robot: Unitree Go2
- Simulator: NVIDIA Isaac Lab 2.3.0, Isaac Sim 5.1
- Hardware: RTX 5070 Ti laptop, 12GB VRAM, ~3.7 it/s at 2048 envs
- RL Framework: skrl (primary — PPO, SAC, TD3 benchmarking)
- rsl_rl: used only for the frozen locomotion checkpoint (already trained)
- Python: ~/isaac-sim/kit/python/bin/python3

WHAT ALREADY EXISTS (teammate's side, frozen, don't touch):
- A converged locomotion policy (symmetric actor-critic PPO via rsl_rl),
  trained on rough terrain, checkpoint at logs/rsl_rl/unitree_go2_rough/
- The locomotion policy takes as input: base_ang_vel, projected_gravity,
  velocity_commands (vx, vy, yaw_rate), joint_pos_rel, joint_vel_rel,
  last_action — 45-dimensional observation
- It outputs 12 joint position targets
- A navigation task shell (NeuroGait-Navigation-Rubble-v0) is registered
  in Isaac Lab, currently running the frozen locomotion policy with a
  fixed velocity command (CP1 done)
- A standalone A* planner (planner.py) working on 2D occupancy grids
  (CP2 done)
- Depth camera added to the scene, occupancy grid conversion function
  written and tested in isolation (CP3 in progress)

WHAT YOU ARE BUILDING (algorithm track):
The navigation policy — a learned PPO policy (via skrl) that sits ABOVE
the frozen locomotion policy and outputs velocity commands (vx, vy,
yaw_rate) each step, instead of the random velocity sampler that was
there before. This is the high-level RL controller that decides WHERE
to walk; the frozen locomotion policy decides HOW to walk there.

OVERALL ARCHITECTURE (for context):
Layer 1: A* planner over a learned terrain cost map (global route,
         runs once per goal, classical — no RL here)
Layer 2: Dual PPO navigation policy (progress policy + caution policy,
         blended by a risk-prediction head) — THIS IS YOUR MAIN WORK
Layer 3: Proprioception feedback loop (torque/orientation signal
         corrects terrain cost) — feeds into Layer 1

INTEGRATION POINT:
My teammate will have a working occupancy grid (40x40, 0.2m resolution,
8m x 8m coverage) available as a 2D tensor from the depth camera by the
time you've completed your solo checkpoints (roughly your AT4-5 below).
Until then, you work with PRIVILEGED OBSERVATIONS (read obstacle
positions directly from sim state — cheaper stand-in for real perception,
lets you prove the RL policy works before the real perception is ready).

NOVEL ARCHITECTURE PAPER REFERENCES:
- TOP-Nav (Ren et al., CoRL 2025, arxiv.org/abs/2404.15256):
  terrain cost feedback loop, proprioception advisor
- ABS / Agile But Safe (He et al., RSS 2024, arxiv.org/abs/2401.17583):
  dual policy (agile + recovery) with reach-avoid value network
- The hybrid design combines these: terrain-aware A* (TOP-Nav) +
  dual progress/caution policy with risk-head blending (ABS-inspired)

YOUR WORKING DIRECTORY: neurogait/tasks/manager_based/navigation/
YOUR AGENT CONFIGS LIVE IN: neurogait/tasks/manager_based/navigation/
                             config/go2/agents/
TRAINING LOGS: logs/skrl/navigation/

Please help me work through the algorithm-track checkpoints listed
below, one at a time. Start with AT1.
```

---

## ALGORITHM TRACK CHECKPOINTS

These align with your teammate's simulation-track checkpoints (CP1-15).
The integration point is AT5 ↔ CP5/CP6.

---

### AT1 — skrl PPO config for navigation (1 day)
**What to build:** a clean skrl PPO agent config for the navigation
policy. This is NOT the locomotion config (that's rsl_rl). This is a
fresh skrl config for a smaller network appropriate for navigation
(navigation is a lower-dimensional problem than locomotion).

**Passing bar:** config loads without errors, agent instantiates
correctly with the right network shapes, no import errors.

**Key decisions to make here:**
- Actor hidden dims: start smaller than locomotion's [512,256,128]
  — try [256,128,64] as a first guess, tune later
- Critic: symmetric for now (same obs as actor, privileged obs added later)
- Normalization: RunningStandardScaler MUST be active — leaving it null
  causes the robot to immediately splay and fall (learned the hard way
  on the locomotion side)
- Timesteps: skrl uses total timesteps, not iterations —
  timesteps = max_iterations × num_steps_per_env × num_envs,
  confusing these terminates training far too early

**What to show your professor:** config file + agent instantiates +
network architecture printed to terminal showing correct layer shapes.

---

### AT2 — reward function design (1-2 days)
**What to build:** the navigation reward terms in
mdp/rewards.py — four named functions, each returning its own scalar
(critical for phase 03's multi-objective work later):

1. `reward_progress(env)` — reduction in distance to current waypoint
   each step. Most important term. Use Euclidean distance between robot's
   base position and the next A* waypoint.

2. `reward_heading(env)` — cosine similarity between robot's forward
   direction and the direction toward the next waypoint. Rewards facing
   the right way, penalizes crab-walking or reversing toward goal.

3. `penalty_collision(env)` — negative reward when the robot contacts
   any obstacle. Use the existing contact sensor already in the scene.

4. `reward_smoothness(env)` — negative reward proportional to
   step-to-step change in velocity command. Penalizes jerky, erratic
   commands. Formula: -||cmd_t - cmd_{t-1}||

**Passing bar:** all four terms log separately to TensorBoard from step 1.
The weighted sum matches what you'd expect. No term is orders of magnitude
larger than another (scale matters enormously for PPO).

**Key decision — reward scaling:** this is the most common failure mode.
Run 100 steps with RANDOM actions first and print the mean/std of each
reward term. If progress is ~0.1 and collision is ~-100, PPO will
optimize collision avoidance exclusively and ignore progress entirely.
Target: all terms roughly in the same order of magnitude after weighting.

**What to show:** TensorBoard screenshot with all 4 terms logged
separately + their weighted sum, after a short (30 min) training run.

---

### AT3 — observation space design (1 day)
**What to build:** the navigation policy's observation group in
mdp/observations.py. This is what the learned policy actually SEES.

For the privileged version (before real perception is ready):
```
- goal_direction: unit vector from robot to current A* waypoint (3-dim)
- goal_distance: scalar distance to current waypoint (1-dim)
- robot_base_vel: base linear + angular velocity (6-dim)
- waypoint_sequence: next 3 waypoints as relative positions (9-dim)
- obstacle_positions_privileged: positions of nearest N obstacles
  read directly from sim state — NOT from camera. This is the
  "privileged" part that gets swapped for real perception later. (varies)
```

**Passing bar:** observation group instantiates, shape is fixed and
consistent, no NaN values on first reset.

**Key thing to know:** your teammate's real occupancy grid (40x40=1600
cells, or a CNN-encoded version of it) replaces the
obstacle_positions_privileged term later. Design the observation group
so swapping that one term is a clean substitution, not a rebuild.

**What to show:** observation shape printed on env reset, confirmed
non-NaN, confirmed fixed size across resets.

---

### AT4 — first training run + TensorBoard validation (3-5 days)
**What to build:** get PPO training actually running end-to-end with
the privileged observation version. This is the "does it learn anything
at all" checkpoint — not a converged policy, just evidence of learning.

**Before launching any long run:**
1. Set CPU governor to performance mode:
   `sudo cpupower frequency-set -g performance`
   (powersave mode throttles throughput significantly — confirmed issue)
2. Run 500 steps with random actions, print reward statistics,
   confirm no NaNs and reasonable scales
3. Run a 30-min training session, check TensorBoard — is anything
   trending? If total reward is flat or oscillating wildly, something
   is wrong with reward scales or observation normalization

**Passing bar:**
- Total reward curve trending upward (even slowly) over training
- No individual reward term flatlined at zero or exploding
- Policy entropy not collapsing to zero in the first few thousand steps
  (entropy collapse = policy got stuck in a local mode too early,
  usually means learning rate too high or entropy_coef too low)

**What to show:** TensorBoard screenshot showing reward trending up +
all 4 individual terms behaving sanely over a 1-2 hour training run.

**Common failure modes to debug:**
- NaN in rewards: check observation normalization is actually on
- Reward stuck at zero: check that the robot is actually moving
  (frozen locomotion checkpoint loading correctly?)
- Reward oscillating: reward scale mismatch, try reducing learning rate
- Robot immediately falls: normalization is off, or command ranges
  are outside what the frozen locomotion policy was trained on
  (keep vx in [-1.0, 1.0], vy in [-0.5, 0.5], yaw in [-1.0, 1.0])

---

### AT5 — converged single-policy baseline + eval metrics (1 week)
**What to build:** train to convergence (this is mostly GPU waiting time
— plan other work in parallel), then write eval_metrics.py to get
hard numbers.

**eval_metrics.py should log per episode:**
- success: did the robot reach the goal? (bool)
- time_to_goal: episode length in seconds
- collision_count: number of contact events during episode
- path_efficiency: actual distance traveled / A* optimal path length
  (ratio — 1.0 = perfectly efficient, higher = more wandering)

Run N=50 evaluation episodes, save to CSV.

**Passing bar:**
- Success rate > 60% on the static rubble scene
- path_efficiency < 2.0 (not wandering excessively)
- These numbers become your baseline that every later phase compares to

**What to show:** a printed table + the CSV file. This is your
Phase 01 result, the number your supervisor will point to when
asking "did the navigation policy learn anything."

---

### AT6 — risk-prediction head (1 week, aligns with teammate's CP9)
**What to build:** add a small auxiliary output head to the policy
network predicting P(collision in next N steps), trained via binary
cross-entropy alongside the main PPO loss.

**Why:** this is the ABS-derived mechanism — anticipatory safety
signal that shapes the reward BEFORE contact, not just after.
Without it, the policy only learns "collisions are bad" after they
happen. With it, the policy learns to deflect before contact.

**Implementation in skrl:** skrl supports custom models — you'll need
a custom actor class (inheriting from skrl's GaussianMixin) that has
two output heads: the main action mean (velocity command) and the
auxiliary collision-probability scalar. The auxiliary BCE loss is
computed separately and added to the PPO loss with a small weight
(start with 0.1 × BCE, tune if needed).

**Passing bar:**
- Auxiliary risk loss decreasing in TensorBoard (the head is learning
  to predict something, not outputting constant 0.5)
- Main reward still improving alongside it (auxiliary head didn't
  destabilize the main objective)

**What to show:** TensorBoard with both losses plotted + a recording
showing the robot visibly slowing/deflecting before contact events.

---

### AT7 — dual policy: progress + caution (1 week, aligns with CP12)
**What to build:** split the single navigation policy into two PPO
policies trained with different reward weightings:

- Progress policy: high weight on progress + heading, low weight on
  collision penalty. Optimizes for speed and efficiency.
- Caution policy: high weight on collision penalty + smoothness, low
  weight on progress. Optimizes for safety.

Then blend their output velocity commands using the risk head's output
as the blend weight: `cmd = (1 - risk) × cmd_progress + risk × cmd_caution`

**Important:** train both policies from scratch with different configs,
not by fine-tuning one into two. They need genuinely different behaviors,
not just different temperatures on the same learned distribution.

**Passing bar:**
- Each policy, run in isolation, visibly behaves differently:
  progress policy moves faster and more directly, caution policy
  moves slower and gives wider berth to obstacles
- The blend doesn't produce visible jerks or instability at the
  switch point (use the smoothness reward to mitigate this)

**What to show:** two side-by-side recordings (each policy alone) +
one recording of the blended system, plus collision rate comparison
between: single policy (AT5 baseline) vs blended dual policy.

---

### AT8 — multi-objective Pareto sweep (1-2 weeks GPU time, aligns with CP14)
**What to build:** externalize all reward weights into yaml config files,
then train 3-5 configurations with different weight settings:

```
weights_speed.yaml:    progress=2.0, heading=1.0, collision=-0.5, smooth=-0.1
weights_balanced.yaml: progress=1.0, heading=1.0, collision=-1.0, smooth=-0.5
weights_energy.yaml:   progress=0.5, heading=0.5, collision=-1.0, smooth=-1.5
weights_safe.yaml:     progress=0.5, heading=0.5, collision=-3.0, smooth=-0.5
```

Then eval_metrics.py on all trained checkpoints → plot_pareto.py
plotting energy (Σ|torque × joint_vel|, already computed in sim) vs
time-to-goal for each configuration.

**Passing bar:** a plot showing a visible tradeoff — e.g. speed config
reaches goal faster but has higher collision rate than safe config.
The tradeoff existing and being plotted IS the multi-objective result.

**What to show:** the Pareto plot. Each point is one trained policy.
The curve shape (not a single point) is the contribution.

---

## INTEGRATION CHECKLIST
When your teammate's perception pipeline (CP3-CP6) is ready:

- [ ] Replace obstacle_positions_privileged observation term with the
      real CNN-encoded occupancy grid tensor from the depth camera
- [ ] Retrain AT4-AT5 with real perception (observation shape changes,
      so existing checkpoint is incompatible — expected, not a setback)
- [ ] Compare AT5 metrics (privileged) vs re-trained (real perception)
      to quantify the perception gap
- [ ] The dual policy (AT7) and Pareto sweep (AT8) only need to run
      ONCE on the real-perception version, not duplicated

---

## KEY TECHNICAL DECISIONS YOUR TEAMMATE MADE (don't redo these)
- Flat ground for CP1/AT1-AT3, rubble terrain only from CP3/AT4 onward
- Fixed velocity command ranges: vx∈[-1,1], vy∈[-0.5,0.5], yaw∈[-1,1]
  (outside this = frozen locomotion policy goes out of distribution)
- heading_command=False in the command config (nav policy outputs
  yaw rate directly, not a target heading angle)
- Locomotion checkpoint: logs/rsl_rl/unitree_go2_rough/<run>/model_*.pt
- Task ID: NeuroGait-Navigation-Rubble-v0 (registered in __init__.py)
- skrl is the RL framework for navigation (NOT rsl_rl)

---

## REPO STRUCTURE (your relevant files)
```
neurogait/tasks/manager_based/navigation/
├── config/go2/
│   ├── agents/
│   │   ├── skrl_nav_ppo_cfg.yaml          ← YOUR MAIN CONFIG (AT1)
│   │   ├── weights_speed.yaml             ← AT8
│   │   ├── weights_balanced.yaml          ← AT8
│   │   └── weights_energy.yaml            ← AT8
│   └── navigation_env_cfg.py              ← teammate owns this
├── mdp/
│   ├── observations.py                    ← YOUR FILE (AT3)
│   ├── rewards.py                         ← YOUR FILE (AT2)
│   ├── commands.py                        ← teammate owns this
│   └── terminations.py                    ← shared
└── planner.py                             ← teammate owns this

scripts/
├── skrl/
│   ├── train_nav.py                       ← YOUR TRAINING SCRIPT
│   └── eval_metrics.py                    ← YOUR EVAL SCRIPT (AT5)
└── plot_pareto.py                         ← YOUR PLOT SCRIPT (AT8)
```

---

## TIMELINE ALIGNMENT

| Your checkpoint | Teammate checkpoint | What integrates |
|---|---|---|
| AT1 (config) | CP1 done | Independent — run in parallel |
| AT2-AT3 (reward/obs) | CP2-CP3 in progress | Independent — run in parallel |
| AT4 (first training) | CP4 (rule-based pipeline) | Share the nav env shell |
| AT5 (baseline) | CP6 (phase 01 full demo) | FIRST INTEGRATION POINT |
| AT6 (risk head) | CP9 (risk head demo) | Same mechanism, both sides |
| AT7 (dual policy) | CP12 (dual policy demo) | Policy outputs go to sim |
| AT8 (Pareto) | CP14 (Pareto front) | Same eval script, different inputs |
