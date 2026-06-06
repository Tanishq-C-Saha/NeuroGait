# Understanding `commands.py` — The Velocity Command Configuration

*A learn-as-you-clone walkthrough of the command file in the NeuroGait velocity task.*
*This is the single most important file for understanding **what the task actually is**.*

---

## 1. The big idea: a "command" *is* the task

Stop and absorb this before reading the code, because it reframes everything.

Your robot is not learning to "walk." It's learning to **track a velocity command**. At every moment the task hands the policy a target — "move forward at 0.8 m/s while turning left at 0.3 rad/s" — and the policy's job is to make the real base velocity match that target. The reward terms `tracking_lin_vel` and `tracking_ang_vel` measure exactly how well it does.

So the command is two things at once:
1. **Part of the observation** — the policy *sees* the command as input. This is "goal-conditioned RL": the same policy produces different behavior for different commanded velocities.
2. **The definition of success** — the reward is *defined relative to* the command. No command, no task.

This file configures the **CommandManager** — one of the managers in the manager-based workflow, a sibling to the Observation, Reward, and Termination managers you've already met. Its job: generate a stream of velocity targets for the robot to chase.

---

## 2. Why *random, changing* commands?

```python
base_velocity = env_mdp.UniformVelocityCommandCfg(...)
```

`Uniform` = the command is sampled uniformly at random from a range. `Velocity` = the command is a target velocity (not a position or pose).

The reason it's *random and changing* rather than fixed: if you only ever commanded "walk forward at 1 m/s," the policy would learn that one trick and nothing else. By sampling random velocities (and re-sampling them periodically), you force the policy to learn a *whole skill surface* — forward, backward, sideways, turning, standing — in one training run. That's what makes the resulting locomotion controller general and reusable.

---

## 3. Field by field

```python
asset_name="robot",
```
Which asset the command drives. `"robot"` matches the articulation slot defined in the scene (the one that was `MISSING` and gets filled by the Go2 config). The command and the robot are wired together by this name.

```python
resampling_time_range=(10.0, 10.0),
```
Every **10 seconds** the robot is given a *fresh* random command. Both bounds are equal, so the interval is fixed at 10 s; a range like `(5.0, 12.0)` would randomize the interval too. Re-sampling mid-episode is what teaches the policy to *transition* between gaits/directions on the fly, not just hold one velocity.

```python
rel_standing_envs=0.02,
```
**2% of the parallel environments** are given a *stand-still* command (zero velocity) instead of a motion command. This is easy to overlook but important: without it, a robot trained only on motion commands often can't stand still cleanly — it fidgets. Reserving a slice of envs for "stand still" teaches a stable standing behavior. (`rel_` = a *relative fraction* of all envs.)

```python
rel_heading_envs=1.0,
heading_command=True,
heading_control_stiffness=0.5,
```
These three work together and contain the subtlest behavior in the file:

- **`heading_command=True`** changes *how yaw is commanded*. Instead of directly commanding a yaw *rate*, the task commands a target *heading* (a compass direction), and the yaw rate is computed automatically to steer the robot toward it.
- **`heading_control_stiffness=0.5`** is the proportional gain in that steering: `commanded_yaw_rate = 0.5 × (target_heading − current_heading)`. Higher = turns toward the target heading more aggressively.
- **`rel_heading_envs=1.0`** means **100%** of environments use this heading-based scheme.

**The consequence (important):** because every env uses heading control, the `ang_vel_z` sampling range below is largely *bypassed* — the yaw command is derived from heading error, not sampled directly. If you later set `rel_heading_envs=0.0`, the robot would instead receive raw sampled yaw-rate commands and the `ang_vel_z` range would govern turning. Knowing which mode you're in matters when you debug "why won't my robot turn the way I expect."

```python
debug_vis=True,
```
Draws the command as **arrows on the robot** in the viewport — typically the commanded velocity vs. the actual velocity. Genuinely useful: when you watch training, you can *see* whether the robot is following its command. Set `False` for headless runs (no effect on learning).

```python
ranges=env_mdp.UniformVelocityCommandCfg.Ranges(
    lin_vel_x=(-1.0, 1.0),
    lin_vel_y=(-1.0, 1.0),
    ang_vel_z=(-1.0, 1.0),
    heading=(-math.pi, math.pi),
),
```
The sampling bounds — the "envelope" of commands the robot must master:
- **`lin_vel_x=(-1.0, 1.0)`** — forward/backward, ±1 m/s.
- **`lin_vel_y=(-1.0, 1.0)`** — lateral (strafing) left/right, ±1 m/s.
- **`ang_vel_z=(-1.0, 1.0)`** — yaw rate, ±1 rad/s. *(Mostly bypassed here because of heading control — see above.)*
- **`heading=(-math.pi, math.pi)`** — the target heading, a full ±180°. Since heading control is on, this is the *active* turning command.

These numbers define how athletic the policy must be. They're conservative defaults — a starting envelope, not a law.

---

## 4. The NeuroGait insight: this file is the locomotion↔navigation seam

This is the most important thing to take away for *your* project.

Right now, `UniformVelocityCommandCfg` is a **random number generator** feeding velocity targets to the low-level policy. But look at what it *is* structurally: it's "the thing that decides what velocity the robot should move at." In your hierarchical architecture, that is **exactly the job of the high-level navigation policy.**

So when you reach the navigation phase (Week 4), the move is conceptually clean:

```
NOW (locomotion benchmark):
   random sampler ─→ velocity command ─→ low-level policy ─→ joints

LATER (navigation):
   high-level nav policy ─→ velocity command ─→ FROZEN low-level policy ─→ joints
```

You **replace this random command source with the output of the navigation policy.** The low-level policy doesn't even know the difference — it still just tracks a commanded velocity. That clean interface is *why* the hierarchical decomposition works, and this file is where the interface lives. Understanding `commands.py` is understanding the bolt that joins your two layers.

---

## 5. Conventions you just learned

| Concept | What it means | Why it matters |
|---|---|---|
| Command = task | The MDP's goal is "track this command" | Reward is defined relative to the command; it's also a policy input |
| Goal-conditioned RL | Policy sees the command and adapts behavior | One policy, many behaviors (forward/back/turn/stand) |
| Resampling | Commands change mid-episode | Teaches transitions, not a single fixed gait |
| `rel_standing_envs` | Fraction commanded to stand still | Produces a clean stand, not fidgeting |
| `heading_command` | Command a heading, derive the yaw rate | More natural steering; but it bypasses the `ang_vel_z` range |
| `ranges` | The command envelope | Defines how athletic/general the policy must be |
| CommandManager | The manager that owns all this | A sibling to Observation/Reward/Termination managers |

---

## 6. What you could add / change on your journey

**Soon — for the locomotion benchmark (Weeks 2–3)**
- **Command curriculum.** Start with a narrow envelope (e.g. `lin_vel_x=(-0.5, 0.5)`) and widen it as the policy improves, via the `CurriculumManager`. Easier early learning, more capable final policy. Common in published work.
- **Match the envelope to the Go2's real capability.** ±1 m/s is modest; the Go2 can go faster. If your benchmark cares about high-speed locomotion, widen `lin_vel_x` — but do it identically across PPO/SAC/TD3 so the comparison stays fair.
- **Stress-test standing.** Bump `rel_standing_envs` if your robot struggles to hold still (relevant before adding obstacles, where stopping matters).

**Mid — the navigation phase (Week 4)**
- **Swap the command source.** Replace (or wrap) `UniformVelocityCommandCfg` so commands come from your high-level navigation policy instead of the random sampler — the seam described in §4.
- **Consider a pose/position command** for the high level. IsaacLab also ships position/pose command terms; the navigation policy might output "go to this point" which the system converts to velocities.

**Later — research polish**
- **Reproducibility:** fix the command-sampling seed so all three algorithms see the *same* command stream during benchmarking. Otherwise you're partly measuring luck-of-the-draw in commands, not algorithm quality.
- **Report the command envelope in your paper/thesis.** Reviewers need to know the velocity range to interpret tracking-error numbers — it's a standard methods-section detail.

---

## 7. Industrial & research conventions to carry forward

- **Goal-conditioning is the standard locomotion formulation.** Virtually all modern legged-RL (ETH, MIT, Unitree) trains a *velocity-conditioned* policy, not a fixed-gait one. You're learning the field-standard interface.
- **The command range is a reported experimental parameter**, like learning rate — not an afterthought. Pin it, log it, report it.
- **Hierarchical control via a command interface** (high level outputs commands, low level tracks them) is a mainstream architecture for navigation-over-locomotion. The clean seam you see here is *why* it's mainstream — the layers stay decoupled and independently trainable.
- **Curriculum on commands and terrain together** is how labs get policies that are both fast and robust. Expect to tune both curricula.

---

## 8. One-paragraph summary

`commands.py` configures the **CommandManager** to generate velocity targets the robot must track — the very definition of the velocity task. It samples random linear (x, y) and angular/heading commands within a set envelope, re-samples them every 10 s to teach transitions, reserves 2% of environments for standing still, and uses *heading control* (so yaw is driven by a target heading rather than a raw yaw-rate command, which means the `ang_vel_z` range is mostly bypassed). The command is both an *input* to the policy (goal-conditioning) and the *reference* for the tracking rewards. For NeuroGait specifically, this file is the **locomotion↔navigation seam**: today a random sampler fills it, but your high-level navigation policy will later supply these commands to the frozen low-level controller — which is exactly what makes the hierarchical design work.