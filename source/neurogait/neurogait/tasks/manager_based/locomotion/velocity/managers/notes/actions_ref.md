# Understanding `actions.py` — The Action Space Configuration

*A learn-as-you-clone walkthrough of the action file in the NeuroGait velocity task.*
*This file defines **what the policy actually controls** — and it hides two decisions that make or break sim-to-real.*

---

## 1. The big idea: the policy does *not* command torques

This is the line most newcomers get wrong in their mental model, so lead with it:

**Your policy outputs target joint *positions*, not motor torques.**

The neural network's output is interpreted as "where should each joint go," and a low-level **PD controller** then computes the torque needed to drive each joint toward that target. The flow is:

```
policy network ─→ target joint positions ─→ PD controller ─→ torques ─→ motors
        (this file configures the first arrow)   (lives in the robot's actuator config)
```

Why position control instead of letting the policy output torque directly? Because:
- **It's vastly easier to learn.** "Put the knee at 0.6 rad" is a stabler, smoother target than "apply 12.3 N·m," which the policy would have to chase at high frequency.
- **It transfers to real hardware.** Real quadruped motors (the Go2 included) are commanded with position targets + PD gains. A policy that outputs positions matches how the robot is actually driven. Torque-control policies are notoriously hard to get across the sim-to-real gap.

This is the field standard — ETH, MIT, and Unitree all use position control for learned locomotion. You're learning the convention, not a quirk.

---

## 2. The file, in one block

```python
@configclass
class ActionsCfg:
    joint_pos = env_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.5,
        use_default_offset=True,
    )
```

`ActionsCfg` configures the **ActionManager** — another sibling of the Observation / Reward / Command / Termination managers. Its job: take the raw numbers the network spits out and turn them into something the simulator can apply to the robot. Here there's a single action term, `joint_pos`, controlling all joints by position.

---

## 3. Field by field

```python
asset_name="robot",
```
Which articulation these actions drive — the same `"robot"` slot the scene left `MISSING` and the Go2 config fills in. Actions, commands, and the robot are all wired together by this shared name.

```python
joint_names=[".*"],
```
A regex selecting **which joints** the policy controls. `.*` matches *all* of them — for the Go2 that's all 12 actuated joints (hip, thigh, calf × 4 legs). The policy outputs a 12-dimensional action vector, one target per joint. You *could* restrict this (e.g. `".*_hip_.*"` to control only hips) but for whole-body locomotion you want everything.

```python
scale=0.5,
```
The raw network output is **multiplied by 0.5** before use. Networks tend to output values roughly in the ±1 range; scaling by 0.5 keeps the commanded position offsets modest (about ±0.5 rad here). This matters more than it looks:
- **Too large a scale** → the policy can command violent, joint-limit-slamming motions, especially early in training when outputs are near-random → unstable learning, possible self-collision.
- **Too small** → the robot can't move expressively enough to walk well.
- `0.5` is a sane, well-tested default. It's a real tuning knob, but change it deliberately.

```python
use_default_offset=True,
```
The most important and most subtle line. With this on, the action is applied as an **offset from the robot's default (nominal standing) joint positions**:

```
target_position = raw_action × scale + default_joint_position
```

The payoff: **a zero action means "hold the default standing pose."** So the policy doesn't have to learn the absolute geometry of standing from scratch — it starts from a sensible crouch/stand prior and only learns *corrections* to it. This:
- dramatically speeds up and stabilizes early training (random initial actions produce small wobbles around a stand, not a flailing collapse),
- keeps the learned motion centered on a natural posture,
- is a big reason these policies train "in minutes" rather than fighting from a bad starting point.

Turn this off and the policy must discover the entire standing configuration through reward alone — much slower and more fragile.

---

## 4. Where the rest of the control loop lives (so you're not confused later)

This file is only the *first arrow* in the diagram. Two related pieces live elsewhere, and knowing where saves debugging pain:

- **The PD gains (stiffness `Kp`, damping `Kd`)** that convert position-target → torque live in the **robot's actuator configuration** (the `ArticulationCfg` / actuator model in the Go2 config), **not** here. If the robot is sluggish or twitchy, you tune gains there, not in this file.
- **The control frequency** — how often the policy issues a new action versus how often the PD loop and physics run — is set by **`decimation`** in the *environment* config (`velocity_env_cfg.py`). Typically the policy acts at ~50 Hz while physics steps faster; the PD controller fills the gap. The action here is held between policy steps.

So: **action term (this file) → actuator PD gains (robot config) → decimation/timing (env config).** Three files, one control loop.

---

## 5. The NeuroGait insight: this is purely the *low-level* output

In your hierarchical architecture, keep the layers straight:

```
high-level nav policy ─→ velocity COMMAND  (commands.py)
                                │
low-level locomotion policy ─→ joint POSITION ACTIONS  (THIS FILE) ─→ PD ─→ torques
```

- **Commands** (previous file) are the *input goal* — and eventually the navigation policy supplies them.
- **Actions** (this file) are the *low-level motor output* — and they stay the same whether the command came from a random sampler or your navigation policy.

The navigation policy will **never** touch this file's action space — it operates a level up, in velocity-command space. This file is forever the low-level controller's mouth. Understanding that boundary is understanding why the locomotion policy you train now can be *frozen and reused* under the navigation layer later.

There's also a direct tie to your **jerk metric**: smooth actions = smooth motion. The `action_rate` reward term penalizes large *changes* in this action vector between steps, pushing the policy toward fluid motion. `scale` and the action space defined here are what that smoothness is measured over.

---

## 6. Conventions you just learned

| Concept | What it means | Why it matters |
|---|---|---|
| Position control (not torque) | Policy outputs joint targets; PD makes torque | Easier to learn **and** sim-to-real friendly — the field standard |
| `use_default_offset` | Action = offset from standing pose | Zero action = stand; learns corrections, not absolutes → fast, stable training |
| `scale` | Shrinks raw network output | Keeps motions sane, prevents early-training violence |
| `joint_names=[".*"]` | Regex selects controlled joints | All 12 joints → whole-body control |
| ActionManager | Owns the action processing | Sibling of Observation/Reward/Command managers |
| Separation of action vs gains vs timing | Three files, one loop | Know which file to edit when behavior is wrong |

---

## 7. What you could add / change on your journey

**Soon — locomotion benchmark (Weeks 2–3)**
- **Experiment with `scale`** as a hyperparameter — but identically across PPO/SAC/TD3 to keep the benchmark fair.
- **Per-joint scaling.** You can give hips, thighs, and calves different scales if one joint group needs more or less authority. Some tuned controllers do this.
- **Tune PD gains** (in the robot config, not here) alongside `scale` — they interact. Stiff gains + large scale = aggressive; soft gains + small scale = compliant.

**Mid — sim-to-real prep**
- **Action filtering / rate limiting.** To protect the real Go2's motors, a low-pass filter on actions is common before deployment (this is one of the standard sim-to-real techniques). Keep the action space here clean; add filtering in the deploy path.
- **Match the action space to Unitree's deploy expectations.** Unitree's `unitree_rl_lab` deploy pipeline expects position targets — your position-control choice already aligns, which is exactly why this default is the right one.

**Later — research polish**
- **Document the action space precisely** in your thesis/paper: position control, 12-dim, offset-from-default, scale 0.5. Reviewers need this to interpret your results and reproduce them.

---

## 8. Industrial & research conventions to carry forward

- **Position control + PD is the legged-RL standard**, specifically because it bridges the sim-to-real gap. Torque-control policies look elegant but rarely transfer — don't reach for them without a strong reason.
- **Default-offset actions are a learning prior.** Encoding "start from a reasonable stance and learn deviations" is a small change that buys large stability. It generalizes: giving a policy a sensible default to correct from is good practice well beyond locomotion.
- **Keep action / actuator / timing concerns separated.** The discipline of "action term here, gains in the robot config, decimation in the env" is what keeps large robotics codebases maintainable.
- **Smoothness is a first-class objective**, not an afterthought — hence the `action_rate` penalty and your jerk metric. Real motors and real gearboxes punish jerky commands; a controller that's smooth in sim is far likelier to survive on hardware.

---

## 9. One-paragraph summary

`actions.py` configures the **ActionManager** with a single position-control term: the policy outputs a 12-dim vector interpreted (after a ×0.5 scale) as **offsets from the Go2's default standing joint positions**, which a downstream **PD controller** turns into motor torques. Two decisions dominate: **position control** (not torque) makes the task learnable *and* sim-to-real-friendly, and **`use_default_offset=True`** gives the policy a standing-pose prior so it learns corrections rather than absolute geometry — a major reason these policies train fast and stably. The PD gains live in the robot's actuator config and the control timing in the env config, so this file is just the first link in the chain. For NeuroGait, this is strictly the **low-level** output: the navigation policy never touches it, which is precisely why the locomotion policy you train here can be frozen and reused under the navigation layer.