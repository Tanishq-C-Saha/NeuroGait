# Understanding `events.py` — Domain Randomization & The Sim-to-Real File

*A learn-as-you-clone walkthrough of the event file in the NeuroGait velocity task.*
*This is **the** file that decides whether your policy survives contact with the real Go2. Read it carefully.*

---

## 1. The big idea: events = randomization + resets + perturbations

The **EventManager** is how the task injects *variation and disturbance* into training. It does three kinds of work:

1. **Domain randomization** — randomly vary physical parameters (friction, mass, center of mass) so the policy never overfits to one exact robot/world.
2. **Resets** — decide the state each environment starts an episode in (pose, velocity, joint angles).
3. **Perturbations** — shove the robot mid-episode so it learns to recover balance.

Here's *why* domain randomization is the single most important sim-to-real idea: a simulator is never a perfect copy of reality (the "reality gap"). If you instead train across a *distribution* of simulators — some slippery, some heavy, some with an off-center mass — then the real robot just looks like *one more sample* from that distribution the policy has already seen. **You don't model reality perfectly; you make the policy robust to not knowing it exactly.** This file is where that robustness is manufactured.

> Earlier I told you DR happens in the EventManager, not the scene. This is that file. The scene set *default* friction/mass; the events *perturb* them.

---

## 2. The three `mode`s (the organizing principle)

Every term has a `mode` that decides *when* it fires. This is the key to reading the file:

| `mode` | Fires when | Used for | Per-env effect |
|---|---|---|---|
| `"startup"` | Once, at sim start | Static randomization | Each of the 1000s of cloned envs gets its *own* fixed friction/mass for its whole life |
| `"reset"` | Every episode reset | Initial-state variety | New start pose/velocity/joints each episode |
| `"interval"` | Periodically mid-episode | Disturbances | Random pushes during walking |

So at startup, env_0 might be a 3 kg-heavier, slightly slippery robot and env_42 a 2 kg-lighter, grippier one — and they stay that way. Across the whole batch, the policy trains on a *spread* of robots simultaneously. That spread is the domain randomization.

---

## 3. Startup events — static domain randomization

### Friction (currently OFF — important)

```python
physics_material = EventTerm(
    func=env_mdp.randomize_rigid_body_material,
    mode="startup",
    params={
        "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
        "static_friction_range": (0.8, 0.8),
        "dynamic_friction_range": (0.6, 0.6),
        "restitution_range": (0.0, 0.0),
        "num_buckets": 64,
    },
)
```

- Randomizes the **contact material** of the robot's bodies (`body_names=".*"` = all of them, including feet).
- **The catch:** `(0.8, 0.8)` is a *degenerate range* — min equals max — so friction is being **set to a fixed 0.8, not randomized**. Same for dynamic friction (0.6) and restitution (0). Right now this term does *no* randomization; it just pins values.
- `num_buckets=64` is a performance trick: randomizing materials per-body is expensive in PhysX, so it precomputes 64 material "buckets" and assigns bodies among them.
- **This is your friction-cone hook.** When you reach sim-to-real, *this* is the line you widen — e.g. `static_friction_range=(0.4, 1.2)` — so the policy learns to walk on everything from near-ice to high-grip rubber. Leaving it at `(0.8, 0.8)` is fine for the locomotion benchmark but **insufficient for hardware transfer.**

### Base mass (currently ON)

```python
add_base_mass = EventTerm(
    func=env_mdp.randomize_rigid_body_mass,
    mode="startup",
    params={
        "asset_cfg": SceneEntityCfg("robot", body_names="base"),
        "mass_distribution_params": (-5.0, 5.0),
        "operation": "add",
    },
)
```

- **Adds** a random offset between **−5 and +5 kg** to the base mass (`operation="add"` → added to nominal, not replacing it).
- This range is *real* (non-degenerate), so this one *is* randomizing. It simulates payload variation and the fact that the sim's mass never exactly matches the real robot. A policy trained on a ±5 kg spread won't fall over when the real Go2 weighs slightly more than its URDF says, or when you strap a sensor to it.

### Center of mass (currently ON)

```python
base_com = EventTerm(
    func=env_mdp.randomize_rigid_body_com,
    mode="startup",
    params={
        "asset_cfg": SceneEntityCfg("robot", body_names="base"),
        "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
    },
)
```

- Shifts the base's **center of mass** by up to ±5 cm horizontally, ±1 cm vertically.
- CoM is something you *never* know exactly on a real robot (battery placement, wiring, payload). Randomizing it makes the balance controller robust to that uncertainty. The tighter z-range reflects that vertical CoM shifts are smaller and matter differently than horizontal ones.

---

## 4. Reset events — initial-state variety

### External force at reset (currently OFF)

```python
base_external_force_torque = EventTerm(
    func=env_mdp.apply_external_force_torque,
    mode="reset",
    params={"force_range": (0.0, 0.0), "torque_range": (-0.0, 0.0)},
)
```
A placeholder hook to apply a random force/torque at reset — currently zeroed, so disabled. Another knob you can switch on for extra disturbance robustness.

### Reset base pose & velocity (ON)

```python
reset_base = EventTerm(
    func=env_mdp.reset_root_state_uniform,
    mode="reset",
    params={
        "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
        "velocity_range": {"x": (-0.5,0.5), "y": (-0.5,0.5), "z": (-0.5,0.5),
                           "roll": (-0.5,0.5), "pitch": (-0.5,0.5), "yaw": (-0.5,0.5)},
    },
)
```
- Each episode, the robot is placed at a random offset (±0.5 m, *any* yaw 0–360°) **and** given a random initial velocity in all 6 DOF.
- Starting from varied positions, headings, and even tumbling slightly means the policy can't memorize one perfect start — it must handle *any* reasonable state. The full-range yaw is what lets it follow heading commands in every direction.

### Reset joints (ON)

```python
reset_robot_joints = EventTerm(
    func=env_mdp.reset_joints_by_scale,
    mode="reset",
    params={"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)},
)
```
- Initial joint positions are the default pose **scaled by a random factor 0.5–1.5**, with zero initial joint velocity.
- So the robot starts each episode in a slightly different crouch/stance. More starting-posture variety → a policy that can stabilize from many configurations, not just the perfect stand.

---

## 5. Interval events — the push (ON, and crucial)

```python
push_robot = EventTerm(
    func=env_mdp.push_by_setting_velocity,
    mode="interval",
    interval_range_s=(10.0, 15.0),
    params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
)
```

- Every **10–15 seconds** during an episode, the robot's base velocity is randomly perturbed by up to ±0.5 m/s — effectively a **shove**.
- This is the classic robustness trick: a policy that's only ever walked undisturbed will topple the first time the real world bumps it. Training under random pushes teaches active balance recovery. It's a big part of why learned controllers look so sturdy when researchers kick the robot in demo videos.
- For sim-to-real you often *increase* push magnitude/frequency to harden the policy further.

---

## 6. The NeuroGait insight: this file IS your adaptation problem

This is the deepest connection in the whole project, so sit with it.

Remember **RMA / implicit terrain adaptation** — where an adaptation module estimates the environment's hidden "extrinsics" (friction, payload, terrain) from proprioception? **The extrinsics RMA estimates are *exactly the things this file randomizes*** — base mass, CoM, friction, pushes. The teacher policy in RMA/DreamWaQ gets to *see* these randomized values directly (privileged info); the student learns to *infer* them from how the robot moves.

So the chain for your novel contribution is:

```
events.py randomizes mass/friction/CoM/pushes  ──►  these are the "extrinsics"
        │                                                      │
teacher policy sees them directly (privileged)      student infers them from
                                                    proprioception → implicit adaptation
```

**Widening the ranges in this file is what creates the adaptation problem worth solving.** If nothing varies, there's nothing to adapt to. Your implicit-terrain-adaptation work and this file are two ends of the same idea.

Two more NeuroGait-specific points:

- **Benchmark fairness:** for your PPO-vs-SAC-vs-TD3 comparison, the event config must be **identical and seeded** across all three. Different randomization = you're comparing luck, not algorithms.
- **Phase staging:** keep randomization modest for the *benchmark* (clean comparison), then *widen* it for the *sim-to-real* phase (robustness). Don't conflate the two.

---

## 7. Conventions you just learned

| Concept | What it means | Why it matters |
|---|---|---|
| Domain randomization | Vary sim params across envs | Bridges the reality gap; the #1 sim-to-real technique |
| `mode` (startup/reset/interval) | When a term fires | Static DR vs per-episode resets vs mid-episode pushes |
| Degenerate range `(x, x)` | min == max → no randomization | Spot these! `(0.8,0.8)` friction is *off*, not randomized |
| Per-env randomization | Each clone gets its own params | The batch becomes a *distribution* of robots |
| Push events | Mid-episode disturbances | Active balance recovery, sturdiness |
| `num_buckets` | Precomputed material sets | Performance optimization for PhysX |
| Privileged extrinsics | The randomized values | What RMA's teacher sees and the student infers |

---

## 8. What you could add / change on your journey

**For sim-to-real robustness (the main upgrades)**
- **Widen friction** — turn `(0.8, 0.8)` into a real range like `(0.3, 1.25)`. *The* friction-cone randomization you flagged at the start of the project.
- **Randomize actuator gains (Kp/Kd).** Not present here yet, and one of the most impactful sim-to-real additions — the real Go2's effective gains differ from sim. Add a startup event randomizing PD stiffness/damping.
- **Enable & randomize the reset force/torque** and **increase push magnitude** for a tougher policy.
- **Add observation/sensor noise and latency.** Note: obs noise is configured in the *observation* config, not here — but it's part of the same DR philosophy. The real robot's sensors are noisy and delayed; train for it.

**For research polish**
- **Curriculum the randomization** — start narrow, widen as the policy improves (via the CurriculumManager), so early learning isn't drowned by disturbance.
- **Log/report your randomization ranges.** They're a core part of the methods section; sim-to-real results are meaningless without them.

---

## 9. Industrial & research conventions to carry forward

- **Domain randomization is the field-standard bridge to reality** (Tobin et al. 2017 onward; used everywhere from OpenAI's dexterous hand to ETH's ANYmal). Randomize what you're *uncertain* about (mass, friction, CoM, latency); don't randomize what you know exactly.
- **Push/disturbance training = robustness you can see.** It's why learned quadrupeds shrug off kicks. Standard practice.
- **The randomized extrinsics are the basis of adaptation methods** (RMA, DreamWaQ, HIM). Your privileged-teacher / proprioceptive-student design draws its signal from exactly these terms — the connection in §6 is the conceptual core of your contribution.
- **Separate "benchmark" randomization from "deployment" randomization.** Light and fixed for fair algorithm comparison; wide and curriculum'd for hardware transfer. Conflating them muddies both your benchmark and your sim-to-real story.
- **Watch for degenerate ranges in cloned configs.** Inherited configs often ship with randomization *defined but disabled* (`(x, x)`). Always check whether a term is actually randomizing before you trust your robustness.

---

## 10. One-paragraph summary

`events.py` configures the **EventManager** — the home of domain randomization, episode resets, and disturbances. At **startup** it (optionally) randomizes friction, base mass (±5 kg), and center of mass per-environment so the policy trains on a *distribution* of robots; at **reset** it varies initial pose, velocity, and joint angles for state diversity; at **intervals** it shoves the robot every 10–15 s to teach balance recovery. Crucially, **several ranges are currently degenerate** (friction `(0.8,0.8)`, reset force `(0,0)`) — defined but switched off — and those are exactly what you'll widen for sim-to-real. The deep NeuroGait link: the parameters randomized here (mass, friction, CoM, pushes) *are* the hidden "extrinsics" that RMA-style implicit adaptation learns to infer — so this file both creates the reality-gap robustness *and* defines the adaptation problem your novel contribution solves.