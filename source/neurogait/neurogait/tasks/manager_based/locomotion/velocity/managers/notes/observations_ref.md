# Understanding `observations.py` — What the Policy *Sees* (and the Door to Your Novelty)

*A learn-as-you-clone walkthrough of the observation file in the NeuroGait velocity task.*
*This file defines the neural network's **input**. It's also where your implicit-terrain-adaptation contribution will be built — via a privileged critic group.*

---

## 1. The big idea: observations are the policy's senses

The **ObservationManager** assembles everything the policy *perceives* each step into a vector — the input to the neural network. If it's not in the observation, the policy is blind to it. So this file literally defines the robot's window onto its world.

Three categories of observation live here, and telling them apart is the key to the whole file:

| Category | What it is | On the real robot? | In this file |
|---|---|---|---|
| **Proprioception** | The robot sensing its *own body* | ✅ Yes (IMU, joint encoders) | base vel, gravity, joint pos/vel, last action |
| **Exteroception** | The robot sensing the *external world* | ⚠️ Hard/expensive (lidar, depth) | `height_scan` |
| **Task input** | The *goal* (not a sensor) | n/a — given by the task | `velocity_commands` |

Why this split matters: **whatever the actor observes, the real robot must be able to measure.** Proprioception transfers for free; exteroception is costly and noisy on hardware. That tension is the seed of the advanced design in §4.

---

## 2. The structure: groups

```python
class ObservationsCfg:
    class PolicyCfg(ObsGroup):
        ...
    policy: PolicyCfg = PolicyCfg()
```

Observations are organized into **groups**. Right now there's one group, `policy` — the observations the *actor* network receives. The nested-class pattern (`PolicyCfg` inside `ObservationsCfg`) is how IsaacLab namespaces a group. **The big upgrade in §4 is adding a *second* group (`critic`)** — so hold that structure in mind.

---

## 3. The `policy` group, term by term

Order matters — terms are concatenated in the order written, and that order must match at deployment.

```python
base_lin_vel = ObsTerm(func=env_mdp.base_lin_vel, noise=Unoise(-0.1, 0.1))
base_ang_vel = ObsTerm(func=env_mdp.base_ang_vel, noise=Unoise(-0.2, 0.2))
projected_gravity = ObsTerm(func=env_mdp.projected_gravity, noise=Unoise(-0.05, 0.05))
velocity_commands = ObsTerm(func=env_mdp.generated_commands, params={"command_name": "base_velocity"})
joint_pos = ObsTerm(func=env_mdp.joint_pos_rel, noise=Unoise(-0.01, 0.01))
joint_vel = ObsTerm(func=env_mdp.joint_vel_rel, noise=Unoise(-1.5, 1.5))
actions = ObsTerm(func=env_mdp.last_action)
height_scan = ObsTerm(func=env_mdp.height_scan, params={"sensor_cfg": SceneEntityCfg("height_scanner")},
                      noise=Unoise(-0.1, 0.1), clip=(-1.0, 1.0))
```

- **`base_lin_vel` / `base_ang_vel`** — how fast the body is translating and rotating. The core feedback for *tracking* the velocity command. (On hardware these come from the IMU + state estimator.)
- **`projected_gravity`** — a clever one. Instead of feeding a raw orientation quaternion, it gives the **gravity vector expressed in the robot's body frame** — i.e. "which way is down, relative to me." That directly encodes roll and pitch (the tilt that matters for balance). Yaw isn't observable from gravity (it's symmetric about vertical), which is fine because heading is handled by the command. Compact, intuitive, and robust — the standard way to give a legged policy its orientation.
- **`velocity_commands`** — the *goal*: the commanded velocity from `commands.py`. **No noise** — it's a task input, not a sensor reading. This is what makes the policy goal-conditioned; it *sees* what it's supposed to do.
- **`joint_pos` / `joint_vel`** (relative to default) — the 12 joint angles and speeds, the robot's body configuration. `_rel` means measured as offsets from the nominal pose, consistent with the default-offset actions you saw in `actions.py`.
- **`last_action`** — the previous action the policy output. Giving the policy its own last command provides a sense of *continuity/memory*, helping it produce smooth sequential motion (and it pairs with the `action_rate` reward).
- **`height_scan`** — the only **exteroceptive** term: the terrain height grid from the `height_scanner` sensor in `scenes.py`. `clip=(-1.0, 1.0)` bounds the values so a cliff or spike doesn't feed a huge outlier into the network. This is what lets the policy *anticipate* terrain — but it's also the term that's hardest to get on the real Go2 (see §4).

### The `__post_init__` — two important switches

```python
def __post_init__(self):
    self.enable_corruption = True
    self.concatenate_terms = True
```

- **`enable_corruption = True`** turns ON the per-term `noise`. This is **observation noise injection** — a sim-to-real technique. Real sensors are noisy; training with noise stops the policy relying on impossibly clean readings. (Notice `joint_vel` noise is large, ±1.5 — real joint-velocity estimates *are* very noisy, so the sim mirrors that.) **This is the part of domain randomization that lives in the observation file, not `events.py`** — exactly as flagged in the events doc. You typically *disable* corruption at eval/deployment (`play.py` does this) so you measure the policy on clean inputs.
- **`concatenate_terms = True`** flattens all terms into a single vector. With it off, you'd get a dictionary of named tensors (useful when different terms feed different network branches, e.g. a CNN for images + MLP for proprioception).

---

## 4. Making it advanced — multiple groups (this is your novelty)

You asked how to extend this with an action group, critic group, etc. Here's the high-value version, and it ties straight into NeuroGait's research contribution.

### 4a. A **critic group** → asymmetric actor-critic + privileged observations

In actor-critic RL (PPO/SAC/TD3 all are), there are two networks: the **actor** (the policy you deploy) and the **critic** (the value function, used *only during training* to compute advantages). Here's the insight:

> **The critic is never deployed, so it can see things the real robot can't.**

This is **asymmetric actor-critic**: give the *critic* extra "privileged" information — true friction, true base mass, the *clean, full* terrain height map, contact forces, the magnitude of the random pushes from `events.py` — while the *actor* sees only what the real robot can measure (noisy proprioception). The critic's better value estimates stabilize and speed up training; the actor stays deployable.

You'd add a second group:

```python
class CriticCfg(ObsGroup):
    # everything the policy sees, but CLEAN (no corruption) ...
    base_lin_vel = ObsTerm(func=env_mdp.base_lin_vel)
    # ... plus PRIVILEGED terms the real robot can't measure:
    base_mass        = ObsTerm(func=...)   # the randomized mass from events.py
    friction         = ObsTerm(func=...)   # the randomized friction
    full_height_scan = ObsTerm(func=env_mdp.height_scan, params={...})  # clean, wide
    applied_push     = ObsTerm(func=...)   # the interval push velocity
    def __post_init__(self):
        self.enable_corruption = False   # critic gets clean truth
        self.concatenate_terms = True

policy: PolicyCfg = PolicyCfg()
critic: CriticCfg = CriticCfg()   # RL library routes this to the value network
```

**This is the heart of your contribution.** Remember the chain from the events doc: `events.py` randomizes the extrinsics (mass, friction, CoM, pushes) → the **teacher/critic sees them directly (privileged)** → the **student/actor learns to infer them from proprioception** → that inference *is* implicit terrain/dynamics adaptation (RMA, DreamWaQ). The critic group is the concrete mechanism. And note the deployment move: **move `height_scan` out of the actor's `policy` group and into the critic-only group**, so the deployed actor relies on proprioception alone and *infers* terrain — exactly the robust, hardware-friendly design we discussed.

*(Library note: rsl_rl and skrl both support a privileged/critic observation group for asymmetric training — confirm the exact key name your wrapper expects, usually `critic` or `privileged`.)*

### 4b. **Observation history** → the input that makes adaptation possible

A single frame can't reveal dynamics like friction — you have to watch *how the robot responds over time*. So RMA-style adaptation feeds the policy a **history** of the last N observations:

```python
class PolicyCfg(ObsGroup):
    ...
    def __post_init__(self):
        self.history_length = 5          # stack the last 5 frames
        self.flatten_history_dim = True
```

The adaptation module then infers the extrinsics from this short history of states and actions — literally how RMA estimates friction/payload. **Observation history is the input substrate of implicit adaptation; the critic group is its training signal.** Together they're your method.

### 4c. Other group types you may meet

- **AMP group** — for Adversarial Motion Priors (imitating reference motions for natural gaits). Adds a group of features a discriminator compares to motion-capture data.
- **Teacher/student groups** — explicit two-policy distillation: a `teacher` group (privileged) and a `student` group (proprioceptive). The actor-critic split in 4a is the lighter-weight cousin.

---

## 5. What to add for *other* projects (transferable)

The same group/term machinery generalizes. What changes is *what the agent needs to perceive*:

- **Your own navigation phase (next):** the high-level policy needs **exteroception of obstacles** — a depth/RGB **camera** (with `concatenate_terms=False` so images route to a CNN branch), plus **goal direction/distance** as task input, and possibly an **occupancy/elevation map**. Dynamic obstacles add their relative positions/velocities.
- **Manipulation:** object pose, end-effector pose, target pose, gripper state, force-torque readings.
- **Vision-based locomotion:** raw camera tensors instead of (or alongside) the height-scan, encoded by a CNN — the modern alternative to hand-built elevation maps.
- **Memory-heavy tasks:** longer observation history or a recurrent (LSTM/GRU) policy, fed by a non-concatenated group.

General rule: **add to the observation only what the agent genuinely needs and can (eventually) measure** — every extra dimension is something the network must learn to use and the real robot must supply.

---

## 6. Conventions you just learned

| Concept | What it means | Why it matters |
|---|---|---|
| Observation = NN input | What the policy perceives | Not observed = invisible to the policy |
| Proprioception vs exteroception | Own body vs external world | Determines what transfers to hardware |
| `projected_gravity` | Gravity in body frame = tilt | Compact, robust orientation signal |
| Command as observation | Goal is an input | Makes the policy goal-conditioned |
| `enable_corruption` | Turns on observation noise | Sim-to-real; the DR that lives *here*, not in events |
| `concatenate_terms` | Flat vector vs dict | Dict needed for multi-branch nets (e.g. images) |
| Term order | Fixed at train & deploy | Mismatched order = garbage at deployment |
| **Critic/privileged group** | Extra info for the value net only | **Asymmetric actor-critic — the core of your adaptation method** |
| **Observation history** | Stack of recent frames | Lets the policy infer dynamics (RMA) |

---

## 7. The NeuroGait insight, summarized

This file is where your project goes from "reproduction" to "contribution":
- **Now (benchmark):** keep the single `policy` group as-is so PPO/SAC/TD3 compare fairly on identical observations (and keep the noise seed fixed).
- **Terrain phase:** add a **critic group** with privileged extrinsics + **observation history** on the actor → implicit terrain adaptation. Move the height-scan to critic-only for a proprioceptive deployable actor.
- **Navigation phase:** add exteroceptive obstacle perception (camera) and goal observations to the *high-level* policy's own observation group.

The actor's observation list is also your **sim-to-real contract**: every term in the deployed `policy` group must have a real Go2 counterpart, in the same order, with realistic noise. Get that contract right and the policy transfers; get it wrong and it fails on hardware no matter how good training looked.

---

## 8. Industrial & research conventions to carry forward

- **Asymmetric actor-critic is standard for sim-to-real legged RL.** Privileged critic + proprioceptive actor is how ETH/Unitree-style policies stay both trainable and deployable. It's likely *the* most important advanced technique in this file.
- **Observation noise is mandatory, not optional.** Real sensors are noisy; train with corruption on, evaluate with it off. Report your noise model.
- **Mind the observation order and the deployment contract.** The deployed controller must build the *exact same* observation vector in the *exact same order* — a classic, painful sim-to-real bug.
- **`projected_gravity` over raw quaternions.** The field convention for orientation input; more robust and lower-dimensional.
- **Only observe what's measurable.** Privileged info goes to the critic, never the deployed actor. Confusing the two is how a policy "works in sim" and dies on the robot.

---

## 9. One-paragraph summary

`observations.py` configures the **ObservationManager** — the policy's senses — assembling a single `policy` group of **proprioception** (base linear/angular velocity, `projected_gravity` for tilt, relative joint positions/velocities, last action), one **exteroceptive** term (`height_scan` of the terrain), and the **velocity command** as task input, concatenated in a fixed order with per-term **noise** enabled (`enable_corruption=True`) as a sim-to-real measure. The advanced extension you asked about is **multiple groups**: adding a **critic group** with *privileged* observations (true mass, friction, clean terrain, applied pushes) that only the value network sees — asymmetric actor-critic — combined with **observation history** so the actor can *infer* those hidden extrinsics from proprioception. That actor/critic split plus history *is* the implicit terrain-adaptation method (RMA/DreamWaQ) at the center of NeuroGait's novelty; the deployable actor's observation list doubles as your sim-to-real contract, where every term must have a real Go2 counterpart in the same order.