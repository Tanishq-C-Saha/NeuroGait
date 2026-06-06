# Understanding `scenes.py` — The Locomotion Scene Configuration

*A learn-as-you-clone walkthrough of the scene file in the NeuroGait velocity task.*
*Goal: by the end you should understand not just **what** this file does, but **why** every line is there and **where** it fits in the IsaacLab architecture.*

---

## 1. Where this file sits in the big picture

IsaacLab builds a task in layers. From the ground up:

```
Simulation (Isaac Sim / PhysX)
        │
   ┌────┴─────┐
   │  SCENE   │  ← THIS FILE. The physical world: terrain, robot, sensors, lights.
   └────┬─────┘
   ┌────┴──────────┐
   │  ENVIRONMENT  │  Adds the MDP: observations, actions, rewards, terminations, events.
   └────┬──────────┘
   ┌────┴────┐
   │  TASK   │  A registered, trainable gym env (e.g. Isaac-Velocity-Rough-Go2-v0).
   └─────────┘
```

**The scene answers one question: "What physically exists in the world, and what can sense it?"** It does *not* decide rewards or what counts as success — that's the environment layer (`velocity_env_cfg.py`). Keeping these separate is a deliberate design choice (more on that in §6).

This file defines `MySceneCfg`, which subclasses `InteractiveSceneCfg`. "Interactive" + "Cfg" tells you two important things:
- **Interactive** = it supports *massive parallelism* — IsaacLab clones this scene hundreds or thousands of times on the GPU and trains in all of them at once.
- **Cfg** = it's a *configuration object*, not runtime code. It describes what to build; the framework does the building.

---

## 2. The `@configclass` pattern (read this first)

```python
@configclass
class MySceneCfg(InteractiveSceneCfg):
```

`@configclass` is IsaacLab's decorator built on Python dataclasses. Every IsaacLab config you'll touch uses it. Two rules to internalize:

1. **You declare *attributes*, not logic.** Each line below is a field describing a thing to spawn. There's no `__init__` writing imperative setup code. This is "configuration as data."
2. **`MISSING` is a deliberate hole.** `from dataclasses import MISSING` gives a sentinel meaning *"a value is required here, but I'm not providing it at this level — a more specific config must fill it in."* That's the key to how this scene is shared across robots (§4).

This declarative style is why you can swap the Go2 for an ANYmal by changing one field, without touching scene logic — composition over hard-coding.

---

## 3. The terrain

```python
terrain = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=ROUGH_TERRAINS_CFG,
    max_init_terrain_level=5,
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(...),
    visual_material=sim_utils.MdlFileCfg(...),
    debug_vis=False,
)
```

This is the most important block for a *locomotion* project. Field by field:

- **`prim_path="/World/ground"`** — the address of this object in the USD scene graph. USD (OpenUSD) is the scene-description format Isaac uses; everything lives at a path like a filesystem. `/World/ground` is where the terrain mesh gets placed.
- **`terrain_type="generator"`** — the terrain is *procedurally generated*, not a fixed mesh. The alternatives are `"plane"` (flat infinite ground) and `"usd"` (load a fixed mesh file).
- **`terrain_generator=ROUGH_TERRAINS_CFG`** — the recipe for that procedural terrain. This preset produces a grid of sub-terrains: slopes, stairs (up and down), rough/random noise, discrete obstacles. **This is what makes your locomotion policy terrain-robust** — it trains across many surface types at once.
- **`max_init_terrain_level=5`** — the terrain is a *curriculum*: difficulty rises in levels (flat-ish → very rough). This caps the *initial* difficulty so the robot doesn't start on impossible terrain and never learn. The `CurriculumManager` promotes robots to harder levels as they succeed. *(This is the hook your Week-3 terrain work plugs into.)*
- **`collision_group=-1`** — puts the terrain in its own collision group so collision filtering is clean across the thousands of cloned environments.
- **`physics_material=RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, ...)`** — the *physical* surface properties. **Friction is the single most sim-to-real-critical parameter for a walking robot.** These fixed 1.0 values are your default; in the domain-randomization phase you'll randomize friction so the policy doesn't overfit to one surface (see §5).
- **`visual_material=MdlFileCfg(...)`** — purely cosmetic (a marble texture from NVIDIA's asset server). It affects rendering, *not* physics. Safe to ignore for training; matters only for nice videos.
- **`debug_vis=False`** — toggle to `True` to visualize terrain origins/levels when debugging the curriculum.

---

## 4. The robot — a deliberate blank

```python
robot: ArticulationCfg = MISSING
```

This is the cleverest line in the file, and the easiest to misread as a bug. It is **intentional**.

An *articulation* is a multi-jointed rigid-body system — i.e. your quadruped (12 actuated joints + a floating base). But this scene is **robot-agnostic**: the same terrain, sensors, and lights work for a Go2, an ANYmal, a Spot. So the scene declares *"a robot goes here"* via `MISSING`, and the **robot-specific config** (`config/go2/...`) fills it in:

```python
# elsewhere, in the Go2 config:
self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
```

This is why your repo has one shared `velocity_env_cfg.py` plus thin per-robot configs. **Learn this pattern — it's how all of IsaacLab achieves reuse.**

---

## 5. The sensors — these feed your RL directly

The two sensors are not optional extras. Each one supplies data that your observations and rewards consume. If you don't understand these, the reward function won't make sense.

### Height scanner (exteroception)

```python
height_scanner = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base",
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    ray_alignment="yaw",
    pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
    debug_vis=False,
    mesh_prim_paths=["/World/ground"],
)
```

- **What it is:** a grid of downward rays — "scandots" — that measure the terrain *height around the robot*. This is **exteroceptive** sensing: the robot perceives the world beyond its own body.
- **`prim_path=".../Robot/base"`** — attached to the robot's base, so it moves with the robot.
- **`offset pos=(0,0,20.0)`** — the ray origins sit 20 m *above* the base, then cast straight down. Casting from high up guarantees the rays clear the robot's own body and hit the ground cleanly.
- **`ray_alignment="yaw"`** — the grid rotates with the robot's heading (yaw) but ignores roll/pitch, so the height map is always "forward-relative" and stable when the body tilts.
- **`GridPatternCfg(resolution=0.1, size=[1.6, 1.0])`** — a 1.6 m × 1.0 m grid at 10 cm spacing → roughly a 17×11 grid of height samples. That's the robot's "view" of the terrain shape ahead.
- **`mesh_prim_paths=["/World/ground"]`** — it only ray-casts against the terrain.
- **`{ENV_REGEX_NS}`** — the magic token that resolves per-clone (e.g. `/World/envs/env_0/Robot`, `env_1`, …). This is how one config spawns thousands of independent robots. You'll see it everywhere.

**Why it matters for NeuroGait:** the height-scan is the classic input that lets a policy *anticipate* terrain (step up before hitting a stair). It's also exactly the **privileged information** you'll lean on for the teacher/asymmetric-critic setup in your implicit-terrain-adaptation work: the critic can see the clean height-scan while the student actor learns to infer terrain from proprioception alone.

### Contact sensor (proprioception-adjacent)

```python
contact_forces = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/.*",
    history_length=3,
    track_air_time=True,
)
```

- **`prim_path=".../Robot/.*"`** — the `.*` regex means *every body* on the robot, so contacts are tracked on all feet (and the trunk, calves, etc.).
- **`history_length=3`** — keeps the last 3 timesteps of contact data, which smooths noisy single-frame contact spikes.
- **`track_air_time=True`** — measures how long each foot has been *off* the ground. **This directly powers the `feet_air_time` reward** (the term that encourages proper stepping instead of foot-dragging). It's also how you detect a fall: sustained trunk contact → terminate the episode.

**Takeaway:** rewards aren't computed from thin air. `feet_air_time` exists *because* this sensor tracks air time; base-contact termination exists *because* this sensor reports trunk collisions. Sensor → observation/reward is the chain to keep in your head.

---

## 6. The lighting

```python
sky_light = AssetBaseCfg(
    prim_path="/World/skyLight",
    spawn=sim_utils.DomeLightCfg(intensity=750.0, texture_file="...kloofendal...hdr"),
)
```

A dome light with an HDRI sky texture — it lights the scene for *rendering* (cameras, videos, the GUI). For headless training it's essentially irrelevant to learning. Don't spend time here unless you add cameras for the navigation phase.

---

## 7. The conventions you just learned (worth naming explicitly)

These show up in every IsaacLab file — recognizing them is half the battle:

| Convention | What it means | Why it exists |
|---|---|---|
| `@configclass` | Declarative config, not imperative code | Separates *what to build* from *how it runs*; enables reuse and overrides |
| `MISSING` | "Required, filled in by a more specific config" | Lets one scene serve many robots (composition) |
| `prim_path` | USD scene-graph address | OpenUSD is the standard scene description; everything is path-addressed |
| `{ENV_REGEX_NS}` | Per-clone namespace token | The mechanism behind GPU-parallel training across thousands of envs |
| `*_NUCLEUS_DIR` | Path to NVIDIA's asset server | Assets (textures, robots) stream from a shared store, not your disk |
| `physics_material` vs `visual_material` | Physics reality vs appearance | Never confuse them — only the former affects learning |
| Curriculum (`max_init_terrain_level`) | Difficulty ramps over training | Robots learn on easy terrain first, then hard |

---

## 8. What you could add next (your project journey)

Roughly in the order they fit your NeuroGait roadmap:

**Soon — locomotion robustness (Weeks 2–3)**
- **Domain randomization of friction/mass.** Note: you randomize these via the **`EventManager`** in the *environment* config, *not* by editing this scene. The scene sets the *defaults* (the `1.0` frictions above); events perturb them at reset. Keep that separation clear.
- **An IMU sensor** (`ImuCfg`) on the base if you want explicit linear-acceleration/angular-velocity observations matching what the real Go2's IMU provides — useful for sim-to-real fidelity.
- **Tuning the height-scan grid** — finer resolution = richer terrain perception but heavier compute. A knob to experiment with.

**Mid — the navigation layer (Week 4)**
- **Obstacle assets.** Add static obstacles as `RigidObjectCfg` / `AssetBaseCfg` prims (boxes, cylinders) for your static-obstacle phase, then scripted movers for dynamic obstacles. This is where the scene grows beyond pure terrain.
- **A camera** (`CameraCfg` / `TiledCameraCfg`, depth or RGB) on the base — the high-level navigation policy needs perception of obstacles, and tiled cameras are the GPU-efficient way to do it at scale.

**Later — sim-to-real & research polish**
- **A custom terrain generator** that emphasizes the *specific* surfaces your demo needs (a dedicated stairs/ramp sub-terrain mix) rather than the generic `ROUGH_TERRAINS_CFG`.
- **Privileged-vs-observed split** made explicit: route the height-scanner to the critic only (asymmetric actor-critic) so the deployable actor relies on proprioception — the backbone of your implicit-terrain-adaptation contribution.

---

## 9. Industrial & research conventions to carry forward

Things real labs (ETH RSL, NVIDIA, Unitree) and production teams do that this file quietly models:

- **Config/logic separation.** Scene, environment, agent are separate, overridable configs. This is *the* maintainability lesson — it's why you can update IsaacLab without rewriting your task, and why a reviewer can read your reward config without reading engine code.
- **Procedural + curriculum terrain over hand-built maps.** Generating terrain with difficulty levels is the standard for generalizable locomotion. Memorizing one map is the anti-pattern.
- **Massive parallelism by design.** The `{ENV_REGEX_NS}` cloning model is why Isaac trains in minutes, not days. Write everything to be clone-safe.
- **Sensors mirror the real robot.** Choose sim sensors (IMU, contact, height-scan-as-privileged) that map to what the *physical* Go2 actually has, or sim-to-real will bite you. Privileged sensors (height-scan) belong on the *critic*, not the deployed actor.
- **Asset provenance & reproducibility.** Assets stream from versioned Nucleus paths; terrain generators take seeds. For a *benchmark* project specifically: pin the terrain seed and asset versions so PPO/SAC/TD3 all train on the identical world. Comparability is the whole point.
- **Read the upstream source.** This file is adapted from IsaacLab's own velocity task. The professional habit is to read the framework's reference implementation, understand it, then specialize — exactly what you did.

---

## 10. One-paragraph summary (for your README/notes)

`scenes.py` defines the *physical world* of the velocity locomotion task: a procedurally-generated rough terrain with curriculum difficulty, a robot slot left intentionally blank (`MISSING`) for a per-robot config to fill, a downward-ray **height scanner** for terrain perception, a **contact sensor** that powers the air-time reward and fall detection, and a sky light for rendering. It deliberately contains *no* reward or task logic — that lives in the environment layer — and it's written to be cloned thousands of times on the GPU via the `{ENV_REGEX_NS}` namespace. Understanding this file means understanding how IsaacLab separates *world* from *task*, how it reuses one scene across many robots, and how sensors feed the RL signals downstream.