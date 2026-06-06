# NeuroGait — Project Concept & Study Guide

*A reference for the team: what we are building, the concepts behind it, and exactly what to study and where.*

---

## 0. Where we are going (the one-sentence goal)

> **Build NeuroGait: a benchmarked, terrain-adaptive hierarchical RL system for a quadruped (Unitree Go2) in IsaacLab — where a low-level locomotion policy with *implicit* terrain estimation tracks velocity commands across diverse terrains, and a high-level navigation policy steers it through static and then dynamic obstacle fields — with a clear path toward sim-to-real on the Go2 and on-the-fly online adaptation.**

Everything in this document supports that sentence. Read Section 1 first; it is the mental model that makes the rest click.

---

## 1. The core mental model: two layers, not one

The single most important idea. The project spans **two control layers**, and most early confusion ("are we doing locomotion or navigation?") dissolves once you separate them:

| Layer | Question it answers | Concern | When it appears |
|---|---|---|---|
| **Low level — Locomotion** | "Given a velocity command, how do I move my 12 joints to follow it without falling on *this* terrain?" | *How* to walk | From day 1 |
| **High level — Navigation** | "Given a goal and obstacles, what velocity command should I send down?" | *Where* to walk | Only once obstacles enter |

- Terrain (stairs, ramp, friction) is a **locomotion** problem → handled by the low level.
- Obstacle avoidance, especially **moving** obstacles, is a **navigation** problem → handled by the high level.
- The clean architecture is a **hierarchical policy**: high-level outputs velocity commands → low-level executes them.
- **Implicit terrain adaptation lives in the LOW level. Obstacle navigation (static → dynamic) lives in the HIGH level.**

This is why "sir mixing locomotion and navigation" is actually correct: the project genuinely needs both, just at different layers.

---

## 2. Concept map (what depends on what)

```
                    NeuroGait System
                          │
        ┌─────────────────┴──────────────────┐
        │                                     │
   HIGH LEVEL                            LOW LEVEL
   Navigation policy                    Locomotion policy
   (outputs velocity cmd)               (outputs joint targets)
        │                                     │
   ┌────┴─────┐                    ┌──────────┼───────────┐
 static     dynamic           velocity    implicit      domain
 obstacles  obstacles         tracking    terrain       randomization
                                          estimation    + actuator model
                                              │              │
                                          (RMA /         (sim-to-real
                                           DreamWaQ)       robustness)
```

The spine is the low-level locomotion policy. Build it first and well; everything else hangs off it.

---

## 3. Study Track A — Reinforcement Learning foundations

**Goal:** understand the MDP framing, on-policy vs off-policy, and the three algorithms we benchmark.

### What to know
- **MDP / RL loop:** state, action, reward, policy, episode, return, discount factor. The locomotion task is an MDP: observe → act → reward → repeat at ~50 Hz.
- **On-policy vs off-policy** (this distinction drives our whole benchmark):
  - **PPO** (on-policy): stable, robust, the field default for legged robots. Less sample-efficient.
  - **SAC** (off-policy): sample-efficient, max-entropy, but hyperparameter-sensitive on locomotion.
  - **TD3** (off-policy): deterministic, twin-critic; also sensitive.
- **Why PPO usually wins on legged robots:** not because it's "better," but because it has better defaults and tolerates reward-shaping noise. Budget tuning time for SAC/TD3 before declaring a winner, or you're just measuring "PPO has better defaults."

### Where to study
| Resource | Use it for |
|---|---|
| **OpenAI Spinning Up** (`spinningup.openai.com`) | The single best from-scratch RL intro. Read "Intro to RL" + the PPO, SAC, TD3 algorithm pages. |
| **PPO paper** — Schulman et al. 2017 (arXiv `1707.06347`) | The clipped-objective intuition. |
| **SAC paper** — Haarnoja et al. 2018 (arXiv `1801.01290`) | Max-entropy RL, why it's sample-efficient. |
| **TD3 paper** — Fujimoto et al. 2018 (arXiv `1802.09477`) | Twin critics, delayed updates. |
| **Sutton & Barto, *Reinforcement Learning: An Introduction*** (free online) | Reference for any gap in fundamentals. |

---

## 4. Study Track B — IsaacLab & the manager-based workflow

**Goal:** be able to define our task as a manager-based environment and train it.

### What to know
- **Why IsaacLab/Isaac Sim:** GPU-parallel simulation — thousands of robots at once, far faster than PyBullet/RaiSim, with high physics fidelity.
- **Manager-based vs Direct workflow:** we use **manager-based**. The task is split into `ObservationManager`, `RewardManager`, `TerminationManager`, `EventManager` (domain randomization), `CurriculumManager`. Reward/observation *terms* are declared in config — ideal for a benchmarking project where we toggle and reweight terms constantly.
- **Direct multi-agent:** NOT for us (that's for multiple *learning* agents; our dynamic obstacles are scripted, not RL agents).
- **Gym task ID convention:** e.g. `Isaac-Velocity-NeuroGait-Go2-v0` (Pascal/hyphen string), while the Python package is lowercase `neurogait`.

### Where to study
| Resource | Use it for |
|---|---|
| **IsaacLab docs** (`isaac-sim.github.io/IsaacLab`) | Tutorials: "Creating a Manager-Based RL Environment," "Training with an RL Agent." Start here. |
| **IsaacLab `velocity` locomotion task** (in the repo) | The template to copy. Our env is a fork of this. |
| **`legged_gym`** (`github.com/leggedrobotics/legged_gym`) | The ancestor of the IsaacLab velocity task; the original ETH curriculum + reward structure. |

---

## 5. Study Track C — RL libraries (which tool, when)

**Goal:** know why skrl is our benchmark harness and rsl_rl is our deploy path.

### What to know
- **skrl** — *our primary library for the benchmark.* It implements PPO, SAC, TD3 (and more) behind **one unified API**, so swapping algorithms changes only the algorithm — a fair, controlled comparison. Runs on GPU tensors (keeps IsaacLab's speed). Pairs with **Optuna** for hyperparameter optimization.
- **rsl_rl** — *PPO-only, but the deployment standard.* It's what published quadruped work and Unitree's own pipeline use. If PPO wins our benchmark, we port to rsl_rl for sim-to-real.
- **rl_games** — fast, narrower algorithm set, fiddly configs. Not our choice.
- **Stable-Baselines3** — broadest classic algorithms but slow here (CPU numpy interface). Sanity checks only.

### Our strategy
1. Benchmark PPO/SAC/TD3 in **skrl** on the identical task.
2. Optimize the winner in **skrl** (Optuna).
3. If PPO wins → port to **rsl_rl** for Go2 deployment. **A port is a re-validation, not a copy-paste** (different defaults; re-verify and possibly re-tune).

### Where to study
| Resource | Use it for |
|---|---|
| **skrl docs** (`skrl.readthedocs.io`) | API, agent configs, IsaacLab wrapper, Optuna integration. |
| **rsl_rl** (`github.com/leggedrobotics/rsl_rl`) | The PPO implementation used for deployment. |
| **IsaacLab "RL Library Comparison"** docs page | Side-by-side of all four libraries. |

---

## 6. Study Track D — Reward design (the heart of the project)

**Goal:** design a balanced reward, understand there is no universal "magic ratio."

### What to know
- **No universal best ratio.** Weights are robot-, task-, and control-frequency-specific. There is a shared **skeleton** (ETH's `legged_gym`) that everyone forks, then re-tunes.
- **The structure:** *task rewards* (positive, drive behavior) + *regularization penalties* (negative, shape how).
- **Canonical starting weights** (legged_gym velocity task — our seed values):

| Term | Approx. scale | Purpose |
|---|---|---|
| `tracking_lin_vel` | +1.0 | follow commanded x/y velocity |
| `tracking_ang_vel` | +0.5 | follow commanded yaw rate |
| `feet_air_time` | +1.0 (≈0.5 s target) | proper stepping, not shuffling |
| `lin_vel_z` | −2.0 | no vertical bouncing |
| `ang_vel_xy` | −0.05 | no roll/pitch wobble |
| `torques` | −1e-5 | energy/effort |
| `dof_acc` | −2.5e-7 | smoothness |
| `action_rate` | −0.01 | penalize jerk (→ our jerk metric) |
| `collision` | −1.0 | avoid self/world collision |
| `dof_pos_limits` | −10.0 | stay off joint stops |

- Tracking terms use an **exponential kernel** `exp(−error²/σ²)`, `σ ≈ 0.25`. Tuning σ changes how forgiving tracking is.
- **THE KEY TRAP:** raw coefficients are NOT the ratio that matters. Torque is O(10–50), acceleration O(thousands), velocity error O(0.1). What to balance is each term's **contribution to total reward per step**, not its coefficient. **Log per-term contributions during training** and tune so no penalty drowns the task reward. This is exactly what the Optuna sweep should optimize.

### Lab-by-lab (honest version)
- **ETH (RSL / Hutter):** origin of the skeleton. Copy this philosophy — it's open and it's our codebase's ancestor.
- **MIT (Margolis & Agrawal, "Walk These Ways"):** adds gait-conditioning auxiliary rewards (contact schedule, swing height, stance width, frequency). Relevant to terrain adaptation.
- **Stanford / Berkeley:** more method innovation than a single reward template (DayDreamer is Berkeley).
- **Boston Dynamics:** does **not** publish reward functions; Spot was historically MPC, recent RL is proprietary. No ratio to copy — ignore claims otherwise.

### Where to study
| Resource | Use it for |
|---|---|
| `legged_gym` reward config (`legged_robot.py` rewards) | The exact term implementations + default scales. |
| **Unitree `unitree_rl_lab`** Go2 task config | Go2-specific reward weights to diff against ETH defaults. |
| **"Walk These Ways"** — Margolis & Agrawal 2022 (arXiv `2212.03238`) | Gait-auxiliary rewards. |

---

## 7. Study Track E — Terrain adaptation (our novelty, low level)

**Goal:** implement *implicit* terrain estimation, not a brittle terrain classifier.

### What to know
- **The fork:** an explicit terrain *classifier* that hard-switches between separate policies is fragile — it fails at terrain boundaries and at misclassification. The field has moved to **implicit** terrain estimation inside **one** adaptive policy.
- **RMA (Rapid Motor Adaptation):** an adaptation module estimates environment "extrinsics" (friction, payload) from a short history of states/actions. Teacher uses privileged sim info; student infers it from proprioception.
- **DreamWaQ:** implicit terrain "imagination" via a context-aided estimator + **asymmetric actor-critic** (critic sees privileged info, actor doesn't). State of the art for proprioceptive Go2-class robots.
- **Teacher-student / privileged learning:** the general robustness recipe — train a teacher with privileged info, distill into a deployable proprioception-only student.
- **Our implementation:** one locomotion policy + continuous latent terrain/dynamics estimate + asymmetric actor-critic + domain randomization. "Different terrains" becomes a property of ONE policy, not a policy bank.

### Where to study
| Resource | Use it for |
|---|---|
| **RMA** — Kumar et al. 2021 (arXiv `2107.04034`) | The adaptation-module pattern. |
| **DreamWaQ** — Nahrendra et al. 2023 (ICRA; search title) | Implicit terrain estimation + asymmetric actor-critic. |
| **"Learning quadrupedal locomotion over challenging terrain"** — Lee et al., *Science Robotics* 2020 (DOI `10.1126/scirobotics.abc5986`) | The teacher-student privileged-learning blueprint. |
| **"Learning robust perceptive locomotion…"** — Miki et al., *Science Robotics* 2022 | Combining proprioception + exteroception. |

---

## 8. Study Track F — Hierarchical navigation (high level)

**Goal:** add obstacle avoidance on top of a frozen locomotion policy.

### What to know
- Train the high level **only after** the low level is solid. Freeze locomotion, then train navigation to output velocity commands.
- **Static obstacles first** (flat-with-obstacles → stairs/ramp with obstacles), **dynamic obstacles second** (much harder; needs real-time reaction).
- Dynamic obstacles are scripted movers, **not** RL agents — so this stays single-agent RL (no multi-agent workflow needed).

### Where to study
| Resource | Use it for |
|---|---|
| **PRELUDE** — "Learning to Walk by Steering" (2022) | Hierarchical perceptive navigation in dynamic environments — closest match to our high level. |
| Hierarchical DRL sections of **both review papers** (see Section 11) | Latent/velocity command interfaces between layers. |

---

## 9. Study Track G — Sim-to-real on the Unitree Go2

**Goal:** get the trained policy onto the physical Go2 without it face-planting.

### What to know — the lucky break
- **Unitree maintains an official IsaacLab-based RL repo with a full Go2 sim-to-real pipeline.** Train in Isaac Sim with rsl_rl → export ONNX → deploy via C++ controllers + Unitree SDK2. The painful deployment plumbing is already written *by the manufacturer*. **Align NeuroGait's structure to it.**
- **Always do sim-to-sim before sim-to-real:** validate the policy in **MuJoCo** first to catch physics-engine overfitting. IsaacLab also has a sim-to-sim transfer path tested on the Go2 (needs a joint-mapping YAML). This step saves robots.
- **The consensus sim-to-real toolkit:**
  - Domain randomization (friction, mass, motor strength, latency)
  - Actuator model
  - Observation noise + latency injection
  - Action filtering (protects motors from high-frequency commands)
  - Teacher-student / RMA distillation to a proprioception-only student
- **Deploy implication:** Unitree's pipeline is **rsl_rl + ONNX**, so a PPO winner ported to rsl_rl drops straight onto hardware.

### Where to study
| Resource | Use it for |
|---|---|
| **`github.com/unitreerobotics/unitree_rl_lab`** | The official Go2 train→deploy pipeline. Study its reward config + deployment FSM. |
| **`github.com/unitreerobotics/unitree_sdk2`** | Hardware communication layer. |
| **IsaacLab "Sim-to-Real" + "Sim-to-Sim" docs** | The official transfer guidance + Go2-tested workflow. |
| Sim-to-real sections of **both review papers** | Domain randomization + reality-gap methods catalog. |

---

## 10. Study Track H — On-the-fly online learning (future / thesis frontier)

**Goal:** understand the "learn live" idea — and why it's scoped LAST.

### What to know
- This is **DayDreamer**: a world-model approach where the deployed policy keeps improving by training on live sensory experience.
- It's the most exciting *and* most dangerous part: risks hardware damage, hard to stabilize, a research contribution in its own right.
- **Scope it as Phase 3 / future work.** Get the benchmark + adaptive single policy + static navigation working first. Online learning is the cherry, not the cake.

### Where to study
| Resource | Use it for |
|---|---|
| **DayDreamer** — Wu et al., CoRL 2022 (arXiv `2206.14176`) | World-model real-world robot learning. |
| **"A Walk in the Park"** — Smith et al. 2022 | Fast real-world RL on the A1 (20-minute walking). |

---

## 11. The two anchor review papers (read these first)

| Paper | What it gives you |
|---|---|
| **Zhang, He & Wang (2022), "Deep RL for real-world quadrupedal locomotion: a comprehensive review"** (*Intelligence & Robotics*, DOI `10.20517/ir.2022.20`) | The big map: algorithm trends, simulators, hardware, and a huge appendix table of papers by algorithm/state/action/reward. **Use the appendix as your reward + observation menu.** |
| **Gurram, Uttam & Ohol (2025), "RL for Quadrupedal Locomotion: Current Advancements and Future Perspectives"** (ICMERR 2025) | The implementation-focused walkthrough: teacher-student, RMA, domain randomization, DreamWaQ, ZSL-RPPO, DayDreamer. **Closest to our actual pipeline.** |

---

## 12. Study roadmap mapped to the 4-week plan

| Week | Build milestone | Study in parallel |
|---|---|---|
| **0 (now)** | Read both review papers; set up IsaacLab + skrl | Tracks A (RL basics) + B (IsaacLab) |
| **1** | Non-learned baseline + eval harness (time, jerk, energy, stability, tracking error) on flat terrain | Track B (manager-based) + D (reward skeleton) |
| **2** | Benchmark PPO/SAC/TD3 in skrl; pick winner | Track C (libraries) + A (algorithm details) |
| **3** | Optuna optimization of winner + terrain + implicit estimation | Track D (reward tuning) + E (terrain adaptation) |
| **4** | High-level navigation: static obstacles solid, dynamic-obstacle demo | Track F (navigation) |
| **Phase 3** | Robust dynamic nav, Go2 sim-to-real, online adaptation | Tracks G (sim-to-real) + H (online) |

### Suggested team division (adapt to your size)
- **Person 1 — Locomotion & rewards:** Tracks A, D. Owns the low-level policy + reward design.
- **Person 2 — Infra & benchmark:** Tracks B, C. Owns the IsaacLab env, skrl harness, eval metrics, Optuna.
- **Person 3 — Terrain & sim-to-real:** Tracks E, G. Owns implicit estimation + Go2 deployment.
- **(Navigation, Track F, is a shared push in Week 4.)**

---

## 13. What makes this masters / ETH-level (state it in the proposal)

1. **A controlled, multi-metric benchmark** of on- vs off-policy algorithms on *identical* terrain-adaptive locomotion (most papers report one algorithm; rigorous, reproducible comparison is a real contribution).
2. **Implicit terrain adaptation integrated with a hierarchical navigation layer** — not an off-the-shelf combination.
3. **On-the-fly online adaptation** as the clearly-scoped forward-looking thrust.

Honest scoping is a strength: "we rigorously benchmark and build the hierarchical terrain-adaptive system within the month; robust dynamic navigation and online adaptation are the research extensions."

---

## 14. Quick link index

| Topic | Link |
|---|---|
| OpenAI Spinning Up | `spinningup.openai.com` |
| IsaacLab docs | `isaac-sim.github.io/IsaacLab` |
| legged_gym | `github.com/leggedrobotics/legged_gym` |
| rsl_rl | `github.com/leggedrobotics/rsl_rl` |
| skrl | `skrl.readthedocs.io` |
| Unitree RL Lab (Go2 deploy) | `github.com/unitreerobotics/unitree_rl_lab` |
| Unitree SDK2 | `github.com/unitreerobotics/unitree_sdk2` |
| PPO | arXiv 1707.06347 |
| SAC | arXiv 1801.01290 |
| TD3 | arXiv 1802.09477 |
| RMA | arXiv 2107.04034 |
| Walk These Ways | arXiv 2212.03238 |
| Learn to Walk in Minutes (legged_gym paper) | arXiv 2109.11978 |
| DayDreamer | arXiv 2206.14176 |
| Review 1 (Zhang et al.) | DOI 10.20517/ir.2022.20 |

---

*Verify exact arXiv IDs/links when you open them — titles + venues are given so you can find each even if a number drifts. The Unitree repo and IsaacLab docs links are confirmed current.*