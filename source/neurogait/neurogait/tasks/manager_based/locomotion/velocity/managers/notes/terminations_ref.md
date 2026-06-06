# Understanding `terminations.py` — When an Episode Ends (and Why It Matters More Than You Think)

*A learn-as-you-clone walkthrough of the termination file in the NeuroGait velocity task.*
*Short file, deep concept: the difference between **failing** and **timing out** quietly shapes everything your agent learns.*

---

## 1. The big idea: terminations decide when to stop and reset

The **TerminationManager** answers one question every step: *"Should this environment's episode end right now?"* When it says yes, that environment resets (via the reset events you saw in `events.py`) and starts a fresh episode.

Episodes end for two fundamentally different reasons, and **the distinction is the whole point of this file**:

| Kind | Meaning | Example here | RL treatment |
|---|---|---|---|
| **Termination** (failure/success) | The task genuinely ended | `base_contact` — the robot fell | Terminal state, do **not** bootstrap |
| **Time-out** (truncation) | We artificially cut it short at a time limit | `time_out` — episode hit max length | Not really terminal, **do** bootstrap |

If you remember one thing from this file: **a fall is a real ending; a time-out is not.** Treating them the same is one of the most common silent bugs in RL.

---

## 2. The file, line by line

```python
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=env_mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=env_mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
```

### `time_out` — the truncation

```python
time_out = DoneTerm(func=env_mdp.time_out, time_out=True)
```

- `env_mdp.time_out` returns `True` once the episode reaches its maximum length (set by `episode_length_s` in the env config). It's a clock, nothing more.
- **The critical part is `time_out=True`.** This flag tells IsaacLab "this ending is a *truncation*, not a failure." The environment then reports it in a separate `truncated` channel rather than `terminated`.
- **Why it matters:** when an episode is *truncated* (cut off by the clock), the robot was doing fine — it just ran out of time. The RL algorithm should still estimate the value of the state it was in and **bootstrap** from it (`V(s) ← r + γV(s')`). If you instead told the algorithm "this was terminal," it would assume all future reward is zero — effectively teaching the robot that *surviving to the time limit is bad*. That produces bizarre behavior near the horizon (e.g. the robot "giving up" as the clock runs down). The `time_out=True` flag is what prevents this. **It is not optional bookkeeping — it's correctness.**

### `base_contact` — the fall detector (true termination)

```python
base_contact = DoneTerm(
    func=env_mdp.illegal_contact,
    params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
)
```

- `env_mdp.illegal_contact` returns `True` when the contact force on the named bodies exceeds a threshold. Here: if the **`base`** (the trunk) is hit with more than **1.0 N**, the episode ends.
- **Translation: the robot fell.** A quadruped's trunk should never touch the ground during normal locomotion — if it does, it has tipped over or collapsed. So trunk contact is the canonical "fall" signal.
- **Note there is NO `time_out=True` here** → this is a *real* termination. The state is genuinely terminal (the robot is on its face), so the algorithm correctly assigns it no future value.
- It reads the **`contact_forces`** sensor from `scenes.py` — the same scene→manager chain you saw feeding the `feet_air_time` reward. One sensor, many consumers.

---

## 3. How terminations interact with rewards (the implicit penalty)

A fall ends the episode, which means the robot **forfeits all the future reward** it would have earned by continuing to walk. That lost future reward is, in effect, a large *implicit* penalty for falling — often more powerful than any explicit penalty term. This is why you don't strictly need a giant "don't fall" reward: terminating the episode does most of the work.

That said, many configs *also* add a small explicit **termination penalty** reward term (a negative reward on the step the episode fails) to sharpen the signal. You don't have one yet — it's a reasonable thing to add if the robot learns to "fall early" to escape accumulating penalties (a known failure mode when penalties outweigh the task reward — tying back to the contribution-balance trap from the rewards doc).

---

## 4. The NeuroGait insight: terminations define success *and* failure — and you'll add success later

Right now both terminations are about **failure** (`base_contact`) or **time** (`time_out`). There is **no success termination**, and that's correct for a *velocity-tracking* task: there's no finish line — the robot just tracks commands forever until it falls or times out. Success is continuous (tracking well), not a discrete event.

But your **navigation phase changes this.** Once you add goals and obstacles, you'll introduce:
- a **success termination** — "reached the goal" (episode ends *well*), and
- a **collision termination** — "hit an obstacle" (episode ends *badly*).

So this file will grow meaningfully when you move from locomotion to navigation. The pattern stays identical (a `DoneTerm` pointing at a function + threshold); only the *conditions* change. Recognizing that terminations are how you encode "what counts as done, well or badly" is the transferable lesson.

One benchmark note: keep terminations **identical across PPO/SAC/TD3**. Episode-ending conditions change the effective reward horizon, so different terminations would make your algorithm comparison unfair.

---

## 5. Conventions you just learned

| Concept | What it means | Why it matters |
|---|---|---|
| Termination vs time-out | Real ending vs clock cutoff | Different value-bootstrapping → correctness, not bookkeeping |
| `time_out=True` | Marks a term as truncation | Prevents the agent learning "surviving is bad" |
| `illegal_contact` | Contact force > threshold ends episode | The standard fall detector |
| Implicit fall penalty | Ending forfeits future reward | Often stronger than an explicit penalty |
| Sensor reuse | Contact sensor feeds rewards *and* terminations | One scene definition, many consumers |
| No success term (yet) | Velocity tracking has no finish line | Success terms appear with navigation goals |

---

## 6. Further terminations to add — specifically for a quadruped

These catch failures earlier or more precisely than waiting for the trunk to hit the floor. Roughly in order of usefulness:

- **Bad orientation / excessive tilt** (`bad_orientation`): terminate if roll or pitch exceeds, say, ~0.7–1.0 rad. Catches a tip-over *before* the base actually hits the ground → cleaner, earlier failure signal and less time wasted simulating a doomed fall. This is the single most common addition.
- **Base height too low:** terminate if the trunk drops below a height threshold (e.g. the robot has buckled/collapsed but no single body triggered an illegal contact). Complements `base_contact`.
- **Illegal contact on more bodies:** extend the contact check to `THIGH`/`calf`/`hip` (you already *penalize* thigh contact in rewards; you might also *terminate* on it if it reliably means a fall). Decide per body whether contact = "bad gait" (penalize) or "fallen" (terminate).
- **Out-of-bounds:** terminate if the robot leaves the terrain patch (its x/y exceeds the environment bounds). Useful on bounded terrains and essential once you add a navigation arena.
- **Numerical-safety guard:** terminate on absurd base velocity/acceleration (a sign the sim has gone unstable for that env) so one blown-up environment doesn't poison the batch.

A design tip: prefer **earlier, orientation-based** termination during initial training (fast failure = faster learning), and be cautious that *too-aggressive* termination can also hurt — if you kill episodes for minor tilts, the robot never learns to *recover* from a stumble. Tune the thresholds; they're a real knob.

---

## 7. Further terminations for *other* projects (transferable patterns)

The same `DoneTerm` machinery encodes "done" in any task. What changes is the condition:

- **Your navigation phase (next for NeuroGait):**
  - *Success:* `goal_reached` — robot within a radius of the target. Ends the episode **positively**.
  - *Failure:* `obstacle_collision` — illegal contact with an obstacle body. Ends it **negatively**.
  - *(These are exactly the two you'll add when the high-level policy comes online.)*
- **Manipulation (arm/hand):**
  - *Success:* object reached target pose.
  - *Failure:* object dropped / left the workspace, or arm self-collision.
- **Bipeds / humanoids:** same orientation + base-height + fall-contact pattern as quadrupeds, often stricter (bipeds fall more easily).
- **Wheeled / drone navigation:** out-of-bounds, collision, goal-reached, battery/time budget exhausted.
- **General principle:** every task has (optionally) a **success** condition, one or more **failure** conditions, and a **time-out**. Name them explicitly; don't rely on the agent inferring "done" from rewards alone.

---

## 8. Industrial & research conventions to carry forward

- **Always flag time-outs correctly.** `time_out=True` for clock-based endings is standard and non-negotiable for correct value estimation. Reviewers and reproducers expect it.
- **Terminate early on clear failure.** Don't waste compute simulating a robot that's already lost; orientation/height checks end doomed episodes fast and speed up training.
- **Separate "bad gait" (penalize) from "fallen" (terminate).** A thigh brushing the ground might be a style issue (reward penalty) or a fall (termination) — decide deliberately per body.
- **Encode success explicitly when it exists.** Goal-conditioned/navigation tasks should have a success termination, not just a reward bump — it cleanly bounds episodes and gives unambiguous success metrics for your results tables.
- **Keep terminations fixed across a benchmark.** They alter the effective horizon, so varying them between algorithms invalidates comparisons.

---

## 9. One-paragraph summary

`terminations.py` configures the **TerminationManager**, which ends and resets an episode. It has two terms: `time_out` (the episode hit its max length) flagged with `time_out=True` so it's treated as a **truncation** — the agent still bootstraps future value, preventing it from learning that "surviving to the limit is bad" — and `base_contact`, which ends the episode as a **true failure** when the trunk is hit (>1 N), i.e. the robot fell, reading the same contact sensor that feeds the rewards. The failure→reset mechanism forfeits future reward, acting as a strong *implicit* fall penalty. There's deliberately no success termination because velocity tracking has no finish line — but your navigation phase will add `goal_reached` (success) and `obstacle_collision` (failure). For a quadruped specifically, the highest-value additions are an **orientation/tilt** and **base-height** termination to catch falls earlier; the same `DoneTerm` pattern generalizes to success/failure conditions in any task.