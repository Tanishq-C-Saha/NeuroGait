<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>NeuroGait — Demo Checkpoints: What to Show Your Professor, When</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root{
    --bg:#15181b; --panel:#1d2126; --panel-2:#252a31;
    --ink:#ece7dc; --ink-dim:#9aa0a6;
    --line:rgba(236,231,220,0.10); --line-strong:rgba(236,231,220,0.20);
    --data:#5fd0c9; --signal:#ff8c42; --terrain:#b9a87c; --comm:#c792ea; --warn:#e2533a; --go:#7fc97f;
    --font-display:'Space Grotesk',sans-serif;
    --font-body:'Inter',system-ui,sans-serif;
    --font-mono:'JetBrains Mono',monospace;
  }
  *{box-sizing:border-box;}
  html,body{margin:0;}
  body{ background:var(--bg); color:var(--ink); font-family:var(--font-body); line-height:1.65; -webkit-font-smoothing:antialiased; }
  .shell{ display:flex; min-height:100vh; }
  .rail{ flex:0 0 250px; border-right:1px solid var(--line); background:var(--panel); padding:26px 16px; position:sticky; top:0; height:100vh; overflow-y:auto; }
  .rail .brand{ font-family:var(--font-mono); font-size:10.5px; letter-spacing:.14em; color:var(--go); margin-bottom:4px; }
  .rail h1{ font-family:var(--font-display); font-size:16.5px; margin:0 0 20px; font-weight:600; line-height:1.3; }
  .rail nav a{ display:block; font-family:var(--font-mono); font-size:11px; color:var(--ink-dim); text-decoration:none; padding:6px 0; border-left:2px solid transparent; padding-left:10px; margin-left:-10px; transition:color .15s, border-color .15s; }
  .rail nav a:hover, .rail nav a.active{ color:var(--ink); border-color:var(--go); }
  .rail .phasetag{ display:block; font-size:9px; color:var(--ink-dim); opacity:.75; margin:14px 0 3px 0; text-transform:uppercase; letter-spacing:.12em; }
  .content{ flex:1; min-width:0; padding:46px 54px 130px; max-width:920px; }
  @media(max-width:900px){ .rail{ display:none; } .content{ padding:30px 18px 100px; } }
  section{ margin-bottom:58px; scroll-margin-top:24px; }
  .eyebrow{ font-family:var(--font-mono); font-size:11px; letter-spacing:.16em; text-transform:uppercase; color:var(--go); }
  h2{ font-family:var(--font-display); font-size:clamp(21px,3vw,28px); font-weight:600; margin:8px 0 16px; letter-spacing:-0.01em; }
  h3{ font-family:var(--font-display); font-size:16px; font-weight:600; margin:26px 0 10px; }
  h4{ font-family:var(--font-mono); font-size:11px; letter-spacing:.1em; text-transform:uppercase; color:var(--ink-dim); margin:18px 0 8px; }
  p{ font-size:14px; color:var(--ink); margin:0 0 12px; }
  .dim{ color:var(--ink-dim); }
  code, .mono{ font-family:var(--font-mono); font-size:0.92em; }
  code{ background:var(--panel-2); padding:1px 5px; border-radius:4px; }
  .banner{ border:1px solid var(--line-strong); background:var(--panel); border-radius:10px; padding:16px 18px; margin-bottom:10px; }
  .banner p{ margin:8px 0 0; }

  .checkpoint{
    background:var(--panel); border:1px solid var(--line-strong); border-radius:14px;
    padding:20px 22px; margin:18px 0; position:relative;
  }
  .checkpoint::before{ content:""; position:absolute; left:0; top:0; bottom:0; width:4px; border-radius:14px 0 0 14px; background:var(--cpcolor,var(--data)); }
  .cp-head{ display:flex; align-items:baseline; justify-content:space-between; gap:12px; flex-wrap:wrap; margin-bottom:8px; }
  .cp-num{ font-family:var(--font-mono); font-size:11px; color:var(--cpcolor,var(--data)); letter-spacing:.1em; text-transform:uppercase; }
  .cp-when{ font-family:var(--font-mono); font-size:11px; color:var(--ink-dim); }
  .cp-title{ font-family:var(--font-display); font-size:17px; font-weight:600; margin:2px 0 10px; color:var(--ink); }

  .show-box{
    background:var(--panel-2); border-radius:10px; padding:12px 14px; margin:10px 0;
    border-left:3px solid var(--go);
  }
  .show-box .show-label{ font-family:var(--font-mono); font-size:10px; letter-spacing:.1em; text-transform:uppercase; color:var(--go); margin-bottom:5px; }
  .show-box p{ margin:0; font-size:13px; }

  .needbox{
    background:var(--panel-2); border-radius:10px; padding:12px 14px; margin:10px 0;
    border-left:3px solid var(--cpcolor,var(--data));
  }
  .needbox .need-label{ font-family:var(--font-mono); font-size:10px; letter-spacing:.1em; text-transform:uppercase; color:var(--ink-dim); margin-bottom:5px; }
  .needbox ul{ margin:0; padding-left:18px; }
  .needbox li{ font-size:13px; margin-bottom:4px; }

  .fallback-box{
    background:rgba(226,83,58,0.06); border-radius:10px; padding:12px 14px; margin:10px 0;
    border-left:3px solid var(--warn);
  }
  .fallback-box .fb-label{ font-family:var(--font-mono); font-size:10px; letter-spacing:.1em; text-transform:uppercase; color:var(--warn); margin-bottom:5px; }
  .fallback-box p{ margin:0; font-size:13px; color:var(--ink-dim); }

  .progress-strip{ display:flex; gap:6px; margin:18px 0; flex-wrap:wrap; }
  .progress-pip{
    flex:1; min-width:90px; text-align:center; font-family:var(--font-mono); font-size:10px;
    padding:8px 6px; border-radius:8px; background:var(--panel); border:1px solid var(--line);
    color:var(--ink-dim);
  }
  .progress-pip.active{ border-color:var(--cpcolor,var(--data)); color:var(--ink); }

  .legend-row{ display:flex; gap:16px; flex-wrap:wrap; margin:10px 0 18px; font-family:var(--font-mono); font-size:11px; color:var(--ink-dim); }
  .legend-row .swatch{ display:inline-block; width:9px; height:9px; border-radius:2px; margin-right:6px; vertical-align:middle; }

  table{ width:100%; border-collapse:collapse; font-size:12.5px; margin:14px 0; }
  th,td{ text-align:left; padding:8px 10px; border-bottom:1px solid var(--line); vertical-align:top; }
  th{ font-family:var(--font-mono); font-size:10px; letter-spacing:.08em; text-transform:uppercase; color:var(--ink-dim); }
  td.mono{ font-family:var(--font-mono); font-size:11px; color:var(--data); }

  footer{ font-family:var(--font-mono); font-size:10.5px; color:var(--ink-dim); padding-top:18px; border-top:1px solid var(--line); }
  .topgap{ scroll-margin-top:80px; }
</style>
</head>
<body>
<div class="shell">

  <aside class="rail">
    <div class="brand">NEUROGAIT // DEMO CHECKPOINTS</div>
    <h1>What to show your professor, at every stage</h1>
    <nav id="toc">
      <a href="#how-to-use">How to use this</a>
      <a href="#overview">All checkpoints at a glance</a>

      <span class="phasetag" style="color:var(--data)">Foundation</span>
      <a href="#cp1">CP1 — frozen locomotion sanity check</a>
      <a href="#cp2">CP2 — A* planner standalone</a>
      <a href="#cp3">CP3 — perception pipeline visualized</a>

      <span class="phasetag" style="color:var(--signal)">Phase 01 deliverable</span>
      <a href="#cp4">CP4 — rule-based pipeline plumbing</a>
      <a href="#cp5">CP5 — first trained nav policy</a>
      <a href="#cp6">CP6 — phase 01 full demo + metrics</a>

      <span class="phasetag" style="color:var(--terrain)">Phase 02 deliverable</span>
      <a href="#cp7">CP7 — dynamic obstacles moving</a>
      <a href="#cp8">CP8 — frame-stacking validated</a>
      <a href="#cp9">CP9 — risk head + reactive demo</a>

      <span class="phasetag" style="color:var(--comm)">Hybrid upgrade</span>
      <a href="#cp10">CP10 — learned terrain cost</a>
      <a href="#cp11">CP11 — feedback loop correcting cost</a>
      <a href="#cp12">CP12 — dual policy, hard switch</a>
      <a href="#cp13">CP13 — risk-blended dual policy demo</a>

      <span class="phasetag" style="color:var(--warn)">Stretch / later</span>
      <a href="#cp14">CP14 — multi-objective Pareto front</a>
      <a href="#cp15">CP15 — multi-robot baseline</a>

      <span class="phasetag">Reference</span>
      <a href="#recording-tips">How to record each demo</a>
      <a href="#summary-table">Full checkpoint summary table</a>
    </nav>
  </aside>

  <main class="content">

    <section id="how-to-use" class="topgap">
      <span class="eyebrow">Read me first</span>
      <h2>How to use this document</h2>
      <p>
        Every checkpoint below is built around one question: <b>what's the smallest
        thing I can put in front of my professor that proves this piece actually
        works?</b> Not "I wrote the code" — something visible, like a screen
        recording, a printed number, a TensorBoard screenshot, or a before/after
        comparison. Each checkpoint has four parts: what to build to reach it,
        exactly what to show, what counts as passing, and a fallback if you're
        short on time — because a smaller honest demo beats a bigger broken one.
      </p>
      <div class="banner">
        <span class="eyebrow">A note on order</span>
        <p>
          Checkpoints 1-9 are your original single-policy plan (phases 01-02) —
          build and demo these regardless of whether the hybrid architecture
          happens, since they're the working foundation everything else sits on.
          Checkpoints 10-13 are the hybrid upgrade (learned terrain cost, feedback
          loop, dual policy) from the architecture doc. 14-15 are further out and
          explicitly optional depending on remaining time.
        </p>
      </div>
    </section>

    <section id="overview">
      <h3>All checkpoints at a glance</h3>
      <div class="progress-strip">
        <div class="progress-pip" style="--cpcolor:#5fd0c9" class="active">CP1-3<br>Foundation</div>
        <div class="progress-pip" style="--cpcolor:#ff8c42">CP4-6<br>Phase 01</div>
        <div class="progress-pip" style="--cpcolor:#b9a87c">CP7-9<br>Phase 02</div>
        <div class="progress-pip" style="--cpcolor:#c792ea">CP10-13<br>Hybrid</div>
        <div class="progress-pip" style="--cpcolor:#e2533a">CP14-15<br>Stretch</div>
      </div>
      <p class="dim" style="font-size:13px">Fifteen checkpoints total. Each is something you can demo live or via a short recording in a 5-10 minute supervisor meeting — none require your professor to read code.</p>
    </section>

    <section id="cp1" class="topgap" style="--cpcolor:#5fd0c9">
      <span class="eyebrow">Foundation</span>
      <h2>CP1 — frozen locomotion sanity check</h2>
      <div class="checkpoint" style="--cpcolor:#5fd0c9">
        <div class="cp-head"><span class="cp-num">Checkpoint 1</span><span class="cp-when">~half a day</span></div>
        <div class="cp-title">The frozen locomotion checkpoint walks correctly when given a hand-typed velocity command, inside the new navigation env.</div>
        <div class="needbox" style="--cpcolor:#5fd0c9">
          <div class="need-label">Build before this checkpoint</div>
          <ul>
            <li>New navigation env shell that loads the frozen <code>agent_32000.pt</code> (or your latest converged checkpoint)</li>
            <li>A way to feed a constant, hand-typed velocity command (e.g. "walk forward at 0.5 m/s") instead of a learned policy's output</li>
          </ul>
        </div>
        <div class="show-box">
          <div class="show-label">What to show</div>
          <p>A short screen recording of the Isaac Lab GUI: the Go2 walking forward smoothly and staying upright for at least 10-15 seconds with a constant command, inside the navigation env shell (not the original locomotion env).</p>
        </div>
        <div class="needbox" style="--cpcolor:#5fd0c9">
          <div class="need-label">Passing bar</div>
          <ul>
            <li>Robot doesn't tip over, drift sideways unexpectedly, or jitter</li>
            <li>Confirms the checkpoint loading and plumbing into a new env works before anything else is built on top</li>
          </ul>
        </div>
        <div class="fallback-box">
          <div class="fb-label">If short on time</div>
          <p>This one shouldn't be skipped — it's the cheapest possible checkpoint and catches the exact class of bug (shape mismatches, wrong checkpoint path) you've already hit once.</p>
        </div>
      </div>
    </section>

    <section id="cp2">
      <h2>CP2 — A* planner standalone</h2>
      <div class="checkpoint" style="--cpcolor:#5fd0c9">
        <div class="cp-head"><span class="cp-num">Checkpoint 2</span><span class="cp-when">~half a day</span></div>
        <div class="cp-title">A* returns a sensible, obstacle-avoiding path on a known toy grid — entirely outside Isaac Lab.</div>
        <div class="needbox" style="--cpcolor:#5fd0c9">
          <div class="need-label">Build before this checkpoint</div>
          <ul>
            <li><code>planner.py</code> as plain Python, no simulator dependency</li>
            <li>One or two hand-made test grids with known obstacle placements</li>
          </ul>
        </div>
        <div class="show-box">
          <div class="show-label">What to show</div>
          <p>A matplotlib plot: the grid, the obstacles in one color, and the returned path overlaid in another color, clearly routing around the obstacles rather than through them.</p>
        </div>
        <div class="needbox" style="--cpcolor:#5fd0c9">
          <div class="need-label">Passing bar</div>
          <ul>
            <li>Path never crosses a marked obstacle cell</li>
            <li>Path length is reasonably close to optimal on a grid small enough to verify by eye</li>
          </ul>
        </div>
        <div class="fallback-box">
          <div class="fb-label">If short on time</div>
          <p>None needed — this is a couple hours of work and de-risks every later checkpoint that depends on the planner.</p>
        </div>
      </div>
    </section>

    <section id="cp3">
      <h2>CP3 — perception pipeline visualized</h2>
      <div class="checkpoint" style="--cpcolor:#5fd0c9">
        <div class="cp-head"><span class="cp-num">Checkpoint 3</span><span class="cp-when">1-2 days</span></div>
        <div class="cp-title">The depth camera's occupancy grid visibly lines up with where obstacles actually are in the scene.</div>
        <div class="needbox" style="--cpcolor:#5fd0c9">
          <div class="need-label">Build before this checkpoint</div>
          <ul>
            <li>Depth camera sensor added to the scene config</li>
            <li>Depth-image → occupancy-grid projection function</li>
          </ul>
        </div>
        <div class="show-box">
          <div class="show-label">What to show</div>
          <p>Side-by-side: a screenshot of the Isaac Lab scene from a top-down or angled view, next to a heatmap/grid visualization of the computed occupancy grid at that same instant. The occupied cells should visibly correspond to where the rubble/obstacles are in the screenshot.</p>
        </div>
        <div class="needbox" style="--cpcolor:#5fd0c9">
          <div class="need-label">Passing bar</div>
          <ul>
            <li>No obvious misalignment (occupied cells where there's clearly open ground, or vice versa)</li>
            <li>Grid updates sensibly as the robot's viewpoint changes</li>
          </ul>
        </div>
        <div class="fallback-box">
          <div class="fb-label">If short on time</div>
          <p>A single static side-by-side image is enough — you don't need to show this across multiple viewpoints to make the point.</p>
        </div>
      </div>
    </section>

    <section id="cp4" class="topgap" style="--cpcolor:#ff8c42">
      <span class="eyebrow">Phase 01 deliverable</span>
      <h2>CP4 — rule-based pipeline plumbing</h2>
      <div class="checkpoint" style="--cpcolor:#ff8c42">
        <div class="cp-head"><span class="cp-num">Checkpoint 4</span><span class="cp-when">1-2 days</span></div>
        <div class="cp-title">A non-learned, hand-coded "always move toward the current waypoint" rule gets the robot most of the way to a goal through the rubble.</div>
        <div class="needbox" style="--cpcolor:#ff8c42">
          <div class="need-label">Build before this checkpoint</div>
          <ul>
            <li>CP1 (frozen locomotion) + CP2 (planner) + CP3 (perception) all wired into one env</li>
            <li>A simple rule replacing "RL policy output" with "velocity vector pointing at the current A* waypoint"</li>
          </ul>
        </div>
        <div class="show-box">
          <div class="show-label">What to show</div>
          <p>A screen recording: robot spawns, A* path appears (overlay or printed waypoint list), robot walks along it and reaches the goal area in the rubble scene, no RL involved at all yet.</p>
        </div>
        <div class="needbox" style="--cpcolor:#ff8c42">
          <div class="need-label">Passing bar</div>
          <ul>
            <li>Robot reaches the goal, or gets close, on at least one clean run</li>
            <li>This is proof the entire plumbing chain (perception → planner → frozen locomotion) works before any training time is spent</li>
          </ul>
        </div>
        <div class="fallback-box">
          <div class="fb-label">If short on time</div>
          <p>Don't skip this one — it's the cheapest way to catch an integration bug before you've sunk hours into a training run that fails for a plumbing reason, not a learning reason.</p>
        </div>
      </div>
    </section>

    <section id="cp5">
      <h2>CP5 — first trained nav policy</h2>
      <div class="checkpoint" style="--cpcolor:#ff8c42">
        <div class="cp-head"><span class="cp-num">Checkpoint 5</span><span class="cp-when">3-5 days</span></div>
        <div class="cp-title">PPO training is running, reward is trending upward, and nothing is diverging.</div>
        <div class="needbox" style="--cpcolor:#ff8c42">
          <div class="need-label">Build before this checkpoint</div>
          <ul>
            <li>Observation terms (occupancy grid + goal vector) and reward terms (progress, heading, collision, smoothness) written and logged separately</li>
            <li>skrl PPO agent config for navigation</li>
          </ul>
        </div>
        <div class="show-box">
          <div class="show-label">What to show</div>
          <p>A TensorBoard screenshot: total reward curve trending up over training iterations, plus the individual reward term curves (no term flatlined at zero or exploding).</p>
        </div>
        <div class="needbox" style="--cpcolor:#ff8c42">
          <div class="need-label">Passing bar</div>
          <ul>
            <li>Reward is increasing, even slowly — doesn't need to be converged yet</li>
            <li>No NaNs, no term dominating the others by orders of magnitude</li>
          </ul>
        </div>
        <div class="fallback-box">
          <div class="fb-label">If short on time</div>
          <p>A short (~1hr) training run's curve is enough to show momentum even if the policy isn't good yet — "it's learning, here's the trend" is a legitimate checkpoint on its own.</p>
        </div>
      </div>
    </section>

    <section id="cp6">
      <h2>CP6 — phase 01 full demo + metrics</h2>
      <div class="checkpoint" style="--cpcolor:#ff8c42">
        <div class="cp-head"><span class="cp-num">Checkpoint 6</span><span class="cp-when">end of phase 01</span></div>
        <div class="cp-title">The trained nav policy reliably reaches goals through static rubble, with logged metrics to back it up.</div>
        <div class="needbox" style="--cpcolor:#ff8c42">
          <div class="need-label">Build before this checkpoint</div>
          <ul>
            <li>Training run to convergence</li>
            <li><code>eval_metrics.py</code> running N episodes, logging success rate / time-to-goal / collision rate</li>
          </ul>
        </div>
        <div class="show-box">
          <div class="show-label">What to show</div>
          <p>A <code>play.py</code> recording across 2-3 different start/goal pairs in the rubble scene, plus a printed table: success rate, average time-to-goal, collision count over N evaluation episodes.</p>
        </div>
        <div class="needbox" style="--cpcolor:#ff8c42">
          <div class="need-label">Passing bar</div>
          <ul>
            <li>Success rate meaningfully above the rule-based CP4 baseline, or at least comparable with smoother motion</li>
            <li>This is your phase 01 baseline number — everything later compares back to it</li>
          </ul>
        </div>
        <div class="fallback-box">
          <div class="fb-label">If short on time</div>
          <p>Even a partial success rate (e.g. 60-70%) with honest numbers is a legitimate phase 01 result — "here's where it stands, here's what I'd improve next" is a fine thing to bring to a meeting.</p>
        </div>
      </div>
    </section>

    <section id="cp7" class="topgap" style="--cpcolor:#b9a87c">
      <span class="eyebrow">Phase 02 deliverable</span>
      <h2>CP7 — dynamic obstacles moving</h2>
      <div class="checkpoint" style="--cpcolor:#b9a87c">
        <div class="cp-head"><span class="cp-num">Checkpoint 7</span><span class="cp-when">1-2 days</span></div>
        <div class="cp-title">Scripted obstacles move in the scene in a sane, bounded way, independent of any policy.</div>
        <div class="needbox" style="--cpcolor:#b9a87c">
          <div class="need-label">Build before this checkpoint</div>
          <ul>
            <li>2-4 movable assets added to the scene</li>
            <li><code>events.py</code> motion script (bounded random walk or linear back-and-forth)</li>
          </ul>
        </div>
        <div class="show-box">
          <div class="show-label">What to show</div>
          <p>A short GUI recording of just the obstacles moving — robot can even be stationary for this one, the point is purely to show the obstacles behave correctly across a full episode reset.</p>
        </div>
        <div class="needbox" style="--cpcolor:#b9a87c">
          <div class="need-label">Passing bar</div>
          <ul>
            <li>Obstacles stay within bounds, don't clip through walls or fly off</li>
            <li>Motion resets correctly at episode start</li>
          </ul>
        </div>
        <div class="fallback-box">
          <div class="fb-label">If short on time</div>
          <p>One obstacle moving correctly is enough to demonstrate the mechanism — you can add more before the next checkpoint.</p>
        </div>
      </div>
    </section>

    <section id="cp8">
      <h2>CP8 — frame-stacking validated</h2>
      <div class="checkpoint" style="--cpcolor:#b9a87c">
        <div class="cp-head"><span class="cp-num">Checkpoint 8</span><span class="cp-when">1-2 days</span></div>
        <div class="cp-title">The CNN encoder's output measurably differs between a moving-obstacle sequence and a static one.</div>
        <div class="needbox" style="--cpcolor:#b9a87c">
          <div class="need-label">Build before this checkpoint</div>
          <ul>
            <li>Rolling buffer of last 3 occupancy grids, stacked as channels</li>
            <li>Encoder's input layer updated for the new channel count</li>
          </ul>
        </div>
        <div class="show-box">
          <div class="show-label">What to show</div>
          <p>A small standalone script output: feed the encoder a hand-crafted "obstacle moving left" sequence and a hand-crafted "obstacle static" sequence, print or plot the encoder's output embedding for each — they should differ noticeably.</p>
        </div>
        <div class="needbox" style="--cpcolor:#b9a87c">
          <div class="need-label">Passing bar</div>
          <ul>
            <li>A measurable difference exists between the two embeddings — doesn't need to be interpretable, just non-trivial</li>
          </ul>
        </div>
        <div class="fallback-box">
          <div class="fb-label">If short on time</div>
          <p>This can be a quick printed vector-distance number rather than a visualization — "embedding distance: 0.34 for moving vs 0.02 for two static frames" is a perfectly good thing to show.</p>
        </div>
      </div>
    </section>

    <section id="cp9">
      <h2>CP9 — risk head + reactive demo</h2>
      <div class="checkpoint" style="--cpcolor:#b9a87c">
        <div class="cp-head"><span class="cp-num">Checkpoint 9</span><span class="cp-when">end of phase 02</span></div>
        <div class="cp-title">The trained policy visibly deflects around moving obstacles, and the risk-head ablation comparison has real numbers.</div>
        <div class="needbox" style="--cpcolor:#b9a87c">
          <div class="need-label">Build before this checkpoint</div>
          <ul>
            <li>Risk-prediction head added and trained alongside the main policy</li>
            <li>A no-risk-head ablation trained for comparison (even briefly)</li>
          </ul>
        </div>
        <div class="show-box">
          <div class="show-label">What to show</div>
          <p>A side-by-side recording: the risk-head policy clearly slowing or deflecting before contact, next to the ablation policy colliding or reacting later, plus a printed collision-rate comparison table between the two.</p>
        </div>
        <div class="needbox" style="--cpcolor:#b9a87c">
          <div class="need-label">Passing bar</div>
          <ul>
            <li>Collision rate with the risk head is meaningfully lower than without it</li>
            <li>This comparison is your actual phase 02 result, not just "it doesn't collide" — the ablation is what makes it evidence rather than anecdote</li>
          </ul>
        </div>
        <div class="fallback-box">
          <div class="fb-label">If short on time</div>
          <p>If the ablation run can't be trained in time, a single qualitative recording of anticipatory deflection is still a real result — just be upfront that you haven't quantified the improvement yet.</p>
        </div>
      </div>
    </section>
    <section id="cp10" class="topgap" style="--cpcolor:#c792ea">
      <span class="eyebrow">Hybrid upgrade</span>
      <h2>CP10 — learned terrain cost</h2>
      <div class="checkpoint" style="--cpcolor:#c792ea">
        <div class="cp-head"><span class="cp-num">Checkpoint 10</span><span class="cp-when">2-3 days</span></div>
        <div class="cp-title">A small learned network produces a traversability cost that visibly differs from the hand-coded version, and A* still plans correctly over it.</div>
        <div class="needbox" style="--cpcolor:#c792ea">
          <div class="need-label">Build before this checkpoint</div>
          <ul>
            <li>A small terrain-estimator network (input: local sensor patch, output: a cost scalar per cell)</li>
            <li>A* wired to plan over this learned cost instead of the hand-coded one from CP2/CP6</li>
          </ul>
        </div>
        <div class="show-box">
          <div class="show-label">What to show</div>
          <p>Two side-by-side cost-map heatmaps over the same scene — the old hand-coded cost map and the new learned one — plus the A* path overlaid on each, showing the route changes where the learned cost disagrees with the hand-coded guess.</p>
        </div>
        <div class="needbox" style="--cpcolor:#c792ea">
          <div class="need-label">Passing bar</div>
          <ul>
            <li>The learned cost map isn't just a noisy copy of the hand-coded one — it should show at least one region where it disagrees in a sensible way</li>
            <li>A* still returns a valid, obstacle-avoiding path over the new cost</li>
          </ul>
        </div>
        <div class="fallback-box">
          <div class="fb-label">If short on time</div>
          <p>If the learned estimator isn't trained well yet, showing the mechanism working end-to-end with an undertrained network is still a legitimate "the pipeline works, the network needs more training" checkpoint.</p>
        </div>
      </div>
    </section>

    <section id="cp11">
      <h2>CP11 — feedback loop correcting cost</h2>
      <div class="checkpoint" style="--cpcolor:#c792ea">
        <div class="cp-head"><span class="cp-num">Checkpoint 11</span><span class="cp-when">2-3 days</span></div>
        <div class="cp-title">The proprioception signal visibly changes the terrain cost map mid-episode, in the direction you'd expect.</div>
        <div class="needbox" style="--cpcolor:#c792ea">
          <div class="need-label">Build before this checkpoint</div>
          <ul>
            <li>Joint-torque-magnitude + orientation-deviation signal computed from the frozen locomotion policy each step</li>
            <li>EMA correction rule updating the learned terrain cost from CP10</li>
          </ul>
        </div>
        <div class="show-box">
          <div class="show-label">What to show</div>
          <p>A before/after cost-map pair from the same episode: the cost map at episode start, and the cost map after the robot has struggled (high torque/tilt) crossing a specific region — that region's cost should visibly increase in the "after" map.</p>
        </div>
        <div class="needbox" style="--cpcolor:#c792ea">
          <div class="need-label">Passing bar</div>
          <ul>
            <li>The correction moves the right direction — harder-to-traverse regions get more expensive, not less</li>
            <li>The correction doesn't blow up or oscillate wildly run to run</li>
          </ul>
        </div>
        <div class="fallback-box">
          <div class="fb-label">If short on time</div>
          <p>A printed before/after cost value for one specific grid cell, rather than a full heatmap pair, is enough to demonstrate the mechanism is alive.</p>
        </div>
      </div>
    </section>

    <section id="cp12">
      <h2>CP12 — dual policy, hard switch</h2>
      <div class="checkpoint" style="--cpcolor:#c792ea">
        <div class="cp-head"><span class="cp-num">Checkpoint 12</span><span class="cp-when">1 week</span></div>
        <div class="cp-title">Two separately trained policies — progress and caution — each behave distinctly, and a hard switch between them doesn't break the robot.</div>
        <div class="needbox" style="--cpcolor:#c792ea">
          <div class="need-label">Build before this checkpoint</div>
          <ul>
            <li>Progress policy trained with a speed/heading-weighted reward</li>
            <li>Caution policy trained with a collision-avoidance/stability-weighted reward</li>
            <li>A simple hard-threshold switch using the existing risk head's output</li>
          </ul>
        </div>
        <div class="show-box">
          <div class="show-label">What to show</div>
          <p>Two separate recordings of each policy running alone (progress policy moving fast and direct; caution policy moving slower and more conservatively), then one recording of the switched combination reacting to an obstacle by visibly handing control to the caution policy.</p>
        </div>
        <div class="needbox" style="--cpcolor:#c792ea">
          <div class="need-label">Passing bar</div>
          <ul>
            <li>The two policies are visibly different in behavior, not just numerically different in reward</li>
            <li>The switch doesn't cause a visible jerk or instability bad enough to destabilize the gait</li>
          </ul>
        </div>
        <div class="fallback-box">
          <div class="fb-label">If short on time</div>
          <p>If training two full policies isn't feasible yet, showing the progress policy alone plus a clear plan/diagram for the caution policy and switch is a defensible partial checkpoint — be upfront that the switch isn't implemented yet.</p>
        </div>
      </div>
    </section>

    <section id="cp13">
      <h2>CP13 — risk-blended dual policy demo</h2>
      <div class="checkpoint" style="--cpcolor:#c792ea">
        <div class="cp-head"><span class="cp-num">Checkpoint 13</span><span class="cp-when">end of hybrid build</span></div>
        <div class="cp-title">The full hybrid system — learned terrain cost, feedback loop, and risk-blended dual policy — runs end to end and beats the phase 02 baseline on a real metric.</div>
        <div class="needbox" style="--cpcolor:#c792ea">
          <div class="need-label">Build before this checkpoint</div>
          <ul>
            <li>CP10 + CP11 + CP12 all wired together into one running system</li>
            <li>Continuous blend (not hard switch) between progress and caution policies, weighted by risk-head output</li>
          </ul>
        </div>
        <div class="show-box">
          <div class="show-label">What to show</div>
          <p>A side-by-side recording: your phase 02 single-policy baseline versus the full hybrid system, on the same dynamic-obstacle scene, plus a comparison table (collision rate, time-to-goal, jerk) between the two.</p>
        </div>
        <div class="needbox" style="--cpcolor:#c792ea">
          <div class="need-label">Passing bar</div>
          <ul>
            <li>The hybrid system is measurably better on at least one metric (most likely collision rate, given the design intent) without being drastically worse on the others</li>
            <li>This comparison table is your actual thesis-level result for this part of the project</li>
          </ul>
        </div>
        <div class="fallback-box">
          <div class="fb-label">If short on time</div>
          <p>A neutral or mixed result (better on safety, worse on speed) is still a real, presentable finding — that's the actual tradeoff the design predicts, and showing it honestly is stronger than only showing favorable numbers.</p>
        </div>
      </div>
    </section>

    <section id="cp14" class="topgap" style="--cpcolor:#e2533a">
      <span class="eyebrow">Stretch / later</span>
      <h2>CP14 — multi-objective Pareto front</h2>
      <div class="checkpoint" style="--cpcolor:#e2533a">
        <div class="cp-head"><span class="cp-num">Checkpoint 14</span><span class="cp-when">1-2 weeks, optional</span></div>
        <div class="cp-title">Multiple weight-trained policies plotted against each other show a visible tradeoff.</div>
        <div class="show-box">
          <div class="show-label">What to show</div>
          <p>A scatter plot — energy vs. time-to-goal, or similar — with each trained weight configuration as one point, showing a visible tradeoff curve.</p>
        </div>
        <div class="fallback-box">
          <div class="fb-label">If short on time</div>
          <p>This entire checkpoint is appropriate to defer past your main deadline if the hybrid architecture (CP10-13) takes longer than planned — it's additive, not foundational.</p>
        </div>
      </div>
    </section>

    <section id="cp15">
      <h2>CP15 — multi-robot baseline</h2>
      <div class="checkpoint" style="--cpcolor:#e2533a">
        <div class="cp-head"><span class="cp-num">Checkpoint 15</span><span class="cp-when">research-frontier, optional</span></div>
        <div class="cp-title">Multiple robots running independent copies of the trained policy in one scene, without interfering with each other's physics.</div>
        <div class="show-box">
          <div class="show-label">What to show</div>
          <p>A recording of 2-3 robots simultaneously navigating to separate goals in the same scene.</p>
        </div>
        <div class="fallback-box">
          <div class="fb-label">If short on time</div>
          <p>Treat this entire checkpoint, and any communication layer beyond it, as documented future work rather than something to rush — it was always scoped as a stretch goal.</p>
        </div>
      </div>
    </section>

    <section id="recording-tips" class="topgap">
      <span class="eyebrow">Reference</span>
      <h2>How to record each demo</h2>
      <p>
        A consistent, low-effort recording habit makes every checkpoint above
        easy to actually produce, rather than something you scramble to capture
        the night before a meeting.
      </p>
      <table>
        <tr><th>What</th><th>How</th></tr>
        <tr><td>GUI behavior (robot walking, obstacles moving)</td><td class="mono">OS screen recorder, 15-30 sec clips, trim to the relevant moment</td></tr>
        <tr><td>TensorBoard curves</td><td class="mono">Screenshot at a zoom level where the trend is visible — don't crop out the axis labels</td></tr>
        <tr><td>Before/after comparisons</td><td class="mono">Same camera angle, same scene, side-by-side image or two clips back to back</td></tr>
        <tr><td>Metrics tables</td><td class="mono">Plain printed terminal output or a simple CSV — no need to format nicely for a working meeting</td></tr>
      </table>
      <p class="dim" style="font-size:13px">Keep a single running folder (e.g. <code>demo_recordings/</code>) with one subfolder per checkpoint — this becomes the raw material for your final thesis presentation too, so it's not wasted effort.</p>
    </section>

    <section id="summary-table">
      <h3>Full checkpoint summary table</h3>
      <table>
        <tr><th>CP</th><th>Proves</th><th>Time</th></tr>
        <tr><td class="mono">1</td><td>Frozen locomotion loads and walks in the new env</td><td class="mono">~half day</td></tr>
        <tr><td class="mono">2</td><td>A* avoids obstacles on a toy grid</td><td class="mono">~half day</td></tr>
        <tr><td class="mono">3</td><td>Occupancy grid matches the real scene</td><td class="mono">1-2 days</td></tr>
        <tr><td class="mono">4</td><td>Full plumbing works with a rule, no RL yet</td><td class="mono">1-2 days</td></tr>
        <tr><td class="mono">5</td><td>PPO training is alive and trending</td><td class="mono">3-5 days</td></tr>
        <tr><td class="mono">6</td><td>Phase 01 baseline numbers exist</td><td class="mono">end of phase 01</td></tr>
        <tr><td class="mono">7</td><td>Dynamic obstacles move correctly</td><td class="mono">1-2 days</td></tr>
        <tr><td class="mono">8</td><td>Frame-stacking carries motion info</td><td class="mono">1-2 days</td></tr>
        <tr><td class="mono">9</td><td>Risk head measurably reduces collisions</td><td class="mono">end of phase 02</td></tr>
        <tr><td class="mono">10</td><td>Terrain cost is learned, not hand-coded</td><td class="mono">2-3 days</td></tr>
        <tr><td class="mono">11</td><td>Feedback loop corrects the cost map</td><td class="mono">2-3 days</td></tr>
        <tr><td class="mono">12</td><td>Two specialized policies behave differently</td><td class="mono">1 week</td></tr>
        <tr><td class="mono">13</td><td>Full hybrid beats phase 02 baseline</td><td class="mono">end of hybrid build</td></tr>
        <tr><td class="mono">14</td><td>Pareto tradeoff across objectives</td><td class="mono">1-2 weeks, optional</td></tr>
        <tr><td class="mono">15</td><td>Multiple robots coexist without breaking</td><td class="mono">optional</td></tr>
      </table>
    </section>

    <footer>NEUROGAIT · DEMO CHECKPOINT ROADMAP · GO2 / ISAAC LAB 2.3 / SKRL PPO</footer>

  </main>
</div>
<script>
  const links = document.querySelectorAll('#toc a');
  const sections = [...links].map(l => document.querySelector(l.getAttribute('href'))).filter(Boolean);
  function onScroll(){
    let current = sections[0];
    const y = window.scrollY + 100;
    sections.forEach(s => { if (s && s.offsetTop <= y) current = s; });
    links.forEach(l => l.classList.toggle('active', current && l.getAttribute('href') === '#' + current.id));
  }
  document.addEventListener('scroll', onScroll);
  onScroll();
</script>
</body>
</html>
