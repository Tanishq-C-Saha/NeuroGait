# NeuroGait: Hierarchical Legged Navigation on Rubble

This project serves as a production development repository and Omniverse extension for **NeuroGait**, a hierarchical legged navigation system that interfaces a high-level PPO policy (trained via **skrl**) with a frozen locomotion policy (trained via **rsl_rl** on a Unitree Go2 robot) in **Isaac Lab 2.3.0** (Isaac Sim 5.1).

---

## 🛠️ Installation

1. Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

2. Clone this repository separately from the Isaac Lab installation directory and install the package in editable mode:
   ```bash
   pip install -e .
   python -m pip install -e source/neurogait
   ```

3. Verify the installation list:
   ```bash
   python scripts/list_envs.py
   ```

---

## 📂 Codebase Architecture

```
d:/CAPSTONE/Navigation/
├── source/                      # Teammate's core Omniverse extension source
├── scripts/                     # Evaluation, training, and random agent scripts
├── configs/                     # Hydra configuration system (compositional YAMLs)
│   ├── config.yaml              # Entry point config mapping defaults
│   ├── agent/                   # Agent parameters (PPO, SAC, TD3, etc.)
│   ├── env/                     # Simulation metrics, frequency configurations
│   ├── robot/                   # Robot specifications (Go2, R1, R2)
│   └── logger/                  # Logging (Tensorboard / Weights & Biases)
├── neurogait/                   # Source package
│   ├── train.py                 # Core trainer script
│   ├── eval.py                  # Evaluator script compiling metrics reports
│   ├── agents/                  # Policy controllers and metaheuristic optimizers
│   │   ├── base_agent.py        # Abstract Agent template interface
│   │   ├── ppo_agent.py         # skrl PPO adapter with Risk prediction head
│   │   ├── dual_policy.py       # Blended Progress/Caution coordinator (AT7)
│   │   └── metaheuristics/      # Phase 2 metaheuristic optimizers
│   ├── envs/                    # Gymnasium simulations wrappers
│   │   ├── navigation_env.py    # NeuroGait-Navigation-Rubble-v0 implementation
│   │   └── locomotion/          # Frozen rsl_rl locomotion loader
│   ├── perception/              # Real-time depth projection & costmap compilers
│   ├── planning/                # Path planning modules (A* planner)
│   ├── communication/           # Graph Attention Networks (GAT) multi-agent sync
│   └── utils/                   # Rewards, metrics, auxiliary risk heads, pareto helpers
└── tests/                       # Unit testing suite
```

---

## 🚀 Execution Pipelines

### 1. Training (AT4)
To start training the high-level PPO policy:
```bash
python neurogait/train.py
```
Or run the specialized CLI script:
```bash
python train_nav.py --num_envs 1024 --device cuda:0 --epochs 100
```

### 2. Evaluation (AT5)
To evaluate a trained checkpoint and compile a benchmark report:
```bash
python eval_metrics.py --checkpoint checkpoints/neurogait_final.pt --episodes 20
```

---

## 📄 Team Assignments
- **RL Team**: Responsible for `neurogait/agents/*`, config sweeps, and Pareto evaluations.
- **Simulation Team**: Coordinates `neurogait/envs/*` and asset imports.
- **Navigation Team**: Manages A* search, depth registration, costmaps, and rewards.
- **Communication Team**: Implements multi-robot Graph Attention Networks (GAT).
