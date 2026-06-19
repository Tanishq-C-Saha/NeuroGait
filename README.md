# NeuroGait: Hierarchical Legged Navigation on Rubble

This repository contains the production code layout for **NeuroGait**, a hierarchical legged navigation system that interfaces a high-level PPO policy (trained via **skrl**) with a frozen locomotion policy (trained via **rsl_rl** on a Unitree Go2 robot) in **Isaac Lab 2.3.0** (Isaac Sim 5.1).

---

## 🛠️ Installation

Ensure you have Isaac Sim 5.1 and Isaac Lab installed. Then install NeuroGait:

```bash
# Clone the repository and install dependencies
cd d:/CAPSTONE/Navigation
pip install -e .
```

---

## 📂 Codebase Architecture

```
d:/CAPSTONE/Navigation/
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
│   │   ├── dual_policy.py       # Agile/Conservative coordinator (AT7)
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

To run sweep configurations via Hydra's multirun:
```bash
python neurogait/train.py --multirun agent.learning_rate=1e-4,3e-4,5e-4
```

### 2. Evaluation (AT5)
To evaluate a trained checkpoint:
```bash
python neurogait/eval.py checkpoint_path=outputs/checkpoints/neurogait_final.pt
```

---

## 📄 Team Assignments
- **RL Team**: Responsible for `neurogait/agents/*`, config sweeps, and Pareto evaluations.
- **Simulation Team**: Coordinates `neurogait/envs/*` and asset imports.
- **Navigation Team**: Manages A* search, depth registration, costmaps, and rewards.
- **Communication Team**: Implements multi-robot Graph Attention Networks (GAT).
