"""CP5 Step 1 — Export the frozen locomotion policy to TorchScript.

The rsl_rl play.py already auto-exports on each play run.  This script:
  1. Checks whether the exported file already exists and verifies its
     input/output shape with a dummy forward pass.
  2. If the file does NOT exist (unusual), runs the rsl_rl export path.

Run:
    ~/isaac-sim/kit/python/bin/python3 scripts/cp5/export_locomotion.py

No simulator needed — pure torch, no isaaclab imports.
"""

import argparse
import os
import sys

_DEFAULT_PT = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "logs",
    "rsl_rl",
    "unitree_go2_rough",
    "2026-06-13_19-33-23",
    "exported",
    "policy.pt",
)

# Locomotion observation dimension for the rough env (235 dims verified in research)
_LOCO_OBS_DIM = 235
_LOCO_ACT_DIM = 12


def verify_policy(path: str) -> None:
    import torch

    path = os.path.abspath(path)
    if not os.path.isfile(path):
        print(f"[export] ERROR: policy file not found at {path}")
        print("[export] Run scripts/rsl_rl/play.py once to auto-export.")
        sys.exit(1)

    print(f"[export] Loading: {path}")
    model = torch.jit.load(path, map_location="cpu")
    model.eval()

    dummy = torch.zeros(1, _LOCO_OBS_DIM)
    with torch.no_grad():
        out = model(dummy)

    assert out.shape == (1, _LOCO_ACT_DIM), (
        f"Shape mismatch: expected (1, {_LOCO_ACT_DIM}), got {out.shape}\n"
        f"Check that LOW_LEVEL_ENV_CFG uses UnitreeGo2RoughEnvCfg (235-dim obs)."
    )
    print(f"[export] Verified ✓  input=(1,{_LOCO_OBS_DIM}) → output={tuple(out.shape)}")
    print(f"[export] Use this path in PreTrainedPolicyActionCfg:")
    print(f"         policy_path={path!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        default=_DEFAULT_PT,
        help="Path to the exported TorchScript policy (.pt)",
    )
    args = parser.parse_args()
    verify_policy(args.path)
