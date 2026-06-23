# 05: CNN vs MLP for Occupancy Grid

## The confusion
Why not just flatten the 40×40 grid and pass it to an MLP?
The grid IS 1600 floats — an MLP can take any fixed-size input.

## The answer
A flat MLP treats `grid[5,20]` and `grid[5,21]` as completely independent features.
A CNN exploits spatial locality: adjacent cells are processed together via shared kernels.
Fewer parameters, faster convergence, built-in translation invariance.

## How it works

### Spatial structure lost in flat MLP

```
Grid cell (row=5, col=20) → input index 220
Grid cell (row=5, col=21) → input index 221
Grid cell (row=6, col=20) → input index 260   ← 40 apart in the flat vector
```

An MLP's first layer learns a 1600×128 = 204,800 weight matrix. Neighboring cells
are 40 indices apart — the MLP must learn spatial structure from scratch with no inductive
bias. This requires many more samples and parameters.

### CNN inductive bias

A 5×5 convolutional kernel explicitly processes each 5×5 neighborhood. The kernel weights
are shared across all (row, col) positions — translation equivariance is built in.

```
Layer 1: Conv2d(1, 16, 5×5) → 40×40 grid × 16 channels = 400 parameters (vs 204,800)
Layer 2: Conv2d(16, 32, 3×3) + stride=2 → spatial compression
Layer 3: Conv2d(32, 32, 3×3) → refined features
FC: flatten → 128 → 64
```

### Parameter count comparison

| Architecture | Grid params | Total params |
|-------------|------------|--------------|
| Flat MLP (1600→256→128→64→3) | 454,144 | ~454K |
| CNN (3 conv + 2 FC) + MLP scalars | ~87K | ~240K |

~47% fewer parameters → faster training, less overfitting.

### Why not use a pretrained ResNet?

ResNet was pretrained on ImageNet (natural images: cats, dogs, textures).
Our grid is binary 0/1 occupancy — completely different domain.
Transfer learning from ResNet would require fine-tuning from scratch anyway,
with additional overhead of 11M+ parameters. Custom small CNN wins.

## What breaks if you get this wrong

Using a flat MLP on 1600 binary inputs is not catastrophically wrong — it can in
principle learn navigation — but requires significantly more samples (estimated 3–5×
more environment steps) and may fail to generalize to new obstacle configurations
because it memorizes pixel positions rather than spatial patterns.

## Code reference

- `models/navigation_policy.py` — `NavigationPolicy.grid_cnn` (3-layer CNN)
- Obs order: grid must be indices [0:1600] so `obs[:, :1600].view(-1, 1, 40, 40)` works
- See `01_two_obs_managers.md` for why obs ordering is critical
