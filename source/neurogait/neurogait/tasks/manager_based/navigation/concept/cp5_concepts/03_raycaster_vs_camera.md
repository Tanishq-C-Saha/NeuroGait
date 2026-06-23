# 03: RayCasterCamera vs CameraCfg

## The confusion
Both produce depth images. Why does one limit us to 12 envs while the other scales to 1024+?

## The answer
`CameraCfg` uses the **RTX rendering pipeline** (GPU texture memory per env).
`MultiMeshRayCasterCameraCfg` uses **Warp GPU raycasting** against PhysX meshes.
Same depth output. ~100× less GPU memory.

## How it works

| | CameraCfg | MultiMeshRayCasterCameraCfg |
|-|-----------|------------------------------|
| Pipeline | Omniverse RTX renderer | Warp kernel on PhysX collision meshes |
| GPU cost | ~500 MB per env | ~0.5 MB per env |
| Max envs (24 GB VRAM) | ~12–48 envs | 1024+ envs |
| Lighting effects | Yes (realistic) | No (pure geometry) |
| Dynamic meshes | Yes | Yes (PhysX updates) |
| Import | `isaaclab.sensors.CameraCfg` | `isaaclab.sensors.MultiMeshRayCasterCameraCfg` |
| mesh_prim_paths | n/a | Required: list of prim paths to cast against |

### Data access (identical interface)

```python
# Both cameras: access depth the same way
depth = env.scene["front_cam"].data.output["distance_to_image_plane"]  # (E, H, W)
K     = env.scene["front_cam"].data.intrinsic_matrices                  # (E, 3, 3)
```

### Why RTX is so expensive

The RTX pipeline allocates an independent render buffer for every environment because each env
can have different lighting, materials, and camera positions. For training with many envs,
this creates one render context per env, exhausting VRAM at low env counts.

### Why Warp raycasting is cheap

Warp casts rays directly against PhysX's BVH collision geometry. All envs share the same Warp
kernel call; the per-env overhead is proportional to the number of rays, not a full render context.

## CP5 Attempt 1 bug

```python
# WRONG — RTX camera, failed at 12 envs
self.scene.camera = CameraCfg(prim_path="{ENV_REGEX_NS}/Robot/base/front_cam", ...)
# Only 12 envs could be created before running out of VRAM (3 hours of debugging)
```

## Code reference

- `IsaacLab/source/isaaclab/isaaclab/sensors/ray_caster/multi_mesh_ray_caster_camera_cfg.py`
- `IsaacLab/source/isaaclab/isaaclab/sensors/ray_caster/multi_mesh_ray_caster_camera.py`
- CP5 config: `config/go2/navigation_env_cfg.py` — scene.front_cam
