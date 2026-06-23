"""Spawn visual-only landmark pillars in the Isaac Sim viewport.

Import only after AppLauncher has started.
"""

import isaaclab.sim as sim_utils


def spawn_marker(prim_path: str, xyz: tuple, color_rgb: tuple) -> None:
    """Spawn a tall coloured visual-only pillar at world xyz.  No physics.

    prim_path must be directly under /World/ (e.g. /World/marker_start) because
    intermediate parents are not created automatically.
    """
    cfg = sim_utils.CuboidCfg(
        size=(0.3, 0.3, 1.8),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=color_rgb,
            opacity=1.0,
        ),
    )
    cfg.func(prim_path, cfg, translation=xyz)
