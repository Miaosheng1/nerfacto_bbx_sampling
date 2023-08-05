#!/usr/bin/env python
"""
eval.py
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
import tyro
from rich.console import Console

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.marching_cubes import get_surface_occupancy, get_surface_sliding,get_density_voxel

CONSOLE = Console(width=120)


@dataclass
class ExtractMesh:
    """Load a checkpoint, run marching cubes, extract mesh, and save it to a ply file."""

    # Path to config YAML file.
    load_config: Path
    # Marching cube resolution.
    resolution: int = 256
    # Name of the output file.
    output_path: Path = Path("output.ply")
    # Whether to simplify the mesh.
    simplify_mesh: bool = False
    # extract the mesh using occupancy field (unisurf) or SDF, default sdf
    is_occupancy: bool = False
    """Minimum of the bounding box."""
    bounding_box_min: Tuple[float, float, float] = (-1, -0.3, -0.8) ## for kitti360
    # bounding_box_min: Tuple[float, float, float] = (-1, -0.3, -2.5)
    """Maximum of the bounding box."""
    bounding_box_max: Tuple[float, float, float] = (1, 0.3, 2.5)  ## for kitti360
    # bounding_box_max: Tuple[float, float, float] = (1, 0.3, 0)
    # used for sdf-based method
    is_sdf: bool = False
    # used for nerfacto, vanilla-nerf, other density-based method
    is_extract_density: bool = True
    # extract semantic voxel grid
    is_use_semantic: bool = False


    def main(self) -> None:
        """Main function."""
        assert str(self.output_path)[-4:] == ".ply"
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        _, pipeline, _ = eval_setup(self.load_config)

        CONSOLE.print("Extract mesh with marching cubes and may take a while")

        if self.is_occupancy:
            # for unisurf
            get_surface_occupancy(
                occupancy_fn=lambda x: torch.sigmoid(
                    10 * pipeline.model.field.forward_geonetwork(x)[:, 0].contiguous()
                ),
                resolution=self.resolution,
                bounding_box_min=self.bounding_box_min,
                bounding_box_max=self.bounding_box_max,
                level=0.5,
                device=pipeline.model.device,
                output_path=self.output_path,
            )
        elif self.is_sdf:
            assert self.resolution % 512 == 0
            # for sdf we can multi-scale extraction.
            get_surface_sliding(
                # sdf=lambda x: pipeline.model.field.forward_geonetwork(x)[:, 0].contiguous(), for sdf-based method
                sdf=lambda x: pipeline.model.field.get_pos_density(x)[0] - 1,  ## for nerfacto
                resolution=self.resolution//2,
                bounding_box_min=self.bounding_box_min,
                bounding_box_max=self.bounding_box_max,
                coarse_mask=pipeline.model.scene_box.coarse_binary_gird,
                output_path=self.output_path,
                simplify_mesh=self.simplify_mesh,
            )
        elif self.is_extract_density:
            # assert self.resolution % 512 == 0
            density_fn = lambda x: pipeline.model.field.get_pos_density(x)[0]
            semantic_fn = lambda x: pipeline.model.field.get_pos_semantic_class(x)
            get_density_voxel(
                density_fn = density_fn,
                semantic_fn = semantic_fn,
                resolution = self.resolution,
                bounding_box_min=self.bounding_box_min,
                bounding_box_max=self.bounding_box_max,
                output_path=self.output_path,
            )




def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(tyro.conf.FlagConversionOff[ExtractMesh]).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(ExtractMesh)  # noqa
