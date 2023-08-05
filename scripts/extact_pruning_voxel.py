from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt
from typing import Tuple
import mediapy as media
import numpy as np
import torch
import tyro
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

import open3d as o3d
from typing_extensions import Literal, assert_never
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from skimage.metrics import structural_similarity
from nerfstudio.cameras.camera_paths import get_path_from_json, get_spiral_path
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.configs.base_config import Config  # pylint: disable=unused-import
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.model_components.ray_samplers import UniformSampler
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import ItersPerSecColumn
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse
import cv2
from nerfstudio.data.utils.label import id2label,labels,assigncolor
# from nerfstudio.data.utils.waymo_lable import id2label,labels,assigncolor

CONSOLE = Console(width=120)

@dataclass
class ExtractPruningVoxel():
    """Load a checkpoint, run marching cubes, extract mesh, and save it to a ply file."""

    # Path to config YAML file.
    load_config: Path = Path("no_palce")
    # Marching cube resolution.
    resolution: int = 256
    # Name of the output file.
    output_path: Path = Path("output.ply")
    """Minimum of the bounding box."""
    bounding_box_min: Tuple[float, float, float] = (-1, -0.3, -0.8) ## for kitti360
    # bounding_box_min: Tuple[float, float, float] = (-1, -0.3, -2.5)
    """Maximum of the bounding box."""
    bounding_box_max: Tuple[float, float, float] = (1, 0.3, 2.5)  ## for kitti360
    # bounding_box_max: Tuple[float, float, float] = (1, 0.3, 0)

    # used for nerfacto, vanilla-nerf, other density-based method
    is_extract_density: bool = True
    # extract semantic voxel grid
    is_use_semantic: bool = False
    # Voxel Size of each Voxel
    voxelsize: float= 0.01
    # Density Threshold
    density_threshold: float = 2.0
    ## Transmittance Threshold
    transmittance_threshold: float = 0.9

    num_rays_per_chunk: int = 4096

    extra_depth: float = 0.1

    output_dir: Path = Path("pointcloud")

    def __init__(self, parser_path):
        self.load_config = Path(parser_path.load_config)
        self.output_path = parser_path.output_path
        self.uniform_sampler = UniformSampler(single_jitter=False)




    def main(self) -> None:
        """Main function."""
        assert str(self.output_path)[-4:] == ".ply"
        _, pipeline, _ = eval_setup(self.load_config)

        CONSOLE.print("Extract Voxel Field with density threshold: {} , transmittance threshold: {} ".format(self.density_threshold,self.transmittance_threshold))
        progress = Progress(
            TextColumn(":cloud: Computing Voxel Field :cloud:"),
            BarColumn(),
            TaskProgressColumn(show_speed=True),
            TimeRemainingColumn(elapsed_when_finished=True, compact=True),
        )
        ## we use traindatasets to extract voxel field
        DataCache = pipeline.datamanager.train_dataset
        cameras = DataCache.cameras.to(pipeline.device)

        points = []
        rgbs = []
        with progress:
            for camera_idx in progress.track(range(0,cameras.size,3), description="Calcul cameras"):
                camera_ray_bundle = cameras.generate_rays(camera_indices=camera_idx)
                with torch.no_grad():
                    outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
                depth = outputs['depth'].flatten() + self.extra_depth

                camera_ray_bundle.nears = torch.zeros_like(depth)
                num_rays = len(camera_ray_bundle)
                for i in range(0, num_rays, self.num_rays_per_chunk):
                    start_idx = i
                    end_idx = i + self.num_rays_per_chunk
                    ray_bundle = camera_ray_bundle.flatten()[start_idx:end_idx]
                    depths = depth.flatten()[start_idx:end_idx]
                    ray_bundle.fars = depths.unsqueeze(dim=-1)
                    ray_bundle.nears = torch.zeros_like(depths).unsqueeze(dim=-1)
                    point,rgb = self.sample_voxel_output(ray_bundle=ray_bundle,pipeline=pipeline)
                    points.append(point)
                    rgbs.append(rgb)

        ## Visualize Pointcloud
        points = torch.cat(points, dim=0)
        rgbs = torch.cat(rgbs, dim=0)

        ## eliminate sky pointcloud
        Nosky_region = ~(rgbs == 23)
        points = points[Nosky_region]
        rgbs = rgbs[Nosky_region]


        rgbs = torch.from_numpy(assigncolor(rgbs.reshape(-1)))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.float().cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(rgbs.float().cpu().numpy())

        CONSOLE.print("Cleaning Point Cloud")
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=10.0)
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Cleaning Point Cloud")

        torch.cuda.empty_cache()

        CONSOLE.print(f"[bold green]:white_check_mark: Generated {pcd}")
        CONSOLE.print("Saving Point Cloud...")
        o3d.io.write_point_cloud(str(self.output_dir / "point_cloud.ply"), pcd)
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Saving Point Cloud")


    def sample_voxel_output(self,ray_bundle,pipeline = None, num_samples=25):
        semantic_fn = lambda x: pipeline.model.field.get_pos_semantic_class(x)

        samples = self.uniform_sampler(ray_bundle, num_samples=num_samples)
        field_outputs = pipeline.model.field(samples, compute_normals=False)
        position = samples.frustums.get_positions()

        weights,tranmittance = samples.get_weights(field_outputs[FieldHeadNames.DENSITY],output_transmittance=True)
        density = field_outputs[FieldHeadNames.DENSITY]
        density_mask = density > self.density_threshold
        # transmittance_mask = tranmittance > self.transmittance_threshold
        transmittance_mask = weights > self.transmittance_threshold
        valid_mask = density_mask * transmittance_mask
        valid_mask = valid_mask.squeeze(dim=-1)
        ## semantic
        semantics = semantic_fn(position[valid_mask])
        return position[valid_mask],semantics




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-config', type=str, help='testset or trainset')
    parser.add_argument('--output-path', type=str, help='Config Path')
    config = parser.parse_args()

    ExtractPruningVoxel(config).main()