#!/usr/bin/env python
"""
render.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

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
from typing_extensions import Literal, assert_never

from nerfstudio.cameras.camera_paths import get_path_from_json, get_spiral_path
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.configs.base_config import Config  # pylint: disable=unused-import
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import install_checks
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import ItersPerSecColumn
from nerfstudio.utils import plotly_utils as vis
from nerfstudio.data.utils.label import id2label,labels,assigncolor
# from nerfstudio.data.utils.waymo_lable import assigncolor
CONSOLE = Console(width=120)


def _render_trajectory_video(
    pipeline: Pipeline,
    cameras: Cameras,
    output_filename: Path,
    rendered_output_names: List[str],
    rendered_resolution_scaling_factor: float = 1.0,
    seconds: float = 5.0,
    output_format: Literal["images", "video"] = "video",
) -> None:
    """Helper function to create a video of the spiral trajectory.

    Args:
        pipeline: Pipeline to evaluate with.
        cameras: Cameras to render.
        output_filename: Name of the output file.
        rendered_output_names: List of outputs to visualise.
        rendered_resolution_scaling_factor: Scaling factor to apply to the camera image resolution.
        seconds: Length of output video.
        output_format: How to save output data.
    """
    CONSOLE.print("[bold green]Creating trajectory video")
    images = []
    semantic_image = []
    cameras.rescale_output_resolution(rendered_resolution_scaling_factor)
    cameras = cameras.to(pipeline.device)
    output_semantic_filname = output_filename.with_name("sem_valid.mp4")

    progress = Progress(
        TextColumn(":movie_camera: Rendering :movie_camera:"),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        ItersPerSecColumn(suffix="fps"),
        TimeRemainingColumn(elapsed_when_finished=True, compact=True),
    )
    output_image_dir = output_filename.parent / output_filename.stem
    if output_format == "images":
        output_image_dir.mkdir(parents=True, exist_ok=True)

    render_image = []
    render_semantic = []

    with progress:
        for camera_idx in progress.track(range(cameras.size), description=""):
            camera_ray_bundle = cameras.generate_rays(camera_indices=camera_idx)
            with torch.no_grad():
                ## 这个output 里面包括 rgb map, depth map,nomal map 等
                outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)

            for rendered_output_name in rendered_output_names:
                if rendered_output_name not in outputs:
                    CONSOLE.rule("Error", style="red")
                    CONSOLE.print(f"Could not find {rendered_output_name} in the model outputs", justify="center")
                    CONSOLE.print(f"Please set --rendered_output_name to one of: {outputs.keys()}", justify="center")
                    sys.exit(1)

                output_image = outputs[rendered_output_name].cpu().numpy()
                if rendered_output_name == 'rgb':
                    render_image.append(output_image)

                elif rendered_output_name == "semantics":
                    semantic_labels = torch.argmax(torch.nn.functional.softmax(outputs["semantics"], dim=-1), dim=-1)
                    h, w = semantic_labels.shape[0], semantic_labels.shape[1]
                    semantic_color_map = assigncolor(semantic_labels.reshape(-1)).reshape(h, w, 3)
                    render_semantic.append(semantic_color_map)

        render_image = np.stack(render_image, axis=0)
        render_semantic = np.stack(render_semantic, axis=0)

        if output_format == "images":
            media.write_image(output_image_dir / f"{camera_idx:05d}.png", render_image)
            media.write_image(output_image_dir / f"sem_{camera_idx:05d}.png", render_semantic)
        # else:
        #     images.append(render_image)
        #     semantic_image.append(render_semantic)


    if output_format == "video":
        fps = len(render_image) / seconds
        # make the folder if it doesn't exist
        output_filename.parent.mkdir(parents=True, exist_ok=True)
        with CONSOLE.status("[yellow]Saving video", spinner="bouncingBall"):
            media.write_video(output_filename, render_image, fps=fps)
            ## semantic

        with CONSOLE.status("[yellow]Saving video", spinner="bouncingBall"):
            media.write_video(output_semantic_filname, render_semantic, fps=fps)
    CONSOLE.rule("[green] :tada: :tada: :tada: Success :tada: :tada: :tada:")
    CONSOLE.print(f"[green]Saved video to {output_filename}", justify="center")


@dataclass
class RenderTrajectory:
    """Load a checkpoint, render a trajectory, and save to a video file."""

    # Path to config YAML file.
    load_config: Path
    # Name of the renderer outputs to use. rgb, depth, etc. concatenates them along y axis
    rendered_output_names: List[str] = field(default_factory=lambda: ["rgb","semantics"])
    #  Trajectory to render.
    traj: Literal["spiral", "filename","roam"] = "spiral"
    # Scaling factor to apply to the camera image resolution.
    downscale_factor: int = 1
    # Filename of the camera path to render.
    camera_path_filename: Path = Path("camera_path.json")
    # Name of the output file.
    output_path: Path = Path("renders/output.mp4")
    # How long the video should be.
    seconds: float = 8.0
    # How to save output data.
    output_format: Literal["images", "video"] = "video"
    # Specifies number of rays per chunk during eval.
    eval_num_rays_per_chunk: Optional[int] = None


    def render_sptial_view(self, camera: Cameras,steps: int = 30 ,rots: int = 2, zrate: float = 1,radius: Optional[float] = None):

        N_per_seg = 10
        new_c2ws = []
        ## 每次间隔4帧，选取首帧和末帧，转动 360 角度
        for i in range(0,len(camera)-1 //2,4):
            start_point = camera[i].camera_to_worlds[:,3].detach().cpu().numpy()
            end_idx = i + 3
            if end_idx > len(camera)-1:
                end_idx = len(camera)-1
            end_point = camera[end_idx].camera_to_worlds[:, 3].detach().cpu().numpy()

            new_z = np.linspace(start_point[2],end_point[2],N_per_seg)
            new_xyz = []
            for theta in np.linspace(0., 2*np.pi, N_per_seg + 1)[:-1]:
                x = radius * np.cos(theta) + (start_point[0] + end_point[0]) * 0.5
                y = radius * np.sin(-theta) + (start_point[1] + end_point[1]) * 0.5
                new_xyz.append(np.array([x,y]))
            new_xyz = np.concatenate([np.array(new_xyz),new_z[...,None]],axis=1)
            camera_pose = np.eye(4)
            camera_pose = camera_pose[None,...].repeat(repeats=N_per_seg,axis = 0)
            camera_pose[:,:3,:3] = camera[i].camera_to_worlds[:3,:3].detach().cpu().numpy()
            camera_pose[:,:3,3] = new_xyz
            new_c2ws.append(camera_pose)

        new_c2ws = torch.tensor(new_c2ws).reshape(-1,4,4).float() ##[B,N,4,4]---> [B*N,4,4]

        return Cameras(
            fx=camera[0].fx[0],
            fy=camera[0].fy[0],
            cx=camera[0].cx[0],
            cy=camera[0].cy[0],
            height=camera[0].height,
            width=camera[0].width,
            distortion_params=camera[0].distortion_params,
            camera_type=camera[0].camera_type,
            camera_to_worlds=new_c2ws[:,:3,:4],
        )

    def generate_spiral_path(self,cameras: Cameras, bounds, n_frames=40, n_rots=2, zrate=.5):
        """Calculates a forward facing spiral path for rendering."""
        # Find a reasonable 'focus depth' for this dataset as a weighted average
        # of conservative near and far bounds in disparity space.
        NEAR_STRETCH = .9  # Push forward near bound for forward facing render path.
        FAR_STRETCH = 5.  # Push back far bound for forward facing render path.
        FOCUS_DISTANCE = .75  # Relative weighting of near, far bounds for render path.

        poses = [cameras[i].camera_to_worlds.detach().cpu().numpy() for i in range(len(cameras))]
        poses = np.stack(poses)

        near_bound = bounds.min() * NEAR_STRETCH
        far_bound = bounds.max() * FAR_STRETCH
        # All cameras will point towards the world space point (0, 0, -focal).
        focal = 1 / (((1 - FOCUS_DISTANCE) / near_bound + FOCUS_DISTANCE / far_bound))

        # Get radii for spiral path using 90th percentile of camera positions.
        positions = poses[:, :3, 3]
        radii = np.percentile(np.abs(positions), 90, 0)
        radii = np.concatenate([radii, [1.]])

        # Generate poses for spiral path.
        render_poses = []
        cam2world = self.average_pose(poses)
        up = poses[:, :3, 1].mean(0)
        for theta in np.linspace(0., 2. * np.pi * n_rots, n_frames, endpoint=False):
            t = radii * [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]
            position = cam2world @ t
            lookat = cam2world @ [0, 0, -focal, 1.]
            z_axis = position - lookat
            render_poses.append(self.viewmatrix(z_axis, up, position))
        render_poses = np.stack(render_poses, axis=0).astype(np.float32)
        render_poses = torch.from_numpy(render_poses).to(cameras[0].camera_to_worlds.device)

        return Cameras(
                fx=cameras[0].fx[0],
                fy=cameras[0].fy[0],
                cx=cameras[0].cx[0],
                cy=cameras[0].cy[0],
                height=cameras[0].height,
                width=cameras[0].width,
                distortion_params=cameras[0].distortion_params,
                camera_type=cameras[0].camera_type,
                camera_to_worlds=render_poses[:,:3,:4],
            )

    def render_yaw_roam(self,camera: Cameras):
        c2w = camera.camera_to_worlds.detach().cpu().numpy()
        tt = c2w[:3, 3]
        Rot = c2w[:3, :3]
        new_poses = []
        for theta in np.linspace(-0.5 * np.pi, 0.5 * np.pi, 10 + 1)[:-1]:
            theta = -theta
            R = np.array([[np.cos(theta), 0, np.sin(theta)],
                          [0, 1, 0],
                          [-np.sin(theta), 0, np.cos(theta)]])
            new_R = np.dot(Rot, R)
            new_pose = np.eye(4)
            new_pose[:3, :3] = new_R
            new_pose[:3, -1] = tt
            new_poses.append(new_pose)
        new_poses = np.stack(new_poses).astype(np.float32)
        new_poses = torch.from_numpy(new_poses)
        return Cameras(
            fx=camera.fx[0],
            fy=camera.fy[0],
            cx=camera.cx[0],
            cy=camera.cy[0],
            height=camera.height,
            width=camera.width,
            distortion_params=camera.distortion_params,
            camera_type=camera.camera_type,
            camera_to_worlds= new_poses[:, :3, :4],
        )

    def average_pose(self,poses):
        """New pose using average position, z-axis, and up vector of input poses."""
        position = poses[:, :3, 3].mean(0)
        z_axis = poses[:, :3, 2].mean(0)
        up = poses[:, :3, 1].mean(0)
        cam2world = self.viewmatrix(z_axis, up, position)
        return cam2world

    def viewmatrix(self,lookdir, up, position):
        """Construct lookat view matrix."""
        def normalize(x):
            """Normalization helper function."""
            return x / np.linalg.norm(x)

        vec2 = normalize(lookdir)
        vec0 = normalize(np.cross(up, vec2))
        vec1 = normalize(np.cross(vec2, vec0))
        m = np.stack([vec0, vec1, vec2, position], axis=1)
        return m



    def main(self) -> None:
        """Main function."""
        _, pipeline, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="test" if self.traj == "spiral" else "inference",
        )

        install_checks.check_ffmpeg_installed()

        seconds = self.seconds

        # TODO(ethan): use camera information from parsing args
        if self.traj == "spiral":
            # num_cameras = len(pipeline.datamanager.eval_dataset.filenames)
            # cameras = [pipeline.datamanager.eval_dataloader.get_camera(image_idx=i) for i in range(num_cameras)]
            cameras = pipeline.datamanager.train_dataset.cameras
            # TODO(ethan): pass in the up direction of the camera
            # camera_path = get_spiral_path(camera_start, steps=30, radius=0.1)
            camera_path = self.render_sptial_view(cameras, steps=30, radius=0.01)
            # camera_path = self.generate_spiral_path(cameras,bounds =np.array([0,1]) )
        elif self.traj == "roam":
            camera = pipeline.datamanager.train_dataset.cameras[2]
            camera_path = self.render_yaw_roam(camera)
            seconds = 4.0
            self.output_format = "images"
        elif self.traj == "filename":
            with open(self.camera_path_filename, "r", encoding="utf-8") as f:
                camera_path = json.load(f)
            seconds = camera_path["seconds"]
            camera_path = get_path_from_json(camera_path)
        else:
            assert_never(self.traj)

        '将相机的参数信息，送至这个函数进行渲染'
        # self.rendered_output_names = ['depth']
        # _render_trajectory_video(pipeline,camera_path[:1,...],
        #     output_filename=self.output_path,
        #     rendered_output_names=self.rendered_output_names,
        #     rendered_resolution_scaling_factor=1.0 / 2,
        #     seconds=seconds,
        #     output_format= "images",
        # )
        _render_trajectory_video(
            pipeline,
            camera_path,
            output_filename=self.output_path,
            rendered_output_names=self.rendered_output_names,
            rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
            seconds=seconds,
            output_format=self.output_format,
        )


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(RenderTrajectory).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(RenderTrajectory)  # noqa
