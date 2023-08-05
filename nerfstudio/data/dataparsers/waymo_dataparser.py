# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Optional, Type

import numpy as np
import torch
from PIL import Image
import cv2 as cv
from rich.console import Console
from typing_extensions import Literal

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.io import load_from_json
from nerfstudio.utils import plotly_utils as vis

CONSOLE = Console(width=120)
MAX_AUTO_RESOLUTION = 1600
from nerfstudio.data.utils.annotation_3d import Annotation3D, global2local,id2label


@dataclass
class WaymoDataParserConfig(DataParserConfig):
    """Blender dataset parser config"""

    _target: Type = field(default_factory=lambda: Waymo)
    """target class to instantiate"""
    data: Path = Path("data/nerfstudio/poster")
    """Directory specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    downscale_factor: Optional[int] = None
    """How much to downscale images. If not set, images are chosen such that the max dimension is <1600px."""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    orientation_method: Literal["pca", "up", "none"] = "none"
    """The method to use for orientation."""
    center_poses: bool = True
    """Whether to center the poses."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    train_split_percentage: float = 0.9
    """The percent of images to use for training. The remaining images are for eval."""
    annotation_3d = None
    use_fisheye: bool = False
    """use fisheye """
    include_semantics: bool = True
    """whether or not to include loading of semantics data"""
    mannual_assigned = True


@dataclass
class Waymo(DataParser):
    """Waymo Dataset
    Some of this code comes from https://github.com/yenchenlin/nerf-pytorch/blob/master/load_blender.py#L37.
    """

    """Nerfstudio DatasetParser"""

    config: WaymoDataParserConfig
    downscale_factor: Optional[int] = None

    def _generate_dataparser_outputs(self, split="train"):
        # pylint: disable=too-many-statements

        meta = load_from_json(self.config.data / "transforms.json")
        image_filenames = []
        mask_filenames = []
        poses = []
        num_skipped_image_filenames = 0
        img_index = []

        fx_fixed = "fl_x" in meta
        fy_fixed = "fl_y" in meta
        cx_fixed = "cx" in meta
        cy_fixed = "cy" in meta
        height_fixed = "h" in meta
        width_fixed = "w" in meta
        distort_fixed = False
        for distort_key in ["k1", "k2", "k3", "p1", "p2"]:
            if distort_key in meta:
                distort_fixed = True
                break
        fx = []
        fy = []
        cx = []
        cy = []
        height = []
        width = []
        distort = []
        intristic = []


        for frame in meta["frames"]:
            filepath = PurePath(frame["file_path"])
            fname = self._get_fname(filepath)
            if not fname.exists():
                num_skipped_image_filenames += 1
                continue

            # if not fx_fixed:
            #     assert "fl_x" in frame, "fx not specified in frame"
            #     fx.append(float(frame["fl_x"]))
            # if not fy_fixed:
            #     assert "fl_y" in frame, "fy not specified in frame"
            #     fy.append(float(frame["fl_y"]))
            # if not cx_fixed:
            #     assert "cx" in frame, "cx not specified in frame"
            #     cx.append(float(frame["cx"]))
            # if not cy_fixed:
            #     assert "cy" in frame, "cy not specified in frame"
            #     cy.append(float(frame["cy"]))
            # if not height_fixed:
            #     assert "h" in frame, "height not specified in frame"
            #     height.append(int(frame["h"]))
            # if not width_fixed:
            #     assert "w" in frame, "width not specified in frame"
            #     width.append(int(frame["w"]))
            if not distort_fixed:
                distort.append(
                    camera_utils.get_distortion_params(
                        k1=float(meta["k1"]) if "k1" in meta else 0.0,
                        k2=float(meta["k2"]) if "k2" in meta else 0.0,
                        k3=float(meta["k3"]) if "k3" in meta else 0.0,
                        k4=float(meta["k4"]) if "k4" in meta else 0.0,
                        p1=float(meta["p1"]) if "p1" in meta else 0.0,
                        p2=float(meta["p2"]) if "p2" in meta else 0.0,
                    )
                )

            image_filenames.append(fname)
            intrinsics = torch.tensor(frame["intrinsics"])
            fx.append(intrinsics[0, 0])
            fy.append(intrinsics[1, 1])
            cx.append(intrinsics[0, 2])
            cy.append(intrinsics[1, 2])
            poses.append(np.array(frame["transform_matrix"]))
            if "mask_path" in frame:
                mask_filepath = PurePath(frame["mask_path"])
                mask_fname = self._get_fname(mask_filepath, downsample_folder_prefix="masks_")
                mask_filenames.append(mask_fname)
            if "leader_board" in meta and meta['leader_board']:
                index = str(fname).split('/')[-1]
                img_index.append(index)
        if num_skipped_image_filenames >= 0:
            CONSOLE.log(f"Skipping {num_skipped_image_filenames} files in dataset split {split}.")
        assert (
                len(image_filenames) != 0
        ), """
           No image files found. 
           You should check the file_paths in the transforms.json file to make sure they are correct.
           """
        assert len(mask_filenames) == 0 or (
                len(mask_filenames) == len(image_filenames)
        ), """
           Different number of image and mask filenames.
           You should check that mask_path is specified for every frame (or zero frames) in transforms.json.
           """

        # filter image_filenames and poses based on train/eval split percentage
        num_images = len(image_filenames)
        i_all = np.arange(num_images)
        ## 50% dropout Setting
        if "dropout" in meta and meta['dropout']:
            i_train = []
            for i in range(0, num_images, 2):
                if i % 4 == 0:
                    i_train.extend([i, i + 1])
            i_train = np.array(i_train)
            num_train_images = len(i_train)
            i_eval = np.setdiff1d(i_all, i_train)[:-2]  # Demo kitti360
        elif "leader_board" in meta and meta['leader_board']:
            num_eval_images = int(meta['num_test'])
            i_train = i_all[:(num_images - num_eval_images)]
            i_eval = np.setdiff1d(i_all, i_train)
            # i_train = np.arange(10,79)
            # i_eval = np.arange(83,99)
        elif self.config.mannual_assigned:
            i_eval = np.array([20, 40])
            i_train = np.setdiff1d(i_all, i_eval)
        else:
            self.config.train_split_percentage = 0.8
            num_train_images = math.ceil(num_images * self.config.train_split_percentage)
            num_eval_images = num_images - num_train_images
            i_train = np.linspace(
                0, num_images - 1, num_train_images, dtype=int
            )  # equally spaced training images starting and ending at 0 and num_images-1
            i_eval = np.setdiff1d(i_all, i_train)  # eval images are the remaining images

        if split == "train":
            indices = i_train
            print(f"Train View:  {indices}\n" + f"Train View Num: {len(i_train)}")
        elif split in ["val", "test"]:
            indices = i_eval
            print(f"Test View: {indices}" + f"Test View Num{len(i_eval)}")
        else:
            raise ValueError(f"Unknown dataparser split {split}")

        if "orientation_override" in meta:
            orientation_method = meta["orientation_override"]
            CONSOLE.log(f"[yellow] Dataset is overriding orientation method to {orientation_method}")
        else:
            orientation_method = self.config.orientation_method

        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        fx = torch.stack(fx)
        fy = torch.stack(fy)
        cx = torch.stack(cx)
        cy = torch.stack(cy)

        diff_mean_poses = torch.mean(poses[:, :3, -1], dim=0)
        poses, _ = camera_utils.auto_orient_and_center_poses(
            poses,
            method=orientation_method,
            center_poses=self.config.center_poses,
        )

        # Scale poses[translation]
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= torch.max(torch.abs(poses[:, :3, 3]))

        poses[:, :3, 3] *= scale_factor * self.config.scale_factor
        poses[:, 0:3, 1:3] *= -1

        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        image_filenames = [image_filenames[i] for i in indices]
        mask_filenames = [mask_filenames[i] for i in indices] if len(mask_filenames) > 0 else []
        poses = poses[indices]


        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        if "camera_model" in meta:
            camera_type = CAMERA_MODEL_TO_TYPE[meta["camera_model"]]
        else:
            camera_type = CameraType.PERSPECTIVE

        idx_tensor = torch.tensor(indices, dtype=torch.long)
        ## becasue the waymo camera have the different intristics
        fx = fx[indices]
        fy = fy[indices]
        cx = cx[indices]
        cy = cy[indices]
        height = torch.tensor(1280, dtype=torch.int32)
        width = torch.tensor(1920, dtype=torch.int32)
        if distort_fixed:
            distortion_params = camera_utils.get_distortion_params(
                k1=float(meta["k1"]) if "k1" in meta else 0.0,
                k2=float(meta["k2"]) if "k2" in meta else 0.0,
                k3=float(meta["k3"]) if "k3" in meta else 0.0,
                k4=float(meta["k4"]) if "k4" in meta else 0.0,
                p1=float(meta["p1"]) if "p1" in meta else 0.0,
                p2=float(meta["p2"]) if "p2" in meta else 0.0,
            )
        else:
            distortion_params = torch.stack(distort, dim=0)[idx_tensor]

        ## Where Use bounding box ,if Used,need to read instance image and bbx
        if "use_bbx" in meta and meta['use_bbx']:
            print(f"BBx Abled!")
            bbx2world = np.array(meta["bbx2w"])
            bbox = []
            instance_imgs = []
            data_dir = '/data/datasets/KITTI-360/'
            instance_path = os.path.join(data_dir, 'data_2d_semantics', 'train',
                                         '2013_05_28_drive_0000_sync', 'image_00/instance')
            ## 这里的400 是针对我的数据集的
            for idx in range(400, 400 + 40, 1):
                img_file = os.path.join(instance_path, "{:010d}.png".format(idx))
                instance_imgs.append(cv.imread(img_file, -1))

            ## 这里的 bbx 的坐标都是相当于第一帧相机来说的（第一帧相机是世界坐标系）
            bbx_root = os.path.join(data_dir, 'data_3d_bboxes')
            self.annotation_3d = Annotation3D(os.path.join(bbx_root, 'train'), '2013_05_28_drive_0000_sync')
            all_bbxes = self.load_bbx(instance_imgs=instance_imgs, bbx2w=bbx2world,
                                      scale=scale_factor * self.config.scale_factor,
                                      diff_centor_translation=diff_mean_poses)

            ''' 将3D 的bbx 投影到2D 验证bbx 是否正确'''
            # all_bbxes = self.load_bbx_for_test(instance_imgs=instance_imgs, bbx2w=bbx2world)
            # self.project2Dbbx(bbx=all_bbxes, img_idx=0, f=fx, cx=cx, cy=cy, img_file=image_filenames)
            cameras = Cameras(
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                distortion_params=distortion_params,
                height=height,
                width=width,
                camera_to_worlds=poses[:, :3, :4],
                camera_type=camera_type,
                bounding_box=all_bbxes,
                test_idx=i_eval,
                train_idx=i_train,
            )
        else:
            print(f"BBx Unabled!")
            cameras = Cameras(
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                distortion_params=distortion_params,
                height=height,
                width=width,
                camera_to_worlds=poses[:, :3, :4],
                camera_type=camera_type,
            )

        ## semantic
        if self.config.include_semantics:
            empty_path = Path()
            replace_this_path = str("waymodata_noisy/")
            with_this_path = str("waymodata_noisy/semantics/pred/")
            seg_filenames = [
                Path(str(image_filename).replace(replace_this_path, with_this_path))
                for image_filename in image_filenames
            ]

            # ## rename to suit waymo noisy
            # base_path = '/data/smiao/datasets/waymodata_noisy/semantics/pred/'
            # #base_path = '/data/smiao/datasets/waymodata/semantics/pred/'
            # new_segfilenames = []
            # for i in range(len(seg_filenames)):  # 从000.png到010.png
            #     new_filename = f'{i:08d}_pred.png'  # 格式化为八位数的文件名
            #     new_path = base_path + new_filename
            #     new_segfilenames.append(new_path)

        assert self.downscale_factor is not None
        cameras.rescale_output_resolution(scaling_factor=1.0 / self.downscale_factor)

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames if len(mask_filenames) > 0 else None,
            metadata={"semantics": seg_filenames} if self.config.include_semantics else {},
        )

        ## add fisheye param  如果要加 fisheye 记得在json 文件里 加上use_fisheye 的选项
        if self.config.use_fisheye and 'use_fisheye' in meta:
            fisheye_poses, fisheye_imgs, fisheye_meta, mask = self.load_fish_eye_param(diff_mean_poses=diff_mean_poses,
                                                                                       scale_factor=scale_factor,
                                                                                       config_data=self.config.data)
            fisheye_dict = {
                'pose': fisheye_poses,
                'imgs': fisheye_imgs,
                'meta': fisheye_meta,
                'mask': mask,
            }
            dataparser_outputs.fisheye_dict = fisheye_dict
        return dataparser_outputs

    def load_fish_eye_param(self, diff_mean_poses, scale_factor, config_data=None, use_mask=True):

        meta = load_from_json(config_data / Path("transforms_fisheye.json"))
        image_filenames = []
        poses = []
        images = []

        for frame in meta["frames"]:
            filepath = PurePath(frame["file_path"])
            fname = Path(config_data) / filepath
            pil_image = Image.open(fname)
            image = np.array(pil_image, dtype="uint8") / 255.0
            images.append(image)
            image_filenames.append(fname)
            poses.append(np.array(frame["transform_matrix"]))

        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        images = torch.from_numpy(np.array(images).astype(np.float32))

        poses[:, :3, -1] -= diff_mean_poses
        poses[:, :3, 3] *= scale_factor

        fisheye_mask = cv.imread(str(config_data) + "/mask.png") / 255.0
        fisheye_mask = fisheye_mask.astype(np.int32)[..., 0]
        return poses, images, meta, fisheye_mask

    def _get_fname(self, filepath: PurePath, downsample_folder_prefix="images_") -> Path:
        """Get the filename of the image file.
        downsample_folder_prefix can be used to point to auxillary image data, e.g. masks
        """

        if self.downscale_factor is None:
            if self.config.downscale_factor is None:
                test_img = Image.open(self.config.data / filepath)
                h, w = test_img.size
                max_res = max(h, w)
                df = 0
                while True:
                    if (max_res / 2 ** (df)) < MAX_AUTO_RESOLUTION:
                        break
                    if not (self.config.data / f"{downsample_folder_prefix}{2 ** (df + 1)}" / filepath.name).exists():
                        break
                    df += 1

                self.downscale_factor = 2 ** df
                CONSOLE.log(f"Auto image downscale factor of {self.downscale_factor}")
            else:
                self.downscale_factor = self.config.downscale_factor

        if self.downscale_factor > 1:
            return self.config.data / f"{downsample_folder_prefix}{self.downscale_factor}" / filepath.name
        return self.config.data / filepath
