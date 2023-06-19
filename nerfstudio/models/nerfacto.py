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

"""
NeRF implementation that combines many recent advancements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import numpy as np
import torch
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.fields.nerfacto_field import TCNNNerfactoField
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
)
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    NormalsRenderer,
    RGBRenderer,
)
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.utils.colors import get_color
from scipy.interpolate import RegularGridInterpolator
import torch.nn.functional as F
from nerfstudio.model_components.ray_samplers import PDFSampler, UniformSampler,UniformLinDispPiecewiseSampler

@dataclass
class NerfactoModelConfig(ModelConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: NerfactoModel)
    near_plane: float = 0.05
    """How far along the ray to start sampling."""
    far_plane: float = 1000.0
    """How far along the ray to stop sampling."""
    background_color: Literal["random", "last_sample", "white", "black"] = "last_sample"
    """Whether to randomize the background color."""
    num_levels: int = 16
    """Number of levels of the hashmap for the base mlp."""
    max_res: int = 1024
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size: int = 19
    """Size of the hashmap for the base mlp"""
    num_proposal_samples_per_ray: Tuple[int] = (256, 96)
    """Number of samples per ray for the proposal network."""
    # num_nerf_samples_per_ray: int = 32
    num_nerf_samples_per_ray: int = 48
    # num_nerf_samples_per_ray: int = 64 (效果PSNR 略微降低)
    """Number of samples per ray for the nerf network."""
    proposal_update_every: int = 5
    """Sample every n steps after the warmup"""
    proposal_warmup: int = 5000
    """Scales n from 1 to proposal_update_every over this many steps"""
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""
    use_same_proposal_network: bool = False
    """Use the same proposal network. Otherwise use different ones."""
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 64},
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 256},
        ]
    )
    """Arguments for the proposal density fields."""
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    distortion_loss_mult: float = 0.002
    """Distortion loss multiplier."""
    orientation_loss_mult: float = 0.0001
    """Orientation loss multipier on computed noramls."""
    pred_normal_loss_mult: float = 0.001
    alpha_loss_mult: float = 1.0
    voxformer_coff: float = 0.5
    """Voxformer Coefficient init is 0.5"""
    """Predicted normal loss multiplier."""
    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""
    use_average_appearance_embedding: bool = False
    """Whether to use average appearance embedding or zeros for inference."""
    use_individual_appearance_embedding: bool = True
    """Whether to use individual appearance embedding or zeros for inference."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""
    predict_normals: bool = False
    """Whether to predict normals or not."""
    inference_dataset: Literal["off", "trainset", "testset"] = "off"
    """When in inference, which dataset will be loaded."""



class NerfactoModel(Model):
    """Nerfacto model

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: NerfactoModelConfig

    def populate_modules(self,voxformer_occupancy=None):
        """Set the fields and modules."""
        super().populate_modules()

        scene_contraction = SceneContraction(order=float("inf"))

        # Fields
        self.field = TCNNNerfactoField(
            self.scene_box.aabb,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            log2_hashmap_size=self.config.log2_hashmap_size,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            use_individual_appearance_embedding=self.config.use_individual_appearance_embedding,
            inference_dataset = self.config.inference_dataset
        )


        self.voxformer_occupancy = voxformer_occupancy
        self.Cam2Velo = [
            [0.04307104, -0.08829286, 0.99516293, 0.80439144],
            [-0.99900437, 0.00778461, 0.04392797, 0.29934896],
            [-0.01162549, -0.99606414, -0.08786967, -0.17702258],
        ]

        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = HashMLPDensityField(self.scene_box.aabb, spatial_distortion=scene_contraction, **prop_net_args)
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = HashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    **prop_net_args,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in self.proposal_networks])

            ## 为voxformer 添加一个可以输出 负值的 density
            self.neg_density_fn = network.get_neg_alpha

        # Samplers
        update_schedule = lambda step: np.clip(
            np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
            1,
            self.config.proposal_update_every,
        )
        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
        )

        ## PDF sampler used in NeRF
        self.initial_sampler = UniformLinDispPiecewiseSampler(single_jitter=False)
        self.sampler_pdf = PDFSampler(include_original=False, single_jitter=False,num_samples=128)


        # Collider
        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        # renderers
        background_color = (
            get_color(self.config.background_color)
            if self.config.background_color in set(["white", "black"])
            else self.config.background_color
        )
        self.renderer_rgb = RGBRenderer(background_color=background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method='expected')
        self.renderer_normals = NormalsRenderer()

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity()

        ## filename_index
        self.img_filename = None

        self.voxformer_step = 0

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["fields"] = list(self.field.parameters())
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                train_frac = np.clip(step / N, 0, 1)
                bias = lambda x, b: (b * x) / ((b - 1) * x + 1)
                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )
        return callbacks

    def get_outputs(self, ray_bundle: RayBundle):
        if self.voxformer_occupancy is not None:
            """ Nerfacto 的 Proposal Network 采样"""
            ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle,
                                                                                density_fns=self.density_fns,
                                                                                voxformer_fn=self.get_occupancy_with_voxformer,
                                                                                # neg_alpha_fn = self.neg_density_fn,
                                                                                )
            field_outputs = self.field(ray_samples, compute_normals=self.config.predict_normals)
            voxformer_alpha = self.get_occupancy_with_voxformer(ray_samples.frustums.get_positions())
            # neg_alpha = self.neg_density_fn(ray_samples)
            weights,alphas_dict = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY] , voxformer_alpha = voxformer_alpha,neg_alpha=None)
            # weights, alphas_dict = ray_samples.get_weights(torch.zeros_like(voxformer_alpha),
            #                                                voxformer_alpha=voxformer_alpha)
            t = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2

            if ray_samples.shape[0] < 5:
                print("Debug Fixed Ray")
                from pathlib import Path
                np.save( Path("piexl_alphas.npy"), alphas_dict.get('alphas').detach().cpu().numpy())
                np.save( Path("piexl_weight.npy"), weights.detach().cpu().numpy())
                np.save( Path("pixel_t.npy"), t.detach().cpu().numpy())
                exit()

        else:
            ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
            field_outputs = self.field(ray_samples, compute_normals=self.config.predict_normals)
            weights,alphas_dict = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])

            t = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2

        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "weight":weights,
            "density":field_outputs[FieldHeadNames.DENSITY],
            "alphas":alphas_dict.get('alphas'),
            "voxformer_alpha":alphas_dict.get('voxformer_alpha'),
            "orign_alpha":alphas_dict.get('orign_alpha'),
            "t":t,
            # "neg_alpha":neg_alpha
            "alpha_loss":alphas_dict.get('alpha_loss'),
        }

        if self.config.predict_normals:
            outputs["normals"] = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            outputs["pred_normals"] = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)

        # These use a lot of GPU memory, so we avoid storing them for eval.
        if True or self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        return outputs


    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        image = batch["image"].to(self.device)
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        if self.training:
            metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        loss_dict["rgb_loss"] = self.rgb_loss(image, outputs["rgb"])
        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]
            if self.config.predict_normals:
                # orientation loss for computed normals
                loss_dict["orientation_loss"] = self.config.orientation_loss_mult * torch.mean(
                    outputs["rendered_orientation_loss"]
                )

                # ground truth supervision for normals
                loss_dict["pred_normal_loss"] = self.config.pred_normal_loss_mult * torch.mean(
                    outputs["rendered_pred_normal_loss"]
                )
            if self.voxformer_occupancy is not None:
                loss_dict["alpha_loss"] =  self.config.alpha_loss_mult * outputs['alpha_loss']
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:

        image = batch["image"].to(self.device)
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)


        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}

        # normals to RGB for visualization. TODO: use a colormap
        if "normals" in outputs:
            images_dict["normals"] = (outputs["normals"] + 1.0) / 2.0
        if "pred_normals" in outputs:
            images_dict["pred_normals"] = (outputs["pred_normals"] + 1.0) / 2.0

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        return metrics_dict, images_dict

    def get_occupancy_with_voxformer(
            self,
            positions = None,
    ):
        self.voxformer_step += 1
        scale = self.voxformer_occupancy["scale_factor"]
        diff_translation = self.voxformer_occupancy["diff_mean_poses"].to('cuda')
        voxformer_occupancy = self.voxformer_occupancy["voxformer_occupancy"]

        shape = positions.shape[:2]
        ## 将camera 系的 coor 转到 lidar 系，去查询 occupancy
        positions = positions/scale + diff_translation  ## 转到Cam1 坐标系
        positions = positions.detach().cpu().numpy().reshape(-1,3)
        positions = np.hstack((positions, np.ones((positions.shape[0], 1))))
        positions = np.dot(positions, np.transpose(self.Cam2Velo))

        """ 根据采样点的 position 去查询 voxformer 的 occupancy 初始值"""
        Occupancy = np.clip(voxformer_occupancy[..., -1], 0, 1)
        x_range = (0, 51.2)
        y_range = (-25.6, 25.6)
        z_range = (-2, 4.4)

        x = np.linspace(x_range[0], x_range[1], 256 + 1)[:-1] + 0.1
        y = np.linspace(y_range[0], y_range[1], 256 + 1)[:-1] + 0.1
        z = np.linspace(z_range[0], z_range[1], 32 + 1)[:-1] + 0.1

        ## Torch 实现
        voxformer_volume = Occupancy.transpose(2, 1, 0)
        voxformer_volume = torch.from_numpy(voxformer_volume)[None, None, ...].double()
        # positions = torch.from_numpy(pos)  相对于 padding_mode = border
        positions = torch.from_numpy(positions).to(self.device)
        x_normalized = 2 * (positions[..., 0] - x[0]) / (x[-1] - x[0]) - 1
        y_normalized = positions[..., 1] / y[-1]
        z_normalized = 2 * (positions[..., 2] - z[0]) / (z[-1] - z[0]) - 1
        grid = torch.stack([x_normalized, y_normalized, z_normalized], dim=-1).double()
        interpolated_values = F.grid_sample(voxformer_volume.to(self.device), grid[None, None, None, ...], mode='bilinear',
                                            align_corners=True,padding_mode='zeros').float()

        voxformer_coff = self.config.voxformer_coff
        return interpolated_values.view(*shape,-1) * voxformer_coff

    # def coff_anneal(self):
    #     fina_step = 10000
    #     if self.voxformer_step < fina_step:
    #         coff = 1 / -fina_step * self.voxformer_step + 1
    #     else:
    #         coff = 0
    #     return coff

