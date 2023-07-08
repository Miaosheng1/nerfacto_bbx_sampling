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
Code for sampling pixels.
"""

import random
from typing import Dict
import cv2
import torch

from nerfstudio.utils.images import BasicImages


def collate_image_dataset_batch(batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False):
    """
    Operates on a batch of images and samples pixels to use for generating rays.
    Returns a collated batch which is input to the Graph.
    It will sample only within the valid 'mask' if it's specified.

    Args:
        batch: batch of images to sample from
        num_rays_per_batch: number of rays to sample per batch
        keep_full_image: whether or not to include a reference to the full image in returned batch
    """
    device = batch["image"].device
    num_images, image_height, image_width, _ = batch["image"].shape

    # only sample within the mask, if the mask is in the batch
    if "mask" in batch:
        nonzero_indices = torch.nonzero(batch["mask"][..., 0].to(device), as_tuple=False)
        chosen_indices = random.sample(range(len(nonzero_indices)), k=num_rays_per_batch)
        indices = nonzero_indices[chosen_indices]
    else:
        indices = torch.floor(
            torch.rand((num_rays_per_batch, 3), device=device)
            * torch.tensor([num_images, image_height, image_width], device=device)
        ).long()

    ## c 是 image_indice (4096,)  y 是 image_height (4096,) x 是image_width (4096,)
    c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
    ## 使用 字典生成式 得到 对应的 indice 的 rgb 真值
    collated_batch = {
        key: value[c, y, x]
        for key, value in batch.items()
        if key not in ("image_idx", "src_imgs", "src_idxs", "sparse_sfm_points") and value is not None
    }

    assert collated_batch["image"].shape == (num_rays_per_batch, 3), collated_batch["image"].shape

    if "sparse_sfm_points" in batch:
        collated_batch["sparse_sfm_points"] = batch["sparse_sfm_points"].images[c[0]]

    # Needed to correct the random indices to their actual camera idx locations.
    indices[:, 0] = batch["image_idx"][c]
    collated_batch["indices"] = indices  # with the abs camera indices

    if keep_full_image:
        collated_batch["full_image"] = batch["image"]

    return collated_batch

def collect_image_patch(batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False,patch_size=32):
    device = batch["image"].device
    num_images, image_height, image_width, _ = batch["image"].shape
    patch_num = num_rays_per_batch // (patch_size*patch_size)
    patch_height = image_height - patch_size
    patch_width = image_width - patch_size

    ## indices is the left_top corner point
    indices = torch.floor(
        torch.rand((patch_num, 3), device=device)
        * torch.tensor([num_images, patch_height, patch_width], device=device)
    ).long()

    c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))

    y_list = []
    x_list = []
    for i in range(patch_num):
        pixel_y,pixel_x = torch.meshgrid(torch.linspace(y[i],y[i]+patch_size-1,patch_size),torch.linspace(x[i],x[i]+patch_size-1,patch_size))
        y_list.append(pixel_y)
        x_list.append(pixel_x)
    y = torch.concat(y_list).flatten().long()
    x = torch.concat(x_list).flatten().long()
    c = c.unsqueeze(-1).repeat(1,patch_size*patch_size).flatten()

    collated_batch = {
        key : value[c,y,x]
        for key, value in batch.items()
        if key not in ("image_idx", "src_imgs", "src_idxs", "sparse_sfm_points") and value is not None
    }

    assert collated_batch["image"].shape == (num_rays_per_batch, 3), collated_batch["image"].shape

    if "sparse_sfm_points" in batch:
        collated_batch["sparse_sfm_points"] = batch["sparse_sfm_points"].images[c[0]]

    # Needed to correct the random indices to their actual camera idx locations.
    ## batch["image_idx"][c] y x
    # indices[:, 0] = batch["image_idx"][c]
    collated_batch["indices"] = torch.stack([batch["image_idx"][c].to(device),y,x],dim=0).permute(1,0)  # with the abs camera indices

    if keep_full_image:
        collated_batch["full_image"] = batch["image"]
    # ## debug image
    # a = collated_batch['image'].reshape(patch_num,patch_size,patch_size,-1)*255.0
    # cv2.imwrite("patch1.png",a[1,:,:,[2,1,0]].detach().cpu().numpy())

    return collated_batch


def collate_image_dataset_batch_list(batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False):
    """
    Does the same as collate_image_dataset_batch, except it will operate over a list of images / masks inside
    a list.

    We will use this with the intent of DEPRECIATING it as soon as we find a viable alternative.
    The intention will be to replace this with a more efficient implementation that doesn't require a for loop, but
    since pytorch's ragged tensors are still in beta (this would allow for some vectorization), this will do

    Args:
        batch: batch of images to sample from
        num_rays_per_batch: number of rays to sample per batch
        keep_full_image: whether or not to include a reference to the full image in returned batch
    """

    device = batch["image"][0].device
    num_images = len(batch["image"])

    # only sample within the mask, if the mask is in the batch
    all_indices = []
    all_images = []
    all_fg_masks = []

    if "mask" in batch:
        num_rays_in_batch = num_rays_per_batch // num_images
        for i in range(num_images):
            if i == num_images - 1:
                num_rays_in_batch = num_rays_per_batch - (num_images - 1) * num_rays_in_batch
            # nonzero_indices = torch.nonzero(batch["mask"][i][..., 0], as_tuple=False)
            nonzero_indices = batch["mask"][i]

            chosen_indices = random.sample(range(len(nonzero_indices)), k=num_rays_in_batch)
            indices = nonzero_indices[chosen_indices]
            indices = torch.cat([torch.full((num_rays_in_batch, 1), i, device=device), indices], dim=-1)
            all_indices.append(indices)
            all_images.append(batch["image"][i][indices[:, 1], indices[:, 2]])
            if "fg_mask" in batch:
                all_fg_masks.append(batch["fg_mask"][i][indices[:, 1], indices[:, 2]])

    else:
        num_rays_in_batch = num_rays_per_batch // num_images
        for i in range(num_images):
            image_height, image_width, _ = batch["image"][i].shape
            if i == num_images - 1:
                num_rays_in_batch = num_rays_per_batch - (num_images - 1) * num_rays_in_batch
            indices = torch.floor(
                torch.rand((num_rays_in_batch, 3), device=device)
                * torch.tensor([1, image_height, image_width], device=device)
            ).long()
            indices[:, 0] = i
            all_indices.append(indices)
            all_images.append(batch["image"][i][indices[:, 1], indices[:, 2]])
            if "fg_mask" in batch:
                all_fg_masks.append(batch["fg_mask"][i][indices[:, 1], indices[:, 2]])

    indices = torch.cat(all_indices, dim=0)

    c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
    collated_batch = {
        key: value[c, y, x]
        for key, value in batch.items()
        if key != "image_idx"
        and key != "image"
        and key != "mask"
        and key != "fg_mask"
        and key != "sparse_pts"
        and value is not None
    }

    collated_batch["image"] = torch.cat(all_images, dim=0)
    if len(all_fg_masks) > 0:
        collated_batch["fg_mask"] = torch.cat(all_fg_masks, dim=0)

    if "sparse_pts" in batch:
        rand_idx = random.randint(0, num_images - 1)
        collated_batch["sparse_pts"] = batch["sparse_pts"][rand_idx]

    assert collated_batch["image"].shape == (num_rays_per_batch, 3), collated_batch["image"].shape

    # Needed to correct the random indices to their actual camera idx locations.
    indices[:, 0] = batch["image_idx"][c]
    collated_batch["indices"] = indices  # with the abs camera indices

    if keep_full_image:
        collated_batch["full_image"] = batch["image"]

    return collated_batch


class PixelSampler:  # pylint: disable=too-few-public-methods
    """Samples 'pixel_batch's from 'image_batch's.

    Args:
        num_rays_per_batch: number of rays to sample per batch
        keep_full_image: whether or not to include a reference to the full image in returned batch
    """

    def __init__(self, num_rays_per_batch: int, keep_full_image: bool = False) -> None:
        self.num_rays_per_batch = num_rays_per_batch
        self.keep_full_image = keep_full_image

    def set_num_rays_per_batch(self, num_rays_per_batch: int):
        """Set the number of rays to sample per batch.

        Args:
            num_rays_per_batch: number of rays to sample per batch
        """
        self.num_rays_per_batch = num_rays_per_batch

    def sample(self, image_batch: Dict,step):
        """Sample an image batch and return a pixel batch.

        Args:
            image_batch: batch of images to sample from
        """
        if isinstance(image_batch["image"], list):
            image_batch = dict(image_batch.items())  # copy the dictioary so we don't modify the original
            pixel_batch = collate_image_dataset_batch_list(
                image_batch, self.num_rays_per_batch, keep_full_image=self.keep_full_image
            )
        elif isinstance(image_batch["image"], BasicImages):
            image_batch = dict(image_batch.items())  # copy the dictioary so we don't modify the original
            image_batch["image"] = image_batch["image"].images
            if "mask" in image_batch:
                image_batch["mask"] = image_batch["mask"].images
            # TODO clean up
            if "fg_mask" in image_batch:
                image_batch["fg_mask"] = image_batch["fg_mask"].images
            if "sparse_pts" in image_batch:
                image_batch["sparse_pts"] = image_batch["sparse_pts"].images
            pixel_batch = collate_image_dataset_batch_list(
                image_batch, self.num_rays_per_batch, keep_full_image=self.keep_full_image
            )
        elif isinstance(image_batch["image"], torch.Tensor):
            # if step > 20000 and step % 10 == 0:  ## After 20000 step we add lpips loss to optimize
            #     pixel_batch = collect_image_patch(image_batch, self.num_rays_per_batch,keep_full_image=self.keep_full_image)
            # else:
            #     pixel_batch = collate_image_dataset_batch(image_batch, self.num_rays_per_batch, keep_full_image=self.keep_full_image)

            pixel_batch = collate_image_dataset_batch(image_batch, self.num_rays_per_batch,keep_full_image=self.keep_full_image)

        else:
            raise ValueError("image_batch['image'] must be a list or torch.Tensor")
        return pixel_batch

    def sample_fisheye_with_patch(self,image_batch,step,patch_size = 1):
        num_images,image_width,image_height,_ = image_batch.shape
        device = image_batch.device
        num_patch = self.num_rays_per_batch // (patch_size * patch_size)

        patch_height = image_height - patch_size
        patch_width = image_width - patch_size

        ## indices is the left_top corner point
        indices = torch.floor(
            torch.rand((num_patch, 1)).to(device)
            * torch.tensor([num_images], device=device)
        ).long()
        c = indices.flatten()

        y_list = []
        x_list = []
        for i in range(num_patch):
            mask = image_batch[c[i],:,:,-1]
            region_mask = 0
            while region_mask < patch_size*patch_size:
                nonzero_indices = torch.nonzero(mask, as_tuple=False)
                chosen_indices = random.sample(range(len(nonzero_indices)), k=1)
                indices = nonzero_indices[chosen_indices]
                ## only sample in mask region
                if indices[0,0] + patch_size - 1 < image_width and indices[0,1] + patch_size-1 < image_height:
                    pixel_y, pixel_x = torch.meshgrid(torch.linspace(indices[0,0] , indices[0,0]  + patch_size - 1, patch_size),
                                                      torch.linspace(indices[0,1], indices[0,1] + patch_size - 1, patch_size))
                    region_mask = mask[pixel_y.flatten().long(),pixel_x.flatten().long()].sum()


            y_list.append(pixel_y)
            x_list.append(pixel_x)
        y = torch.concat(y_list).flatten().long().to(device)
        x = torch.concat(x_list).flatten().long().to(device)
        c = c.unsqueeze(-1).repeat(1, patch_size * patch_size).flatten()

        ray_data = image_batch[c,y,x]
        collated_batch = {
            "ray_o": ray_data[...,0:3],
            "ray_d": ray_data[..., 3:6],
            "image":ray_data[...,6:9],
            "indices":torch.stack([c,y,x],dim=0).permute(1,0)
        }

        # ## debug image
        # a = collated_batch['image'].reshape(num_patch,patch_size,patch_size,-1)*255.0
        # cv2.imwrite("patch1.png",a[0,:,:,[2,1,0]].detach().cpu().numpy())

        return collated_batch

    def sample_fisheye(self, image_batch, step, patch_size=1):
        num_images, image_width, image_height, _ = image_batch.shape
        mask = image_batch[:,:,:,-1]
        nonzero_indices = torch.nonzero(mask, as_tuple=False)
        chosen_indices = random.sample(range(len(nonzero_indices)), k=self.num_rays_per_batch)
        indices = nonzero_indices[chosen_indices]
        c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))

        ray_data = image_batch[c, y, x].to('cuda')
        collated_batch = {
            "ray_o": ray_data[..., 0:3],
            "ray_d": ray_data[..., 3:6],
            "image": ray_data[..., 6:9],
            "indices": torch.stack([c, y, x], dim=0).permute(1, 0)
        }

        return collated_batch


def collate_image_dataset_batch_equirectangular(batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False):
    """
    Operates on a batch of equirectangular images and samples pixels to use for
    generating rays. Rays will be generated uniformly on the sphere.
    Returns a collated batch which is input to the Graph.
    It will sample only within the valid 'mask' if it's specified.

    Args:
        batch: batch of images to sample from
        num_rays_per_batch: number of rays to sample per batch
        keep_full_image: whether or not to include a reference to the full image in returned batch
    """
    # TODO(kevinddchen): make more DRY
    device = batch["image"].device
    num_images, image_height, image_width, _ = batch["image"].shape

    # only sample within the mask, if the mask is in the batch
    if "mask" in batch:
        # TODO(kevinddchen): implement this
        raise NotImplementedError("Masking not implemented for equirectangular images.")

    # We sample theta uniformly in [0, 2*pi]
    # We sample phi in [0, pi] according to the PDF f(phi) = sin(phi) / 2.
    # This is done by inverse transform sampling.
    # http://corysimon.github.io/articles/uniformdistn-on-sphere/
    num_images_rand = torch.rand(num_rays_per_batch, device=device)
    phi_rand = torch.acos(1 - 2 * torch.rand(num_rays_per_batch, device=device)) / torch.pi
    theta_rand = torch.rand(num_rays_per_batch, device=device)
    indices = torch.floor(
        torch.stack((num_images_rand, phi_rand, theta_rand), dim=-1)
        * torch.tensor([num_images, image_height, image_width], device=device)
    ).long()

    c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
    collated_batch = {key: value[c, y, x] for key, value in batch.items() if key != "image_idx" and value is not None}

    assert collated_batch["image"].shape == (num_rays_per_batch, 3), collated_batch["image"].shape

    # Needed to correct the random indices to their actual camera idx locations.
    indices[:, 0] = batch["image_idx"][c]
    collated_batch["indices"] = indices  # with the abs camera indices

    if keep_full_image:
        collated_batch["full_image"] = batch["image"]

    return collated_batch


class EquirectangularPixelSampler(PixelSampler):  # pylint: disable=too-few-public-methods
    """Samples 'pixel_batch's from 'image_batch's. Assumes images are
    equirectangular and the sampling is done uniformly on the sphere.

    Args:
        num_rays_per_batch: number of rays to sample per batch
        keep_full_image: whether or not to include a reference to the full image in returned batch
    """

    # overrides base method
    def sample(self, image_batch: Dict):

        pixel_batch = collate_image_dataset_batch_equirectangular(
            image_batch, self.num_rays_per_batch, keep_full_image=self.keep_full_image
        )
        return pixel_batch
