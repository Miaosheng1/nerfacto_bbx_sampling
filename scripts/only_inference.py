from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt

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
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional import structural_similarity_index_measure
from skimage.metrics import structural_similarity
from nerfstudio.cameras.camera_paths import get_path_from_json, get_spiral_path
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.configs.base_config import Config  # pylint: disable=unused-import
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import install_checks
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import ItersPerSecColumn
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse
import cv2
from nerfstudio.data.utils.label import id2label,labels,assigncolor
# from nerfstudio.data.utils.waymo_lable import id2label,labels,assigncolor

CONSOLE = Console(width=120)

class RenderDatasets():
    """Load a checkpoint, render the trainset and testset rgb,normal,depth, and save to the picture"""
    def __init__(self,parser_path):
        self.load_config = Path(parser_path.config)

        # self.rendered_output_names = ['rgb_fine', 'depth_fine']
        exp_method = str(self.load_config).split('/')[-3]
        if exp_method == 'nerfacto':
            self.rendered_output_names = ['rgb', 'depth',"semantics"]
        elif exp_method =='vanillanerf':
            self.rendered_output_names = ['rgb_fine', 'depth_fine']
        else:
            self.rendered_output_names = ['rgb', 'depth', 'normal']
        self.root_dir = Path('exp_psnr_' + exp_method)
        if self.root_dir.is_dir():
            os.system(f"rm -rf {self.root_dir}")
        self.task = parser_path.task
        self.is_leaderboard = parser_path.is_leaderboard
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity()

    def generate_errorMap(self,ssim,index):
        ssim = np.mean(ssim,axis=-1).clip(0,1)
        ssim = ssim*2 -1
        ## 当ssim 为1 的时候，error 为0 ，代表图像中黑色的区域
        error_map = 1 - 0.5*(1+ssim)
        media.write_image(self.root_dir/f'{self.task}_error_{index:02d}.png',error_map)

    def generate_MSE_map(self,redner_img,gt_img,index):
        mse = np.mean((redner_img - gt_img) ** 2,axis=-1)
        plt.close('all')
        plt.figure(figsize=(15, 5))  ## figure 的宽度:1500 height: 500
        ax = plt.subplot()
        sc = ax.imshow((mse), cmap='jet')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(sc, cax=cax)
        plt.savefig(os.path.join(str(self.root_dir)+"/error_map"+ f'/{self.task}_{index:02d}_mse.png'), bbox_inches='tight')
        return

    def search_Camera_index(self,train_names,test_list):
        train_idx =[]
        test_idx = []
        for name in train_names:
            name = str(name).split('/')[-1][:-4]
            train_idx.append(name)
        for name in test_list:
            name = str(name).split('/')[-1][:-4]
            test_idx.append(name)
        result = []
        i = 0
        for element in test_idx:
            while i < len(train_idx) and train_idx[i] < element:
                i += 1
            result.append(i)

        return result

    def main(self):
        config, pipeline, _ = eval_setup(
            self.load_config,
            test_mode= "test",
        )


        trainDataCache = pipeline.datamanager.train_dataset
        testDatasetCache = pipeline.datamanager.eval_dataset
        os.makedirs(self.root_dir / "error_map", exist_ok=True)
        os.makedirs(self.root_dir / "gt_rgb", exist_ok=True)
        os.makedirs(self.root_dir / "semantics", exist_ok=True)

        if self.task == 'trainset':
            DataCache = trainDataCache
            pipeline.model.inference_dataset ="trainset"
            config.pipeline.model.inference_dataset ="trainset"
            self.is_leaderboard = False
        elif self.task == 'testset':
            DataCache = testDatasetCache
            pipeline.model.inference_dataset = "testset"
            config.pipeline.model.inference_dataset = "testset"
            if self.is_leaderboard:
                Test_orderInTrainlist = self.search_Camera_index(trainDataCache.filenames,testDatasetCache.filenames)
                sequece_id = ''.join([x for x in str(config.data) if x.isdigit()])
                test_filename = Path('data_leader/test_name/test_{}'.format(sequece_id)).with_suffix('.txt')
                test_file = []
                with open(test_filename, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        lineData = list(line.strip().split('/'))[-1]
                        seq = list(line.strip().split('/'))[0][-9:-4]
                        test_file.append(seq + lineData)

            else:
                num_images = len(DataCache.image_cache) // 2
                Test_orderInTrainlist = [8+2*i for i in range(num_images)]
                # Test_orderInTrainlist = [14,16,18,20,22]   ##  在20张的 demo 中，test_id 是[4,9,14,18]
                # Test_orderInTrainlist = [20, 39]  ##waymo
                # Test_orderInTrainlist = [24, 26, 28, 30, 32]  ## 0652
            pipeline.model.field.testset_embedding_index = Test_orderInTrainlist
        else:
            raise print("Task Input is trainset or testset")

        config.print_to_terminal()
        'Read the image and save in target directory'
        os.makedirs(self.root_dir / "render_rgb",exist_ok=True)

        CONSOLE.print(f"[bold yellow]Rendering {len(DataCache.image_cache)} Images")
        bbx = DataCache.cameras.bbx
        test_id, train_id = DataCache.cameras.test_idx, DataCache.cameras.train_idx
        cameras = DataCache.cameras.to(pipeline.device)
        cameras.bbx = bbx
        cameras.test_idx = test_id
        cameras.train_idx = train_id


        progress = Progress(
            TextColumn(":movie_camera: Rendering :movie_camera:"),
            BarColumn(),
            TaskProgressColumn(show_speed=True),
            ItersPerSecColumn(suffix="fps"),
            TimeRemainingColumn(elapsed_when_finished=True, compact=True),
        )
        render_image = []
        render_depth = []
        render_normal = []
        render_semantics = []


        with progress:
            for camera_idx in progress.track(range(cameras.size), description=""):
                camera_ray_bundle = cameras.generate_rays(camera_indices=camera_idx)
                with torch.no_grad():
                    outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
                for rendered_output_name in self.rendered_output_names:
                    if rendered_output_name not in outputs:
                        CONSOLE.rule("Error", style="red")
                        CONSOLE.print(f"Could not find {rendered_output_name} in the model outputs", justify="center")
                        CONSOLE.print(f"Please set --rendered_output_name to one of: {outputs.keys()}",
                                      justify="center")
                        sys.exit(1)
                    output_image = outputs[rendered_output_name].cpu().numpy()
                    if rendered_output_name == 'rgb':
                        render_image.append(output_image)
                    elif rendered_output_name == 'depth':
                        render_depth.append(output_image)
                    elif rendered_output_name == 'normal':
                        render_normal.append(output_image)
                    elif rendered_output_name == "semantics":
                        semantic_labels = torch.argmax(torch.nn.functional.softmax(outputs["semantics"], dim=-1),dim=-1)
                        h, w = semantic_labels.shape[0], semantic_labels.shape[1]
                        ## waymo dataset or kitti360 have different colormap
                        semantic_color_map = assigncolor(semantic_labels.reshape(-1)).reshape(h, w, 3)
                        render_semantics.append(semantic_color_map)
        CONSOLE.print("[bold green]Rendering Images Finished")

        ''' Output rgb depth and normal image'''
        sum_psnr = 0
        sum_lpips = 0
        sum_ssim = 0
        for i in range(len(DataCache.image_cache.items()) // 2):
        # for i,image in sorted(DataCache.image_cache.items()):
            image = DataCache.image_cache.get(i)                       ## rgb_gt
            # semantic_gt = DataCache.image_cache.get("semantic_" + str(i))  ## seamntic_gt
            if self.is_leaderboard and self.task == 'testset':
                media.write_image(self.root_dir /"render_rgb"/ test_file[i], render_image[i])
            else:
                media.write_image(self.root_dir / "render_rgb" / f'{self.task}_{i:02d}_redner_rgb.png', render_image[i])
                media.write_image(self.root_dir/"gt_rgb" / f'{self.task}_{i:02d}_gtrgb.png', (image.detach().cpu().numpy()))
                # _,ssim_matrix = self.ssim(render_image[i],image.detach().cpu().numpy(),multichannel=True,full=True)
                # self.generate_errorMap(ssim_matrix,i)
                self.generate_MSE_map(image.detach().cpu().numpy(),render_image[i],i)
                psnr = -10. * np.log10(np.mean(np.square(image.detach().cpu().numpy() - render_image[i])))
                lpips = self.lpips(image.unsqueeze(0).permute(0,3,1,2),torch.from_numpy(render_image[i]).unsqueeze(0).permute(0,3,1,2))
                ssim = self.ssim(image.unsqueeze(0).permute(0,3,1,2),torch.from_numpy(render_image[i]).unsqueeze(0).permute(0,3,1,2))

                # ## 求出限制 汽车特定区域的 PSNR
                # os.makedirs(self.root_dir / "car_mse", exist_ok=True)
                # left, right, top, bottom = 450, 670, 182, 292
                # backup_img = image.detach().cpu().numpy()
                # ori_img = render_image[i].copy()
                # ori_img[top:bottom, left:right] =  backup_img[top:bottom, left:right]
                # psnr_car = -10. * np.log10(np.mean(np.square(ori_img - render_image[i])))
                # media.write_image(self.root_dir / "car_mse"/f'{self.task}_{i:02d}_car.png',ori_img)

                sum_psnr += psnr
                sum_lpips += lpips
                sum_ssim += ssim
                print("{} Mode image {} PSNR:{} LPIPS: {} SSIM: {}".format(self.task,i,psnr,lpips,ssim))

                ## semantics
                media.write_image(self.root_dir / "semantics" / f'{self.task}_{i:02d}_pred.png', render_semantics[i])
                # media.write_image(self.root_dir / "semantics" / f'{self.task}_{i:02d}_gt.png', semantic_gt[i])


        print(f"Average PSNR: {sum_psnr / len(DataCache.image_cache) * 2 }")
        print(f"Average LPIPS: {sum_lpips / len(DataCache.image_cache) *2}")
        print(f"Average LPIPS: {sum_ssim / len(DataCache.image_cache) * 2}")

        for i in range(len(render_depth)):
            pred_depth = render_depth[i].squeeze(2)
            pred_depth = pred_depth.clip(0,20)
            print(f"Predecited Depth Max:{pred_depth.max()}  clip max = 20 ")
            plt.close('all')
            ax = plt.subplot()
            sc = ax.imshow((pred_depth), cmap='jet')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(sc, cax=cax)
            plt.savefig(os.path.join(str(self.root_dir)+f'/{self.task}_{i:02d}_depth.png'))
            # pred_depth = cv2.applyColorMap(cv2.convertScaleAbs(((pred_depth / pred_depth.max()) * 255).astype(np.uint8), alpha=2), cv2.COLORMAP_JET)
            # cv2.imwrite(str(self.root_dir)+f'/{self.task}_{i:02d}_depth.png',pred_depth)
            if "normal" in self.rendered_output_names:
                media.write_image(self.root_dir/f'{self.task}_{i:02d}_normal.png', render_normal[i])
        CONSOLE.print(f"[bold blue] Store image to {self.root_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, help='testset or trainset')
    parser.add_argument('--config',type=str,help='Config Path')
    parser.add_argument('--is_leaderboard',action='store_true')
    config = parser.parse_args()

    RenderDatasets(config).main()



