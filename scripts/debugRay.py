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

CONSOLE = Console(width=120)

class RenderDatasets():
    """Load a checkpoint, render the trainset and testset rgb,normal,depth, and save to the picture"""
    def __init__(self,parser_path):
        self.load_config = Path(parser_path.config)

        # self.rendered_output_names = ['rgb_fine', 'depth_fine']
        exp_method = str(self.load_config).split('/')[-3]
        if exp_method == 'nerfacto':
            self.rendered_output_names = ['rgb', 'depth']
        elif exp_method =='vanillanerf':
            self.rendered_output_names = ['rgb_fine', 'depth_fine']
        else:
            self.rendered_output_names = ['rgb', 'depth', 'normal']
        self.root_dir = Path('exp_psnr_' + exp_method)
        if self.root_dir.is_dir():
            os.system(f"rm -rf {self.root_dir}")
        self.task = parser_path.task
        self.is_leaderboard = parser_path.is_leaderboard
        self.ssim = structural_similarity
        self.camera_index = 0

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
                num_images = len(DataCache.image_cache)
                # Test_orderInTrainlist = [2+i for i in range(num_images)]
                Test_orderInTrainlist = [4,8,12,14]   ##  在20张的 demo 中，test_id 是[4,9,14,18]
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


        camera_ray_bundle = cameras.generate_rays(camera_indices=self.camera_index)
        with torch.no_grad():
            outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
        H,W = outputs['rgb'].shape[0],outputs['rgb'].shape[1]
        np.save(self.root_dir/"density.npy",outputs['density'].detach().cpu().numpy())
        np.save(self.root_dir/"depth.npy",outputs['depth'].detach().cpu().numpy())
        np.save(self.root_dir/"color.npy",outputs['rgb'].detach().cpu().numpy())
        np.save(self.root_dir/'weight.npy',outputs["weight"].detach().cpu().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, help='testset or trainset')
    parser.add_argument('--config',type=str,help='Config Path')
    parser.add_argument('--is_leaderboard',action='store_true')
    config = parser.parse_args()

    RenderDatasets(config).main()