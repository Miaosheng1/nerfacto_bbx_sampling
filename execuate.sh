## Neus Render
# python scripts/only_inference.py --config outputs/datasets-kitti360_Neus/neus/2023-03-31_100645/config.yml --task testset

## Nefacto Train
python scripts/train.py nerfacto --pipeline.model.collider-params near_plane 0.0 far_plane 6.0
        --pipeline.datamanager.camera-optimizer.mode off --vis tensorboard --trainer.max-num-iterations 20000 --data datasets/Car_bbx

## Nerfaco Render (camera pose refinement off, orientened pose = 'none')
python scripts/only_inference.py --config outputs/datasets-kitti360_nerfacto/nerfacto/2023-04-10_100018/config.yml --task testset --is_leaderboard

## Nefacto Render(For fixed pixel to visualize Ray samples in bbx experiment)
python scripts\render_pixel.py --config outputs/datasets-Car_bbx/nerfacto/2023-04-20_153306/config.yml --task testset

********************************************           Add   Fisheye                   ************************************************
## Nerfacto Render with fisheye
python scripts/train.py  nerfacto --pipeline.model.collider-params near_plane 0.0 far_plane 6.0
        --pipeline.datamanager.camera-optimizer.mode off --vis tensorboard --trainer.max-num-iterations 20000 --data datasets/kitti360_nerfacto_fisheye/

##Nerfacto Render yaw viewpoint for fisheye.
python scripts/render.py  --load-config outputs/datasets-kitti360_nerfacto_fisheye/nerfacto/2023-05-08_222418/config.yml --traj roam --output-path 00_roam.mp4

## extact mesh
python scripts/extract_mesh.py --load-config outputs/datasets-kitti360_nerfacto_fisheye/nerfacto/2023-05-09_161147/config.yml --output-path meshes/kitti.ply

## run voxformer & unisurf
python scripts/train.py unisurf --pipeline.model.sdf-field.inside-outside False --vis tensorboard sdfstudio-data --data datasets/kitti360_neus/


## run voxformer & nerfacto
python scripts/train.py  nerfacto --pipeline.datamanager.camera-optimizer.mode off --vis tensorboard
  --trainer.max-num-iterations 20000 --data /data/smiao/datasets/kitti360_nerfacto_3353_50/

python scripts/only_inference.py --config  outputs/-data-smiao-datasets-kitti360_nerfacto_3353_50/nerfacto/2023-06-02_181503/config.yml --task testset

## abalation fisheye capacity
CUDA_VISIBLE_DEVICES=3 python scripts/train.py nerfacto --pipeline.datamanager.camera-optimizer.mode off --vis tensorboard  --trainer.max-num-iterations 30000
 --pipeline.model.log2-hashmap-size 21 --pipeline.model.feature-per-level 4 --descriptor fisheye_lpips --data /data/smiao/datasets/valid_03_fisheye/

## render semantic video and rgb video
CUDA_VISIBLE_DEVICES=3 python scripts/render.py --load-config outputs/-data-smiao-datasets-waymodata/nerfacto/60_add_semantic/config.yml --traj spiral --output-path valid03.mp4

## add new waymodata perser in nerfacto method
python scripts/train.py nerfacto --pipeline.datamanager.camera-optimizer.mode off --vis tensorboard  --trainer.max-num-iterations 30000  \
--descriptor add_semantic waymo-data --data /data/smiao/datasets/waymodata

## extracct semantic pointcloud from render depth
python scripts/exporter.py pointcloud --load-config outputs/-data-smiao-datasets-waymodata_noisy/nerfacto/nvida_semantic/config.yml \
--output-dir pointcloud_waymo \
--output_semantic_pointcloud True

## run wo semantics  [best res:8192_23]
python scripts/train.py nerfacto --pipeline.model.pred_semantic_loss_mult 0.0 --descriptor None --data /data/smiao/datasets/valid_0652_test20/