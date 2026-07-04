<div align="center">
<h1>SF-Mamba</h1>
<h3>[ICML2026] SF-Mamba: Rethinking State Space Model for Vision</h3>

Homepage ([homepage](https://sony.github.io/SF-Mamba/)), arXiv Preprint ([arXiv 2411.03855](https://arxiv.org/abs/2603.16423)), ICML 2026 ([conference paper](https://openreview.net/forum?id=R9AUrEgZEq))

</div>
this is the official implementation of SF-Mamba.  


## Setup

Our code is tested on CUDA 12.6, Pytorch 2.7.0, and Python 3.10.  
For example,
```sh
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126
```
Then,
```sh
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0,<2.2.0"
mim install "mmdet>=3.3.0" mmpretrain "mmsegmentation>=1.0.0"
pip install -r docker/cuda12.6/requirements.txt
# fix mmpretrain for newer timm version
python docker/cuda12.6/mmpretrain_fix_patch.py
# install cuda kernels
pip install personal_lib/external_packages/mamba_bfold --user  
pip install personal_lib/external_packages/non_causal_conv1d --user
```

#### If you want to use docker instead of venv/conda...
```sh
cd docker/cuda12.6/ 
docker build -t mambapeft:cuda126 .
docker run -itd --device nvidia.com/gpu=all --shm-size=16gb -h docker --name trainer1 -v [PATH-to-SF-MAMBA-DIRECTORY]:/work -v [PATH-to-YOUR-DATA-DIRECTORY]:/data sfmamba:cuda126
docker exec -it trainer1 /bin/bash

# install cuda kernels
pip install personal_lib/external_packages/mamba_bfold --user  
pip install personal_lib/external_packages/non_causal_conv1d --user
```


# Train / Test

This is an example for the tiny model. You can try other model sizes by changing the config.  

#### ImageNet (8GPU)
1. Update `data_root` in the config as needed.
2. Run the commands below:
```
CONFIG_NAME=mmpretrain/sf_mamba/imagenet1k/1_tiny_8gpu
CHECKPOINT_DIR=`date "+%Y%m%d_%H%M%S_%N"`
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_train.sh mmpretrain configs/${CONFIG_NAME}.py 8 --work-dir work_dirs/${CONFIG_NAME}/${CHECKPOINT_DIR}
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_test.sh mmpretrain configs/${CONFIG_NAME}.py work_dirs/${CONFIG_NAME}/${CHECKPOINT_DIR}/last_checkpoint 8 | tee work_dirs/${CONFIG_NAME}/${CHECKPOINT_DIR}/result.txt
```

#### COCO Cascaded Mask-RCNN (8GPU)
1. Update `data_root` and `checkpoint` (ImageNet pretrained) in the config as needed.
2. Run the commands below:
```
CONFIG_NAME=mmdet/sf_mamba/coco_all/1_cmask_rcnn/5_tiny_3x_fit_window_8gpu
CHECKPOINT_DIR=`date "+%Y%m%d_%H%M%S_%N"`
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_train.sh mmdet configs/${CONFIG_NAME}.py 8 --work-dir work_dirs/${CONFIG_NAME}/${CHECKPOINT_DIR}
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_test.sh mmdet configs/${CONFIG_NAME}.py work_dirs/${CONFIG_NAME}/${CHECKPOINT_DIR}/last_checkpoint 8 | tee work_dirs/${CONFIG_NAME}/${CHECKPOINT_DIR}/result.txt
```

Note: Config files with "twind" are SF-Mamba&spades; in the paper. (Memory efficient version by using small window size in Transformer blocks)

#### ADE20K UperNet (4GPU)
1. Update `data_root` and `checkpoint` (ImageNet pretrained) in the config as needed.
2. Run the commands below:
```
CONFIG_NAME=mmseg/sf_mamba/ade20k/1_uper/5_tiny
CHECKPOINT_DIR=`date "+%Y%m%d_%H%M%S_%N"`
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh mmseg configs/${CONFIG_NAME}.py 4 --work-dir work_dirs/${CONFIG_NAME}/${CHECKPOINT_DIR}
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_test.sh mmseg configs/${CONFIG_NAME}.py work_dirs/${CONFIG_NAME}/${CHECKPOINT_DIR}/last_checkpoint 4 | tee work_dirs/${CONFIG_NAME}/${CHECKPOINT_DIR}/result.txt
```

# How to Port SF-Mamba to Other Codebases
1. Copy [personal_lib/external_packages](personal_lib/external_packages) to your target project and install it with pip.
2. Copy [personal_lib/mmpretrain/models/backbones/sf_mamba](personal_lib/mmpretrain/models/backbones/sf_mamba) to your target project.
3. If you remove `@MODELS.register_module()` from [personal_lib/mmpretrain/models/backbones/sf_mamba/sf_mamba.py](personal_lib/mmpretrain/models/backbones/sf_mamba/sf_mamba.py), it should work without installing the MMLab package.  
4. About weights: backbone-compatible weights can be loaded as shown below.  
```
import re

backbone = SFMamba(
        window_size = (None, None, [84,84], [42,42]),
        classify=False,
        use_window=True) # SF-Mamba-T for COCO backbone

checkpoint = torch.load(filename, map_location='cpu', weights_only=False)
state_dict = checkpoint['ema_state_dict'] # or state_dict = checkpoint['state_dict']

# Rename keys like this:
revise_keys=[(r'^module\.backbone\.', '')]
for p, r in revise_keys:
    state_dict = OrderedDict(
        {re.sub(p, r, k): v
        for k, v in state_dict.items()})
backbone.load_state_dict(state_dict)
```

# Citation
```
@article{yoshimura2026sf,
  title={SF-Mamba: Rethinking State Space Model for Vision},
  author={Yoshimura, Masakazu and Hayashi, Teruaki and Hoshino, Yuki and Wang, Wei-Yao and Ohashi, Takeshi},
  journal={arXiv preprint arXiv:2603.16423},
  year={2026}
}
```