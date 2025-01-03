# Locally Controlled 3D Generation
Final Project for CS468.

[Slides](https://docs.google.com/presentation/d/1myU7y-pp23LhgC_WUwzfndJdyFfPOIKOnqyVpwOtskk/edit?usp=sharing) | [Paper](https://www.overleaf.com/read/bmtjsptvfgfh#4576ba)

Generate 3D assets (left) with locally controlled geometry (right).

<p align="center">
    <img src = "https://github.com/user-attachments/assets/763f685a-528a-47d4-ab9f-2ceb52f12afd" width="80%">
    <img src = "https://github.com/user-attachments/assets/62e69579-3af4-4417-a3ba-4d7a0ba29f7f" width="80%">
    <img src = "https://github.com/user-attachments/assets/65bdf75d-e05c-40d8-be36-6681dcf1b01c" width="80%">
</p>


## Installation 
This part is the same as original [MVDream-threestudio](https://github.com/bytedance/MVDream-threestudio). Skip it if you already have installed the environment.


### Install ControlDreamer

Install additional dependency:
```sh
cd threestudio/util/lib_shape_prior
python setup.py build_ext --inplace
```

ControlDreamer using multi-view ControlNet is provided in a different codebase. Install it by:
```sh
export PYTHONPATH=$PYTHONPATH:./extern/MVDream
export PYTHONPATH=$PYTHONPATH:./extern/ImageDream
pip install -e extern/MVDream 
```
Further, to provide depth-conditioned MV-ControlNet, download from url or please put midas ckpt file on:
```ControlDreamer/extern/MVDream/mvdream/annotator/ckpts```

## Quickstart
Please download the model from [MV-ControlNet](https://drive.google.com/file/d/1hOdpfVTkKvUXGQStcmeFnzY0P_q4ZSod/view?usp=sharing) under ```./extern/MVDream/mvdream/ckpt```

In the paper, we use the configuration with soft-shading for source generation. An A40 GPU is required, and we recommend setting num_samples_per_ray to 256 (originally 512) to prevent out-of-memory issues in most cases. Additionally, we provide an example source NeRF representation of [Hulk](https://drive.google.com/drive/folders/1rSemwNII8dQsY4YlkEoT2mtUj9RmPgLi?usp=sharing), generated from MVDream. If you want to use this, put this file into ```outputs/source```. 

To get the source representation:
```sh
python launch.py --config configs/mvdream-sd21-shading.yaml \
    --train --gpu 0 \
    system.prompt_processor.prompt="A high-resolution rendering of a Hulk, 3d asset"
```

After generation, refine the source representation using MV-ControlNet by transforming it into DMTet:
```sh
CFG_PATH=configs/controldreamer-sd21-shading.yaml
LOADPATH=outputs/source/Hulk/ckpts/last.ckpt

python launch.py --config ${CFG_PATH} \
    --train --gpu 0 \
    system.prompt_processor.prompt="A high-resolution rendering of an Iron Man, 3d asset" \
    system.geometry_convert_from=${LOADPATH} \
    system.geometry_convert_override.isosurface_threshold=10.
```

Note: Please refer to our [3D edit prompt bench](https://www.dropbox.com/scl/fi/do3b5fibwva3crr2ww1pd/3D_edit_bench.yaml?rlkey=ee9rjwzlsdck9upyhlfkgw1op&dl=0) for creative 3D generation.
 
## Credits
- This code is developed using [threestudio](https://github.com/threestudio-project/threestudio), [MVDream](https://github.com/bytedance/MVDream-threestudi), and [ImageDream](https://github.com/bytedance/ImageDream).

## Citing
If you find ControlDreamer helpful, please consider citing it:

``` bibtex
@article{oh2023controldreamer,
  title={ControlDreamer: Stylized 3D Generation with Multi-View ControlNet},
  author={Oh, Yeongtak and Choi, Jooyoung and Kim, Yongsung and Park, Minjun and Shin, Chaehun and Yoon, Sungroh},
  journal={arXiv preprint arXiv:2312.01129},
  year={2023}
}
```
