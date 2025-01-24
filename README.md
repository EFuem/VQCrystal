# VQCrystal

This repository is the official implementation for *VQCrystal : Leveraging Vector Quantization for Discovery of Stable Crystal Structures*

[[Paper]](https://arxiv.org/abs/2409.06191)

## Table of Contents

- Installation
- Datasets and Model weights
- Training and Evaluation
- Generate New Crystals
- Authors and acknowledgements
- Citation
- Contact

## Installation

### 1. Virtual Environment Setup

This implementation requires a Linux operating system with CUDA-enabled GPU hardware. The environment is managed using Conda package manager with the following specific dependencies:

* cuda/11.8
* cudnn/8.8.0_cu11x
* gcc/11.2.0
* nccl/2.16.5-1_cuda11.8
* anaconda/2022.10

First we initialize a new conda environment named vqcrystal

```bash
conda create -n vqcrystal python=3.10.14
conda activate vqcrystal
# installing torch==2.5.1+cu118
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Then install additional dependencies by executing the installation script from the VQCrystal root directory

```bash
# cd VQCrystal
bash install.sh
```

### 2. OpenLAM installation

In our work, we utilize a third-party package [OpenLAM](https://github.com/deepmodeling/openlam/tree/main) for fast structral relaxation.

* Rather than installing the full OpenLAM package, we have extracted the essential Python functions required for our workflow
* The relevant scripts have been relocated to the `./lam_optimize` directory to streamline the installation process
* Minor modifications (add saving control, etc) have been made to optimize integration with our pipeline
* The pre-trained model weights are stored in `./dp0529.pth`, which is fetched from [the original repository](https://github.com/deepmodeling/openlam/tree/main).

### 3. Genetic Algorithm

geneticalgorithm==1.0.2 should have been installed after running

```
bash install.sh
```

However, we also make several minor adjustions to better suit out pipeline. Therefore we need to substitute the original file with our provided `./geneticalgorithm_temp.py`

First, run ***pip show geneticalgorithm*** to locate the file, you should expect an output like this

```
Name: geneticalgorithm
Version: 1.0.2
Summary: An easy implementation of genetic-algorithm (GA) to solve continuous and combinatorial optimization problems with real, integer, and mixed variables in Python
Home-page: https://github.com/rmsolgi/geneticalgorithm
Author: Ryan (Mohammad) Solgi
Author-email: ryan.solgi@gmail.com
License: UNKNOWN
Location: /{your root}/.conda/envs/vqcrystal/lib/python3.10/site-packages
Requires: func-timeout, numpy
Required-by: 
```

Then, run

```
rm /{your root}/.conda/envs/vqcrystal/lib/python3.10/site-packages/geneticalgorithm/geneticalgorithm.py
cp ./geneticalgorithm_temp.py /{your root}/.conda/envs/vqcrystal/lib/python3.10/site-packages/geneticalgorithm/geneticalgorithm.py
```

## Datasets and Model weights

Because of the file size limit of github, the related datasets (c2db, carbon_24, mp_20, and perov_5), final model checkpoints and model weight of OpenLAM can not be uploaded directly. Therefore we upload them to google drive. The link is:

*https://drive.google.com/drive/folders/1VT9-mJCQ1HWlL9iSRPQrb3_bHgkjFOGH?usp=sharing*

The users should sustitude the current blank folder `./ckpt` and `./data` with those in the link above.

`dp0529.pth` should be put directly under the root directory of VQCrystal.

## Training and Evaluation

We take perov_5 dataset as an example, the training script can be launched with

```bash
python main.py --data_path ./data/perov_5 --config_path ./config/config_perov_5.yaml
```

Also, you may pass `--wandb True` to enable online logging and `--save True` to save the best checkpoint.

To evaluate the trained model, we can launch

```bash
python validate.py --data_path ./data/perov_5 --config_path ./config/config_perov_5.yaml
```

## Generate New Crystals

To generate new crystals with VQCrystal, run:

```
bash generate.sh
```

The script `generate.sh` executes `./generate.py` in parallel, with `NUM_JOBS` controlling the maximum number of concurrent processes.

Each parallel process performs the following operations:

1. Selects a seed crystal from the dataset
2. Fixes its local indices
3. Applies genetic sampling algorithms to generate structural variants
4. Performs rapid structural relaxation on the variants

The process creates a new directory under `./denovo` with the following hierarchical structure:

```bash
./denovo/
└── {expriment-name}/
    ├── base/          # Contains the original seed crystal structure
    ├── sample/        # Stores structures generated through genetic optimization
    └── optimize/      # Houses successfully relaxed structures that achieved convergence
```

## Authors and acknowledgements

Zijie Qiu and Luozhijie Jin contribute equally to this implementation.

[OpenLAM](https://github.com/deepmodeling/openlam/tree/main) is used as a third-party fast structural relaxation software in this work.

For certain components of the neural network and evaluation, we adapt the code implement of [CGCNN](https://github.com/txie-93/cgcnn), [CDVAE](https://github.com/txie-93/cdvae) and [DiffCSP](https://github.com/jiaor17/DiffCSP)

More details and the sources of the datasets are presented in the [paper](https://arxiv.org/abs/2409.06191).

## Citation

Please consider citing the following paper if you find our work useful.

```
@article{qiu2024vqcrystal,
  title={VQCrystal: Leveraging Vector Quantization for Discovery of Stable Crystal Structures},
  author={Qiu, ZiJie and Jin, Luozhijie and Du, Zijian and Chen, Hongyu and Cen, Yan and Sun, Siqi and Mei, Yongfeng and Zhang, Hao},
  journal={arXiv preprint arXiv:2409.06191},
  year={2024}
}
```

## Contact

Please leave an issue or contact Zijie Qiu (22307140109@m.fudan.edu.cn) if you have any questions.
