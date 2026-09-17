# XAI2I: Explainability Methods for Image-to-Image Diffusion Models

This repository contains the code for my Master's thesis, **"Explainability Methods for Image-to-Image Diffusion Models: A Systematic Comparison"**. It systematically compares six XAI methods — LIME, KernelSHAP, RISE, Grad-CAM, Integrated Gradients, and I²AM — applied to the inpainting task using the StableVITON pipeline on the VITON-HD dataset, evaluated with the Quantus and Co-12 frameworks.

This codebase is built on top of **I²AM**, and the evaluation pipeline (`xai_eval.py`) extends their attention-based attribution framework with additional XAI methods and evaluation metrics.

## Acknowledgements

This work builds directly on:

> **I2AM: Interpreting Image-to-Image Latent Diffusion Models via Bi-Attribution Maps**
> Park & Jang, ICLR 2025
> [Paper](https://openreview.net/forum?id=bBNUiErs26) · [Original repository](https://github.com/qkrwnstj306/I2AM)

All credit for the original I²AM framework, StableVITON integration, and attention hooking implementation goes to the original authors. Hyeryung Jang is the corresponding author of the original work.

## Environment

```bash
git clone https://gitlab.uni-ulm.de/pcg76/xai2i
cd xai2i

conda create --name I2AM python=3.10 -y
conda activate I2AM

# install packages
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117
pip install pytorch-lightning==1.5.0
pip install einops
pip install opencv-python==4.7.0.72
pip install matplotlib
pip install omegaconf
pip install albumentations
pip install transformers==4.33.2
pip install xformers==0.0.19
pip install triton==2.0.0
pip install open-clip-torch==2.19.0
pip install diffusers==0.20.2
pip install scipy==1.10.1
pip install clean-fid
pip install scikit-image
conda install -c anaconda ipython -y
```

## Weights and Data

- StableVITON checkpoint (from I²AM): [download here](https://kaistackr-my.sharepoint.com/:f:/g/personal/rlawjdghek_kaist_ac_kr/EjzAZHJu9MlEoKIxG4tqPr0BM_Ry20NHyNw5Sic2vItxiA?e=5mGa1c)
- VITON-HD dataset: [download here](https://github.com/shadow2496/VITON-HD)

## Thesis Evaluation Pipeline
The core contribution of this repository is `xai_eval.py`, which runs the six compared XAI methods against the I²AM/StableVITON pipeline and computes Quantus and Co-12 metrics.

```bash
CUDA_VISIBLE_DEVICES=0 python xai_eval.py \
 --config_path ./configs/VITONHD.yaml \
 --batch_size 1 \
 --model_load_path ./ckpts/VITONHD.ckpt \
 --data_root_dir ./zalando-hd-resized \
 --save_dir ./results
```

## Original I²AM Inference

The original I²AM inference script (`inference.py`) is preserved for generating attribution maps directly:

- Reference-to-Generated attribution maps: `--generated_image True --controlnet False --new_attention_map False`
- Generated-to-Reference attribution maps: `--generated_image False --controlnet True --new_attention_map False`
- Specific-Reference attribution maps (5th layer): `--generated_image False --controlnet True --new_attention_map True`
- Time-level / head-level attribution maps: `--per_time_step True` or `--per_attention_head True`

```bash
#### paired
CUDA_VISIBLE_DEVICES=0 python inference.py \
 --config_path ./configs/VITONHD.yaml \
 --batch_size 1 \
 --model_load_path ./ckpts/VITONHD.ckpt \
 --save_dir ./results \
 --data_root_dir ../VITON/dataset/zalando-hd-resized \
 --generated_image True \
 --per_time_step False \
 --per_attention_head False \
 --controlnet False \
 --new_attention_map False

#### unpaired
CUDA_VISIBLE_DEVICES=1 python inference.py \
 --config_path ./configs/VITONHD.yaml \
 --batch_size 1 \
 --model_load_path ./ckpts/VITONHD.ckpt \
 --unpair \
 --data_root_dir ../VITON/dataset/zalando-hd-resized \
 --save_dir ./results \
 --generated_image True \
 --per_time_step False \
 --per_attention_head False \
 --controlnet False \
 --new_attention_map False
```
=======
## Cluster Environment (bwUniCluster 3.0)

Experiments were run on bwUniCluster 3.0 (`gpu_a100_il` partition, NVIDIA A100s) via SLURM, using a JupyterHub-based interactive environment. Before running any script or job, apply:

```bash
unset PYTHONPATH
export PATH="$HOME/miniconda3/envs/I2AM/bin:$PATH"
```
