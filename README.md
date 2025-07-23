<p align="center">
  <h1 align="center">
    SmartSplat: Feature-Smart Gaussians for Scalable Compression of Ultra-High-Resolution Images
  </h1>
  <h3 align="center"><a href="https://anonymous.4open.science/w/SmartSplat-BECD/">🌐Anonymous Website</a> 
  </h3>
  <div align="center"></div>
</p>

<div align="center">
  <a href="">
    <video src="assets/teaser.mp4" autoplay loop muted playsinline style="width:100%; border-radius: 8px;"></video>
  </a>
  <p style="margin-top: 8px; font-size: 14px; color: #555;">
   Raw Image info: 16320×10848, 189 MB
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#datasets">Datasets</a>
    </li>
    <li>
      <a href="#benchmarking">Benchmarking</a>
    </li>
    <li>
      <a href="#acknowledgement">Acknowledgement</a>
    </li>
  </ol>
</details>

## Installation

```bash
conda create -n smartsplat python==3.12
conda activate smartsplat

# install torch
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

pip install setuptools==78.0.1

pip install -r requirements.txt


# install Gaussian Rasterization
cd submodules/fused-ssim
pip install -e .
cd ../gsplat
pip install -e .
cd ../gsplat2d
pip install -e .
cd ../simple-knn-2d-qr
pip install -e .
```

## Datasets
You can download the DIV8K dataset from [huggingface](https://huggingface.co/datasets/Iceclear/DIV8K_TrainingSet), and the DIV16K dataset will be made publicly available after the paper is accepted.

## Benchmarking
This codebase integrates multiple GS-based image representation methods, including [GaussianImage](https://github.com/Xinjie-Q/GaussianImage), [ImageGS](https://arxiv.org/abs/2407.01866), [3DGS](https://github.com/nerfstudio-project/gsplat/blob/main/examples/image_fitting.py), and [LIG](https://arxiv.org/abs/2502.09039).

 All our experiments were conducted on the A800 cluster. You can find the relevant run scripts in the `slurm` folder, and the experimental test logs are available in the `slurm_logs` folder.


## Acknowledgement
We thank the authors of the following repositories for their open-source code:

- [GaussianImage](https://github.com/Xinjie-Q/GaussianImage)
- [Gsplat](https://github.com/nerfstudio-project/gsplat)