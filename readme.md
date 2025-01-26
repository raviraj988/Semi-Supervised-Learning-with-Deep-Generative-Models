

### `README.md`



# Abstract
This study explores the semi-supervised and unsupervised learning capabilities of deep generative modeling approaches, starting with an analysis and implementation of the M2 model, originally proposed by Kingma et al., through in-depth mathematical and experimental analyses. Building on the original proposed framework, in the *Research Section*, I introduce an Optimized-ELBO objective that addresses key challenges in the M2 model, such as classifier entropy misalignment, limited mutual information between inputs and latent variables, and insufficient utilization of labeled data. The method incorporates enhancements, including entropy penalty terms, mutual information maximization, and label smoothing, to improve both generative and discriminative performance. Extensive experiments on the MNIST and CIFAR-10 datasets demonstrate the efficacy of the proposed framework, with significant accuracy improvements on both datasets, highlighting its generalizability across diverse domains. The research section provides rigorous theoretical justifications and novel extensions to the ELBO objective, while the results validate the enhanced alignment of decision boundaries with low-density regions and improved learning from labeled data. This work sets the stage for further investigation into deeper architectures and advanced regularization techniques for semi-supervised learning.

---

# M2 Variational Autoencoder (VAE) Training

This repository contains the code to train M2 VAEs with either the standard M2 loss or the optimized ELBO loss(Reserch extention). The models support training on MNIST and CIFAR-10 datasets.

---

## Prerequisites

1. **Python**: Ensure you have Python 3.8 or later installed.
2. **Conda**: Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/).

---

## Setup Instructions

### Step 1: T0 create and Activate the Conda Environment

```bash
# Create a new environment named "vae-env"
conda create -n vae-env python=3.8 -y

# Activate the environment
conda activate vae-env
```

---

### Step 2: Install Dependencies

#### Install PyTorch
Install PyTorch with CUDA 11.8 support:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

If you do not have a CUDA-capable GPU, install the CPU version of PyTorch:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### Install Other Required Libraries
```bash
pip install matplotlib numpy
```

---

### Step 3: Dataset Preparation

The MNIST and CIFAR-10 datasets will be automatically downloaded and saved in the `../data` directory when the script is run. No manual setup is required for datasets.

---

## How to Run the Code

### Training on MNIST

#### Standard M2 Loss:
```bash
python ../src/main.py --dataset MNIST
```

#### Optimized ELBO Loss:
```bash
python ../src/main.py --optimized_elbo --dataset MNIST
```

---

### Training on CIFAR-10

#### Standard M2 Loss:
```bash
python ../src/main.py --dataset CIFAR10
```

#### Optimized ELBO Loss:
```bash
python ../src/main.py --optimized_elbo --dataset CIFAR10
```

---

