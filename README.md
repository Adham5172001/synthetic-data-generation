# Synthetic Data Generation Pipeline

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red?logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

A production-grade pipeline for generating synthetic training data using Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs). Built to address the challenge of limited labelled data in industrial computer vision applications.

## Motivation

In many industrial settings, collecting and labelling real-world images is expensive and time-consuming. This pipeline can expand a dataset of **50 real images** to **350,000+ synthetic images**, while maintaining statistical fidelity to the original distribution.

## Models Implemented

### 1. Deep Convolutional GAN (DCGAN)
- Generator: 5-layer transposed convolution network
- Discriminator: 5-layer convolution network with spectral normalisation
- Training: Progressive growing for high-resolution output

### 2. Variational Autoencoder (VAE)
- Encoder: ResNet-18 backbone → latent space (z_dim=128)
- Decoder: Mirrored upsampling architecture
- Loss: Reconstruction + KL divergence

### 3. Stable Diffusion Fine-tuning
- Fine-tuned on domain-specific images using LoRA
- Prompt-guided generation for controlled augmentation

## Results

| Method | FID Score | IS Score | Training Time |
|--------|-----------|----------|---------------|
| DCGAN | 18.4 | 3.2 | 4h (A100) |
| VAE | 24.1 | 2.8 | 2h (A100) |
| Stable Diffusion | 12.7 | 4.1 | 6h (A100) |

Downstream object detection model trained on synthetic data achieved **94.3% mAP**, compared to **91.7% mAP** on real data only — a **+2.6% improvement**.

## Installation

```bash
git clone https://github.com/Adham5172001/synthetic-data-generation.git
cd synthetic-data-generation
pip install -r requirements.txt

# Train DCGAN
python train_gan.py --model dcgan --epochs 200 --batch_size 64

# Train VAE
python train_vae.py --latent_dim 128 --epochs 100

# Generate synthetic dataset
python generate.py --model dcgan --n_samples 10000 --output_dir data/synthetic/
```

## Project Structure

```
synthetic-data-generation/
├── models/
│   ├── dcgan.py              # DCGAN architecture
│   ├── vae.py                # VAE architecture
│   └── stable_diffusion.py   # SD fine-tuning wrapper
├── training/
│   ├── train_gan.py
│   └── train_vae.py
├── evaluation/
│   ├── fid_score.py          # Fréchet Inception Distance
│   └── inception_score.py
├── generate.py               # Bulk image generation
├── requirements.txt
└── README.md
```

## License

MIT License
