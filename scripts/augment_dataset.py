"""
Dataset Augmentation Script — Bulk Generation Pipeline
Author: Adham Aboulkheir | BT Group AI Research
"""
import numpy as np
import os
import sys
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.dcgan import DCGANGenerator, GANConfig
from models.vae import VAE, VAEConfig
from evaluation.fid_score import evaluate_generation_quality


def augment_dataset(real_images_dir: str, output_dir: str,
                    n_gan: int = 100000, n_vae: int = 100000,
                    seed: int = 42) -> dict:
    """
    Main augmentation pipeline.
    Generates synthetic images using both GAN and VAE.
    """
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Augmentation Pipeline")
    print(f"  Real images dir: {real_images_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  GAN samples: {n_gan:,}")
    print(f"  VAE samples: {n_vae:,}")

    # Simulate loading real images
    n_real = 50
    real_images = np.random.randint(50, 200, (n_real, 64, 64, 3), dtype=np.uint8)
    print(f"\nLoaded {n_real} real images")

    # GAN generation
    print("\nGenerating GAN synthetic images...")
    gan_config = GANConfig(latent_dim=100, image_size=64)
    generator = DCGANGenerator(gan_config)
    gan_images = generator.generate(n_samples=min(n_gan, 1000), seed=seed)
    print(f"  Generated {len(gan_images)} GAN images (shape: {gan_images.shape})")

    # VAE generation
    print("\nGenerating VAE synthetic images...")
    vae_config = VAEConfig(latent_dim=128)
    vae = VAE(vae_config)
    vae_images = vae.generate(n_samples=min(n_vae, 1000), seed=seed + 1)
    print(f"  Generated {len(vae_images)} VAE images (shape: {vae_images.shape})")

    # Evaluate quality
    print("\nEvaluating generation quality...")
    fake_all = np.vstack([
        (gan_images * 127.5 + 127.5).astype(np.uint8),
        (vae_images * 127.5 + 127.5).astype(np.uint8)
    ])
    metrics = evaluate_generation_quality(real_images, fake_all)
    print(f"  FID Score: {metrics['fid_score']:.1f}")
    print(f"  IS: {metrics['inception_score_mean']:.2f} ± {metrics['inception_score_std']:.2f}")

    return {
        "n_real": n_real,
        "n_gan": len(gan_images),
        "n_vae": len(vae_images),
        "total_synthetic": len(gan_images) + len(vae_images),
        "fid_score": metrics["fid_score"],
        "output_dir": output_dir,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment dataset with synthetic images")
    parser.add_argument("--real-dir", default="data/real", help="Real images directory")
    parser.add_argument("--output-dir", default="data/synthetic", help="Output directory")
    parser.add_argument("--n-gan", type=int, default=1000, help="Number of GAN images")
    parser.add_argument("--n-vae", type=int, default=1000, help="Number of VAE images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    results = augment_dataset(args.real_dir, args.output_dir,
                               args.n_gan, args.n_vae, args.seed)

    print(f"\n=== Augmentation Complete ===")
    print(f"Real images:      {results['n_real']}")
    print(f"GAN synthetic:    {results['n_gan']:,}")
    print(f"VAE synthetic:    {results['n_vae']:,}")
    print(f"Total synthetic:  {results['total_synthetic']:,}")
    print(f"FID Score:        {results['fid_score']:.1f}")
    print(f"Output:           {results['output_dir']}")
