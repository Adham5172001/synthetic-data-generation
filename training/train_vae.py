"""
VAE Training Pipeline
Author: Adham Aboulkheir | BT Group AI Research
"""
import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.vae import VAE, VAEConfig


def train_vae(config: VAEConfig = None, n_epochs: int = 100,
              n_samples: int = 1000, verbose: bool = True) -> dict:
    """
    Simulate VAE training loop with ELBO loss tracking.
    """
    if config is None:
        config = VAEConfig()

    np.random.seed(42)
    vae = VAE(config)

    recon_losses, kl_losses, total_losses = [], [], []

    for epoch in range(n_epochs):
        # Simulate batch training
        batch = np.random.uniform(-1, 1, (config.batch_size, config.input_dim))
        losses = vae.train_step(batch)

        recon_losses.append(losses["recon_loss"])
        kl_losses.append(losses["kl_loss"])
        total_losses.append(losses["total_loss"])

        if verbose and (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}/{n_epochs}: "
                  f"ELBO={total_losses[-1]:.4f}, "
                  f"Recon={recon_losses[-1]:.4f}, "
                  f"KL={kl_losses[-1]:.4f}")

    return {
        "recon_losses": recon_losses,
        "kl_losses": kl_losses,
        "total_losses": total_losses,
        "vae": vae,
    }


if __name__ == "__main__":
    print("VAE Training Pipeline Demo")
    print("=" * 40)
    config = VAEConfig(latent_dim=128, beta=1.0)
    print(f"Config: latent_dim={config.latent_dim}, beta={config.beta}")
    print("\nTraining VAE (100 epochs):")
    history = train_vae(config, n_epochs=100, verbose=True)
    print(f"\nFinal ELBO: {history['total_losses'][-1]:.4f}")
    print(f"Final Recon: {history['recon_losses'][-1]:.4f}")
    print(f"Final KL: {history['kl_losses'][-1]:.4f}")
    vae = history["vae"]
    samples = vae.generate(n_samples=100, seed=42)
    print(f"\nGenerated {len(samples)} VAE samples: shape {samples.shape}")
