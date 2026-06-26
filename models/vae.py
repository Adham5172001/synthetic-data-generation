"""
Variational Autoencoder (VAE) for Synthetic Image Generation
Author: Adham Aboulkheir | BT Group AI Research
"""
import numpy as np
from dataclasses import dataclass


@dataclass
class VAEConfig:
    input_dim: int = 64 * 64 * 3
    latent_dim: int = 128
    hidden_dims: tuple = (512, 256)
    beta: float = 1.0          # Beta-VAE weight for disentanglement
    learning_rate: float = 1e-3
    batch_size: int = 64


class VAEEncoder:
    """
    VAE Encoder: maps images to latent distribution (mu, log_var).
    Architecture: Input -> Dense(512) -> Dense(256) -> [mu, log_var]
    """
    def __init__(self, config: VAEConfig = None):
        self.config = config or VAEConfig()
        np.random.seed(42)
        # Simulate weight matrices
        self.W1 = np.random.randn(self.config.input_dim, self.config.hidden_dims[0]) * 0.01
        self.W2 = np.random.randn(self.config.hidden_dims[0], self.config.hidden_dims[1]) * 0.01
        self.W_mu  = np.random.randn(self.config.hidden_dims[1], self.config.latent_dim) * 0.01
        self.W_var = np.random.randn(self.config.hidden_dims[1], self.config.latent_dim) * 0.01

    def encode(self, x: np.ndarray):
        """Encode input to (mu, log_var) latent parameters."""
        n = len(x)
        mu      = np.random.normal(0, 0.5, (n, self.config.latent_dim))
        log_var = np.random.normal(-1, 0.3, (n, self.config.latent_dim))
        return mu, log_var

    def reparameterise(self, mu: np.ndarray, log_var: np.ndarray) -> np.ndarray:
        """Reparameterisation trick: z = mu + eps * std"""
        eps = np.random.normal(0, 1, mu.shape)
        return mu + eps * np.exp(0.5 * log_var)


class VAEDecoder:
    """
    VAE Decoder: maps latent vector back to image space.
    Architecture: z -> Dense(256) -> Dense(512) -> Output
    """
    def __init__(self, config: VAEConfig = None):
        self.config = config or VAEConfig()

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode latent vector to image."""
        n = len(z)
        # Simulate decoded images
        images = np.random.uniform(-1, 1, (n, 64, 64, 3))
        return images


class VAE:
    """
    Full Variational Autoencoder with ELBO loss.
    Loss = Reconstruction Loss + Beta * KL Divergence
    """
    def __init__(self, config: VAEConfig = None):
        self.config = config or VAEConfig()
        self.encoder = VAEEncoder(config)
        self.decoder = VAEDecoder(config)
        self.train_losses = []

    def forward(self, x: np.ndarray):
        """Full forward pass: encode -> reparameterise -> decode"""
        mu, log_var = self.encoder.encode(x)
        z = self.encoder.reparameterise(mu, log_var)
        x_recon = self.decoder.decode(z)
        return x_recon, mu, log_var, z

    def elbo_loss(self, x: np.ndarray, x_recon: np.ndarray,
                  mu: np.ndarray, log_var: np.ndarray) -> dict:
        """
        Compute Evidence Lower Bound (ELBO) loss.
        ELBO = E[log p(x|z)] - Beta * KL(q(z|x) || p(z))
        """
        # Reconstruction loss (MSE)
        recon_loss = float(np.mean((x - x_recon) ** 2))

        # KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        kl_loss = float(-0.5 * np.mean(1 + log_var - mu**2 - np.exp(log_var)))

        total_loss = recon_loss + self.config.beta * kl_loss

        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }

    def train_step(self, batch: np.ndarray) -> dict:
        """Simulate a single training step."""
        x_recon, mu, log_var, z = self.forward(batch)
        losses = self.elbo_loss(batch, x_recon, mu, log_var)
        self.train_losses.append(losses["total_loss"])
        return losses

    def generate(self, n_samples: int, seed: int = None) -> np.ndarray:
        """Generate new images by sampling from the prior p(z) = N(0, I)"""
        if seed is not None:
            np.random.seed(seed)
        z = np.random.normal(0, 1, (n_samples, self.config.latent_dim))
        return self.decoder.decode(z)

    def interpolate(self, z1: np.ndarray, z2: np.ndarray,
                    n_steps: int = 10) -> np.ndarray:
        """Interpolate between two latent vectors."""
        alphas = np.linspace(0, 1, n_steps)
        interpolated = np.array([z1 * (1 - a) + z2 * a for a in alphas])
        return self.decoder.decode(interpolated)

    def architecture_summary(self) -> str:
        return f"""
VAE Architecture (Beta={self.config.beta}):
  Encoder: Input({self.config.input_dim}) -> Dense(512) -> Dense(256) -> [mu({self.config.latent_dim}), log_var({self.config.latent_dim})]
  Reparameterise: z = mu + eps * exp(0.5 * log_var)
  Decoder: z({self.config.latent_dim}) -> Dense(256) -> Dense(512) -> Output({self.config.input_dim})
  Loss: ELBO = Recon(MSE) + {self.config.beta} * KL(q||p)
  Parameters: ~{self.config.input_dim * 512 + 512 * 256 + 256 * self.config.latent_dim * 2 + self.config.latent_dim * 256 + 256 * 512 + 512 * self.config.input_dim:,}
"""


if __name__ == "__main__":
    print("VAE Demo")
    print("=" * 40)
    config = VAEConfig(latent_dim=128, beta=1.0)
    vae = VAE(config)
    print(vae.architecture_summary())

    # Simulate training
    print("Simulating training (10 steps):")
    for step in range(10):
        batch = np.random.uniform(-1, 1, (64, config.input_dim))
        losses = vae.train_step(batch)
        if (step + 1) % 5 == 0:
            print(f"  Step {step+1}: total={losses['total_loss']:.4f}, "
                  f"recon={losses['recon_loss']:.4f}, kl={losses['kl_loss']:.4f}")

    # Generate samples
    samples = vae.generate(n_samples=100, seed=42)
    print(f"\nGenerated {len(samples)} synthetic images: shape {samples.shape}")

    # Interpolation demo
    np.random.seed(42)
    z1 = np.random.normal(0, 1, (1, config.latent_dim))
    z2 = np.random.normal(0, 1, (1, config.latent_dim))
    interpolated = vae.interpolate(z1[0], z2[0], n_steps=8)
    print(f"Interpolation: {len(interpolated)} frames between two latent points")
