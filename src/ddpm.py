import torch
import torch.nn as nn
import numpy as np
from typing import Optional

class DDPMSampler(nn.Module):
    def __init__(self, generator: torch.Generator, num_training_steps: int = 1000, beta_start: float = 0.00085, beta_end: float = 0.0120):
        super().__init__()

        betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_training_steps, dtype=torch.float32) ** 2
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("sqrt_alpha_cumprod", alpha_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_cumprod", (1.0 - alpha_cumprod).sqrt())
        self.register_buffer("one_scalar", torch.tensor(1.0))

        self.generator = generator
        self.num_training_steps = num_training_steps
        self.inference_timesteps = torch.arange(num_training_steps - 1, -1, -1, dtype=torch.long)

    def sample_train_timesteps(self, batch_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
        device = device or self.alpha_cumprod.device
        return torch.randint(0, self.num_training_steps, (batch_size,), device=device)

    def set_inference_timesteps(self, num_inference_steps: int = 50) -> None:
        floats = np.linspace(0, self.num_training_steps - 1, num_inference_steps)
        ints = np.round(floats).astype(np.int64)[::-1].copy()
        self.inference_timesteps = torch.from_numpy(ints)

    def _get_prev_t(self, idx: int) -> int:
        if idx + 1 < len(self.inference_timesteps):
            return int(self.inference_timesteps[idx + 1])
        return -1

    def _get_variance(self, t: int, t_prev: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        alpha_t = self.alpha_cumprod[t]
        alpha_prev = (
            self.alpha_cumprod[t_prev]
            if t_prev >= 0
            else self.one_scalar.to(device=device, dtype=dtype)
        )

        beta_t = 1 - alpha_t / alpha_prev
        var = (1 - alpha_prev) / (1 - alpha_t) + beta_t
        return var.clamp(min=1e-20)

    def step(self, latents, model_output, t: int, idx: int) -> torch.Tensor:
        device, dtype = latents.device, latents.dtype
        t_prev = self._get_prev_t(idx)

        # fetch alphas & betas
        beta_t      = self.betas[t]                 # β_t
        alpha_t     = self.alphas[t]                # α_t
        alpha_bar   = self.alpha_cumprod[t]         # \bar α_t
        alpha_bar_prev = (
            self.alpha_cumprod[t_prev]
            if t_prev >= 0
            else torch.tensor(1.0, device=device, dtype=dtype)
        )

        # 1) compute the posterior mean μ_t(x_t)
        #    equation: 1/√α_t * (x_t - [β_t/√(1-ᾱ_t)] * εθ)
        recip_sqrt_alpha = alpha_t.rsqrt()          # 1/sqrt(alpha_t)
        coeff_eps = beta_t / (1.0 - alpha_bar).sqrt()
        mean_pred = recip_sqrt_alpha * (latents - coeff_eps * model_output)

        # 2) compute posterior variance
        if t_prev >= 0:
            beta_post = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
            noise = torch.randn(latents.shape, generator=self.generator,
                                device=device, dtype=dtype)
            return mean_pred + beta_post.sqrt() * noise
        else:
            return mean_pred

    def add_noise(self, original_samples: torch.Tensor, timesteps: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        device, dtype = original_samples.device, original_samples.dtype
        ts = timesteps.to(device)

        if noise is None:
            noise = torch.randn(original_samples.shape, generator=self.generator, device=device, dtype=dtype)

        sqrt_alpha = self.sqrt_alpha_cumprod[ts].view(
            (original_samples.shape[0],) + (1,) * (original_samples.ndim - 1)
        )

        sqrt_1m_alpha = self.sqrt_one_minus_alpha_cumprod[ts].view(
            (original_samples.shape[0],) + (1,) * (original_samples.ndim - 1)
        )

        return sqrt_alpha * original_samples + sqrt_1m_alpha * noise
