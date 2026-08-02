"""
Diffusion process: forward pass (adding noise), reverse pass (denoising), and DDPM sampling.
"""

import math
import torch
import torch.nn as nn


def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    """Linear schedule for noise variance."""
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule for noise variance (more stable)."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class DiffusionSchedule:
    """Precomputed diffusion schedule for efficient sampling."""
    
    def __init__(self, timesteps=1000, schedule_type="linear"):
        self.timesteps = timesteps
        
        if schedule_type == "linear":
            betas = linear_beta_schedule(timesteps)
        elif schedule_type == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule_type}")
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # Precompute variance schedule
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
        # For reverse process
        self.register_buffer('posterior_variance', 
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(self.posterior_variance, min=1e-20))
        )
        self.register_buffer('posterior_mean_coef1',
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer('posterior_mean_coef2',
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )
    
    def register_buffer(self, name, tensor):
        """Register buffer for device management."""
        setattr(self, name, tensor)
    
    def q_sample(self, x_0, t, noise=None):
        """
        Forward diffusion: add noise to image.
        q(x_t | x_0) = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * eps
        
        Args:
            x_0: Original image (batch_size, channels, H, W)
            t: Timestep (batch_size,)
            noise: Optional noise (if None, sample from N(0, I))
        Returns:
            x_t: Noisy image at timestep t
            noise: The noise added
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Reshape for broadcasting
        while len(sqrt_alpha.shape) < len(x_0.shape):
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
        while len(sqrt_one_minus_alpha.shape) < len(x_0.shape):
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
        
        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
        return x_t, noise
    
    def p_mean_variance(self, model, x_t, t, c, clip_denoised=True):
        """
        Compute posterior mean and variance for reverse step.
        
        Args:
            model: Denoising model
            x_t: Noisy image at timestep t
            t: Timestep
            c: Class conditioning
            clip_denoised: Clip denoised image to [-1, 1]
        Returns:
            mean: Posterior mean
            variance: Posterior variance
            log_variance: Log posterior variance
        """
        # Model predicts noise
        predicted_noise = model(x_t, t, c)
        
        # Compute mean
        coef1 = self.posterior_mean_coef1[t]
        coef2 = self.posterior_mean_coef2[t]
        
        # x_0 from predicted noise
        x_0_pred = (x_t - torch.sqrt(1.0 - self.alphas_cumprod[t]).view(-1, 1, 1, 1) * predicted_noise) / \
                   torch.sqrt(self.alphas_cumprod[t]).view(-1, 1, 1, 1)
        
        if clip_denoised:
            x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)
        
        # Reshape coefficients
        while len(coef1.shape) < len(x_t.shape):
            coef1 = coef1.unsqueeze(-1)
        while len(coef2.shape) < len(x_t.shape):
            coef2 = coef2.unsqueeze(-1)
        
        mean = coef1 * x_0_pred + coef2 * x_t
        
        variance = self.posterior_variance[t]
        log_variance = self.posterior_log_variance_clipped[t]
        
        return mean, variance, log_variance
    
    @torch.no_grad()
    def p_sample(self, model, x_t, t, c, clip_denoised=True):
        """
        Reverse diffusion step (DDPM).
        
        Args:
            model: Denoising model
            x_t: Noisy image
            t: Timestep
            c: Class conditioning
        Returns:
            x_{t-1}: Denoised image
        """
        mean, variance, _ = self.p_mean_variance(model, x_t, t, c, clip_denoised)
        
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, 1, 1, 1)
        
        return mean + nonzero_mask * torch.sqrt(variance).view(-1, 1, 1, 1) * noise
    
    @torch.no_grad()
    def p_sample_guided(self, model, x_t, t, c, target_amp,
                        guidance_scale=0.4, clip_denoised=True):
        """
        Reverse diffusion step with GS-projection guidance (inference only).

        After predicting x̂₀, apply one GS measurement projection toward the
        target far-field amplitude, then blend on the unit circle with
        λ_t = guidance_scale · √ᾱ_t (weak at high noise, strong near t=0).
        Gradient-free: costs ~2 extra FFTs per step, no backprop through model.

        Args:
            model:          Denoising model
            x_t:            Noisy image at timestep t
            t:              Timestep tensor (B,)
            c:              Class conditioning
            target_amp:     (B, 1, H, W) target far-field amplitude
            guidance_scale: λ_max in [0, 1]; 0 = plain DDPM step
        Returns:
            x_{t-1}: Denoised image
        """
        from sample_utils import gs_project_phase  # lazy import, avoids cycle

        predicted_noise = model(x_t, t, c)

        # x_0 from predicted noise (same as p_mean_variance)
        x_0_pred = (x_t - torch.sqrt(1.0 - self.alphas_cumprod[t]).view(-1, 1, 1, 1) * predicted_noise) / \
                   torch.sqrt(self.alphas_cumprod[t]).view(-1, 1, 1, 1)
        if clip_denoised:
            x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)

        # GS projection + unit-circle blending (handles ±π wraparound)
        if guidance_scale > 0:
            lam = (guidance_scale * self.sqrt_alphas_cumprod[t]).view(-1, 1, 1, 1)
            phase = x_0_pred * math.pi
            phase_proj = gs_project_phase(phase, target_amp)
            z = (1.0 - lam) * torch.exp(1j * phase) + lam * torch.exp(1j * phase_proj)
            x_0_pred = torch.angle(z) / math.pi

        # Posterior mean/variance from (possibly projected) x_0
        coef1 = self.posterior_mean_coef1[t].view(-1, 1, 1, 1)
        coef2 = self.posterior_mean_coef2[t].view(-1, 1, 1, 1)
        mean = coef1 * x_0_pred + coef2 * x_t
        variance = self.posterior_variance[t].view(-1, 1, 1, 1)

        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, 1, 1, 1)

        return mean + nonzero_mask * torch.sqrt(variance) * noise
    
    @torch.no_grad()
    def p_sample_ddim(self, model, x_t, t, t_next, c, eta=0.0, clip_denoised=True):
        """
        DDIM reverse step (deterministic or stochastic based on eta).
        
        Args:
            model: Denoising model
            x_t: Noisy image at timestep t
            t: Current timestep (batch tensor or scalar)
            t_next: Next timestep (t-1, batch tensor or scalar)
            c: Class conditioning
            eta: Stochasticity parameter (0.0 = deterministic, 1.0 = stochastic LIKE DDPM)
            clip_denoised: Clip denoised prediction to [-1, 1]
        Returns:
            x_prev: Denoised image at previous timestep
        """
        # Model predicts noise
        predicted_noise = model(x_t, t, c)
        
        # Extract scalar timestep values for indexing alphas_cumprod
        # If t is a batch tensor, all elements should be the same (from torch.full)
        if isinstance(t, torch.Tensor):
            t_scalar = t.view(-1)[0].item()  # Get first element and convert to scalar
            t_next_scalar = t_next.view(-1)[0].item()
        else:
            t_scalar = t
            t_next_scalar = t_next
        
        # Get alpha values
        alpha_t = self.alphas_cumprod[t_scalar]
        alpha_next = self.alphas_cumprod[t_next_scalar] if t_next_scalar >= 0 else torch.ones_like(alpha_t)
        
        # Predict x_0 from x_t
        sqrt_alpha_t = torch.sqrt(alpha_t).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t).view(-1, 1, 1, 1)
        
        x_0_pred = (x_t - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t
        
        if clip_denoised:
            x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)
        
        # DDIM variance schedule
        sqrt_alpha_next = torch.sqrt(alpha_next).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_next = torch.sqrt(1.0 - alpha_next).view(-1, 1, 1, 1)
        
        # Sigma calculation (controls stochasticity)
        sigma = eta * torch.sqrt((1.0 - alpha_next) / (1.0 - alpha_t) * (1.0 - alpha_t / alpha_next)).view(-1, 1, 1, 1)
        
        # Direction from x_t to x_0
        direction = (x_t - sqrt_alpha_t * x_0_pred) / sqrt_one_minus_alpha_t
        
        # Noise for stochasticity
        noise = torch.randn_like(x_t) if eta > 0 else torch.zeros_like(x_t)
        
        # DDIM step: x_prev = sqrt(alpha_next) * x_0 + sqrt(1 - alpha_next - sigma^2) * direction + sigma * noise
        x_prev = sqrt_alpha_next * x_0_pred + sqrt_one_minus_alpha_next * direction + sigma * noise
        
        return x_prev
    
    @torch.no_grad()
    def sample(self, model, num_samples, num_classes, device, class_labels=None):
        """
        Generate samples using DDPM sampling.
        
        Args:
            model: Denoising model
            num_samples: Number of samples to generate
            num_classes: Number of classes
            device: Device to use
            class_labels: If provided, use these labels (else sample uniformly)
        Returns:
            samples: Generated images (num_samples, 1, 28, 28) in [-1, 1]
        """
        model.eval()
        
        if class_labels is None:
            # Sample one image per class
            class_labels = torch.arange(num_classes, device=device)
        
        class_labels = class_labels.to(device)
        
        # Start from pure noise
        x_t = torch.randn(len(class_labels), 1, 28, 28, device=device)
        
        # Reverse process
        for t in reversed(range(self.timesteps)):
            t_tensor = torch.full((len(class_labels),), t, dtype=torch.long, device=device)
            x_t = self.p_sample(model, x_t, t_tensor, class_labels, clip_denoised=True)
        
        return x_t


    @torch.no_grad()
    def sample_ddim(self, model, num_samples, num_classes, device, 
                   num_steps=50, eta=0.0, class_labels=None, img_size=256):
        """
        Generate samples using DDIM (fast sampling).
        
        Args:
            model: Denoising model
            num_samples: Number of samples to generate
            num_classes: Number of classes
            device: Device to use
            num_steps: Number of DDIM steps (default 50 for ~30s at 256×256)
            eta: Stochasticity (0.0 = deterministic, 1.0 = stochastic)
            class_labels: If provided, use these labels (else sample uniformly)
            img_size: Image size - 28 (MNIST) or 256 (phase hologram). Auto-detect from model if possible.
        Returns:
            samples: Generated images (num_samples, 1, img_size, img_size) in [-1, 1]
        """
        model.eval()
        
        if class_labels is None:
            # Sample one image per class
            class_labels = torch.arange(num_classes, device=device)
        
        class_labels = class_labels.to(device)
        
        # Auto-detect img_size from model if it has img_size attribute
        if hasattr(model, 'img_size'):
            img_size = model.img_size
        elif hasattr(model, 'module') and hasattr(model.module, 'img_size'):
            img_size = model.module.img_size
        
        # Start from pure noise (adaptive to img_size)
        x_t = torch.randn(len(class_labels), 1, img_size, img_size, device=device)
        
        # Create DDIM timesteps using linspace for even spacing in noise level
        # Avoids starting at t=999 where alpha_cumprod≈0 causes 64000x error amplification
        # Schedule: evenly spaced from 0 to T-1, reversed, with -1 appended for final full denoise
        ddim_timesteps = torch.linspace(0, self.timesteps - 1, num_steps, dtype=torch.long, device=device)
        ddim_timesteps = torch.flip(ddim_timesteps, [0])  # Reverse: high t → low t
        # Append -1 so the last step fully denoises (alpha_next=1.0)
        ddim_timesteps = torch.cat([ddim_timesteps, torch.tensor([-1], dtype=torch.long, device=device)])
        
        # Reverse process with DDIM
        for i in range(len(ddim_timesteps) - 1):
            t_curr = ddim_timesteps[i].item()
            t_next = ddim_timesteps[i + 1].item()
            
            t_tensor = torch.full((len(class_labels),), t_curr, dtype=torch.long, device=device)
            t_next_tensor = torch.full((len(class_labels),), t_next, dtype=torch.long, device=device)
            
            # DDIM step
            x_t = self.p_sample_ddim(model, x_t, t_tensor, t_next_tensor, class_labels, eta=eta, clip_denoised=True)
        
        return x_t


def create_diffusion(timesteps=1000, schedule_type="cosine"):
    """Create a diffusion schedule."""
    return DiffusionSchedule(timesteps, schedule_type)


if __name__ == "__main__":
    # Test
    diffusion = create_diffusion(timesteps=1000)
    x_0 = torch.randn(2, 1, 28, 28)
    t = torch.tensor([100, 500])
    x_t, noise = diffusion.q_sample(x_0, t)
    
    print(f"x_0 shape: {x_0.shape}, x_t shape: {x_t.shape}")
    print(f"Noise shape: {noise.shape}")
