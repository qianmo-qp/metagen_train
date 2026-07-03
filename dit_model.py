"""
Minimal Conditional DiT (Diffusion Transformer) for MNIST.
Architecture: Patchified pixels -> Linear projection -> Transformer blocks with AdaLN -> Unpatchify
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        """
        Args:
            t: Tensor of shape (batch_size,) with timestep values in [0, 1000)
        Returns:
            Embedding of shape (batch_size, dim)
        """
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class MLPEmbedding(nn.Module):
    """MLP projection for time/class embeddings."""
    
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class PatchEmbed(nn.Module):
    """Convert image to patch embeddings."""
    
    def __init__(self, img_size=28, patch_size=4, in_channels=1, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Linear(in_channels * patch_size * patch_size, embed_dim)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, in_channels, img_size, img_size)
        Returns:
            (batch_size, num_patches, embed_dim)
        """
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size
        
        # Reshape to patches: (B, C, H, W) -> (B, C, num_patches_h, patch_size, num_patches_w, patch_size)
        x = x.reshape(B, C, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size)
        # -> (B, H // patch_size, W // patch_size, C, patch_size, patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        # -> (B, num_patches, C * patch_size * patch_size)
        x = x.reshape(B, -1, C * self.patch_size * self.patch_size)
        
        x = self.proj(x)
        return x


class Unpatchify(nn.Module):
    """Convert patch embeddings back to image."""
    
    def __init__(self, img_size=28, patch_size=4, out_channels=1, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        
        self.proj = nn.Linear(embed_dim, out_channels * patch_size * patch_size)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, num_patches, embed_dim)
        Returns:
            (batch_size, out_channels, img_size, img_size)
        """
        B, num_patches, _ = x.shape
        
        x = self.proj(x)  # (B, num_patches, out_channels * patch_size * patch_size)
        
        num_patches_h = num_patches_w = self.img_size // self.patch_size
        # Reshape: (B, num_patches, out_channels, patch_size, patch_size)
        x = x.reshape(B, num_patches_h, num_patches_w, self.out_channels, self.patch_size, self.patch_size)
        # -> (B, out_channels, num_patches_h, patch_size, num_patches_w, patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        # -> (B, out_channels, img_size, img_size)
        x = x.reshape(B, self.out_channels, self.img_size, self.img_size)
        return x


class AdaLayerNorm(nn.Module):
    """Adaptive Layer Normalization for conditioning."""
    
    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.gamma_proj = nn.Linear(cond_dim, hidden_dim)
        self.beta_proj = nn.Linear(cond_dim, hidden_dim)
        
        # Initialize to identity
        nn.init.ones_(self.gamma_proj.weight)
        nn.init.zeros_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)
    
    def forward(self, x, cond):
        """
        Args:
            x: (batch_size, seq_len, hidden_dim)
            cond: (batch_size, cond_dim)
        Returns:
            (batch_size, seq_len, hidden_dim)
        """
        normalized = self.norm(x)
        gamma = self.gamma_proj(cond).unsqueeze(1)  # (batch_size, 1, hidden_dim)
        beta = self.beta_proj(cond).unsqueeze(1)    # (batch_size, 1, hidden_dim)
        return normalized * gamma + beta


class DiTBlock(nn.Module):
    """Transformer block with AdaLN conditioning + Flash Attention."""
    
    def __init__(self, hidden_dim, num_heads, mlp_ratio=4, cond_dim=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # QKV projection (replaces nn.MultiheadAttention)
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.attn_out = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # MLP feedforward
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_dim)
        )
        
        # Adaptive layer norms
        self.norm1 = AdaLayerNorm(hidden_dim, cond_dim)
        self.norm2 = AdaLayerNorm(hidden_dim, cond_dim)
    
    def forward(self, x, cond):
        """
        Args:
            x: (batch_size, seq_len, hidden_dim)
            cond: (batch_size, cond_dim)
        Returns:
            (batch_size, seq_len, hidden_dim)
        """
        B, L, _ = x.shape
        
        # Self-attention with adaptive norm + Flash Attention
        x_norm = self.norm1(x, cond)
        
        # QKV: (B, L, 3*hidden) -> 3 x (B, heads, L, head_dim)
        qkv = self.qkv(x_norm).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, L, head_dim)
        q, k, v = qkv.unbind(0)            # each: (B, heads, L, head_dim)
        
        # Flash Attention (O(seq) memory, 2-4x faster)
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        attn_out = attn_out.transpose(1, 2).reshape(B, L, self.hidden_dim)
        attn_out = self.attn_out(attn_out)
        x = x + attn_out
        
        # MLP with adaptive norm
        x_norm = self.norm2(x, cond)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out
        
        return x


class ConditionalDiT(nn.Module):
    """
    Conditional Diffusion Transformer for MNIST.
    
    Architecture:
    1. Patchify input (4x4 patches) -> 49 tokens
    2. Linear embedding -> hidden_dim
    3. Add learned position embeddings
    4. 6 DiT blocks with AdaLN conditioning
    5. Un-patchify -> 1x28x28 noise prediction
    
    Conditioning via:
    - Time embedding: Sinusoidal + MLP
    - Class embedding: One-hot (0-9) + MLP
    - Combined and projected to cond_dim
    """
    
    def __init__(
        self,
        img_size=256,
        patch_size=8,
        in_channels=1,
        hidden_dim=768,
        num_heads=12,
        num_layers=12,
        time_dim=256,
        num_classes=10,
        mlp_ratio=4,
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.in_channels = in_channels
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, hidden_dim)
        
        # Positional embedding (learnable)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, hidden_dim) * 0.02)
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            MLPEmbedding(time_dim, time_dim * 2, time_dim)
        )
        
        # Class embedding
        self.class_embed = nn.Embedding(num_classes, time_dim)
        
        # Combine time and class embeddings
        self.cond_proj = nn.Linear(time_dim * 2, hidden_dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, mlp_ratio, cond_dim=hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(hidden_dim)
        
        self.unpatchify = Unpatchify(self.img_size, self.patch_size, in_channels, hidden_dim)
        
        # Output head: predict noise
        self.out_head = nn.Linear(hidden_dim, in_channels * patch_size * patch_size)
    
    def forward(self, x, t, c):
        """
        Args:
            x: Noisy image (batch_size, in_channels, img_size, img_size)
            t: Timestep (batch_size,)
            c: Class label (batch_size,)
        Returns:
            Noise prediction (batch_size, in_channels, img_size, img_size)
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, hidden_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Time and class conditioning
        t_emb = self.time_mlp(t.float())  # (B, time_dim)
        c_emb = self.class_embed(c)       # (B, time_dim)
        cond = torch.cat([t_emb, c_emb], dim=-1)  # (B, time_dim * 2)
        cond = self.cond_proj(cond)       # (B, hidden_dim)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, cond)
        
        # Final layer norm
        x = self.final_norm(x)
        
        # Predict noise per patch
        noise_per_patch = self.out_head(x)  # (B, num_patches, in_channels * patch_size * patch_size)
        
        # Reshape back via unpatchify-like operation
        B, num_patches, patch_out_dim = noise_per_patch.shape
        num_patches_h = num_patches_w = int(math.sqrt(num_patches))
        
        noise_per_patch = noise_per_patch.reshape(
            B, num_patches_h, num_patches_w, self.in_channels, self.patch_size, self.patch_size
        )
        noise_pred = noise_per_patch.permute(0, 3, 1, 4, 2, 5).contiguous()
        noise_pred = noise_pred.reshape(B, self.in_channels, self.img_size, self.img_size)
        
        return noise_pred


if __name__ == "__main__":
    # Quick test: DiT-B level (114M params, Flash Attention)
    model = ConditionalDiT(img_size=256, patch_size=8, hidden_dim=768, num_heads=12, num_layers=12)
    x = torch.randn(2, 1, 256, 256)
    t = torch.randint(0, 1000, (2,))
    c = torch.randint(0, 10, (2,))
    
    out = model(x, t, c)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
