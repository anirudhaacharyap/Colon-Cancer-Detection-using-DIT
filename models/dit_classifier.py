import torch
import torch.nn as nn
import math

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):
    """
    Dummy embedder to mimic the class embedding conditioning in DiT.
    Since we are doing classification, we condition on a single learnable 'task' token.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(1, hidden_size)

    def forward(self, x):
        batch_size = x.size(0)
        task_id = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        return self.embedding(task_id)

class DiTBlock(nn.Module):
    """
    A DiT block with adaLN-Zero conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_size)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        # Initialize adaLN_modulation to zero (adaLN-Zero)
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Attention
        norm1_x = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(norm1_x, norm1_x, norm1_x)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # MLP
        norm2_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(norm2_x)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        
        return x

class DiTClassifier(nn.Module):
    """
    Diffusion Transformer (DiT) adapted for binary classification.
    Takes 1D feature vectors, splits into patches, applies DiT blocks,
    and outputs class logits.
    """
    def __init__(self, feature_dim=3840, patch_size=64, hidden_dim=256, depth=6, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        if feature_dim % patch_size != 0:
            raise ValueError("feature_dim must be divisible by patch_size")
            
        self.num_patches = feature_dim // patch_size
        self.patch_size = patch_size
        
        # Project patches to hidden_dim
        self.patch_embed = nn.Linear(patch_size, hidden_dim)
        
        # CLS token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, hidden_dim))
        
        # Conditioning embedding (mimics class/timestep embedding of DiT)
        self.condition_embed = TimestepEmbedder(hidden_dim)
        
        # DiT Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        
        # Classification Head
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.head = nn.Linear(hidden_dim, 2)
        
        self._init_weights()

    def _init_weights(self):
        # Initialize pos_embed
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize patch_embed
        nn.init.xavier_uniform_(self.patch_embed.weight)
        nn.init.constant_(self.patch_embed.bias, 0)
        
        # Initialize head
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.constant_(self.head.bias, 0)

    def forward(self, x_features):
        """
        Args:
            x_features (Tensor): (B, feature_dim) where feature_dim=3840 typically.
        Returns:
            Tensor: (B, 2) classification logits.
        """
        B = x_features.shape[0]
        
        # Reshape to sequence of patches
        # (B, 3840) -> (B, num_patches, patch_size)
        x = x_features.view(B, self.num_patches, self.patch_size)
        
        # Project
        x = self.patch_embed(x) # (B, num_patches, hidden_dim)
        
        # Append CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # (B, num_patches + 1, hidden_dim)
        
        # Add position embeddings
        x = x + self.pos_embed
        
        # Conditioning
        c = self.condition_embed(x)
        
        # Apply DiT Blocks
        for block in self.blocks:
            x = block(x, c)
            
        # Final layer norm on CLS token
        x_cls = self.norm_final(x[:, 0])
        
        # Classification head
        logits = self.head(x_cls)
        return logits
