import torch
import torch.nn as nn
from models.rope.standard_rope import apply_rotary_emb

class CayleyLearnerPerHead(nn.Module):
    def __init__(self, num_heads: int, head_dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(num_heads, head_dim, head_dim))

    def forward(self):
        A = self.W - self.W.transpose(-1, -2)  # Ensure skew-symmetry per head
        I = torch.eye(A.size(-1), device=A.device).expand_as(A)
        Q = torch.linalg.solve(I + A, I - A)  # Cayley transform
        return Q  # shape: (num_heads, head_dim, head_dim)


class GivensRotationPerHead(nn.Module):
    def __init__(self, num_heads: int, dim: int, num_rotations: int = None):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        if num_rotations is None:
            num_rotations = dim * (dim - 1) // 2
        self.rotation_indices = self._create_rotation_indices(dim, num_rotations)
        self.thetas = nn.Parameter(torch.randn(num_heads, len(self.rotation_indices)) * 0.01)

    def _create_rotation_indices(self, n, num_rotations):
        indices = []
        for i in range(n):
            for j in range(i + 1, n):
                indices.append((i, j))
                if len(indices) >= num_rotations:
                    return indices
        return indices

    def forward(self):
        device = self.thetas.device
        Q = torch.eye(self.dim, device=device).expand(self.num_heads, self.dim, self.dim).clone()  # (H, D, D)
        for rot_idx, (i, j) in enumerate(self.rotation_indices):
            theta = self.thetas[:, rot_idx]  # (H,)
            c = torch.cos(theta).unsqueeze(-1)  # (H, 1)
            s = torch.sin(theta).unsqueeze(-1)  # (H, 1)

            Qi = Q[:, i, :].clone()
            Qj = Q[:, j, :].clone()
            Q[:, i, :] = c * Qi + s * Qj
            Q[:, j, :] = -s * Qi + c * Qj

        return Q  # shape: (H, D, D)


class HouseholderPerHead(nn.Module):
    def __init__(self, num_heads: int, dim: int, num_reflections: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.k = num_reflections
        self.vs = nn.Parameter(torch.randn(num_heads, self.k, dim))

    def forward(self):
        Q = torch.eye(self.dim, device=self.vs.device).expand(self.num_heads, self.dim, self.dim).clone()  # (H, D, D)

        for i in range(self.k):
            v = self.vs[:, i, :]  # (H, D)
            v = v / v.norm(dim=-1, keepdim=True)  # Normalize each head's vector
            H = torch.eye(self.dim, device=v.device) - 2 * torch.einsum("hi,hj->hij", v, v)  # (H, D, D)
            Q = torch.einsum("hij,hjk->hik", H, Q)

        return Q  # shape: (H, D, D)


class LearnerRoPEAttention(Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Q_learner = CayleyLearnerPerHead(self.num_heads, self.head_dim)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q[:, :, 1:], k[:, :, 1:] = apply_rotary_emb(q[:, :, 1:], k[:, :, 1:], freqs_cis=freqs_cis)

        Q = self.Q_learner()  # shape: (num_heads, head_dim, head_dim)

        q = torch.einsum('bhnc,hcd->bhnd', q, Q)
        k = torch.einsum('bhnc,hcd->bhnd', k, Q)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = self.attn_drop(attn.softmax(dim=-1))

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x, Q
