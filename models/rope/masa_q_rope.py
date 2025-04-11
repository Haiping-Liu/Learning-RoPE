import torch
import torch.nn as nn
from .standard_rope import apply_rotary_emb
from timm.models.vision_transformer import Attention

class CayleyLearnerPerHead(nn.Module):
    def __init__(self, num_heads: int, head_dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(num_heads, head_dim, head_dim))

    def forward(self):
        A = self.W - self.W.transpose(-1, -2)  # Ensure skew-symmetry per head
        I = torch.eye(A.size(-1), device=A.device).expand_as(A)
        Q = torch.linalg.solve((I + A).T, (I - A).T).T  # Cayley transform
        return Q  # shape: (num_heads, head_dim, head_dim)


class GivensRotationPerHead(nn.Module):
    def __init__(self, num_heads: int, dim: int, num_rotations: int = 16):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim

        # 默认选择所有可能的 Givens 旋转
        if num_rotations is None:
            num_rotations = dim * (dim - 1) // 2
        self.rotation_indices = self._create_rotation_indices(dim, num_rotations)

        # 初始化旋转角度 theta（每个头独立一组）
        self.thetas = nn.Parameter(torch.randn(num_heads, len(self.rotation_indices)) * 0.01)

    def _create_rotation_indices(self, n, num_rotations):
        """生成 Givens 旋转的坐标对 (i, j)，确保 i < j"""
        indices = []
        for i in range(n):
            for j in range(i + 1, n):
                indices.append((i, j))
                if len(indices) >= num_rotations:
                    return indices
        return indices

    def apply_batch_givens(self, Q, thetas, indices):
        """批量应用 Givens 旋转到 Q，保持计算图安全"""
        Q_new = Q.clone()  # 避免 in-place 写操作破坏 autograd

        for k, (i, j) in enumerate(indices):
            theta = thetas[:, k]  # (H,)
            c = torch.cos(theta).unsqueeze(-1)  # (H, 1)
            s = torch.sin(theta).unsqueeze(-1)  # (H, 1)

            # Clone 各行避免 inplace 导致反向出错
            Qi = Q_new[:, i, :].clone()
            Qj = Q_new[:, j, :].clone()

            Q_new[:, i, :] = c * Qi + s * Qj
            Q_new[:, j, :] = -s * Qi + c * Qj

        return Q_new

    def forward(self):
        """生成每个 head 的正交旋转矩阵 Q，shape: (H, D, D)"""
        device = self.thetas.device
        Q_init = torch.eye(self.dim, device=device).expand(self.num_heads, self.dim, self.dim).clone()
        return self.apply_batch_givens(Q_init, self.thetas, self.rotation_indices)

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
    def __init__(self, freqs_cis: torch.Tensor, learner_type: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        learner_map = {
            'cayley': CayleyLearnerPerHead,
            'givens': GivensRotationPerHead,
            'householder': HouseholderPerHead
        }
        self.Q_learner = learner_map[learner_type](self.num_heads, self.head_dim)
        self.freqs_cis = freqs_cis

    def forward(self, x: torch.Tensor):
        self.freqs_cis = self.freqs_cis.to(x.device)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q[:, :, 1:], k[:, :, 1:] = apply_rotary_emb(q[:, :, 1:], k[:, :, 1:], freqs_cis=self.freqs_cis)

        Q = self.Q_learner()  # shape: (num_heads, head_dim, head_dim)

        q = torch.einsum('bhnc,hcd->bhnd', q, Q)
        k = torch.einsum('bhnc,hcd->bhnd', k, Q)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = self.attn_drop(attn.softmax(dim=-1))

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x
