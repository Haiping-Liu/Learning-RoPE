import torch
from timm.models.vision_transformer import Attention
from typing import Tuple


def init_xy(end_x: int, end_y: int) -> torch.Tensor:
    """
        purpose: 
            from 'row*end_x + col' to get the x and y coordinates
        args:
            end_x: width of the image
            end_y: height of the image
        return:
            t_x: col
            t_x: raw
    """
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode='floor').float()
    return t_x, t_y

def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 100.0):
    """
        purpose:
            Compute the x and y frequency. It's the block style, causing the finaly attention ike [x1,x2,...,xN,y1,y2,...,yN]
            but not [x1,y1,x2,y2,...,xN,yN]
        arg:
            dim: query/key/value dimension
    """

    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: dim // 4].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: dim // 4].float() / dim))

    t_x, t_y = init_xy(end_x, end_y)
    freq_x = torch.outer(t_x, freqs_x)
    freq_y = torch.outer(t_y, freqs_y)
    freq_cis_x = torch.polar(torch.ones_like(freq_x), freq_x)
    freq_cis_y = torch.polar(torch.ones_like(freq_y), freq_y)                                   
    return torch.cat([freq_cis_x, freq_cis_y], dim=-1)

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
        args:
            freqs_cis: [length, head_dim / 2] or [num_heads, length, head_dim / 2]
            x: [batch_size, num_heads, length, head_dim / 2]
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-2 else 1 for i, d in enumerate(x.shape)]  # [1, 1, length, head_dim/2, 2]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-3 else 1 for i, d in enumerate(x.shape)]# [1, num_heads, length, head_dim/2, 2]
        
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
        purpose:
            broadcast the freqs_cis and apply the rotary embedding to the query and key
        args:
            xq/xk: [batch_size, num_heads, length, head_dim]
            freqs_cis: [length, head_dim / 2] or [num_heads, length, head_dim / 2]
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)) # [batch_size, num_heads, length, head_dim/2]
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_) # [1, num_heads, length, head_dim/2]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3) # merge the dimesions from the third dimension to the last dimension
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3) # [batch_size, num_heads, length, head_dim/2, 2] -> [batch_size, num_heads, length, head_dim]
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device) # [batch_size, num_heads, length, head_dim]

class RoPEAttention(Attention):
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        B, N, C = x.shape
        "(B, N, 3, num_heads, head_dim) -> (3, B, num_heads, N, head_dim)"
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) 
        "(B, num_heads, N, head_dim)"
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # drop cls when apply rotary embedding
        q[:, :, 1:], k[:, :, 1:] = apply_rotary_emb(q[:, :, 1:], k[:, :, 1:], freqs_cis=freqs_cis)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x