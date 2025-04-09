import timm
import torch
from timm.models.vision_transformer import VisionTransformer
from models.rope.standard_rope import RoPEAttention, compute_axial_cis
from models.rope.masa_q_rope import LearnerRoPEAttention

def create_vit_tiny(num_classes: int = 200, rope_type: str = 'standard'):
    """
        purpose:
            create a ViT tiny model, support multiple RoPE types
        args:
            num_classes: number of classes
            rope_type: RoPE type, optional 'standard' or 'learner'
    """
    model = VisionTransformer(
        img_size=64,
        patch_size=4,
        embed_dim=192,
        depth=1,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True, 
        num_classes=num_classes
    )
    
    num_patches = (64 // 4) ** 2  # img_size // patch_size
    freqs_cis = compute_axial_cis(
        dim=192 // 3,  # embed_dim // num_heads
        end_x=16,      # img_size // patch_size
        end_y=16
    )
    
    # replace all attention layers
    for block in model.blocks:
        if rope_type == 'standard':
            block.attn = RoPEAttention(
                dim=192,          # embed_dim
                num_heads=3,
                qkv_bias=True,
                attn_drop=0.,
                proj_drop=0.
            )
        elif rope_type == 'learner':
            block.attn = LearnerRoPEAttention(
                dim=192,
                num_heads=3,
                qkv_bias=True,
                attn_drop=0.,
                proj_drop=0.
            )
        else:
            raise ValueError(f"Unsupported rope_type: {rope_type}")
        
    # save the position encoding to the model
    model.register_buffer('freqs_cis', freqs_cis, persistent=False)
    
    # modify the forward method
    original_forward = model.forward

    def new_forward(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
         
        for blk in self.blocks:
            if isinstance(blk.attn, (LearnerRoPEAttention, RoPEAttention)):
                x = blk.attn(x, self.freqs_cis)
            else:
                x = blk.attn(x)
            y = blk.mlp(blk.norm2(x))

            if hasattr(blk, 'drop_path'):
                x = x + blk.drop_path(y)
            else:
                x = x + y

        x = self.norm(x)
        return self.head(x[:, 0])

    model.forward = type(model.forward)(new_forward, model)
    return model
