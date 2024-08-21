from code.deit_modified_ghost_init import VisionTransformer
import numpy as np
import torch
import warnings

warnings.filterwarnings("error")

model = VisionTransformer(
    img_size=224,
    patch_size=16,
    num_classes=200,
    embed_dim=384,
    depth=6,
    num_heads=6,
    mlp_ratio=4,
    in_chans=3,
    qkv_bias=True,
    mha_width=1.0,
    mlp_width=1.0,
    no_ghost=False,
    ghost_mode="simple",
)

print(model.base_model_prefix)
inp = torch.tensor(np.random.rand(10, 3, 224, 224)).float()
out = model(inp)
print(out.shape)
print([k for k, v in model.named_parameters()])
