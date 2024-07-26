# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Tuple
from mmdet.registry import MODELS



class MLPBlock(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            mlp_dim: int,
            act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
@MODELS.register_module(force=True)
class LayerNorm2d(nn.Module):
    """A LayerNorm variant, popularized by Transformers, that performs
    pointwise mean and variance normalization over the channel dimension for
    inputs that have shape (batch_size, channels, height, width)."""

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class Linear_Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)

    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x


class Adapter(nn.Module):
    def __init__(self,
                 in_dim, embeding_feature=48):
        super().__init__()

        self.project1 = nn.Linear(in_dim, embeding_feature)
        self.nonlinear = F.gelu
        self.project2 = nn.Linear(embeding_feature, in_dim)

        self.dropout = nn.Dropout(p=0.1)

        self.conv1 = nn.Conv2d(embeding_feature, embeding_feature, kernel_size=3, padding=3 // 2,
                               groups=embeding_feature)
        self.conv2 = nn.Conv2d(embeding_feature, embeding_feature, kernel_size=3, padding=3 // 2,
                               groups=embeding_feature)
        self.conv3 = nn.Conv2d(embeding_feature, embeding_feature, kernel_size=3, padding=3 // 2,
                               groups=embeding_feature)

        self.norm = nn.LayerNorm(in_dim)
        self.gamma = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.belta = nn.Parameter(torch.ones(in_dim))

    def forward(self, x, hw_shapes=None):
        identity = x
        x = self.norm(x) * self.gamma + x * self.belta
        project1 = self.project1(x)
        project1 = project1.permute(0, 3, 1, 2)

        identity2 = project1
        conv1_x = self.conv1(project1)
        conv2_x = self.conv2(project1)
        conv3_x = self.conv3(project1)
        project1 = (conv1_x + conv2_x + conv3_x) / 3.0 + identity2

        project1 = project1.permute(0, 2, 3, 1)
        nonlinear = self.nonlinear(project1)
        nonlinear = self.dropout(nonlinear)
        project2 = self.project2(nonlinear)
        return project2+identity



class features_feat(nn.Module):
    def __init__(
        self,
        vit_dim,
        transformer_dim=256,
        cnn_dim=256,
    ) -> None:
        super().__init__()
        self.embedding_encoder = nn.Sequential(
                                        nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
                                        LayerNorm2d(transformer_dim // 4),
                                        nn.GELU(),
                                        nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
                                    )
        self.compress_vit_feat = nn.Sequential(
                                        nn.ConvTranspose2d(vit_dim, transformer_dim, kernel_size=2, stride=2),
                                        LayerNorm2d(transformer_dim),
                                        nn.GELU(),
                                        nn.ConvTranspose2d(transformer_dim, transformer_dim // 8, kernel_size=2, stride=2))
        self.compress_cnn_feat = nn.Sequential(
                                        nn.ConvTranspose2d(cnn_dim, cnn_dim // 2, kernel_size=2, stride=2),
                                        LayerNorm2d(cnn_dim // 2),
                                        nn.GELU(),
                                        nn.ConvTranspose2d(cnn_dim // 2, cnn_dim // 4, kernel_size=2, stride=2),
                                        LayerNorm2d(cnn_dim // 4),
                                        nn.GELU(),
                                        nn.ConvTranspose2d(cnn_dim // 4, 32, kernel_size=2, stride=2),
                                    )

    def forward(
        self,
        cnn_feature: torch.Tensor,
        image_embeddings: torch.Tensor,
        interm_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        vit_features = interm_embeddings[2].permute(0, 3, 1, 2) # early-layer ViT feature, after 1st global attention block in ViT
        hq_features = self.embedding_encoder(image_embeddings) + self.compress_vit_feat(vit_features)
        cnn_features_feat = self.compress_cnn_feat(cnn_feature)
        cnn_features_feat = F.interpolate(cnn_features_feat, size=hq_features.shape[2:], mode='bilinear')
        return hq_features + cnn_features_feat


if __name__=="__main__":
    m1=features_feat(768)
    sum_param=0
    for name, p in m1.named_parameters():
        sum_param += p.numel()
    print(f'{sum_param / (2 ** 20)}M')