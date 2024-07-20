import torch.nn as nn
import torch
import torch.nn.functional as F

from .CBAM import CBAM
class EGCM(nn.Module):
    def __init__(self, in_channels,H,W):
        super(EGCM, self).__init__()

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, 1, 1,groups=in_channels//16),
            # nn.LayerNorm([in_channels,H,W],elementwise_affine=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))

        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid())

        self.cbam = CBAM(in_channels)

    def forward(self, edge_input, x, fft_feature):
        residual = x
        xsize=x.shape[1:3]

        edge_input = F.interpolate(edge_input, size=xsize, mode='bilinear', align_corners=True)
        edge_input=edge_input.permute(0,2,3,1)
        edge_feature = x * edge_input

        fusion_feature = torch.cat([edge_feature,fft_feature], dim=3).permute(0,3,1,2)
        fusion_feature = self.fusion_conv(fusion_feature)

        attention_map = self.attention(fusion_feature)
        fusion_feature = fusion_feature * attention_map

        fusion_feature=fusion_feature.permute(0,2,3,1)
        out = fusion_feature + residual
        out=out.permute(0,3,1,2)
        out = self.cbam(out)
        out=out.permute(0,2,3,1)
        return out



if __name__=='__main__':
    egcm=EGCM(768,64,64)
    sum_param = 0
    for name, p in egcm.named_parameters():
        sum_param += p.numel()
    print(f'{sum_param/(2**20)}M')
