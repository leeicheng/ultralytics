from torch import nn
import torch

class GrayScaleLayer(nn.Module):
    """
    Input : B × 3 × H × W   （單張 RGB）
    Output: B × 1 × H × W   （灰階）
    """
    def __init__(self):
        super().__init__()
        # 固定係數轉灰階
        w = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
        self.register_buffer("gray_w", w)   # 非參數，隨 model 存
        self.f = -1
        self.i = 0  # 稍後會被正確覆蓋
        self.type = "grayscale"

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        assert C == 3, "輸入必須是 3-channel RGB"
        # (B,3,H,W) → (B,1,H,W)
        gray = (x * self.gray_w).sum(dim=1, keepdim=True)
        return gray
