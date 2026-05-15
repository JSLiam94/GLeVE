# medformer3d.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# ---------------------------
# Small helper blocks (3D)
# ---------------------------
class ConvNormAct3D(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, norm=nn.InstanceNorm3d, act=nn.GELU):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, k, stride=s, padding=p, bias=False)
        self.norm = norm(out_ch) if norm is not None else nn.Identity()
        self.act = act() if act is not None else nn.Identity()
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class BasicBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, norm=nn.InstanceNorm3d, act=nn.GELU):
        super().__init__()
        self.c1 = ConvNormAct3D(in_ch, out_ch, 3, 1, 1, norm, act)
        self.c2 = ConvNormAct3D(out_ch, out_ch, 3, 1, 1, norm, act)
        self.short = nn.Identity() if in_ch == out_ch else nn.Conv3d(in_ch, out_ch, 1, bias=False)
    def forward(self, x):
        return self.c2(self.c1(x)) + self.short(x)

class PatchMerging3D(nn.Module):
    """Downsample by 2 in each axis (H,W,D)."""
    def __init__(self, in_ch, out_ch, norm=nn.InstanceNorm3d):
        super().__init__()
        self.norm = norm(in_ch * 8)
        self.reduction = nn.Conv3d(in_ch * 8, out_ch, 1, bias=False)

    def forward(self, x):
        # x: B,C,H,W,D
        x000 = x[:, :, 0::2, 0::2, 0::2]
        x001 = x[:, :, 0::2, 0::2, 1::2]
        x010 = x[:, :, 0::2, 1::2, 0::2]
        x011 = x[:, :, 0::2, 1::2, 1::2]
        x100 = x[:, :, 1::2, 0::2, 0::2]
        x101 = x[:, :, 1::2, 0::2, 1::2]
        x110 = x[:, :, 1::2, 1::2, 0::2]
        x111 = x[:, :, 1::2, 1::2, 1::2]
        x = torch.cat([x000,x001,x010,x011,x100,x101,x110,x111], dim=1)  # B,8C,H/2,W/2,D/2
        x = self.reduction(self.norm(x))
        return x

class UpBlock3D(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, norm=nn.InstanceNorm3d, act=nn.GELU):
        super().__init__()
        self.reduction = nn.Conv3d(in_ch + skip_ch, out_ch, 1, bias=False)
        self.norm = norm(in_ch + skip_ch)
        self.b1 = BasicBlock3D(out_ch, out_ch, norm, act)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
        feat = torch.cat([x, skip], dim=1)
        out = self.reduction(self.norm(feat))
        out = self.b1(out)
        return out

# ---------------------------
# MedFormer3D (practical)
# ---------------------------
class MedFormer3D(nn.Module):
    """
    Practical 3D MedFormer-style encoder-decoder:
    - keeps the multi-stage down/up pattern and provides feature maps
    - does not re-implement 2D semantic map attention exactly (too long),
      but keeps "MedFormer" as the visual backbone with encoder/decoder and skip fusions.

    Returns:
      logits: (B,num_classes,H,W,D) or [logits, aux_logits] when aux_loss=True
    Also provides:
      forward_features(x) -> feat (B,C,H/4,W/4,D/4), aux dict with skips.
      decode_features(aux) -> logits from decoder using cached encoder features.
      forward_with_features(x) -> logits_or_list, feat_mid, aux
    """
    def __init__(
        self,
        in_chan=1,
        num_classes=1,
        base_chan=32,
        mid_channels=128,
        norm=nn.InstanceNorm3d,
        act=nn.GELU,
        aux_loss=True,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        c1 = base_chan
        c2 = 2*base_chan
        c3 = 4*base_chan
        c4 = 8*base_chan
        c5 = 16*base_chan

        self.stem = nn.Sequential(
            ConvNormAct3D(in_chan, c1, 3, 1, 1, norm, act),
            BasicBlock3D(c1, c1, norm, act),
        )

        self.down1 = nn.Sequential(PatchMerging3D(c1, c2, norm), BasicBlock3D(c2, c2, norm, act))
        self.down2 = nn.Sequential(PatchMerging3D(c2, c3, norm), BasicBlock3D(c3, c3, norm, act))
        self.down3 = nn.Sequential(PatchMerging3D(c3, c4, norm), BasicBlock3D(c4, c4, norm, act))
        self.down4 = nn.Sequential(PatchMerging3D(c4, c5, norm), BasicBlock3D(c5, c5, norm, act))

        # decoder
        self.up1 = UpBlock3D(c5, c4, c4, norm, act)
        self.up2 = UpBlock3D(c4, c3, c3, norm, act)
        self.up3 = UpBlock3D(c3, c2, c2, norm, act)
        self.up4 = UpBlock3D(c2, c1, c1, norm, act)

        self.aux_loss = aux_loss
        if aux_loss:
            self.aux_out = nn.Conv3d(c3, num_classes, 1)

        self.outc = nn.Conv3d(c1, num_classes, 1)

        # expose a mid-level feature channel for GLeVE similarity (use c3 by default)
        self.mid_proj = nn.Conv3d(c3, mid_channels, 1, bias=False)

    def _run_block(self, module: nn.Module, *args: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training and any(arg.requires_grad for arg in args if torch.is_tensor(arg)):
            return checkpoint(lambda *inputs: module(*inputs), *args, use_reentrant=False)
        return module(*args)

    def forward_features(self, x):
        x0 = self.stem(x)         # B,c1,H,W,D
        x1 = self._run_block(self.down1, x0)       # B,c2,H/2,W/2,D/2
        x2 = self._run_block(self.down2, x1)       # B,c3,H/4,W/4,D/4
        x3 = self._run_block(self.down3, x2)       # B,c4,H/8,W/8,D/8
        x4 = self._run_block(self.down4, x3)       # B,c5,H/16,W/16,D/16

        feat_mid = self.mid_proj(x2)  # B,128,H/4,W/4,D/4
        aux = {"x0": x0, "x1": x1, "x2": x2, "x3": x3, "x4": x4}
        return feat_mid, aux

    def decode_features(self, aux):
        x4 = aux["x4"]; x3 = aux["x3"]; x2 = aux["x2"]; x1 = aux["x1"]; x0 = aux["x0"]

        d1 = self._run_block(self.up1, x4, x3)
        d2 = self._run_block(self.up2, d1, x2)
        aux_logits = None
        if self.aux_loss:
            aux_logits = self.aux_out(d2)
            aux_logits = F.interpolate(aux_logits, size=x0.shape[-3:], mode="trilinear", align_corners=False)
        d3 = self._run_block(self.up3, d2, x1)
        d4 = self._run_block(self.up4, d3, x0)
        logits = self.outc(d4)
        if self.aux_loss:
            return [logits, aux_logits]
        return logits

    def forward_with_features(self, x):
        feat_mid, aux = self.forward_features(x)
        logits = self.decode_features(aux)
        return logits, feat_mid, aux

    def forward(self, x):
        logits_or_list, _, _ = self.forward_with_features(x)
        return logits_or_list
