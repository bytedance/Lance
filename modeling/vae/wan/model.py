# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# coding: utf-8

__all__ = ['WanVideoVAE']

from typing import List
import torch
from torch import Tensor
from einops import rearrange

from common.utils.logging import get_logger
from common.utils.distributed import get_device
from common.utils.misc import AutoEncoderParams
from .vae2_2 import Wan2_2_VAE


def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return eps * std + mu


# ---------------------------------------------------------------------------
# Spatial-tiled VAE decode (see TILED_VAE.md).
#
# The video VAE decode's conv activations for a single frame at full resolution
# OOM a 12 GB card above ~480-512^2. The decode is already streamed temporally
# (one latent frame at a time), so the remaining peak is purely spatial. Tiling
# the latent spatially, decoding each tile through the existing (temporally
# streamed) decode, and feather-blending the outputs bounds the per-tile memory
# to a small frame, lifting the resolution ceiling.
# ---------------------------------------------------------------------------

# Latent spatial size (cells) above which auto-tiling kicks in (vae_tile_size==0).
# 480^2 -> h=30 fits plainly; 512^2 -> 32 fits; 768^2 -> 48 OOMs. Threshold sits between.
_VAE_AUTO_TILE_THRESHOLD = 36
_VAE_DEFAULT_TILE = 32       # latent cells per tile (512 px output at 16x upsample)
_VAE_DEFAULT_OVERLAP = 8     # latent cells of overlap between adjacent tiles


def _tile_starts(n: int, tile: int, stride: int) -> List[int]:
    """Start indices of tiles covering [0, n); the last tile is snapped to the
    edge so the whole extent is covered even when n is not a multiple of stride."""
    if n <= tile:
        return [0]
    starts = list(range(0, n - tile + 1, stride))
    if starts[-1] != n - tile:
        starts.append(n - tile)
    return starts


def _blend_ramp_1d(length: int, ramp: int, ramp_lo: bool, ramp_hi: bool,
                   device, dtype) -> Tensor:
    """1-D blend weight: 1.0 everywhere, linearly ramped toward (but not to) 0 on
    edges that overlap a neighbor. Two adjacent tiles' opposing ramps span the same
    overlap band and sum to ~1; the caller's weight-sum normalization makes the
    blend exact regardless, while single-coverage regions stay at weight 1."""
    w = torch.ones(length, device=device, dtype=dtype)
    r = min(ramp, length // 2)
    if r > 0:
        # values in (0, 1): 1/(r+1) .. r/(r+1) — never exactly 0, so wsum > 0.
        vals = torch.linspace(1.0 / (r + 1), r / (r + 1), r, device=device, dtype=dtype)
        if ramp_lo:
            w[:r] = vals
        if ramp_hi:
            w[length - r:] = vals.flip(0)
    return w


class WanVideoVAE(object):
    __version__ = "v2.2"
    __name__ = "WanVideoVAE"
    __logger__ = None

    def __init__(self, config_path: str = "", **kwargs) -> None:
        if self.__class__.__logger__ is None:
            self.__class__.__logger__ = get_logger(self.__class__.__name__)
        self.logger = self.__class__.__logger__

        self.dtype = kwargs.get("dtype", torch.bfloat16)
        # Allow the VAE to live on a card other than cuda:LOCAL_RANK. Under
        # model-parallel sharding, cuda:0 is the most crowded device (embed, lm_head,
        # ViT, first LLM layers), and the video VAE decode's conv activations OOM it.
        # Placing the VAE on the lightest shard gives the decode room to breathe.
        # Defaults to get_device() so single-GPU behavior is unchanged.
        self.device = torch.device(kwargs.get("device", get_device()))
        self.configure_vae_model()
        self.use_sample = kwargs.get("use_sample", True)

        # Spatial-tiled decode config (latent cells). See TILED_VAE.md.
        #   tile_size > 0 : tile whenever max(h, w) > tile_size
        #   tile_size == 0: auto — tile when max(h, w) > _VAE_AUTO_TILE_THRESHOLD
        #   tile_size <  0: never tile (force plain decode)
        self.tile_size = int(kwargs.get("tile_size", 0) or 0)
        self.tile_overlap = int(kwargs.get("tile_overlap", _VAE_DEFAULT_OVERLAP))

        # wan vae2.2 config is equal to seedance vae
        self.vae_config = AutoEncoderParams(
            downsample_spatial=16,
            downsample_temporal=4,
            z_channels=48,
            # scale_factor=1.0,
            # shift_factor=0.012,
        )

    def configure_vae_model(self):
        device = self.device

        # Read the VAE path from path_default.yaml.
        try:
            from config.config_factory import get_model_path
            vae_path = get_model_path("vae.wan")
        except Exception as e:
            # Fall back to the default local path.
            vae_path = "downloads/Wan2.2_VAE.pth"

        self.vae: Wan2_2_VAE = Wan2_2_VAE(vae_pth=vae_path, device=device, dtype=self.dtype)
        # self.vae.requires_grad_(False).eval()
        # self.vae.to(device=get_device())

    def to(self, device) -> "WanVideoVAE":
        self.device = torch.device(device)
        self.vae.model.to(device=self.device, dtype=self.dtype)
        self.vae.scale = [item.to(device=self.device) for item in self.vae.scale]
        return self

    @torch.no_grad()
    def vae_encode(self, samples: List[Tensor], **kwargs) -> List[Tensor]:
        device = self.device

        latents = []
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            for x in samples:
                x = x.to(device=device).unsqueeze(0)  # 1CTHW

                u, log_var = self.vae.encode(x)  # [1,48,t,h,w], [1,48,t,h,w]

                if self.use_sample:
                    u = reparameterize(u, log_var)  # [1,48,t,h,w]

                u = rearrange(u, "b c ... -> b ... c")  # -> [1,t,h,w,48] for compatibility

                latents.append(u.squeeze(0))  # -> [t,h,w,48]

            return latents

    def _should_tile(self, u: Tensor) -> bool:
        """Decide whether to spatially tile the decode of latent u [1,48,t,h,w]."""
        if self.tile_size < 0:
            return False
        h, w = u.shape[-2], u.shape[-1]
        threshold = self.tile_size if self.tile_size > 0 else _VAE_AUTO_TILE_THRESHOLD
        return max(h, w) > threshold

    def _tiled_decode(self, u: Tensor) -> Tensor:
        """Decode latent u [1,48,t,h,w] in overlapping spatial tiles and
        feather-blend into the full output. Each tile reuses self.vae.decode,
        which resets its own temporal feat_cache, so every tile is a correct
        independent temporal stream. Returns [1,3,T,H,W]."""
        _, _, _, h, w = u.shape
        tile = self.tile_size if self.tile_size > 0 else _VAE_DEFAULT_TILE
        # overlap must leave a positive stride and fit within a tile
        overlap = max(0, min(self.tile_overlap, tile // 2 - 1))
        stride = max(1, tile - overlap)

        row_starts = _tile_starts(h, tile, stride)
        col_starts = _tile_starts(w, tile, stride)

        canvas = None
        wsum = None
        f = None  # spatial upsample factor (pixels per latent cell), inferred from first tile
        for r0 in row_starts:
            r1 = min(r0 + tile, h)
            for c0 in col_starts:
                c1 = min(c0 + tile, w)
                out = self.vae.decode(u[:, :, :, r0:r1, c0:c1])  # [1,3,T,(r1-r0)*f,(c1-c0)*f]

                if canvas is None:
                    f = out.shape[-2] // (r1 - r0)
                    T_out, C_out = out.shape[2], out.shape[1]
                    H, W = h * f, w * f
                    canvas = torch.zeros((1, C_out, T_out, H, W), dtype=out.dtype, device=out.device)
                    wsum = torch.zeros((1, 1, 1, H, W), dtype=out.dtype, device=out.device)

                py0, py1, px0, px1 = r0 * f, r1 * f, c0 * f, c1 * f
                wy = _blend_ramp_1d(py1 - py0, overlap * f, ramp_lo=(r0 != 0), ramp_hi=(r1 != h),
                                    device=out.device, dtype=out.dtype)
                wx = _blend_ramp_1d(px1 - px0, overlap * f, ramp_lo=(c0 != 0), ramp_hi=(c1 != w),
                                    device=out.device, dtype=out.dtype)
                w2d = (wy[:, None] * wx[None, :])[None, None, None, :, :]  # [1,1,1,ph,pw]

                canvas[:, :, :, py0:py1, px0:px1] += out * w2d
                wsum[:, :, :, py0:py1, px0:px1] += w2d
                del out

        return canvas / wsum.clamp(min=1e-6)

    @torch.no_grad()
    def vae_decode(self, latents: List[Tensor], **kwargs) -> List[Tensor]:
        device = self.device

        samples = []
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            for u in latents:
                u = u.unsqueeze(0).to(device=device)  # -> [1,t,h,w,48]
                u = rearrange(u, "b ... c -> b c ...")  # -> [1,48,t,h,w]

                if self._should_tile(u):
                    x_hat = self._tiled_decode(u)  # -> [1,3,T,H,W]
                else:
                    x_hat = self.vae.decode(u)  # -> [1,3,T,H,W]

                samples.append(x_hat.squeeze(0))  # -> List[[3,T,H,W]]

            return samples
