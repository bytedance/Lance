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


class WanVideoVAE(object):
    __version__ = "v2.2"
    __name__ = "WanVideoVAE"
    __logger__ = None

    def __init__(self, config_path: str = "", **kwargs) -> None:
        if self.__class__.__logger__ is None:
            self.__class__.__logger__ = get_logger(self.__class__.__name__)
        self.logger = self.__class__.__logger__

        self.dtype = kwargs.get("dtype", torch.bfloat16)
        self.configure_vae_model()
        self.use_sample = kwargs.get("use_sample", True)

        # wan vae2.2 config is equal to seedance vae
        self.vae_config = AutoEncoderParams(
            downsample_spatial=16,
            downsample_temporal=4,
            z_channels=48,
            # scale_factor=1.0,
            # shift_factor=0.012,
        )

    def configure_vae_model(self):
        device = get_device()

        # 从 path_default.yaml 读取 VAE 路径
        try:
            from config.config_factory import get_model_path
            vae_path = get_model_path("vae.wan")
        except Exception as e:
            # 降级到默认路径
            vae_path = "downloads/Wan2.2_VAE.pth"
        
        self.vae: Wan2_2_VAE = Wan2_2_VAE(vae_pth=vae_path, device=device, dtype=self.dtype)
        # self.vae.requires_grad_(False).eval()
        # self.vae.to(device=get_device())

    @torch.no_grad()
    def vae_encode(self, samples: List[Tensor], **kwargs) -> List[Tensor]:
        device = get_device()

        latents = []
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            for x in samples:
                x = x.to(device=device).unsqueeze(0)  # 1CTHW

                u, log_var = self.vae.encode(x)  # [1,48,t,h,w], [1,48,t,h,w]

                if self.use_sample:
                    u = reparameterize(u, log_var)  # [1,48,t,h,w]

                u = rearrange(u, "b c ... -> b ... c")  # -> [1,t,h,w,48] for 兼容

                latents.append(u.squeeze(0))  # -> [t,h,w,48]

            return latents

    @torch.no_grad()
    def vae_decode(self, latents: List[Tensor], **kwargs) -> List[Tensor]:
        device = get_device()

        samples = []
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            for u in latents:
                u = u.unsqueeze(0).to(device=device)  # -> [1,t,h,w,48]
                u = rearrange(u, "b ... c -> b c ...")  # -> [1,48,t,h,w]

                x_hat = self.vae.decode(u)  # -> [1,3,T,H,W]

                samples.append(x_hat.squeeze(0))  # -> List[[3,T,H,W]]

            return samples
