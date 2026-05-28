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

from itertools import chain
from typing import Dict, List, Tuple
import einops
import torch


def rearrange(
    hid: torch.FloatTensor,  # (L c)
    hid_shape: torch.LongTensor,  # (b n)
    pattern: str,
    **kwargs: Dict[str, int],
) -> Tuple[
    torch.FloatTensor,
    torch.LongTensor,
]:
    return flatten([einops.rearrange(h, pattern, **kwargs) for h in unflatten(hid, hid_shape)])


def repeat(
    hid: torch.FloatTensor,  # (L c)
    hid_shape: torch.LongTensor,  # (b n)
    pattern: str,
    **kwargs: Dict[str, torch.LongTensor],  # (b)
) -> Tuple[
    torch.FloatTensor,
    torch.LongTensor,
]:
    hid = unflatten(hid, hid_shape)
    kwargs = [{k: v[i].item() for k, v in kwargs.items()} for i in range(len(hid))]
    return flatten([einops.repeat(h, pattern, **a) for h, a in zip(hid, kwargs)])


def pack(
    samples: List[torch.Tensor],  # List of (h w c).
) -> Tuple[
    List[torch.Tensor],  # groups [(b1 h1 w1 c1), (b2 h2 w2 c2)]
    List[List[int]],  # reversal indices.
]:
    batches = {}
    indices = {}
    for i, sample in enumerate(samples):
        shape = sample.shape
        batches[shape] = batches.get(shape, [])
        indices[shape] = indices.get(shape, [])
        batches[shape].append(sample)
        indices[shape].append(i)

    batches = list(map(torch.stack, batches.values()))
    indices = list(indices.values())
    return batches, indices


def unpack(
    batches: List[torch.Tensor],
    indices: List[List[int]],
) -> List[torch.Tensor]:
    samples = [None] * (max(chain(*indices)) + 1)
    for batch, index in zip(batches, indices):
        for sample, i in zip(batch.unbind(), index):
            samples[i] = sample
    return samples


# Keep these helpers because rearrange and repeat depend on them.
def flatten(
    hid: List[torch.FloatTensor],  # List of (*** c)
) -> Tuple[
    torch.FloatTensor,  # (L c)
    torch.LongTensor,  # (b n)
]:
    assert len(hid) > 0
    shape = torch.stack([torch.tensor(x.shape[:-1], device=hid[0].device) for x in hid])
    hid = torch.cat([x.flatten(0, -2) for x in hid])
    return hid, shape


def unflatten(
    hid: torch.FloatTensor,  # (L c) or (L ... c)
    hid_shape: torch.LongTensor,  # (b n)
) -> List[torch.Tensor]:  # List of (*** c) or (*** ... c)
    hid_len = hid_shape.prod(-1)
    hid = hid.split(hid_len.tolist())
    hid = [x.unflatten(0, s.tolist()) for x, s in zip(hid, hid_shape)]
    return hid
