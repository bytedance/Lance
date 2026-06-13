# Copyright (c) 2023 OpenGVLab
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: MIT
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025-05-20.
#
# Original file was released under MIT, with the full license text
# available at https://github.com/OpenGVLab/InternVL/blob/main/LICENSE.
#
# This modified file is released under the same license.


import io
import os
import random
import re

import numpy as np
import decord
from PIL import Image


def get_frame_indices(num_frames, vlen, sample='rand', fix_start=None, input_fps=1, max_num_frames=-1):
    if sample in ['rand', 'middle']: # uniform sampling
        acc_samples = min(num_frames, vlen)
        # split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == 'rand':
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif sample == 'middle':
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        if len(frame_indices) < num_frames:  # padded with last frame
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[:len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
    elif 'fps' in sample:  # fps0.5, sequentially sample frames at 0.5 fps
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        delta = 1 / output_fps  # gap between frames, this is also the clip length each frame represents
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            frame_indices = frame_indices[:max_num_frames]
    else:
        raise ValueError
    return frame_indices


def read_frames_decord(video_path, num_frames, sample='rand', fix_start=None, clip=None, min_num_frames=4):
    video_reader = decord.VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    duration = vlen / float(fps)
    if clip:
        start, end = clip
        duration = end - start
        vlen = int(duration * fps)
        start_index = int(start * fps)

    t_num_frames = np.random.randint(min_num_frames, num_frames + 1)

    frame_indices = get_frame_indices(
        t_num_frames, vlen, sample=sample, fix_start=fix_start,
        input_fps=fps
    )
    if clip:
        frame_indices = [f + start_index for f in frame_indices]
    frames = video_reader.get_batch(frame_indices).asnumpy()  # (T, H, W, C), np.uint8
    frames = [Image.fromarray(frames[i]) for i in range(frames.shape[0])]
    return frames


def extract_frame_number(filename):
    # Extract the numeric part from the filename using regular expressions
    match = re.search(r'_(\d+).jpg$', filename)
    return int(match.group(1)) if match else -1


def sort_frames(frame_paths):
    # Extract filenames from each path and sort by their numeric part
    return sorted(frame_paths, key=lambda x: extract_frame_number(os.path.basename(x)))


def read_frames_folder(video_path, num_frames, sample='rand', fix_start=None, min_num_frames=4):
    image_list = sort_frames(list(os.listdir(video_path)))
    frames = []
    for image in image_list:
        fp = os.path.join(video_path, image)
        frame = Image.open(fp).convert('RGB')
        frames.append(frame)
    vlen = len(frames)

    t_num_frames = np.random.randint(min_num_frames, num_frames + 1)

    if vlen > t_num_frames:
        frame_indices = get_frame_indices(
            t_num_frames, vlen, sample=sample, fix_start=fix_start
        )
        frames = [frames[i] for i in frame_indices]
    return frames


class FrameSampler:
    def __init__(self, max_num_frames=-1, min_num_frames=8, sample='rand'):
        self.max_num_frames = max_num_frames
        self.min_num_frames = min_num_frames
        self.sample = sample
    
    def __call__(self, file_name):
        fn = read_frames_folder if file_name.endswith('/') else read_frames_decord
        frames = fn(file_name, num_frames=self.max_num_frames, min_num_frames=self.min_num_frames, sample=self.sample)
        return frames


def decode_video_byte(video_bytes):
    video_stream = io.BytesIO(video_bytes)
    vr = decord.VideoReader(video_stream)
    return vr


def sample_mp4_frames(mp4_p, n_frames=None, fps=None, return_frame_indices=False, random_sample=False):
    if isinstance(mp4_p, str):
        vr = decord.VideoReader(mp4_p, num_threads=1)
    elif isinstance(mp4_p, decord.video_reader.VideoReader):
        vr = mp4_p
    video_fps = vr.get_avg_fps()  # 获取视频的帧率
    video_duration = len(vr) / video_fps
    if n_frames is not None:
        if random_sample:
            frame_indices = sorted(random.sample(range(len(vr)), n_frames))
        else:
            frame_indices = np.linspace(0, len(vr)-1, n_frames, dtype=int).tolist()
    else:
        frame_indices = [int(i) for i in np.arange(0, len(vr)-1, video_fps/fps)]
    frames = vr.get_batch(frame_indices).asnumpy()  # 转换为 numpy 数组
    frames = [Image.fromarray(frame).convert("RGB") for frame in frames]
    if not return_frame_indices:
        return frames, video_duration
    else:
        return frames, video_duration, frame_indices


def sample_mp4_frames_by_indices(mp4_p, frame_indices: list):
    if isinstance(mp4_p, str):
        vr = decord.VideoReader(mp4_p, num_threads=1)
    elif isinstance(mp4_p, decord.video_reader.VideoReader):
        vr = mp4_p
    # sample the frames in frame_indices
    frames = vr.get_batch(frame_indices).asnumpy()  # 转换为 numpy 数组
    frames = [Image.fromarray(frame).convert("RGB") for frame in frames]
    return frames