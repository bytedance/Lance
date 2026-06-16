# coding: utf-8

from abc import ABC, abstractmethod
import os
import random
from typing import Any, Dict, Iterator, List, Optional
import pyarrow.parquet as pq
from PIL import Image
import numpy as np
from io import BytesIO
import torch
from data.datasets_factory.distributed_iterable_dataset import DistributedIterableDataset
from data.parquet_utils import init_arrow_pf_fs, read_parquet_rows
from data.transforms import VideoTransform
from common.utils.logging import get_logger
import decord
import cv2
from decord import VideoReader
from data.video.sampler.frames import FrameSamplerOutput
from config.config_factory import TemplateArguments
from data.common import parse_videochat2it_doubao_caption


class TextCleaner:
    def __call__(self, text: Any) -> str:
        if text is None:
            return ""
        return str(text).strip()


def _collect_target_positions(element_dtype_array: List[str], target_modality: str) -> List[int]:
    # 目标只允许从 index >= 1 里选（排除第0个）
    return [i for i, t in enumerate(element_dtype_array) if i >= 1 and t == target_modality]

def sample_task(task_type, task_type_rate):
    """
    从 task_type 中按 task_type_rate 采样 1 项。
    若权重全为 0 或全为负数，则改为等概率采样。
    """
    if len(task_type) != len(task_type_rate):
        raise ValueError("task_type 与 task_type_rate 长度不一致")
    # 负权重按 0 处理
    weights = [max(0.0, float(w)) for w in task_type_rate]
    if sum(weights) == 0.0:
        weights = [1.0] * len(task_type)  # 退化为均匀
    return random.choices(task_type, weights=weights, k=1)[0]

def data_invert_text_image_pair(interleave_array, element_dtype_array, target_modality):  # 处理单一图像-文本对, 可以不区分位置，简单置换
    if len(element_dtype_array) == 2:
        if element_dtype_array[-1] != target_modality:
            interleave_array = interleave_array[::-1]
            element_dtype_array = element_dtype_array[::-1]
    return interleave_array, element_dtype_array

class BaseMMParquetDataset(DistributedIterableDataset, ABC):
    def __init__(
        self,
        dataset_name: str,
        tokenizer: Any,
        data_dir_list: List[str],
        local_rank: int = 0,
        world_size: int = 1,
        num_workers: int = 8,
        data_status: Optional[Any] = None,
        **kwargs: Any,
    ):
        """
        data_dir_list: list of data directories contains parquet files
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)

        # 只保存配置，延后真正的初始化
        self.tokenizer = tokenizer
        self.data_dir_list = data_dir_list
        self.data_status = data_status
        self.seed = kwargs.get('seed', 42)

        self.caption_key = kwargs.get(
            'caption_key', 'v3_0_long_internlm_caption_en_text'
        )
        self.transform: VideoTransform = kwargs.get('transform')
        self.frame_sampler = kwargs.get("video_frame_sampler") # 视频采样
        self.vae_downsample = kwargs.get(
            'vae_downsample',
            (
                getattr(self.transform, 'stride_temporal', 4),
                getattr(self.transform, 'stride_spatial', 16),
                getattr(self.transform, 'stride_spatial', 16),
            )
        )
        self.max_bytes = kwargs.get('max_bytes', -1)
        self.logger = get_logger()

        # 标记：还没初始化过
        self.data_paths = kwargs.get('all_data_paths')
        self.cpu_count = os.cpu_count() or 1

        self.apply_chat_template = kwargs.get('apply_chat_template', False)
        if self.apply_chat_template:
            self.chat_template = TemplateArguments().chat_template_T2I

        self.vision_stream = kwargs.get("vision_stream", "vae_video")  # 'vae_video' | 'vit_video'
        self.vit_downsample = kwargs.get("vit_downsample", (2, 28, 28))

        if kwargs.get('vit_transform') is not None:
            self.vit_transform: VideoTransform = kwargs.get('vit_transform')
        else:
            self.vit_transform: VideoTransform = kwargs.get('transform')

        self.text_cleaner = TextCleaner()
        self.dataset_type = kwargs.get("dataset_type", "interleave")
        self.force_last_as_gt_prob = kwargs.get("force_last_as_gt_prob", 0.0)
        self.N_target = kwargs.get("N_target", 1)
        self.N_target_random_prob = kwargs.get("N_target_random_prob", 0.0)
        self.max_num_split_vit, self.max_num_split_vae, self.max_num_split_text = kwargs.get("max_num_split", [1000, 1000, 1000])
        self.is_image = kwargs.get("is_image", True)

        self.res_dump = kwargs.get("res_dump", "12fps_192p")
        self.data_mode = kwargs.get("data_mode", "online")
        self.text_template = kwargs.get("text_template", False)

        self.vision_cond_type = kwargs.get("vision_cond_type", ["vit"])

        self.fbyf_group_interval = kwargs.get("fbyf_group_interval", -1)
        self.fbyf_type = kwargs.get("fbyf_type", "group")

        self.sample_task = kwargs.get("sample_task", "t2v") # 作为任务标识，支持联合多任务训练

        if "ocr" in self.dataset_type:
            self.data_filter = kwargs.get("data_filter", {})

        self.save_video_image = kwargs.get("save_video_image", False)

        self.data_config = kwargs

    # ==== 子类需/可覆盖的钩子 ====
    def select_columns(self) -> Optional[List[str]]:
        """可选：返回需要从 parquet 读的列名，减少 IO。None 表示读全部。"""
        if "interleave" in self.dataset_type:
            return None # ["element_dtype_array", "interleave_array"]
        elif "ffhq" in self.dataset_type or "imagenet" in self.dataset_type:
            return ["tos_url", self.caption_key]  # 选取子列
        elif "hav" in self.dataset_type:
            return ["media_url", "properties"]
        elif "vertical" in self.dataset_type:
            return ["meta_url"]
        elif "audio_human" in self.dataset_type:
            return ["video_meta_url"]

        return None

    def lazy_init_clients(self):
        """Compatibility hook. Local parquet training does not initialize remote clients."""
        return

    @staticmethod
    def _read_decord(video: VideoReader, frame_idx: List[int]) -> List[Image.Image]:
        # 使用 get_batch() 替换循环单帧读取，可以大幅提升性能
        frames_np = video.get_batch(frame_idx).asnumpy()
        return [Image.fromarray(frame) for frame in frames_np]

    def vision_token_count(self, video_tensor: torch.Tensor) -> int:
        _, T, H, W = video_tensor.shape
        if self.vision_stream == "vit_video":
            _T, _H, _W = self.vit_downsample
            return (T // _T) * (H // _H) * (W // _W)
        elif self.vision_stream == "vae_video":
            _T, _H, _W = self.vae_downsample
            return ((T // _T) + 1) * (H // _H) * (W // _W)
        else:
            raise ValueError(f"Unknown vision_stream: {self.vision_stream}")

    def get_thwc_url_new(self, media_url, worker_id):
        raise NotImplementedError("Remote media URLs are not supported. Use local parquet rows with embedded bytes.")

        video_reader = VideoReader(video_stream, ctx=decord.cpu(worker_id % self.cpu_count))
        total_frames = len(video_reader)

        sampler_name = self.frame_sampler.__class__.__name__
        if sampler_name == "MultiClipsFrameSampler":
            fps =24
            try:
                fps = int(round(float(video_reader.get_avg_fps())))
            except Exception:
                pass
            frames_info = {
                    "clip_indices": [(0, total_frames)],  # 左闭右开 默认为单个clip
                    "fps": fps,  # 默认为24
                }
        elif sampler_name == "FixedFrameSampler":
            frames_info = {
                    "start_frame": 0,
                    "end_frame": total_frames,
                    "total_frames": total_frames,
                }
        else:
            raise ValueError(f"Not verified frame sampler type: {sampler_name}")

        frames_sampler_output: FrameSamplerOutput = self.frame_sampler(frames_info)
        video_frames = self._read_decord(video_reader, frames_sampler_output.indices)

        # 默认dit
        video_tensor = self.vit_transform(video_frames)  # fix: use List input
        if self.is_image:
            video_tensor = video_tensor.repeat(1, 2, 1, 1)  # NOTE 对于单张图像，需要复制一份，因为encoder的temporal patch size = 2
        # NOTE: 视频长度必须是偶数
        if video_tensor.shape[1] % 2 == 1:
            last_frame = video_tensor[:, -1:, :, :]
            video_tensor = torch.cat([video_tensor, last_frame], dim=1)

        _, T, H, W = video_tensor.shape

        return (T, H, W)

    def get_video_tensor_online(self, media_url, vision_stream, worker_id=0, element_dtype="image", raw_bytes_input=False) -> torch.Tensor:
        self.vision_stream = vision_stream
        if raw_bytes_input:
            # raise NotImplementedError(f"raw_bytes_input must be True for {vision_stream}")
            video_stream = BytesIO(media_url)
            # # 方法A：直接写入文件（最简单）debug code
            # from datetime import datetime  # 导入时间模块
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            # with open(f"saved_image_{timestamp}.png", "wb") as f:
            #     f.write(video_stream.getvalue())  # getvalue() 获取BytesIO中的所有字节

            # # 方法A：直接写入文件（最简单）debug code
            # from datetime import datetime  # 导入时间模块
            # # 保留最高精度时间戳（年-月-日_时-分-秒-微秒）
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            # # 文件名改为 video，后缀改为 .mp4，写入视频字节数据
            # with open(f"saved_video_{timestamp}.mp4", "wb") as f:
            #     f.write(video_stream.getvalue())  # video_stream 是包含视频bytes的BytesIO对象
        else:
            raise NotImplementedError("Remote media URLs are not supported. Use raw_bytes_input=True with local parquet bytes.")

        if self.is_image and element_dtype == "image":
            image = Image.open(video_stream)
            if image.mode == "P":
                image = image.convert("RGBA")
            if image.mode == "RGBA":
                # 在白底上合成，去掉透明
                bg = Image.new("RGB", image.size, (255, 255, 255))
                bg.paste(image, mask=image.split()[3])  # 用 alpha 通道做掩码
                image = bg
            else:
                image = image.convert("RGB")
            video_frames = [image]

            # 保存图像
            if self.save_video_image:
                self.path = f"{self.path}.jpg"
                image.save(self.path, quality=95)
                print(f"Saved image to {self.path}")
        else:  # for video
            video_reader = VideoReader(video_stream, ctx=decord.cpu(worker_id % self.cpu_count))
            total_frames = len(video_reader)

            # 保存视频
            if self.save_video_image:
                fps = video_reader.get_avg_fps()
                width, height = video_reader[0].shape[1], video_reader[0].shape[0]
                self.path =f"{self.path}.mp4" # 保存视频路径
                # 使用OpenCV保存视频
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(self.path, fourcc, fps, (width, height))

                for frame in video_reader:
                    frame_rgb = cv2.cvtColor(frame.asnumpy(), cv2.COLOR_BGR2RGB)  # 将BGR转换为RGB
                    out.write(frame_rgb)
                out.release()
                print(f"Saved image to {self.path} with fps {fps}")

            sampler_name = self.frame_sampler.__class__.__name__
            if sampler_name == "MultiClipsFrameSampler":
                fps =24
                try:
                    fps = int(round(float(video_reader.get_avg_fps())))
                except Exception:
                    pass
                frames_info = {
                        "clip_indices": [(0, total_frames)],  # 左闭右开 默认为单个clip
                        "fps": fps,  # 默认为24
                    }
            elif sampler_name == "FixedFrameSampler":
                frames_info = {
                    "start_frame": 0,
                    "end_frame": total_frames,
                    "total_frames": total_frames,
                }
            else:
                raise ValueError(f"Not verified frame sampler type: {sampler_name}")

            frames_sampler_output: FrameSamplerOutput = self.frame_sampler(frames_info)
            video_frames = self._read_decord(video_reader, frames_sampler_output.indices)

        if vision_stream == "vae_video":
            video_tensor = self.transform(video_frames)  # fix: use List input
        elif vision_stream == "vit_video":
            video_tensor = self.vit_transform(video_frames)  # fix: use List input
            if self.is_image:
                video_tensor = video_tensor.repeat(1, 2, 1, 1)  # NOTE 对于单张图像，需要复制一份，因为encoder的temporal patch size = 2
            # NOTE: 视频长度必须是偶数
            if video_tensor.shape[1] % 2 == 1:
                last_frame = video_tensor[:, -1:, :, :]
                video_tensor = torch.cat([video_tensor, last_frame], dim=1)

        else:
            raise ValueError(f"Unknown vision_stream: {vision_stream}")

        if not (self.is_image and element_dtype == "image"):
            del video_frames, video_reader, video_stream
        return video_tensor, self.vision_token_count(video_tensor)

    def get_video_tensor_offline(self, media_url, vision_stream, worker_id=0) -> torch.Tensor:
        self.vision_stream = vision_stream

        if vision_stream == "vae_video":
            video_tensor = media_url[0]  # [t, h, w, c]
            num_token = video_tensor.shape[0] * video_tensor.shape[1] * video_tensor.shape[2]

        elif vision_stream == "vit_video":
            video_tensor = media_url[1]  # [L, D]
            num_token = video_tensor.shape[0]

        if len(media_url) == 3 and vision_stream == "vit_video":
            if isinstance( media_url[2], str): # 获取thw 信息
                thw = self.get_thwc_url_new(media_url[2], worker_id = worker_id)
            else:
                thw = media_url[2][1:]
            num_token_ = thw[0] * thw[1] * thw[2] // self.vit_downsample[0] // self.vit_downsample[1] // self.vit_downsample[2]
            if num_token_ != num_token:
                raise ValueError(f"Video tensor shape {video_tensor.shape} not match thw {thw}: {num_token_} != {num_token}")
        else:
            thw = None

        return video_tensor, num_token, thw

    def get_video_tensor(self, media_url, vision_stream, worker_id=0, element_dtype="image", raw_bytes_input=False) -> torch.Tensor:

        if isinstance(media_url, tuple):  # offline
            video_tensor, num_tokens_, thw = self.get_video_tensor_offline(media_url, vision_stream=vision_stream)
            self.data_mode = "offline"
            video_tensor = [video_tensor]  # 以list 形式返回，实现和online 的格式区分
        else:  # online
            video_tensor, num_tokens_ = self.get_video_tensor_online(media_url, vision_stream=vision_stream, worker_id=worker_id, element_dtype=element_dtype, raw_bytes_input=raw_bytes_input)
            self.data_mode = "online"
            thw = None
        return video_tensor, num_tokens_, thw

    # 在初始化时获取每个文件的样本数量
    def get_file_sample_counts(self, data_paths):
        sample_counts = []
        for path in data_paths:
            fs = init_arrow_pf_fs(path)
            with fs.open_input_file(path) as f:
                fr = pq.ParquetFile(f)
                # 估算或精确计算文件中的样本数量
                count = sum(fr.metadata.row_group(i).num_rows for i in range(fr.num_row_groups))
                sample_counts.append(count)
        return sample_counts

    def get_condition_target_idx(
        self,
        element_dtype_array,
    ):
        if len(element_dtype_array) == 1 and self.target_modality in element_dtype_array: # 只有一个元素即无condition
            return [], [0], 1

        target_pos_all = _collect_target_positions(element_dtype_array, self.target_modality)  # 获取除 0 位置 外的目标element 位置
        pos_all = list(range(len(element_dtype_array)))
        N_all = len(target_pos_all)
        if N_all == 0:
            # 没有目标类型可选，退化为全量 condition、空 target（上游应丢弃这种样本）
            return None

        if random.random() < self.N_target_random_prob:  # 按概率随机选择目标数量，一般 N_target_random_prob 为 0 则目标数量默认为 self.N_target
            N_target = random.randint(1, N_all)
        else:
            N_target = self.N_target

        if self.target_modality in ["image", "video"] :
            N_target = min(N_target, N_all, self.max_num_split_vae)  # 确保目标数量不大于总数量
        elif self.target_modality == "text":
            N_target = min(N_target, N_all, self.max_num_split_text)

        # --- 选择 target 集合 ---
        choose_last = random.random() < self.force_last_as_gt_prob  # 按概率选择是否强制最后一个为目标
        if choose_last:
            target_idx = target_pos_all[-N_target:]  # 对应目标 element 的索引
        else:
            target_last = random.randint(N_target - 1, N_all - 1)  # 即确保取的最后一个目标 element 索引 能满足 N_target 个目标
            target_idx = target_pos_all[target_last - N_target + 1 : target_last + 1]  # target_last为对应目标 element 的索引，所以+1
        condition_idx = pos_all[: target_idx[0]]  # 对应条件 element 的索引
        return condition_idx, target_idx, N_target

    @abstractmethod
    def _process_row(self, row, parquet_idx, row_group_id, row_idx, worker_id, parquet_file_path):
        pass

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        self.lazy_init_clients()

        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            parquet_start_id = self.data_status[worker_id][0]
            row_group_start_id = self.data_status[worker_id][1]
            row_start_id = self.data_status[worker_id][2] + 1
        else:
            parquet_start_id = 0
            row_group_start_id = 0
            row_start_id = 0

        # log
        if data_paths_per_worker:
            self.logger.info(
                f"Rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
                f"{len(data_paths_per_worker)} parquet files (first: {data_paths_per_worker[0]}, "
                f"last: {data_paths_per_worker[-1]}), "
                f"resuming at parquet#{parquet_start_id}, rg#{row_group_start_id}, row#{row_start_id}"
            )
        else:
            self.logger.warning(f"Rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: " "has 0 parquet files!")

        while True:
            data_paths_per_worker_ = data_paths_per_worker[parquet_start_id:]
            for parquet_idx, parquet_file_path in enumerate(data_paths_per_worker_, start=parquet_start_id):
                fs = init_arrow_pf_fs(parquet_file_path)
                with fs.open_input_file(parquet_file_path) as f:
                    fr = pq.ParquetFile(f)
                    row_group_ids = list(range(fr.num_row_groups))
                    row_group_ids_ = row_group_ids[row_group_start_id:]

                    # 列裁剪：子类可告知 select_columns()
                    cols = self.select_columns()  # 默认为None, 读取全部列

                    for row_group_id in row_group_ids_:
                        rows = read_parquet_rows(fr, row_group_id, columns=cols)
                        rows = rows[row_start_id:]

                        for row_idx, row in enumerate(rows, start=row_start_id):
                            sample = self._process_row(row, parquet_idx, row_group_id, row_idx, worker_id, parquet_file_path)
                            if sample:
                                yield sample
                            # self.logger.info(f"parquet_file_path: {parquet_file_path}, row_idx: {row_idx}, row_group_id: {row_group_id}, worker_id:{worker_id}, self.local_rank:{self.local_rank}") # 方便定位异常数据
                        row_start_id = 0
                    row_group_start_id = 0
            parquet_start_id = 0

            if self.local_rank == 0:
                self.logger.info(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")
                pass

    def transform_row(self, row):
        if self.dataset_type == "text2video_general":
            video_bytes = row["video_bytes"]
            caption = row["caption"]
            interleave_array = [caption, video_bytes] if self.text_first else [video_bytes, caption]
            element_dtype_array = ["text", "video"] if self.text_first else ["video", "text"]
        elif self.dataset_type == "text2image_general":
            image_bytes = row["image_bytes"]
            caption = row["caption"]
            interleave_array = [caption, image_bytes] if self.text_first else [image_bytes, caption]
            element_dtype_array = ["text", "image"] if self.text_first else ["image", "text"]
        elif self.dataset_type == "x2t_general":
            if all(key in row for key in ["caption_i", "caption_q", "caption_a"]):
                caption = [row["caption_i"], row["caption_q"], row["caption_a"]]
            else:
                caption = parse_videochat2it_doubao_caption(row)

            if "image_bytes" in row:
                interleave_array = [row["image_bytes"], caption] if not self.text_first else [caption, row["image_bytes"]]
                element_dtype_array = ["image", "text"] if not self.text_first else ["text", "image"]
            elif "video_bytes" in row:
                interleave_array = [row["video_bytes"], caption] if not self.text_first else [caption, row["video_bytes"]]
                element_dtype_array = ["video", "text"] if not self.text_first else ["text", "video"]
            else:
                interleave_array = [caption]
                element_dtype_array = ["text"]
        elif self.dataset_type == "image2image":
            interleave_array = [row["caption"], row["input_image_bytes"], row["output_image_bytes"]]
            element_dtype_array = ["text", "image", "image"]
            self.force_last_as_gt_prob = 1
            self.N_target = 1
            self.N_target_random_prob = 0
            self.sample_task = "edit"
        elif self.dataset_type == "image2image_online":
            interleave_array = [row["instruction"], row["input_image_url"], row["output_image_url"]]
            element_dtype_array = ["text", "image", "image"]
            self.force_last_as_gt_prob = 1
            self.N_target = 1
            self.N_target_random_prob = 0
            self.sample_task = "edit"
        elif self.dataset_type == "video2video":
            interleave_array = [row["caption"], row["input_video_bytes"], row["output_video_bytes"]]
            element_dtype_array = ["text", "video", "video"]
            self.force_last_as_gt_prob = 1
            self.N_target = 1
            self.N_target_random_prob = 0
            self.sample_task = "edit"
        else:
            raise ValueError(f"dataset_type {self.dataset_type} not supported")

        return interleave_array, np.array(element_dtype_array).astype(dtype=object)
