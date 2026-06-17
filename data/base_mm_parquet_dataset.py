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
    # Targets can only be selected from index >= 1, excluding index 0
    return [i for i, t in enumerate(element_dtype_array) if i >= 1 and t == target_modality]

def sample_task(task_type, task_type_rate):
    """
    Sample one item from task_type according to task_type_rate.
    If all weights are zero or negative, fall back to uniform sampling.
    """
    if len(task_type) != len(task_type_rate):
        raise ValueError("task_type and task_type_rate must have the same length")
    # Treat negative weights as 0
    weights = [max(0.0, float(w)) for w in task_type_rate]
    if sum(weights) == 0.0:
        weights = [1.0] * len(task_type)  # Fall back to uniform weights
    return random.choices(task_type, weights=weights, k=1)[0]

def data_invert_text_image_pair(interleave_array, element_dtype_array, target_modality):  # Handle a single image-text pair by swapping without considering position
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

        # Store config only and delay real initialization
        self.tokenizer = tokenizer
        self.data_dir_list = data_dir_list
        self.data_status = data_status
        self.seed = kwargs.get('seed', 42)

        self.caption_key = kwargs.get(
            'caption_key', 'v3_0_long_internlm_caption_en_text'
        )
        self.transform: VideoTransform = kwargs.get('transform')
        self.frame_sampler = kwargs.get("video_frame_sampler") # Video sampling
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

        # Mark as not initialized yet
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

        self.sample_task = kwargs.get("sample_task", "t2v") # Task identifier for joint multi-task training

        if "ocr" in self.dataset_type:
            self.data_filter = kwargs.get("data_filter", {})

        self.save_video_image = kwargs.get("save_video_image", False)

        self.data_config = kwargs

    # ==== Hooks required or optionally overridden by subclasses ====
    def select_columns(self) -> Optional[List[str]]:
        """Optional: return column names to read from parquet to reduce IO. None reads all columns."""
        if "interleave" in self.dataset_type:
            return None # ["element_dtype_array", "interleave_array"]
        elif "ffhq" in self.dataset_type or "imagenet" in self.dataset_type:
            return ["tos_url", self.caption_key]  # Select a subset of columns
        elif "hav" in self.dataset_type:
            return ["media_url", "properties"]
        elif "vertical" in self.dataset_type:
            return ["meta_url"]
        elif "audio_human" in self.dataset_type:
            return ["video_meta_url"]

        return None

    @staticmethod
    def _read_decord(video: VideoReader, frame_idx: List[int]) -> List[Image.Image]:
        # Use get_batch() instead of reading frames one by one for better performance
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
                    "clip_indices": [(0, total_frames)],  # Left-closed, right-open interval; default is a single clip
                    "fps": fps,  # Default is 24
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

        # Default DIT path
        video_tensor = self.vit_transform(video_frames)  # fix: use List input
        if self.is_image:
            video_tensor = video_tensor.repeat(1, 2, 1, 1)  # NOTE: Duplicate single images because the encoder temporal patch size is 2
        # NOTE: Video length must be even
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
            # # Method A: write directly to file; simplest debug code
            # from datetime import datetime  # Import datetime module
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            # with open(f"saved_image_{timestamp}.png", "wb") as f:
            #     f.write(video_stream.getvalue())  # getvalue() returns all bytes in BytesIO

            # # Method A: write directly to file; simplest debug code
            # from datetime import datetime  # Import datetime module
            # # Keep a high-precision timestamp: year-month-day_hour-minute-second-microsecond
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            # # Use a video filename with .mp4 suffix and write video bytes
            # with open(f"saved_video_{timestamp}.mp4", "wb") as f:
            #     f.write(video_stream.getvalue())  # video_stream is a BytesIO object containing video bytes
        else:
            raise NotImplementedError("Remote media URLs are not supported. Use raw_bytes_input=True with local parquet bytes.")

        if self.is_image and element_dtype == "image":
            image = Image.open(video_stream)
            if image.mode == "P":
                image = image.convert("RGBA")
            if image.mode == "RGBA":
                # Composite on a white background to remove transparency
                bg = Image.new("RGB", image.size, (255, 255, 255))
                bg.paste(image, mask=image.split()[3])  # Use the alpha channel as the mask
                image = bg
            else:
                image = image.convert("RGB")
            video_frames = [image]

            # Save image
            if self.save_video_image:
                self.path = f"{self.path}.jpg"
                image.save(self.path, quality=95)
                print(f"Saved image to {self.path}")
        else:  # for video
            video_reader = VideoReader(video_stream, ctx=decord.cpu(worker_id % self.cpu_count))
            total_frames = len(video_reader)

            # Save video
            if self.save_video_image:
                fps = video_reader.get_avg_fps()
                width, height = video_reader[0].shape[1], video_reader[0].shape[0]
                self.path =f"{self.path}.mp4" # Saved video path
                # Save video with OpenCV
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(self.path, fourcc, fps, (width, height))

                for frame in video_reader:
                    frame_rgb = cv2.cvtColor(frame.asnumpy(), cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
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
                        "clip_indices": [(0, total_frames)],  # Left-closed, right-open interval; default is a single clip
                        "fps": fps,  # Default is 24
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
                video_tensor = video_tensor.repeat(1, 2, 1, 1)  # NOTE: Duplicate single images because the encoder temporal patch size is 2
            # NOTE: Video length must be even
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
            if isinstance( media_url[2], str): # Get THW information
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
            video_tensor = [video_tensor]  # Return as a list to distinguish this from online format
        else:  # online
            video_tensor, num_tokens_ = self.get_video_tensor_online(media_url, vision_stream=vision_stream, worker_id=worker_id, element_dtype=element_dtype, raw_bytes_input=raw_bytes_input)
            self.data_mode = "online"
            thw = None
        return video_tensor, num_tokens_, thw

    # Get the sample count for each file during initialization
    def get_file_sample_counts(self, data_paths):
        sample_counts = []
        for path in data_paths:
            fs = init_arrow_pf_fs(path)
            with fs.open_input_file(path) as f:
                fr = pq.ParquetFile(f)
                # Estimate or exactly compute the sample count in each file
                count = sum(fr.metadata.row_group(i).num_rows for i in range(fr.num_row_groups))
                sample_counts.append(count)
        return sample_counts

    def get_condition_target_idx(
        self,
        element_dtype_array,
    ):
        if len(element_dtype_array) == 1 and self.target_modality in element_dtype_array: # A single element means there is no condition
            return [], [0], 1

        target_pos_all = _collect_target_positions(element_dtype_array, self.target_modality)  # Get target element positions excluding position 0
        pos_all = list(range(len(element_dtype_array)))
        N_all = len(target_pos_all)
        if N_all == 0:
            # If no target type is available, fall back to all condition and empty target; upstream should drop this sample
            return None

        if random.random() < self.N_target_random_prob:  # Randomly choose target count by probability; if N_target_random_prob is 0, default to self.N_target
            N_target = random.randint(1, N_all)
        else:
            N_target = self.N_target

        if self.target_modality in ["image", "video"] :
            N_target = min(N_target, N_all, self.max_num_split_vae)  # Ensure target count does not exceed total count
        elif self.target_modality == "text":
            N_target = min(N_target, N_all, self.max_num_split_text)

        # --- Select target set ---
        choose_last = random.random() < self.force_last_as_gt_prob  # Randomly decide whether to force the last item as target
        if choose_last:
            target_idx = target_pos_all[-N_target:]  # Indexes of target elements
        else:
            target_last = random.randint(N_target - 1, N_all - 1)  # Ensure the selected last target element index can cover N_target targets
            target_idx = target_pos_all[target_last - N_target + 1 : target_last + 1]  # Add 1 because target_last is the target element index
        condition_idx = pos_all[: target_idx[0]]  # Indexes of condition elements
        return condition_idx, target_idx, N_target

    @abstractmethod
    def _process_row(self, row, parquet_idx, row_group_id, row_idx, worker_id, parquet_file_path):
        pass

    def __iter__(self) -> Iterator[Dict[str, Any]]:
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

                    # Column pruning: subclasses can provide select_columns()
                    cols = self.select_columns()  # Default is None, which reads all columns

                    for row_group_id in row_group_ids_:
                        rows = read_parquet_rows(fr, row_group_id, columns=cols)
                        rows = rows[row_start_id:]

                        for row_idx, row in enumerate(rows, start=row_start_id):
                            sample = self._process_row(row, parquet_idx, row_group_id, row_idx, worker_id, parquet_file_path)
                            if sample:
                                yield sample
                            # self.logger.info(f"parquet_file_path: {parquet_file_path}, row_idx: {row_idx}, row_group_id: {row_group_id}, worker_id:{worker_id}, self.local_rank:{self.local_rank}") # Useful for locating bad data
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
