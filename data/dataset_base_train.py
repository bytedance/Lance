# coding: utf-8


import os
EXP_HW_20250819 = os.environ.get("EXP_HW_20250819", "False").lower() == "true"

import random
from dataclasses import dataclass
from typing import Any, Dict, Tuple
import numpy as np
import torch
import yaml
from itertools import chain
import sys

from .data_utils import (
    get_flattened_position_ids_interpolate_video,
    get_flattened_position_ids_extrapolate_video,
    len2weight,
    prepare_attention_mask_per_sample,
    patchify_video_with_merge,
)
from .dataset_info import DATASET_REGISTRY
from data.video.sampler.utils import FRAME_SAMPLER_TYPES
from data.transforms import ImageTransform, VideoTransform
from data.video.video_utils import FrameSampler
from data.parquet_utils import get_parquet_data_paths_balanced
from common.utils.logging import get_logger
from common.utils.basic import get_global_rank
import bisect

sample_task_map = {
    't2v': 0,
    'idip': 1,
    'edit': 2,
    'refedit': 3,
    'maze':3,
}
modality_map = {
    'system_prompt': -1,
    'text': 0,
    'noise': 1,
    'ref_source': 2, # for vae
    'ref_image': 3, # for vae
    'ref_vit': 4 # for ref vit
}

@dataclass
class DataConfig:
    """
    DataConfig variant where vae_downsample is a 3-tuple.
    """
    grouped_datasets: Dict[str, Any]
    text_cond_dropout_prob: float = 0.1
    vit_cond_dropout_prob: float = 0.4
    vae_cond_dropout_prob: float = 0.1

    # Use a 3-tuple for vae_downsample: temporal, height, and width downsampling rates
    vae_downsample: Tuple[int, int, int] = (4, 16, 16)

    max_latent_size: int = 64 # by ModelArguments
    vit_patch_size: int = 14 # by ModelArguments
    vit_patch_size_temporal: int = 2 # by ModelArguments
    vit_max_num_patch_per_side: int = 70 # by ModelArguments
    max_num_frames: int = 25 # by ModelArguments

    latent_patch_size: int = None # by ModelArguments

    @classmethod
    def from_yaml(cls, file_path: str) -> 'DataConfig':
        """
        Create a DataConfig instance from a YAML file.
        """
        with open(file_path, "r") as stream:
            data = yaml.safe_load(stream)
        return cls(grouped_datasets=data)

class PackedDataset(torch.utils.data.IterableDataset):

    def __init__(
        self,
        data_config: DataConfig,
        tokenizer,
        special_tokens,
        local_rank,
        world_size,
        num_workers,
        expected_num_tokens=32768,  # Expected packed sequence length
        max_num_tokens_per_sample=16384,  # Maximum length of a single sample
        max_num_tokens=36864,  # Hard limit for packed sequence length
        prefer_buffer_before=16384,  # Sample-length threshold for preferring the buffer
        max_buffer_size=50,  # Maximum buffer capacity
        interpolate_pos=False,
        use_flex=False,
        data_status=None,
        apply_chat_template=False,
        image_token_id=151655,
        **kwargs,
    ):
        super().__init__()
        self.expected_num_tokens = expected_num_tokens
        self.max_num_tokens_per_sample = max_num_tokens_per_sample
        self.prefer_buffer_before = prefer_buffer_before
        self.max_num_tokens = max_num_tokens
        self.max_buffer_size = max_buffer_size
        self.tokenizer = tokenizer
        self.local_rank = local_rank
        self.world_size = world_size
        self.num_workers = num_workers
        self.use_flex = use_flex
        self.data_config: DataConfig = data_config
        self.max_num_latent_frames = self.data_config.max_num_frames // self.data_config.vae_downsample[0] + 1
        self.apply_chat_template = apply_chat_template
        self.cfg_type = kwargs.get("cfg_type", 0)
        self.cfg_uncond_token_id = kwargs.get("cfg_uncond_token_id", 151643)
        self.image_token_id = image_token_id
        self.data_args = kwargs
        self.expected_num_ce_loss_tokens = kwargs.get("expected_num_ce_loss_tokens", 1000000) # Default 1000000 effectively means unlimited
        self.N_key_frame = kwargs.get("N_key_frame", -1)
        self.fbyf_type = kwargs.get("fbyf_type", "group")
        self.fbyf_group_interval = kwargs.get("fbyf_group_interval", -1)
        self.incre_time_pro = kwargs.get("incre_time_pro", 0)
        self.require_und_gen = kwargs.get("require_und_gen", False)

        # NOTE: Add special tokens such as <|im_start|> and <|im_end|> here
        for k, v in special_tokens.items():
            setattr(self, k, v)

        # add a logger
        self.logger = get_logger()
        # self.log_rank0 = lambda msg: self.logger.info(msg) if get_global_rank() == 0 else None # Log only on rank 0

        grouped_datasets, is_mandatory, grouped_weights, data_type = self.build_datasets(data_config.grouped_datasets, data_status)
        self.grouped_datasets = grouped_datasets
        # self.dataset_iters = [iter(dataset) for dataset in grouped_datasets] # this is dataset iter
        self.dataset_iters = None # Delay creation until __iter__
        self.is_mandatory = is_mandatory
        self.grouped_weights = grouped_weights
        self.interpolate_pos = interpolate_pos
        self.data_type = data_type

    # Log only on rank 0; instance methods can be pickled
    def log_rank0(self, msg: str):
        if get_global_rank() == 0:
            self.logger.info(msg)

    # Avoid sending non-serializable objects such as loggers or iterators to child processes
    def __getstate__(self):
        state = self.__dict__.copy()
        state["logger"] = None
        state["dataset_iters"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.logger = get_logger()
        self.dataset_iters = None

    def set_epoch(self, seed):
        for dataset in self.grouped_datasets:
            dataset.set_epoch(seed)

        # for dataset in self.grouped_datasets:
        #     if hasattr(dataset, 'seed'):
        #         dataset.set_epoch(dataset.seed)
        #     else:
        #         dataset.set_epoch(seed)

    def set_sequence_status(self):
        sequence_status = dict(
            curr                        = 0, # Pointer
            sample_lens                 = [],
            sample_type                 = [],
            sample_N_target             = [],
            packed_position_ids         = [],
            nested_attention_masks      = [],
            split_lens                  = [],
            attn_modes                  = [],
            packed_text_ids             = [],
            packed_text_indexes         = [],
            packed_label_ids            = [],
            ce_loss_indexes             = [],
            ce_loss_weights             = [],
            vae_image_tensors           = [], # image
            vae_video_tensors           = [], # video
            packed_latent_position_ids  = [],
            vae_latent_shapes           = [],
            packed_vae_token_indexes    = [],
            packed_timesteps            = [],
            mse_loss_indexes            = [],
            packed_vit_tokens           = [],
            vit_token_seqlens           = [],
            packed_vit_position_ids     = [],
            packed_vit_token_indexes    = [],
            vit_video_grid_thw          = [], # for vit video
            vae_video_grid_thw          = [], # for vae video
            video_grid_thw              = [], # for all video tensor
            vit_video_tensors           = [], # for vit original video tensor
            # Offline arguments
            vae_video_latent            = [], # for vae video latent offline
            vae_data_mode               = [], # offline or online
            vit_data_mode               = [], # offline or online
            # sample_task for joint training
            sample_task                 = [],
            sample_modality             = [],
        )
        return sequence_status

    def to_tensor(self, sequence_status: Dict[str, Any]):
        data = dict(
            sequence_length=sum(sequence_status['sample_lens']),
            sample_lens=sequence_status['sample_lens'],
            sample_type=sequence_status['sample_type'],
            sample_N_target=sequence_status['sample_N_target'],
            packed_text_ids=torch.tensor(sequence_status['packed_text_ids']),
            packed_text_indexes=torch.tensor(sequence_status['packed_text_indexes']),
            packed_position_ids=torch.tensor(sequence_status['packed_position_ids']),
            vit_data_mode=sequence_status['vit_data_mode'],
            video_grid_thw=sequence_status['video_grid_thw'],
            sample_task=torch.tensor(sequence_status['sample_task']),
            sample_modality=torch.tensor(sequence_status['sample_modality']),
        )

        data['vae_data_mode'] = sequence_status['vae_data_mode']
        if not self.use_flex:
            data['nested_attention_masks'] = sequence_status['nested_attention_masks']
        else:
            sequence_len = data['sequence_length']
            pad_len = self.max_num_tokens - sequence_len
            assert pad_len >= 0, f"pad_len must be greater than 0, but got {pad_len}" # !!!
            data['split_lens'] = sequence_status['split_lens'] + [pad_len]
            data['attn_modes'] = sequence_status['attn_modes'] + ['causal']
            data['sample_lens'] += [pad_len]
            data['sample_type'] += ['pad']
            data['sample_N_target'] += [0]

        # if the model has a convnet vae (e.g., as visual tokenizer)

        data['padded_videos'] = sequence_status.pop('vae_video_tensors')
        if len(data['padded_videos']) > 0:
            # Pack as dynamic resolution
            # NOTE: The following keys are shared between image and video for now
            if 'patchified_vae_latent_shapes' not in data:
                data['patchified_vae_latent_shapes'] = sequence_status['vae_latent_shapes']
                data['packed_latent_position_ids'] = torch.cat(sequence_status['packed_latent_position_ids'], dim=0)
                data['packed_vae_token_indexes'] = torch.tensor(sequence_status['packed_vae_token_indexes'])

        # process for offline data: padding
        if len(sequence_status["vae_video_latent"]) > 0:
            video_latents = sequence_status.pop("vae_video_latent")
            video_sizes = [item.shape for item in video_latents]
            max_video_size = [max(item) for item in list(zip(*video_sizes))]
            padded_videos_latent = torch.zeros(size=(len(video_latents), *max_video_size))
            for i, video_latent in enumerate(video_latents):  # [T, H, W, C]
                t, h, w, c = video_latent.shape
                padded_videos_latent[i, :t, :h, :w, :c] = video_latent
            data["padded_latent"] = padded_videos_latent
            # NOTE: The following keys are shared between image and video for now
            if "patchified_vae_latent_shapes" not in data:
                data["patchified_vae_latent_shapes"] = sequence_status["vae_latent_shapes"]
                data["packed_latent_position_ids"] = torch.cat(sequence_status["packed_latent_position_ids"], dim=0)
                data["packed_vae_token_indexes"] = torch.tensor(sequence_status["packed_vae_token_indexes"])

        # if the model has a vit (e.g., as visual tokenizer)
        if len(sequence_status['packed_vit_tokens']) > 0:
            data['packed_vit_tokens'] = sequence_status.pop('packed_vit_tokens')
            # data['packed_vit_tokens'] = torch.cat(sequence_status['packed_vit_tokens'], dim=0)
            data['packed_vit_position_ids'] = torch.cat(sequence_status['packed_vit_position_ids'], dim=0)
            data['packed_vit_token_indexes'] = torch.tensor(sequence_status['packed_vit_token_indexes'])
            data['vit_token_seqlens'] = torch.tensor(sequence_status['vit_token_seqlens'])

            # Pack as dynamic resolution
            data['padded_videos_vit'] = sequence_status.pop('vit_video_tensors')

        # if the model is required to perform visual generation
        if len(sequence_status['packed_timesteps']) > 0:
            data['packed_timesteps'] = torch.tensor(sequence_status['packed_timesteps'])
            data['mse_loss_indexes'] = torch.tensor(sequence_status['mse_loss_indexes'])

        # if the model is required to perform text generation
        if len(sequence_status['packed_label_ids']) > 0:
            data['packed_label_ids'] = torch.tensor(sequence_status['packed_label_ids'])
            data['ce_loss_indexes'] = torch.tensor(sequence_status['ce_loss_indexes'])
            data['ce_loss_weights'] = torch.tensor(sequence_status['ce_loss_weights'])

        if len(sequence_status['vae_video_grid_thw']) > 0:
            data['vae_video_grid_thw'] = torch.tensor(sequence_status['vae_video_grid_thw'])
        if len(sequence_status['vit_video_grid_thw']) > 0:
            data['vit_video_grid_thw'] = torch.tensor(sequence_status['vit_video_grid_thw'])

        # Memory optimization: release sequence_status contents that are no longer needed
        sequence_status.clear()
        return data

    def build_datasets(self, datasets_metainfo, data_status):
        datasets = []
        is_mandatory = []
        grouped_weights = []
        data_type = []
        for grouped_dataset_name, dataset_args in datasets_metainfo.items():
            if grouped_dataset_name.startswith('D'): # Handle the new multi-level nested logic
                grouped_dataset_name, dataset_args = list(dataset_args.items())[0]
            if '2t' in grouped_dataset_name:
                data_type.append('x2t')
            else:
                data_type.append('x2v')
            is_mandatory.append(dataset_args.pop('is_mandatory', False))
            grouped_weights.append(dataset_args.pop('weight', 0.0))

            if 'frame_sampler_args' in dataset_args.keys():
                # NOTE: NOT for video
                frame_sampler = FrameSampler(**dataset_args.pop('frame_sampler_args'))
                dataset_args['frame_sampler'] = frame_sampler
            if 'image_transform_args' in dataset_args.keys(): # TODO: deprecate this
                transform = ImageTransform(**dataset_args.pop('image_transform_args'))
                dataset_args['transform'] = transform
            if 'video_transform_args' in dataset_args.keys():
                # video
                transform = VideoTransform(**dataset_args.pop('video_transform_args'))
                dataset_args['transform'] = transform
                dataset_args['vae_downsample'] = self.data_config.vae_downsample # fix: pass this in; TODO: consider vae_downsample and vit_downsample, low priority

                # Add the video frame sampler here
                if 'video_frame_sampler_args' in dataset_args:
                    dataset_args['res_dump'] = dataset_args['video_frame_sampler_args']['res_dump'] if 'res_dump' in dataset_args['video_frame_sampler_args'].keys() else ''

                    video_frame_sampler_args = dataset_args.pop('video_frame_sampler_args')

                    video_frame_sampler = FRAME_SAMPLER_TYPES[video_frame_sampler_args.get("type", "fixed")](**video_frame_sampler_args.get("params", {}))

                    dataset_args['video_frame_sampler'] = video_frame_sampler

            if 'vit_video_transform_args' in dataset_args.keys():
                # video
                vit_transform = VideoTransform(**dataset_args.pop('vit_video_transform_args'))
                dataset_args['vit_transform'] = vit_transform

            elif 'vit_image_transform_args' in dataset_args.keys(): # TODO: deprecate this
                vit_transform = ImageTransform(**dataset_args.pop('vit_image_transform_args'))
                dataset_args['vit_transform'] = vit_transform

            assert 'dataset_names' in dataset_args, dataset_args.keys() or "missing 'dataset_names'"
            dataset_names = dataset_args.pop('dataset_names') # NOTE: Pay attention to this pop pattern
            dataset_args['data_dir_list'] = []

            # Iterate and build datasets
            for item, meta_info in dataset_names.items():
                if self.local_rank == 0:
                    self.logger.info(f'Preparing Dataset {grouped_dataset_name}/{item}')

                data_dir = meta_info['data_dir']
                if isinstance(data_dir, str):
                    # If it is a path
                    dataset_args['data_dir_list'].append(meta_info['data_dir'])
                elif isinstance(data_dir, list):
                    # If it is a list
                    dataset_args['data_dir_list'].extend(data_dir)
                else:
                    raise Exception(f'Unknown data_dir type {type(data_dir)}')

                # NOTE: Collect all paths at the outer level, then pass them in
                all_data_paths = get_parquet_data_paths_balanced(
                    data_dir_list=dataset_args.get('data_dir_list'),
                    rank=self.local_rank,
                    world_size=self.world_size,
                    num_repeat=dataset_args.get('num_repeat', 1),
                )

                if 'all_data_paths' in dataset_args.keys():
                    dataset_args['all_data_paths'].extend(all_data_paths)
                else:
                    dataset_args['all_data_paths'] = all_data_paths

            resume_data_status = dataset_args.pop('resume_data_status', True)
            if data_status is not None and grouped_dataset_name in data_status.keys() and resume_data_status:
                data_status_per_group = data_status[grouped_dataset_name]
            else:
                data_status_per_group = None

            dataset_args.update(self.data_args)

            # Update
            dataset_args['vit_cond_dropout_prob'] = self.data_config.vit_cond_dropout_prob
            dataset_args['text_cond_dropout_prob'] = self.data_config.text_cond_dropout_prob
            dataset_args['vae_cond_dropout_prob'] = self.data_config.vae_cond_dropout_prob

            dataset = DATASET_REGISTRY[grouped_dataset_name](
                dataset_name=grouped_dataset_name,
                tokenizer=self.tokenizer,
                local_rank=self.local_rank,
                world_size=self.world_size,
                num_workers=self.num_workers,
                data_status=data_status_per_group,
                apply_chat_template=self.apply_chat_template,
                **dataset_args,
            )
            datasets.append(dataset)

        return datasets, is_mandatory, grouped_weights, data_type

    # Add the video processing branch in pack_sequence
    def pack_sequence(self, sample: Dict[str, Any], sequence_status: Dict[str, Any]):
        image_tensor_list = sample.get('image_tensor_list', []) # just for debug
        video_tensor_list = sample.get('video_tensor_list', [])
        video_latent_list = sample.get('video_latent_list', [])
        sample_N_target = sample.get('N_target', 1)
        text_ids_list = sample['text_ids_list']
        sequence_plan = sample['sequence_plan']
        sample_task = sample.get('sample_task', 't2v')

        split_lens, attn_modes = list(), list()
        curr = sequence_status['curr']
        curr_rope_id = 0
        sample_lens = 0
        sample_type = ''
        curr_split_idx = sequence_status['curr']
        apply_text_template = False
        curr_video_grid_thw = []

        for item in sequence_plan:
            split_start = item.get('split_start', True)
            if split_start:
                curr_split_len = 0

            # TODO: add more item types to help classification
            if item['type'] == 'text':
                sample_type = 'und' # This is overwritten, so only the last item takes effect
                text_ids = text_ids_list.pop(0)
                if item['enable_cfg'] == 1 and random.random() < self.data_config.text_cond_dropout_prob:

                    if self.cfg_type == 0: # 0 fully removes the text condition
                        continue
                    elif self.cfg_type == 1: # 1 keeps only special tokens
                        text_ids = []
                    elif self.cfg_type == 2: # 2 keeps special tokens and replaces middle text tokens with uncond_token
                        text_ids = [self.cfg_uncond_token_id] * len(text_ids)

                if not item.get('special_token_start_nouse'):  # When special_token_start_nouse is None or False
                    shifted_text_ids = [self.bos_token_id] + text_ids  # NOTE: self.bos_token_id=151644 <|im_start|>
                else:
                    shifted_text_ids = text_ids

                sequence_status['packed_text_ids'].extend(shifted_text_ids)
                sequence_status['packed_text_indexes'].extend(range(curr, curr + len(shifted_text_ids)))

                # NOTE: item['loss'] == 1 identifies understanding vs generation
                if item['loss'] == 1:
                    loss_token_shift = item.get('loss_token_shift') or 0
                    sequence_status['ce_loss_indexes'].extend(range(curr - loss_token_shift, curr + len(shifted_text_ids)))
                    sequence_status['ce_loss_weights'].extend([len2weight(len(shifted_text_ids) + loss_token_shift)] * (len(shifted_text_ids) + loss_token_shift))
                    sequence_status['packed_label_ids'].extend(text_ids + [self.eos_token_id])  # NOTE: self.eos_token_id=151645 <|im_end|>
                curr += len(shifted_text_ids)
                curr_split_len += len(shifted_text_ids)

                # add a <|im_end|> token
                if not item.get('special_token_end_nouse'):
                    sequence_status['packed_text_ids'].append(self.eos_token_id)
                    sequence_status['packed_text_indexes'].append(curr)
                    if item['special_token_loss'] == 1: # <|im_end|> may have loss
                        sequence_status['ce_loss_indexes'].append(curr)
                        sequence_status['ce_loss_weights'].append(1.0)
                        sequence_status['packed_label_ids'].append(item['special_token_label'])
                    curr += 1
                    curr_split_len += 1

                # update sequence status
                attn_modes.append("causal")
                #if self.apply_chat_template:
                sequence_status['packed_position_ids'].extend(range(curr_rope_id, curr_rope_id + curr_split_len))
                curr_rope_id += curr_split_len

                sequence_status['sample_modality'].extend([modality_map[item['type']]] * curr_split_len)

            elif item["type"] == "text_template":
                apply_text_template = True
                text_ids = text_ids_list.pop(0)
                # The current template only applies to UND, so ignore cfg for now

                sequence_status["packed_text_ids"].extend(text_ids)
                sample_lens = len(text_ids)
                spans_index = item.get("spans_index", None)  # Vision padding tokens as a list; each item is an index list for the corresponding video/image pad tokens
                curr_sample_modality = []
                caption_index = item.get("cap_index", []) # Indexes for text excluding the system prompt

                for video_id, span_index in enumerate(spans_index):
                    vision_start, vision_end = curr_split_idx + span_index[0], curr_split_idx + span_index[-1]  # Indexes of the first and last '<|video_pad|>'
                    sequence_status["packed_text_indexes"].extend(range(curr, vision_start))
                    if (vision_start - 1) - curr != 0:  # Confirm there is a text split before vision; HACK: changed from the LLaVA version
                        curr_split_len = (vision_start - 1) - curr
                        sequence_status["packed_position_ids"].extend(
                            range(curr_rope_id, curr_rope_id + curr_split_len)
                        )  # Note: use vision_start - 1, not vision_start, because vision_start is the starting token of the video split
                        curr_rope_id += curr_split_len
                        curr_sample_modality.extend([modality_map['system_prompt']] * curr_split_len)
                        if caption_index != [] and caption_index[0] in range(curr, curr + curr_split_len): # NOTE: interleaved text is not supported; text must be contiguous
                            split_len_1 = caption_index[0] - curr # Length of the system prompt before text
                            split_len_2 = len(caption_index) # Text length
                            split_len_3 = curr_split_len - split_len_1 - split_len_2 # Length of the system prompt after text

                            split_len_text = [split_len_1, split_len_2, split_len_3]
                            split_len_text = [x for x in split_len_text if x != 0]
                            attn_modes.extend(["causal"] * len(split_len_text))
                            split_lens.extend(split_len_text)
                        else:
                            attn_modes.append("causal")
                            split_lens.append(curr_split_len)


                    curr_split_len = len(span_index) + 2
                    if sequence_plan[video_id]["type"] == 'vit_video':
                        sequence_status["packed_vit_token_indexes"].extend(range(vision_start, vision_end + 1))
                        attn_modes.append("full")  # TODO: add this check if the GEN branch also uses templates
                        curr_sample_modality.extend([modality_map['ref_vit']] * curr_split_len)
                    elif sequence_plan[video_id]["type"] == 'vae_video':
                        sequence_status["packed_vae_token_indexes"].extend(range(vision_start, vision_end + 1))
                        if sequence_plan[video_id]["loss"] == 0:
                            attn_modes.append("full_noise")  # TODO: add this check if the GEN branch also uses templates
                            if sample_task == 'edit':
                                curr_sample_modality.extend([modality_map['ref_source']] * curr_split_len)
                            elif sample_task == 'idip' or sample_task == 'maze':
                                curr_sample_modality.extend([modality_map['ref_image']] * curr_split_len)
                        else:
                            if frame_condition_idx == []:
                                sequence_status["mse_loss_indexes"].extend(range(vision_start, vision_end + 1))
                            else: # Support f2v; multiple target videos are not currently supported because different videos need separate THW values
                                frame_condition_indexes = []
                                mse_loss_indexes = list(range(vision_start, vision_end + 1))
                                for idx in frame_condition_idx:
                                    if idx == -1:
                                        idx = t - 1
                                        if idx == 1:
                                            continue  # Skip when there are only two frames to avoid using identical condition frames for all frames
                                    frame_condition_indexes.extend(mse_loss_indexes[idx * h * w : (idx + 1) * h * w])
                                mse_loss_indexes = sorted(list(set(mse_loss_indexes) - set(frame_condition_indexes)))
                                sequence_status["mse_loss_indexes"].extend(mse_loss_indexes)


                            attn_modes.append("noise")  # TODO: add this check if the GEN branch also uses templates
                            sample_type = "gen"  # This is overwritten, so only the last item takes effect
                            curr_sample_modality.extend([modality_map['noise']] * curr_split_len)

                    sequence_status["packed_position_ids"].extend([curr_rope_id] * curr_split_len)
                    #attn_modes.append("full")  # TODO: add this check if the GEN branch also uses templates
                    split_lens.append(len(span_index) + 2)
                    curr = vision_end + 1 # Index of the '<|vision_end|>' token
                    curr_rope_id += 1
                    sequence_status["packed_text_indexes"].append(curr)
                    curr += 1 # Starting token of the next sequence

                len_split_last = sample_lens - (curr - curr_split_idx) if spans_index != [] else len(text_ids)
                if len_split_last != 0:  # A trailing text segment remains
                    split_lens.append(len_split_last)
                    sequence_status["packed_text_indexes"].extend(range(curr, curr + len_split_last))
                    sequence_status["packed_position_ids"].extend(range(curr_rope_id, curr_rope_id + len_split_last))
                    attn_modes.append("causal")
                    curr_sample_modality.extend([modality_map['system_prompt']] * len_split_last)

                if item["loss"] == 1: # This marks an understanding task and requires CE loss
                        packed_label_index = item.get("packed_label_index", [])[:]
                        sequence_status["packed_label_ids"].extend(text_ids[packed_label_index[0]:])
                        packed_label_index = np.asarray(packed_label_index, dtype=np.int64) + curr_split_idx
                        ce_loss_indexes = (packed_label_index - 1).tolist()
                        sequence_status["ce_loss_indexes"].extend(ce_loss_indexes)
                        sequence_status["ce_loss_weights"].extend([len2weight(len(packed_label_index))] * (len(packed_label_index)))
                        sample_type = "und"  # This is overwritten, so only the last item takes effect
                # else:  # This marks a generation task
                #         sample_type = "gen"  # This is overwritten, so only the last item takes effect
                        #packed_label_index = item.get("packed_label_index", [])[1:-2]


                # Get caption indexes in text and update their sample_modality
                if caption_index != []:
                    curr_sample_modality[caption_index[0]:caption_index[-1]+1] = [modality_map['text']] * (caption_index[-1] - caption_index[0] + 1)


                curr_split_idx += len(text_ids)
                curr = curr_split_idx
                sequence_status['sample_modality'].extend(curr_sample_modality)

            elif item['type'] == 'vae_video':
                sample_type = 'gen'
                video_tensor = video_tensor_list.pop(0) # CTHW
                if item['enable_cfg'] == 1 and random.random() < self.data_config.vae_cond_dropout_prob:
                    # FIXME: fix VAE dropout in video2video. TODO: confirm whether enable_cfg is needed in the vae_video branch
                    # curr_rope_id += 1
                    continue

                apply_text_template = item.get('apply_text_template', False)


                num_special_tokens = 0
                # Add <|startofimage|> token shared by video and image. TODO: split image and video special tokens?
                if not apply_text_template:
                    sequence_status['packed_text_ids'].append(self.start_of_image) # self.start_of_image=151652, <|vision_start|>
                    sequence_status['packed_text_indexes'].append(curr)
                    curr += 1
                    curr_split_len += 1
                    num_special_tokens += 1

                # In online mode, video_tensor is a tensor; in offline mode, it is a list [latent]
                if isinstance(video_tensor, torch.Tensor):  # online
                    # Preprocess video

                    sequence_status["vae_video_tensors"].append(video_tensor)  # Raw CTHW video, not latent

                    # Assume video_tensor has shape (C, T, H, W)
                    _, T, H, W = video_tensor.shape
                    _T, _H, _W = self.data_config.vae_downsample  # NOTE: Absolute-scale downsample including patchification
                    t = (T - 1) // _T + 1  # k*N+1; normally the T dimension is not patchified. Update this if T patchification is needed
                    h = H // _H
                    w = W // _W
                    sequence_status["vae_data_mode"].append("online")
                else:  # offline
                    # video_latent = video_tensor[0] # [[T, H, W, C]]
                    sequence_status["vae_video_tensors"].append(video_tensor[0])  # video_tensor[0] is a tensor with shape [T, H, W, C]
                    # Assume video_latent has shape [T, H, W, C]
                    T, H, W, _ = video_tensor[0].shape
                    _T, _H, _W = self.data_config.latent_patch_size  # NOTE: Only includes patchification

                    t = T // _T  # k*N+1; normally the T dimension is not patchified. Update this if T patchification is needed
                    h = H // _H
                    w = W // _W

                    sequence_status["vae_data_mode"].append("offline")

                spatial_merge_size = 2  # TODO: must spatial_merge_size always be 2?
                vae_video_grid_thw = [
                    t,
                    h * spatial_merge_size,
                    w * spatial_merge_size,
                 ]  # Qwen-VL RoPE divides by spatial_merge_size by default to match VIT processing, so VAE needs an extra multiply by spatial_merge_size

                if EXP_HW_20250819: # HACK: temporary experiment
                    vae_video_grid_thw = [1, 2, 2]

                sequence_status["vae_video_grid_thw"].append(vae_video_grid_thw)
                curr_video_grid_thw.append(vae_video_grid_thw)

                # Use the native (t, h, w) latent shape
                sequence_status['vae_latent_shapes'].append((t, h, w))

                # Use the 3D-aware position encoding function
                if self.interpolate_pos:
                    # Interpolation
                    packed_latent_position_ids = get_flattened_position_ids_interpolate_video(
                        t, h, w, 1,  # Latent-space patch size is 1
                        max_num_frames=self.max_num_latent_frames,
                        max_num_patches_per_side=self.data_config.max_latent_size
                    )
                else:
                    # Extrapolation
                    packed_latent_position_ids = get_flattened_position_ids_extrapolate_video(
                        t, h, w,
                        max_latent_size=self.data_config.max_latent_size
                    )

                sequence_status['packed_latent_position_ids'].append(packed_latent_position_ids)

                num_vid_tokens = t * h * w

                if not apply_text_template:
                    sequence_status['packed_vae_token_indexes'].extend(range(curr, curr + num_vid_tokens))

                if item["loss"] == 1:
                    pro_fbyf = False
                    if split_start:
                        timestep = np.random.randn()  # NOTE: A sigmoid is applied outside

                    frame_condition_idx = item.get("frame_condition_idx", [])
                    packed_timesteps = [timestep] * num_vid_tokens

                    mse_loss_indexes = list(range(curr, curr + num_vid_tokens))
                    frame_condition_indexes = []
                    for idx in frame_condition_idx:
                        if idx == -1:
                            idx = t - 1
                            if idx == 1:
                                continue  # Skip when there are only two frames to avoid using identical condition frames for all frames
                        frame_condition_indexes.extend(mse_loss_indexes[idx * h * w : (idx + 1) * h * w])
                        packed_timesteps[idx * h * w : (idx + 1) * h * w] = [-sys.float_info.max] * (h * w)
                    if frame_condition_idx:
                        mse_loss_indexes = sorted(list(set(mse_loss_indexes) - set(frame_condition_indexes)))

                    if not apply_text_template:
                        sequence_status["mse_loss_indexes"].extend(mse_loss_indexes)  # range(curr, curr + num_vid_tokens))
                else:
                    if self.incre_time_pro <= 0:
                        timestep = float("-inf")
                    else:
                        timestep = 0.
                    packed_timesteps = [timestep] * num_vid_tokens

                sequence_status['packed_timesteps'].extend(packed_timesteps)


                if not apply_text_template:
                    curr += num_vid_tokens
                    curr_split_len += num_vid_tokens
                    sequence_status["packed_text_ids"].extend([self.image_token_id] * num_vid_tokens)

                    # Add <|endofimage|> token
                    sequence_status['packed_text_ids'].append(self.end_of_image) # self.end_of_image=151653, <|vision_end|>
                    sequence_status['packed_text_indexes'].append(curr)
                    # <|endofimage|> may have loss
                    if item['special_token_loss'] == 1:
                        sequence_status['ce_loss_indexes'].append(curr)
                        sequence_status['ce_loss_weights'].append(1.0)
                        sequence_status['packed_label_ids'].append(item['special_token_label'])
                    curr += 1
                    curr_split_len += 1
                    num_special_tokens += 1

                    # Update sequence status
                    if split_start:
                        if item['loss'] == 1 and 'frame_delta' not in item.keys(): # Is frame_delta for multi-frame generation?
                            attn_modes.append("noise")
                        else:
                            # attn_modes.append("full")
                            attn_modes.append("full_noise")

                    sequence_status['packed_position_ids'].extend([curr_rope_id] * (num_vid_tokens + num_special_tokens)) # NOTE: why is RoPE fixed?
                    if 'frame_delta' in item.keys():
                        curr_rope_id += item['frame_delta']
                    elif item['loss'] == 0:
                        curr_rope_id += 1

                    # update sample sequence modality
                    if item['loss'] == 1:
                        sequence_status['sample_modality'].extend([modality_map['noise']] * curr_split_len)
                    elif item['loss'] == 0 and sample_task == 'edit':
                        sequence_status['sample_modality'].extend([modality_map['ref_source']] * curr_split_len)
                    elif item['loss'] == 0 and (sample_task == 'idip' or sample_task == 'maze'):
                        sequence_status['sample_modality'].extend([modality_map['ref_image']] * curr_split_len)

                del video_tensor, packed_timesteps, packed_latent_position_ids

            elif item['type'] == 'vit_video':
                apply_text_template = item.get('apply_text_template', False)
                video_tensor = video_tensor_list.pop(0) # CTHW
                if item['enable_cfg'] == 1 and random.random() < self.data_config.vit_cond_dropout_prob and not apply_text_template:
                    # FIXME: fix VAE dropout in video2video. TODO: confirm whether enable_cfg is needed in the vit_video branch
                    curr_rope_id += 1 # HACK ？？？？
                    continue

                # Add <|startofimage|> token shared by video and image. TODO: split image and video special tokens?
                if not apply_text_template:
                    sequence_status['packed_text_ids'].append(self.start_of_image) # 151652, <|vision_start|>
                    sequence_status['packed_text_indexes'].append(curr)
                    curr += 1
                    curr_split_len += 1

                # In online mode, video_tensor is a tensor; in offline mode, it is a list [latent]
                if isinstance(video_tensor, torch.Tensor): # online
                    sequence_status['vit_video_tensors'].append(video_tensor) # Raw CTHW video, not latent; used only for validation visualization

                    # preprocess video
                    vit_tokens = patchify_video_with_merge(video_tensor, self.data_config.vit_patch_size, self.data_config.vit_patch_size_temporal) # C T H W -> (T//2 * H//p * W//p) (p*p*2*C)
                    num_video_tokens = vit_tokens.shape[0] // 4  # Qwen2.5-VL also needs merging: 2x2 merges into 1, hardcoded temporarily
                    t, h, w = video_tensor.size(1), video_tensor.size(2), video_tensor.size(3)

                    sequence_status['packed_vit_tokens'].append(vit_tokens)
                    sequence_status['vit_data_mode'].append('online')

                    del vit_tokens

                else: # offline
                    # video_latent, video_tensor = video_tensor # [L,D]
                    # sequence_status['vit_video_tensors'].append(video_tensor[0])
                    num_video_tokens = video_tensor[0].shape[0]

                    sequence_status['packed_vit_tokens'].append(video_tensor[0])
                    sequence_status['vit_data_mode'].append('offline')
                    thw = item.get('thw',None)
                    if thw is not None:
                        t, h, w = thw
                    else:
                        t = None

                if t is not None:
                    vit_video_grid_thw = [
                        t // self.data_config.vit_patch_size_temporal,
                        h // self.data_config.vit_patch_size,
                        w // self.data_config.vit_patch_size,
                    ] # [1, 16, 16]
                    sequence_status["vit_video_grid_thw"].append(vit_video_grid_thw)
                    curr_video_grid_thw.append(vit_video_grid_thw)

                sequence_status['vit_token_seqlens'].append(num_video_tokens)
                sequence_status['packed_vit_position_ids'].append(torch.zeros(num_video_tokens)) # TODO: this may not always be 0; multiple VIT sequences may be problematic, but this is currently only used in vit model:siglip

                if not apply_text_template:
                    sequence_status['packed_vit_token_indexes'].extend(range(curr, curr + num_video_tokens))
                    curr += num_video_tokens
                    curr_split_len += num_video_tokens

                    # NOTE dummy position_ids
                    sequence_status['packed_text_ids'].extend([self.image_token_id]*num_video_tokens)

                    # add a <|endofimage|> token
                    sequence_status['packed_text_ids'].append(self.end_of_image) # 151653, <|vision_end|>
                    sequence_status['packed_text_indexes'].append(curr)
                    if item['special_token_loss'] == 1: # <|endofimage|> may have loss
                        sequence_status['ce_loss_indexes'].append(curr)
                        sequence_status['ce_loss_weights'].append(1.0)
                        sequence_status['packed_label_ids'].append(item['special_token_label'])
                    curr += 1
                    curr_split_len += 1
                    sequence_status['packed_position_ids'].extend([curr_rope_id] * curr_split_len)
                    curr_rope_id += 1

                    # update sequence status
                    attn_modes.append("full")

                    sequence_status['sample_modality'].extend([modality_map['ref_vit']] * curr_split_len)

                del video_tensor

            if item.get('split_end', True) and not apply_text_template:
                if isinstance(curr_split_len, list):
                    split_lens.extend(curr_split_len)
                    sample_lens += sum(curr_split_len)
                else:
                    split_lens.append(curr_split_len)
                    sample_lens += curr_split_len

        sequence_status['curr'] = curr
        sequence_status['sample_lens'].append(sample_lens) # sample_lens is the length of a pair, e.g. <text, video>
        sequence_status['sample_type'].append(sample_type)
        sequence_status['sample_N_target'].append(sample_N_target)
        sequence_status['video_grid_thw'].append(torch.tensor(curr_video_grid_thw)) # video_grid_thw is a sequence split by sample
        # prepare attention mask
        if not self.use_flex:
            sequence_status['nested_attention_masks'].append(
                prepare_attention_mask_per_sample(split_lens, attn_modes)
            )
        else:
            sequence_status['split_lens'].extend(split_lens)
            sequence_status['attn_modes'].extend(attn_modes)
        # add sample_task
        sequence_status['sample_task'].extend([sample_task_map[sample_task]] * sample_lens)
        return sequence_status

    def sample_pro(self, sample, sequence_status, batch_data_indexes, buffer, sample_from_buffer=False, skip_count=0):

        num_tokens = sample['num_tokens'] + 2 * len(sample['sequence_plan']) # NOTE: 2*2 special tokens
        is_pro = False
        if num_tokens < self.max_num_tokens_per_sample and sequence_status['curr'] + num_tokens < self.max_num_tokens :
            sequence_status = self.pack_sequence(sample, sequence_status)
            batch_data_indexes.append(sample['data_indexes'])
            is_pro = True
        elif sequence_status['curr'] + num_tokens > self.max_num_tokens:
            if len(buffer) < self.max_buffer_size and not sample_from_buffer:
                    buffer.append(sample)
            else:
                self.logger.info(f"skip a sample with length {num_tokens}")
                skip_count += 1
        else:
            self.logger.info(f"skip a sample with length {num_tokens}")
            skip_count += 1
        return sequence_status, batch_data_indexes, buffer, is_pro, skip_count


    def __iter__(self):
        # NOTE: Core logic repeatedly calls pack_sequence to pack multimodal data
        # Create underlying dataset iterators at each iteration; spawn/fork safe
        self.dataset_iters = [iter(dataset) for dataset in self.grouped_datasets]

        total_weights = sum(self.grouped_weights)
        assert total_weights > 0.0
        group_cumprobs = [sum(self.grouped_weights[: i + 1]) / total_weights for i in range(len(self.grouped_weights))]
        sequence_status = self.set_sequence_status()
        batch_data_indexes = []

        buffer = []
        data_type_pro = ['x2t', 'x2v']
        skip_count = 0
        max_skips_before_reset = 30  # Threshold for resetting sequence_status
        while True:
            # Ensure at least one sample from each group
            if sequence_status['curr'] == 0:
                for group_index, group_iter in enumerate(self.dataset_iters):
                    if self.is_mandatory[group_index]:
                        while True:
                            sample = next(group_iter)
                            # if a sample is too long, skip it
                            sequence_status, batch_data_indexes, buffer, is_pro, skip_count = self.sample_pro(sample, sequence_status, batch_data_indexes, buffer, skip_count=skip_count)
                            if self.data_type[group_index] in data_type_pro:
                                data_type_pro.remove(self.data_type[group_index])
                            if is_pro:
                                break
            if self.require_und_gen and 'x2t' in self.data_type  and 'x2v' in self.data_type: # NOTE: In joint UND + GEN training, sequences with only one sample may cause communication failures
                while True:
                    if data_type_pro == []:
                        break
                    n = random.random()
                    group_index = bisect.bisect_left(group_cumprobs, n)

                    if self.data_type[group_index] in data_type_pro:
                        sample = next(self.dataset_iters[group_index])
                        sequence_status, batch_data_indexes, buffer, is_pro, skip_count = self.sample_pro(sample, sequence_status, batch_data_indexes, buffer, skip_count=skip_count)
                        if is_pro:
                            data_type_pro.remove(self.data_type[group_index])

                    if skip_count >= max_skips_before_reset: # NOTE: If 30 samples are skipped consecutively, break and reset sequence_status
                        break

            if  skip_count >= max_skips_before_reset: # Reset sequence_status
                self.logger.info(f"Too many skips ({skip_count}), resetting sequence_status")
                sequence_status = self.set_sequence_status()
                batch_data_indexes = []
                data_type_pro = ['x2t', 'x2v']
                skip_count = 0
                continue


            if sequence_status['curr'] < self.prefer_buffer_before and len(buffer) > 0:
                sample = buffer.pop(0)
                sample_from_buffer = True
            else:
                # sample normally across all groups
                n = random.random()
                group_index = bisect.bisect_left(group_cumprobs, n)
                sample = next(self.dataset_iters[group_index])
                sample_from_buffer = False

            # if a sample is too long, skip it
            num_tokens = sample['num_tokens'] + 2 * len(sample['sequence_plan'])
            if num_tokens > self.max_num_tokens_per_sample:
                self.logger.info(f"skip a sample with length {num_tokens}")
                continue


            if sequence_status['curr'] + num_tokens > self.max_num_tokens or len(sequence_status['ce_loss_indexes']) > self.expected_num_ce_loss_tokens:
                if len(buffer) < self.max_buffer_size and not sample_from_buffer:
                    buffer.append(sample)
                # elif self.require_und_gen:
                #     sample_type = 1

                else:
                    # self.logger.info(f"Yielding data with length {sum(sequence_status['sample_lens'])}")
                    data = self.to_tensor(sequence_status)
                    data['batch_data_indexes'] = batch_data_indexes
                    data_type_pro = ['x2t', 'x2v']
                    yield data
                    # Reset sequence_status after yield
                    sequence_status = self.set_sequence_status()
                    batch_data_indexes = []
                continue

            sequence_status = self.pack_sequence(sample, sequence_status)
            batch_data_indexes.append(sample['data_indexes'])

            if sequence_status['curr'] >= self.expected_num_tokens or len(sequence_status['ce_loss_indexes']) >= self.expected_num_ce_loss_tokens:
                data = self.to_tensor(sequence_status)
                data['batch_data_indexes'] = batch_data_indexes
                data_type_pro = ['x2t', 'x2v']
                yield data
                # Reset sequence_status after yield
                sequence_status = self.set_sequence_status()
                batch_data_indexes = []


class SimpleCustomBatch:
    def __init__(self, batch):
        data = batch[0]
        for key, value in data.items():
            setattr(self, key, value)

    def pin_memory(self):
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, key, value.pin_memory())
            elif isinstance(value, list) and value and all(isinstance(i, torch.Tensor) for i in value):
                setattr(self, key, [i.pin_memory() for i in value])

        return self

    def cuda(self, device):
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, key, value.to(device))
            elif isinstance(value, list) and value and all(isinstance(i, torch.Tensor) for i in value):
                setattr(self, key, [i.to(device) for i in value])

        return self

    def to_dict(self):
        return self.__dict__.copy()


def collate_wrapper():
    def collate_fn(batch):
        return SimpleCustomBatch(batch)
    return collate_fn

# Top-level function; pickleable
def simple_custom_collate(batch):
    return SimpleCustomBatch(batch)
