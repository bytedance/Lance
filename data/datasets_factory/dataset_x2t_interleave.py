# coding: utf-8


from typing import Any, Dict, Optional
import random

from data.base_mm_parquet_dataset import BaseMMParquetDataset, sample_task, data_invert_text_image_pair
from data.common import generate_system_prompt, detect_lang_simple
from data.system_prompt_render import render_qwenvl_prompt, expand_and_index_by_token_ids_new

class X2TInterleaveIterableDataset(BaseMMParquetDataset):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("vision_stream", "vae_video")  # NOTE: Must be vae_video, vit_video, or another supported stream
        super().__init__(*args, **kwargs)

        self.target_modality = "text"  # Target element type
        self.task_type = kwargs.get("task_type", "v2t")
        self.task_type_rate = kwargs.get("task_type_rate", 1)
        self.text_first = False
        self.raw_bytes_input = kwargs.get("raw_bytes_input", False) # Whether image and video inputs are raw bytes
        if self.local_rank == 0:
            self.logger.info(f"X2TInterleaveIterableDataset raw_bytes_input: {self.raw_bytes_input}")

    def _process_row(self, row: Any, parquet_idx: int, row_group_id: int, row_idx: int, worker_id: int, parquet_file_path: str) -> Optional[Dict[str, Any]]:
        """
        Process one row.
        Return None if processing fails or the data is invalid.
        """
        try:
            if self.dataset_type == "interleave":
                interleave_array, element_dtype_array = row["interleave_array"], row["element_dtype_array"]
            else:
                try:
                    interleave_array, element_dtype_array = self.transform_row(row)
                except Exception as e:
                    if "ocr" not in self.dataset_type: # OCR has filtering; skip this log
                        self.logger.warning(f"Warning transform row: {e} in self.dataset_type: {self.dataset_type}")
                    return None

            interleave_array, element_dtype_array = data_invert_text_image_pair(interleave_array, element_dtype_array, self.target_modality)

            try:
                condition_idx, target_idx, N_target = self.get_condition_target_idx(element_dtype_array)
            except Exception as e:
                self.logger.warning(f"Warning processing row: num of target element {self.target_modality} is 0, element_dtype_array is {element_dtype_array}")
                return None

            condition_modalities = [element_dtype_array[i] for i in condition_idx]

            # Select task type
            if "text" not in condition_modalities:
                task_type_sample = "v2t"
            elif "image" not in condition_modalities and "video" not in condition_modalities:
                task_type_sample = "t2t"
            elif isinstance(self.task_type, list):
                task_type_sample = sample_task(self.task_type, self.task_type_rate)
            else:
                task_type_sample = self.task_type

            num_tokens, sequence_plan, text_ids_list, video_tensor_list, target_captions = [], [], [], [], []  # sequence_plan stores loss, cfg, and other marker metadata
            num_split_vit, num_split_vae, num_split_text = 0, 0, 0
            text_template_user, text_template_assistant, vit_num_tokens = [], [], []
            for idx in range(target_idx[-1] + 1):
                if idx in condition_idx:  # Process condition element
                    if task_type_sample in ["v2t", "tv2t", "vt2t"] and element_dtype_array[idx] in ["image","video"]:  # Visual condition element
                        if num_split_vit >= self.max_num_split_vit:
                            continue
                        media_url = interleave_array[idx]
                        video_tensor, num_tokens_, thw = self.get_video_tensor(
                            media_url,
                            vision_stream="vit_video",
                            element_dtype=element_dtype_array[idx],
                            raw_bytes_input=self.raw_bytes_input,
                        )

                        video_tensor_list.append(video_tensor)

                        if self.text_template:
                            text_template_user.append({"type": element_dtype_array[idx]})
                            vit_num_tokens.append(num_tokens_)
                        else:
                            num_tokens.append(num_tokens_)  # NOTE: Count vision tokens

                        sequence_plan.append(
                            {
                                "type": "vit_video",
                                "enable_cfg": 0,
                                "loss": 0,
                                "special_token_loss": 0,  # NOTE: Special tokens are provided manually and are not predicted
                                "special_token_label": None,  # eos
                                "apply_text_template": self.text_template,  # Default is false
                                "thw": thw,
                            }
                        )
                        num_split_vit += 1
                    elif task_type_sample in ["t2t", "tv2t", "vt2t"] and element_dtype_array[idx] == "text":  # Process text condition element
                        if num_split_text >= (self.max_num_split_text - N_target):
                            continue
                        caption = self.text_cleaner(interleave_array[idx])
                        if not caption:
                            continue

                        if self.text_template:
                            text_template_user.append({"type": "text", "text": caption})
                        else:
                            text_ids = self.tokenizer.encode(caption)
                            num_tokens.append(len(text_ids))
                            text_ids_list.append(text_ids)
                            sequence_plan.append(
                                {
                                    "type": "text",
                                    "enable_cfg": 0,
                                    "loss": 0,
                                    "special_token_loss": 0,  # NOTE: Special tokens are provided manually and are not predicted
                                    "special_token_label": None,
                                }
                            )
                        num_split_text += 1
                elif idx in target_idx:  # Process target element
                    if num_split_text >= self.max_num_split_text:
                        continue
                    caption = interleave_array[idx]
                    target_captions.append(caption)
                    num_split_text += 1

            if len(target_captions) == 0:
                return None

            target_caption = target_captions[-1]
            if (
                isinstance(target_caption, str)
                and detect_lang_simple(target_caption) != "en"
            ) or (
                not isinstance(target_caption, str)
                and detect_lang_simple(target_caption[1]) != "en"
                and target_caption[1] != ""
            ):
                self.logger.warning(f"Wrong caption: {target_caption}")
                return None

            if self.text_template:
                if isinstance(target_caption, str): # Only one text item
                    if "video" in condition_modalities:
                        vision_type = 'video'
                    else:
                        vision_type = 'image'
                    caption_a = target_caption
                    caption_i = generate_system_prompt(system_prompt_type='caption', vision_type=vision_type)
                    caption_q = ""
                else:
                    caption_i, caption_q, caption_a = target_caption[0], target_caption[1], target_caption[2]

                if self.data_config.get('system_prompt_type') == 'SP2':
                    caption_q = caption_i + " " + caption_q
                    caption_i = "You are a helpful assistant. "
                elif self.data_config.get('system_prompt_type') == 'SP1':
                    # SP1: assistant
                    caption_i = "You are a helpful assistant. " + caption_i

                text_template_assistant.append({"type": "text", "text": caption_a}) # caption
                if caption_q != "":
                    text_template_user.append({"type": "text", "text": caption_q})

                messages = [
                    {
                        "role": "user",
                        "content": text_template_user, # Original usage
                    },
                    {
                        "role": "assistant",
                        "content": text_template_assistant,
                    },
                ]
                caption_all = render_qwenvl_prompt(messages, default_system=caption_i, include_assistant_content=True) # NOTE: Whether to add 'You are a helpful assistant.'

                all_token_id, spans_index, tgt_index, search_index = expand_and_index_by_token_ids_new(
                    rendered_text=caption_all.strip(), tokens=vit_num_tokens, target_text=f"assistant\n", tokenizer=self.tokenizer,search_text=""
                )

                assert len(all_token_id[tgt_index[0] :]) == len(tgt_index)

                num_tokens.append(len(all_token_id))

                text_ids_list.append(all_token_id)
                sequence_plan.append(
                    {
                        "type": "text_template",
                        "enable_cfg": 0,
                        "loss": 1,
                        "special_token_loss": 0,
                        "special_token_label": None,
                        "packed_label_index": tgt_index,
                        "spans_index": spans_index,
                    }
                )
            else:
                caption = " ".join(self.text_cleaner(item) for item in target_captions if self.text_cleaner(item))
                caption = self.text_cleaner(caption)
                text_ids = self.tokenizer.encode(caption)
                num_tokens.append(len(text_ids))
                text_ids_list.append(text_ids)
                sequence_plan.append(
                    {
                        "type": "text",
                        "enable_cfg": 0,
                        "loss": 1,
                        "special_token_loss": 0,
                        "special_token_label": None,
                    }
                )

            sample = dict(
                video_tensor_list=video_tensor_list,
                text_ids_list=text_ids_list,
                num_tokens=sum(num_tokens),
                sequence_plan=sequence_plan,
                N_target=1,  # In theory, N_target is always 1 for the UND branch
                data_indexes={
                    "data_indexes": [parquet_idx, row_group_id, row_idx],
                    "worker_id": worker_id,
                    "dataset_name": self.dataset_name,
                },
            )
            return sample
        except Exception as e:
            # Keep a top-level catch-all for unexpected errors
            self.logger.warning(f"Error processing row: {e} in rg#{row_group_id}, {parquet_file_path}")
            return None
