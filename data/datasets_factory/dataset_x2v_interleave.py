# coding: utf-8


from typing import Any, Dict, Optional
import random
from data.base_mm_parquet_dataset import BaseMMParquetDataset, sample_task, data_invert_text_image_pair
from data.common import generate_system_prompt
from data.system_prompt_render import render_qwenvl_prompt, expand_and_index_by_token_ids_new


class X2VInterleaveIterableDataset(BaseMMParquetDataset):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("vision_stream", "vae_video")  # NOTE: 必须指定为 vae_video or vit_video or 其它⚠️
        super().__init__(*args, **kwargs)

        self.target_modality = kwargs.get("target_modality", "image") #默认 "image"  ,  目标 element 类型: 如果是I2T 则为"text"
        self.task_type = kwargs.get("task_type", "t2v")
        self.task_type_rate = kwargs.get("task_type_rate", 1)
        self.text_first = True
        self.L_video_group = kwargs.get("L_video_group", 1000)  # L_video_group 为视频拆分成的group 数量, 大于t 则为不处理
        self.random_choose_group = kwargs.get("random_choose_group", False) # 是否随机选择 视频组
        self.raw_bytes_input = kwargs.get("raw_bytes_input", False) # 图像和视频是否原始字节输入
        if self.local_rank == 0:
            self.logger.info(f"X2VInterleaveIterableDataset raw_bytes_input: {self.raw_bytes_input}")

    def _process_row(self, row: Any, parquet_idx: int, row_group_id: int, row_idx: int, worker_id: int, parquet_file_path: str) -> Optional[Dict[str, Any]]:
        """
        处理单行数据。
        如果处理失败或数据无效，则返回 None。
        """
        try:
            if self.dataset_type == "interleave":
                interleave_array, element_dtype_array = row["interleave_array"], row["element_dtype_array"]
            else:
                # Local text2video_general / text2image_general data also goes through the shared transform_row path.
                try:
                    interleave_array, element_dtype_array = self.transform_row(row)
                except Exception as e:
                    if "ocr" not in self.dataset_type: # OCR  有过滤， 忽略此项loggr
                        self.logger.warning(f"Warning transform row: {e} in self.dataset_type: {self.dataset_type}")
                    return None

            interleave_array, element_dtype_array = data_invert_text_image_pair(interleave_array, element_dtype_array, self.target_modality)

            try:
                condition_idx, target_idx, N_target = self.get_condition_target_idx(element_dtype_array)
            except Exception as e:
                self.logger.warning(f"Warning processing row: num of target element {self.target_modality} is 0, element_dtype_array is {element_dtype_array}")
                return None

            # 选择 任务类型
            if isinstance(self.task_type, str):
                task_type_sample = self.task_type
            elif "text" not in element_dtype_array[condition_idx]:
                task_type_sample = "v2v"
            elif isinstance(self.task_type, list):
                task_type_sample = sample_task(self.task_type, self.task_type_rate)

            if task_type_sample == 'flf2v': # First and Last Frame to Video Generation
                self.frame_condition_idx = [0, -1]
            elif task_type_sample == 'ff2v': # First Frame to Video Generation
                self.frame_condition_idx = [0]
            else:
                self.frame_condition_idx = []

            num_tokens, sequence_plan, text_ids_list, video_tensor_list = [], [], [], []  # sequence_plan 定义了loss、cfg等标记信息
            text_template_user, text_template_assistant, vit_num_tokens, search_text = [], [], [], ""
            num_split_vit, num_split_vae, num_split_text = 0, 0, 0
            is_offline = False
            for idx in range(target_idx[-1] + 1):
                if idx in condition_idx:  # 处理条件element
                    if task_type_sample in ["t2v", "tv2v", "vt2v", "flf2v", "ff2v"] and element_dtype_array[idx] == "text":  # 文本条件element (必须处理)
                        if num_split_text >= self.max_num_split_text:
                            continue
                        caption = interleave_array[idx]
                        caption = self.text_cleaner(caption)

                        if not caption:
                            continue
                        if self.text_template:
                            if random.random() > self.data_config['text_cond_dropout_prob']: # HACK : cfg 处理, 进行text cond 的drop
                                text_template_user.append({"type": "text", "text": caption})
                                search_text = caption
                        else:
                            text_ids = self.tokenizer.encode(caption)
                            num_tokens.append(len(text_ids))
                            text_ids_list.append(text_ids)
                            sequence_plan.append(
                                {
                                    "type": "text",
                                    "enable_cfg": 1,
                                    "loss": 0,
                                    "special_token_loss": 0,  # NOTE: 需要自己给特殊token，不做predict
                                    "special_token_label": None,
                                }
                            )
                        num_split_text += 1
                    elif task_type_sample in ["v2v", "tv2v", "vt2v", "flf2v", "ff2v"] and element_dtype_array[idx] in ["image", "video"]:  # 处理图像条件element  (可选处理)
                        if num_split_vit >= self.max_num_split_vit:
                            continue
                        if self.text_template and random.random() < self.data_config['vit_cond_dropout_prob']: # HACK : cfg 处理, 进行 vit cond 的drop ， 放在这即 drop vae & vit cond
                            continue

                        media_url = interleave_array[idx]
                        if "vit" in self.vision_cond_type:
                            # if self.text_template and random.random() < self.data_config['vit_cond_dropout_prob']: # HACK : cfg 处理, 进行 vit cond 的drop
                            #     pass
                            # else:
                            video_tensor, num_tokens_, thw = self.get_video_tensor(media_url, vision_stream="vit_video", element_dtype=element_dtype_array[idx], raw_bytes_input=self.raw_bytes_input)
                            video_tensor_list.append(video_tensor)
                            sequence_plan.append(
                                {
                                    "type": "vit_video",
                                    "enable_cfg": 1,
                                    "loss": 0,
                                    "special_token_loss": 0,  # NOTE: 需要自己给特殊token，不做predict
                                    "special_token_label": None,  # eos
                                    "thw": thw,
                                    "apply_text_template": self.text_template,  # 默认false
                                }
                            )
                            num_tokens.append(num_tokens_)  # NOTE: 计算vision token数量
                            num_split_vit += 1

                            if self.text_template:
                                text_template_user.append({"type": element_dtype_array[idx]})
                                vit_num_tokens.append(num_tokens_)

                        del video_tensor
                        if  "vae_Nsplit" in self.vision_cond_type:
                            video_tensor, num_tokens_, thw = self.get_video_tensor(media_url, vision_stream="vae_video", element_dtype=element_dtype_array[idx], raw_bytes_input=self.raw_bytes_input)
                            sequence_plan.append(
                                {
                                    "type": "vae_video",
                                    "enable_cfg": 1,
                                    "loss": 0,
                                    "special_token_loss": 0,  # NOTE: 需要自己给特殊token，不做predict
                                    "special_token_label": None,  # eos
                                    "thw": thw,
                                    "apply_text_template": self.text_template,  # 默认false
                                }
                            )
                            video_tensor_list.append(video_tensor)
                            num_tokens.append(num_tokens_)  # NOTE: 计算vision token数量
                            num_split_vae += 1

                            if self.text_template:
                                text_template_user.append({"type": element_dtype_array[idx]})
                                vit_num_tokens.append(num_tokens_)

                        del video_tensor

                elif idx in target_idx:  # 处理目标element
                    # if num_split_vae >= self.max_num_split_vae:
                    #     continue
                    media_url = interleave_array[idx]
                    video_tensor, num_tokens_, thw = self.get_video_tensor(media_url, vision_stream="vae_video", element_dtype=element_dtype_array[idx], raw_bytes_input=self.raw_bytes_input)

                    if isinstance(video_tensor, list):
                        is_offline = True

                    # 支持group-by-group的生成
                    if is_offline:
                        _ = video_tensor[0].shape[0]
                    else:
                        _ = (video_tensor.shape[1] - 1) // self.vae_downsample[0] + 1

                    video_tensor_list.append(video_tensor)
                    del video_tensor
                    num_tokens.append(num_tokens_)  # NOTE: 计算vision token数量
                    sequence_plan.append(
                        {
                            "type": "vae_video",
                            "enable_cfg": 0,
                            "loss": 1,
                            "special_token_loss": 0,  # NOTE: 需要自己给特殊token，不做predict
                            "special_token_label": None,
                            "frame_condition_idx": self.frame_condition_idx,
                            "L_video_group": self.L_video_group,
                            "random_choose_group": self.random_choose_group,
                            "thw": thw,
                            "apply_text_template": self.text_template,  # 默认false
                            }
                            #"N_key_frame": self.N_key_frame,
                    )
                    num_split_vae += 1

                    if self.text_template:
                        text_template_assistant.append({"type": element_dtype_array[idx]})
                        vit_num_tokens.append(num_tokens_)

                        # 先视频后文本的处理：
                        text_template_user = text_template_user[1:] + text_template_user[:1] # HACK


                        messages = [
                            {
                                "role": "user",
                                "content":  text_template_user, # 原使用
                            },
                            {
                                "role": "assistant",
                                "content": text_template_assistant,
                            },
                        ]

                        caption_all = render_qwenvl_prompt(messages, default_system=generate_system_prompt(self.sample_task, vision_type=element_dtype_array[idx]), include_assistant_content=True)

                        all_token_id, spans_index, tgt_index, search_index = expand_and_index_by_token_ids_new(
                            rendered_text=caption_all.strip(), tokens=vit_num_tokens, target_text=f"assistant\n", tokenizer=self.tokenizer, search_text=search_text
                        )

                        text_ids_list.append(all_token_id)
                        sequence_plan.append(
                            {
                                "type": "text_template",
                                "enable_cfg": 0,
                                "loss": 0,
                                "special_token_loss": 0,
                                "special_token_label": None,
                                "packed_label_index": tgt_index,
                                "spans_index": spans_index,
                                'cap_index': search_index,
                            }
                        )

            sample = dict(
                video_tensor_list=video_tensor_list,
                text_ids_list=text_ids_list,
                num_tokens=sum(num_tokens) if not self.text_template else len(all_token_id),
                sequence_plan=sequence_plan,
                sample_task=self.sample_task,
                N_target=N_target,
                data_indexes={
                    "data_indexes": [parquet_idx, row_group_id, row_idx],
                    "worker_id": worker_id,
                    "dataset_name": self.dataset_name,
                },
            )
            return sample

        except Exception as e:
            # 最外层保留一个总的异常捕获，以防其他未知错误
            self.logger.warning(f"Error processing row: {e} in rg#{row_group_id}, {parquet_file_path}")
            return None
