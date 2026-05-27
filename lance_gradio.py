from __future__ import annotations

import os
import concurrent.futures
import gc
import ipaddress
import threading
import time
import traceback
from collections import deque
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Optional

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    os.getenv("LANCE_GRADIO_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8"),
)

def _sanitize_no_proxy_for_httpx() -> None:
    for env_key in ("no_proxy", "NO_PROXY"):
        value = os.environ.get(env_key)
        if not value:
            continue
        safe_items = []
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            if "://" not in item and item.startswith("["):
                end = item.find("]")
                if end > 1:
                    host = item[1:end]
                    rest = item[end + 1:]
                    is_bracketed_ipv6 = False
                    try:
                        is_bracketed_ipv6 = ipaddress.ip_address(host).version == 6
                    except ValueError:
                        pass
                    if is_bracketed_ipv6 and (not rest or (rest.startswith(":") and rest[1:].isdigit())):
                        item = host
            safe_items.append(item)
        os.environ[env_key] = ",".join(safe_items)

_sanitize_no_proxy_for_httpx()

from safetensors.torch import load_file
import torch
from transformers import set_seed
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig

from common.gradio_utils.helpers import *
from common.gradio_utils.render import build_demo
from common.gradio_utils.settings import *
from common.utils.logging import get_logger
from common.utils.misc import AutoEncoderParams, tuple_mul
from config.config_factory import DataArguments, InferenceArguments, ModelArguments
from data.data_utils import add_special_tokens
from data.dataset_base import DataConfig, simple_custom_collate
from data.datasets_custom import ValidationDataset
from inference_lance import (
    apply_inference_defaults,
    clean_memory,
    init_from_model_path_if_needed,
    save_prompt_results,
    validate_on_fixed_batch,
)
from modeling.lance import Lance, LanceConfig, Qwen2ForCausalLM
from modeling.qwen2 import Qwen2Tokenizer
from modeling.qwen2.modeling_qwen2 import Qwen2Config
from modeling.vae.wan.model import WanVideoVAE
from modeling.vit.qwen2_5_vl_vit import Qwen2_5_VisionTransformerPretrainedModel

class LanceT2VV2TPipeline:
    def __init__(self, device_id: int, model_variant: str = MODEL_VARIANT_VIDEO) -> None:
        self._init_lock = threading.Lock()
        self._generate_lock = threading.Lock()
        self.initialized = False
        self.device = device_id
        self.model_variant = normalize_model_variant(model_variant)
        self.logger = get_logger(f"lance_{self.model_variant}_gpu{device_id}")

        self.model: Optional[Lance] = None
        self.vae_model: Optional[WanVideoVAE] = None
        self.vae_config: Optional[AutoEncoderParams] = None
        self.tokenizer: Optional[Qwen2Tokenizer] = None
        self.new_token_ids: Optional[dict] = None
        self.image_token_id: Optional[int] = None
        self.base_model_args: Optional[ModelArguments] = None
        self.base_data_args: Optional[DataArguments] = None
        self.base_inference_args: Optional[InferenceArguments] = None

    def _log_stage(self, stage_name: str, start_time: float, extra: str = "") -> None:
        elapsed = time.perf_counter() - start_time
        suffix = f" | {extra}" if extra else ""
        print(f"[startup][gpu:{self.device}] {stage_name} done in {elapsed:.2f}s{suffix}", flush=True)

    def _build_base_model_args(self) -> ModelArguments:
        model_path = str(get_model_path(self.model_variant))
        return ModelArguments(
            model_path=model_path,
            vit_type=DEFAULT_VIT_TYPE,
            llm_qk_norm=True,
            llm_qk_norm_und=True,
            llm_qk_norm_gen=True,
            tie_word_embeddings=False,
            max_num_frames=MAX_VIDEO_NUM_FRAMES,
            max_latent_size=64,
            latent_patch_size=[1, 1, 1],
        )

    def _build_base_inference_args(self) -> InferenceArguments:
        return InferenceArguments(
            validation_num_timesteps=DEFAULT_TIMESTEPS,
            validation_timestep_shift=DEFAULT_TIMESTEP_SHIFT,
            copy_init_moe=True,
            visual_und=True,
            visual_gen=True,
            vae_model_type="wan",
            apply_qwen_2_5_vl_pos_emb=True,
            apply_chat_template=False,
            cfg_type=0,
            validation_data_seed=42,
            video_height=DEFAULT_HEIGHT,
            video_width=DEFAULT_WIDTH,
            num_frames=DEFAULT_NUM_FRAMES,
            task=DEFAULT_TASK,
            save_path_gen=str(RESULTS_ROOT),
            resolution=DEFAULT_RESOLUTION,
            text_template=TEXT_TEMPLATE,
            use_KVcache=USE_KVCACHE,
        )

    def initialize(self) -> None:
        with self._init_lock:
            if self.initialized:
                return

            ensure_dirs()
            resolved_model_path = ensure_model_assets(self.model_variant)
            print(
                f"[startup][gpu:{self.device}][{self.model_variant}] Using Lance model path: {resolved_model_path}",
                flush=True,
            )
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable. Lance T2V/V2T Gradio requires a GPU environment.")
            if self.device >= torch.cuda.device_count():
                raise RuntimeError(
                    f"GPU {self.device} is unavailable. Detected {torch.cuda.device_count()} GPU(s)."
                )
            torch.cuda.set_device(self.device)

            model_args = self._build_base_model_args()
            data_args = DataArguments()
            inference_args = self._build_base_inference_args()
            apply_inference_defaults(model_args, data_args, inference_args)
            inference_args.validation_noise_seed = inference_args.validation_data_seed

            self.base_model_args = model_args
            self.base_data_args = data_args
            self.base_inference_args = inference_args

            set_seed(inference_args.global_seed)

            stage_start = time.perf_counter()
            print(
                f"[startup][gpu:{self.device}] Loading LLM config: {Path(model_args.model_path) / 'llm_config.json'}",
                flush=True,
            )
            llm_config: Qwen2Config = Qwen2Config.from_json_file(str(Path(model_args.model_path) / "llm_config.json"))
            self._log_stage("LLM config load", stage_start)

            llm_config.layer_module = model_args.layer_module
            llm_config.qk_norm = model_args.llm_qk_norm
            llm_config.qk_norm_und = model_args.llm_qk_norm_und
            llm_config.qk_norm_gen = model_args.llm_qk_norm_gen
            llm_config.tie_word_embeddings = model_args.tie_word_embeddings
            llm_config.freeze_und = inference_args.freeze_und
            llm_config.apply_qwen_2_5_vl_pos_emb = inference_args.apply_qwen_2_5_vl_pos_emb

            stage_start = time.perf_counter()
            print(f"[startup][gpu:{self.device}] Initializing LLM weights: {model_args.model_path}", flush=True)
            language_model: Qwen2ForCausalLM = Qwen2ForCausalLM(llm_config)
            self._log_stage("LLM weight init", stage_start)

            vit_model = None
            vit_config = None
            if inference_args.visual_und:
                if model_args.vit_type not in ("qwen2_5_vl", "qwen_2_5_vl_original"):
                    raise ValueError(f"Unsupported vit_type: {model_args.vit_type}")
                stage_start = time.perf_counter()
                print(f"[startup][gpu:{self.device}] Loading VIT config: {model_args.vit_path}", flush=True)
                vit_config = Qwen2_5_VLVisionConfig.from_pretrained(model_args.vit_path)
                self._log_stage("VIT config load", stage_start)

                stage_start = time.perf_counter()
                print(
                    f"[startup][gpu:{self.device}] Loading VIT weights: {Path(model_args.vit_path) / 'vit.safetensors'}",
                    flush=True,
                )
                vit_model = Qwen2_5_VisionTransformerPretrainedModel(vit_config)
                vit_weights = load_file(str(Path(model_args.vit_path) / "vit.safetensors"))
                vit_model.load_state_dict(vit_weights, strict=True)
                self._log_stage("VIT weight load", stage_start)
                clean_memory(vit_weights)

            if inference_args.visual_gen:
                stage_start = time.perf_counter()
                print(f"[startup][gpu:{self.device}] Initializing VAE", flush=True)
                vae_model = WanVideoVAE()
                vae_config = deepcopy(vae_model.vae_config)
                self._log_stage("VAE init", stage_start)
            else:
                vae_model = None
                vae_config = None

            config = LanceConfig(
                visual_gen=inference_args.visual_gen,
                visual_und=inference_args.visual_und,
                llm_config=llm_config,
                vit_config=vit_config if inference_args.visual_und else None,
                vae_config=vae_config if inference_args.visual_gen else None,
                latent_patch_size=model_args.latent_patch_size,
                max_num_frames=model_args.max_num_frames,
                max_latent_size=model_args.max_latent_size,
                vit_max_num_patch_per_side=model_args.vit_max_num_patch_per_side,
                connector_act=model_args.connector_act,
                interpolate_pos=model_args.interpolate_pos,
                timestep_shift=inference_args.timestep_shift,
            )
            model: Lance = Lance(
                language_model=language_model,
                vit_model=vit_model if inference_args.visual_und else None,
                vit_type=model_args.vit_type,
                config=config,
                training_args=inference_args,
            )

            stage_start = time.perf_counter()
            print(f"[startup][gpu:{self.device}] Casting Lance model to bf16 on CPU", flush=True)
            model = model.to(dtype=torch.bfloat16)
            self._log_stage("Lance model bf16 cast", stage_start)

            stage_start = time.perf_counter()
            print(f"[startup][gpu:{self.device}] Loading tokenizer: {model_args.model_path}", flush=True)
            tokenizer: Qwen2Tokenizer = Qwen2Tokenizer.from_pretrained(model_args.model_path)
            tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
            self._log_stage("tokenizer load and special token init", stage_start, extra=f"num_new_tokens={num_new_tokens}")

            if inference_args.copy_init_moe:
                language_model.init_moe()

            init_from_model_path_if_needed(model, model_args)

            if num_new_tokens > 0:
                model.language_model.resize_token_embeddings(len(tokenizer))
                model.config.llm_config.vocab_size = len(tokenizer)
                model.language_model.config.vocab_size = len(tokenizer)

            if model_args.vit_type.lower() == "qwen2_5_vl":
                from common.model.hacks import hack_qwen2_5_vl_config

                language_model = hack_qwen2_5_vl_config(language_model)

            image_token_id = language_model.config.video_token_id
            new_token_ids.update({"image_token_id": image_token_id})
            model.update_tokenizer(tokenizer=tokenizer)

            if model_args.tie_word_embeddings:
                model.language_model.untie_lm_head()
                model.language_model.copy_new_token_rows_to_lm_head(num_new_tokens)
                model_args.tie_word_embeddings = False
                llm_config.tie_word_embeddings = False
            else:
                assert (
                    model.language_model.get_input_embeddings().weight.data.data_ptr()
                    != model.language_model.get_output_embeddings().weight.data.data_ptr()
                ), "tie_word_embeddings conflict"

            stage_start = time.perf_counter()
            print(f"[startup][gpu:{self.device}] Moving Lance model to GPU {self.device}", flush=True)
            model = model.to(device=self.device)
            self._log_stage("Lance model move to GPU", stage_start)
            model.eval()
            if vae_model is not None and hasattr(vae_model, "eval"):
                vae_model.eval()

            self.model = model
            self.vae_model = vae_model
            self.vae_config = vae_config
            self.tokenizer = tokenizer
            self.new_token_ids = new_token_ids
            self.image_token_id = image_token_id
            self.initialized = True
            print(
                f"[startup][gpu:{self.device}][{self.model_variant}] Lance multimodal Gradio model loaded and ready for reuse.",
                flush=True,
            )

    def unload(self) -> None:
        with self._init_lock:
            if self.model is not None:
                self.model.cpu()
            if self.vae_model is not None and hasattr(self.vae_model, "vae"):
                vae_inner = self.vae_model.vae
                if hasattr(vae_inner, "model"):
                    vae_inner.model.cpu()

            self.model = None
            self.vae_model = None
            self.vae_config = None
            self.tokenizer = None
            self.new_token_ids = None
            self.image_token_id = None
            self.base_model_args = None
            self.base_data_args = None
            self.base_inference_args = None
            self.initialized = False
            gc.collect()
            if torch.cuda.is_available():
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

    def _build_request_batch(
        self,
        prompt_file: Path,
        model_args: ModelArguments,
        data_args: DataArguments,
        inference_args: InferenceArguments,
    ):
        assert self.tokenizer is not None
        assert self.new_token_ids is not None
        assert self.vae_config is not None

        dataset_config = DataConfig.from_yaml(str(prompt_file))
        if inference_args.visual_und:
            dataset_config.vit_patch_size = model_args.vit_patch_size
            dataset_config.vit_patch_size_temporal = model_args.vit_patch_size_temporal
            dataset_config.vit_max_num_patch_per_side = model_args.vit_max_num_patch_per_side
        if inference_args.visual_gen:
            vae_downsample = tuple_mul(
                tuple(model_args.latent_patch_size),
                (
                    self.vae_config.downsample_temporal,
                    self.vae_config.downsample_spatial,
                    self.vae_config.downsample_spatial,
                ),
            )
            dataset_config.latent_patch_size = model_args.latent_patch_size
            dataset_config.vae_downsample = vae_downsample
            dataset_config.max_latent_size = model_args.max_latent_size
            dataset_config.max_num_frames = model_args.max_num_frames

        dataset_config.text_cond_dropout_prob = model_args.text_cond_dropout_prob
        dataset_config.vae_cond_dropout_prob = model_args.vae_cond_dropout_prob
        dataset_config.vit_cond_dropout_prob = model_args.vit_cond_dropout_prob

        dataset_config.num_frames = inference_args.num_frames
        dataset_config.H = inference_args.video_height
        dataset_config.W = inference_args.video_width
        dataset_config.task = inference_args.task
        dataset_config.resolution = inference_args.resolution
        dataset_config.text_template = inference_args.text_template

        val_dataset = ValidationDataset(
            jsonl_path=str(prompt_file),
            tokenizer=self.tokenizer,
            data_args=data_args,
            model_args=model_args,
            training_args=inference_args,
            new_token_ids=self.new_token_ids,
            dataset_config=dataset_config,
            local_rank=0,
            world_size=1,
        )
        return simple_custom_collate([val_dataset[0]])

    def generate(
        self,
        task: str,
        prompt: str,
        system_prompt: Optional[str],
        input_video: Optional[str],
        input_image: Optional[str],
        height: int,
        width: int,
        num_frames: int,
        seed: int,
        resolution: str,
        validation_num_timesteps: int,
        validation_timestep_shift: float,
        cfg_text_scale: float,
    ):
        self.initialize()
        internal_task = normalize_task(task)
        prompt = (prompt or "").strip()
        input_video = str(input_video).strip() if input_video else ""
        input_image = str(input_image).strip() if input_image else ""

        if internal_task in GENERATION_TASKS and not prompt:
            return None, None, "", "Please enter a prompt."
        if internal_task in UNDERSTANDING_TASKS and not prompt:
            return None, None, "", "Please enter a question."
        if internal_task in {TASK_VIDEO_EDIT, TASK_X2T_VIDEO} and not input_video:
            return None, None, "", "Please upload an input video."
        if internal_task in {TASK_IMAGE_EDIT, TASK_X2T_IMAGE} and not input_image:
            return None, None, "", "Please upload an input image."
        if height <= 0 or width <= 0:
            return None, None, "", "Height and width must be greater than 0."
        if num_frames <= 0:
            return None, None, "", "The number of frames must be greater than 0."

        assert self.model is not None
        assert self.tokenizer is not None
        assert self.new_token_ids is not None
        assert self.image_token_id is not None
        assert self.base_model_args is not None
        assert self.base_data_args is not None
        assert self.base_inference_args is not None
        active_model_path = self.base_model_args.model_path

        with self._generate_lock:
            torch.cuda.set_device(self.device)
            actual_seed = normalize_seed(int(seed))
            prompt_file = create_request_json(
                task=internal_task,
                prompt=prompt,
                input_video=input_video,
                input_image=input_image,
                system_prompt=system_prompt,
            )
            save_dir = build_save_dir(internal_task)
            save_dir.mkdir(parents=True, exist_ok=True)
            request_started_at = datetime.now().isoformat(timespec="seconds")

            request_model_args = deepcopy(self.base_model_args)
            request_model_args.cfg_text_scale = float(cfg_text_scale)

            request_data_args = deepcopy(self.base_data_args)
            request_data_args.val_dataset_config_file = str(prompt_file)

            request_inference_args = deepcopy(self.base_inference_args)
            request_inference_args.validation_num_timesteps = int(validation_num_timesteps)
            request_inference_args.validation_timestep_shift = float(validation_timestep_shift)
            request_inference_args.validation_data_seed = actual_seed
            request_inference_args.validation_noise_seed = actual_seed
            request_inference_args.video_height = int(height)
            request_inference_args.video_width = int(width)
            request_inference_args.num_frames = int(num_frames)
            display_resolution = str(resolution)
            backend_resolution = normalize_resolution_for_backend(display_resolution, internal_task)
            request_inference_args.resolution = backend_resolution
            request_inference_args.save_path_gen = str(save_dir)
            request_inference_args.task = internal_task
            request_inference_args.text_template = TEXT_TEMPLATE
            request_inference_args.prompt_data_dict = {}

            try:
                print(
                    "[lance_gradio_t2v_v2t] Start generation "
                    f"| task={internal_task} | gpu={self.device} | seed={actual_seed} | "
                    f"size={height}x{width} | frames={num_frames} | resolution={display_resolution}",
                    flush=True,
                )
                val_data_cpu = self._build_request_batch(
                    prompt_file=prompt_file,
                    model_args=request_model_args,
                    data_args=request_data_args,
                    inference_args=request_inference_args,
                )
                # Keep the allocator from fragmenting before the heavy forward pass.
                clean_memory()
                generate_start = time.perf_counter()
                validate_on_fixed_batch(
                    fsdp_model=self.model,
                    vae_model=self.vae_model,
                    tokenizer=self.tokenizer,
                    val_data_cpu=val_data_cpu,
                    training_args=request_inference_args,
                    model_args=request_model_args,
                    inference_args=request_inference_args,
                    new_token_ids=self.new_token_ids,
                    image_token_id=self.image_token_id,
                    device=self.device,
                    save_source_video=False,
                    save_path_gen=request_inference_args.save_path_gen,
                    save_path_gt="",
                )
                elapsed = time.perf_counter() - generate_start
                save_prompt_results(request_inference_args.prompt_data_dict, request_inference_args.save_path_gen, self.logger)
                clean_memory()

                video_path = find_generated_video(save_dir) if internal_task in {TASK_T2V, TASK_VIDEO_EDIT} else None
                image_path = find_generated_image(save_dir) if internal_task in {TASK_T2I, TASK_IMAGE_EDIT} else None
                text_result = extract_text_result(save_dir) if internal_task in UNDERSTANDING_TASKS else ""
                record = {
                    "request_started_at": request_started_at,
                    "request_finished_at": datetime.now().isoformat(timespec="seconds"),
                    "status": "success",
                    "task": internal_task,
                    "model_variant": self.model_variant,
                    "model_path": active_model_path,
                    "gpu": self.device,
                    "prompt": prompt,
                    "system_prompt": normalize_understanding_system_prompt(internal_task, system_prompt)
                    if internal_task in UNDERSTANDING_TASKS
                    else "",
                    "input_video": input_video,
                    "input_image": input_image,
                    "seed": actual_seed,
                    "height": int(height),
                    "width": int(width),
                    "num_frames": int(num_frames),
                    "resolution": display_resolution,
                    "backend_resolution": backend_resolution,
                    "validation_num_timesteps": int(validation_num_timesteps),
                    "validation_timestep_shift": float(validation_timestep_shift),
                    "cfg_text_scale": float(cfg_text_scale),
                    "elapsed_seconds": round(elapsed, 3),
                    "prompt_file": str(prompt_file),
                    "output_dir": str(save_dir),
                    "video_path": str(video_path) if video_path is not None else "",
                    "image_path": str(image_path) if image_path is not None else "",
                    "text_result": text_result,
                }
                if internal_task in {TASK_T2V, TASK_VIDEO_EDIT} and video_path is None:
                    record["status"] = "completed_without_video"
                if internal_task in {TASK_T2I, TASK_IMAGE_EDIT} and image_path is None:
                    record["status"] = "completed_without_image"
                if internal_task in UNDERSTANDING_TASKS and not text_result:
                    record["status"] = "completed_without_text"
                save_generation_record(record, save_dir)

                if internal_task in {TASK_T2V, TASK_VIDEO_EDIT}:
                    if video_path is None:
                        status = (
                            "Inference completed, but no output video was found.\n\n"
                            f"- Task: `{internal_task}`\n"
                            f"- Model: `{self.model_variant}`\n"
                            f"- Model path: `{active_model_path}`\n"
                            f"- GPU: `{self.device}`\n"
                            f"- Actual seed: `{actual_seed}`\n"
                            f"- Output directory: `{save_dir}`"
                        )
                        return None, None, "", status
                    # status = (
                    #     "Inference completed.\n\n"
                    #     f"- Task: `{internal_task}`\n"
                    #     f"- Model: `{self.model_variant}`\n"
                    #     f"- Model path: `{active_model_path}`\n"
                    #     f"- GPU: `{self.device}`\n"
                    #     f"- Actual seed: `{actual_seed}`\n"
                    #     f"- Output directory: `{save_dir}`\n"
                    #     f"- Result file: `{video_path}`"
                    # )
                    status = ""
                    return str(video_path), None, "", status

                if internal_task in {TASK_T2I, TASK_IMAGE_EDIT}:
                    if image_path is None:
                        status = (
                            "Inference completed, but no output image was found.\n\n"
                            f"- Task: `{internal_task}`\n"
                            f"- Model: `{self.model_variant}`\n"
                            f"- Model path: `{active_model_path}`\n"
                            f"- GPU: `{self.device}`\n"
                            f"- Actual seed: `{actual_seed}`\n"
                            f"- Output directory: `{save_dir}`"
                        )
                        return None, None, "", status
                    # status = (
                    #     "Inference completed.\n\n"
                    #     f"- Task: `{internal_task}`\n"
                    #     f"- Model: `{self.model_variant}`\n"
                    #     f"- Model path: `{active_model_path}`\n"
                    #     f"- GPU: `{self.device}`\n"
                    #     f"- Actual seed: `{actual_seed}`\n"
                    #     f"- Output directory: `{save_dir}`\n"
                    #     f"- Result file: `{image_path}`"
                    # )
                    status = ""
                    return None, str(image_path), "", status

                # status = (
                #     "Understanding completed.\n\n"
                #     f"- Task: `{task}`\n"
                #     f"- Model: `{self.model_variant}`\n"
                #     f"- Model path: `{active_model_path}`\n"
                #     f"- GPU: `{self.device}`\n"
                #     f"- Actual seed: `{actual_seed}`\n"
                #     f"- Output directory: `{save_dir}`"
                # )
                status = ""
                return None, None, text_result, status
            except Exception:
                error_trace = traceback.format_exc()
                print(error_trace, flush=True)
                record = {
                    "request_started_at": request_started_at,
                    "request_finished_at": datetime.now().isoformat(timespec="seconds"),
                    "status": "failed",
                    "task": internal_task,
                    "model_variant": self.model_variant,
                    "model_path": active_model_path,
                    "gpu": self.device,
                    "prompt": prompt,
                    "input_video": input_video,
                    "input_image": input_image,
                    "seed": actual_seed,
                    "height": int(height),
                    "width": int(width),
                    "num_frames": int(num_frames),
                    "resolution": display_resolution,
                    "backend_resolution": backend_resolution,
                    "validation_num_timesteps": int(validation_num_timesteps),
                    "validation_timestep_shift": float(validation_timestep_shift),
                    "cfg_text_scale": float(cfg_text_scale),
                    "prompt_file": str(prompt_file),
                    "output_dir": str(save_dir),
                    "video_path": "",
                    "image_path": "",
                    "text_result": "",
                    "error": error_trace,
                }
                save_generation_record(record, save_dir)
                clean_memory()
                status = (
                    "Inference failed.\n\n"
                    f"- Task: `{internal_task}`\n"
                    f"- Model: `{self.model_variant}`\n"
                    f"- Model path: `{active_model_path}`\n"
                    f"- GPU: `{self.device}`\n"
                    f"- Actual seed: `{actual_seed}`\n"
                    f"- Resolution: `{display_resolution}`\n"
                    f"- Output directory: `{save_dir}`"
                )
                return None, None, "", status

class PipelinePool:
    def __init__(self, gpu_ids: list[int], model_variant: str = MODEL_VARIANT_VIDEO) -> None:
        if not gpu_ids:
            raise ValueError("At least one GPU must be configured.")
        self.gpu_ids = gpu_ids
        self.model_variant = normalize_model_variant(model_variant)
        self.pipelines = [
            LanceT2VV2TPipeline(device_id=gpu_id, model_variant=self.model_variant)
            for gpu_id in gpu_ids
        ]
        self._available = deque(self.pipelines)
        self._condition = threading.Condition()

    @property
    def size(self) -> int:
        return len(self.pipelines)

    @property
    def is_initialized(self) -> bool:
        return all(pipeline.initialized for pipeline in self.pipelines)

    def initialize_all(self) -> None:
        if self.is_initialized:
            return
        print(f"[startup][{self.model_variant}] Preparing parallel GPU preload: {self.gpu_ids}", flush=True)
        exceptions: list[Exception] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.size) as executor:
            futures = {
                executor.submit(pipeline.initialize): pipeline.device for pipeline in self.pipelines
            }
            for future in concurrent.futures.as_completed(futures):
                gpu_id = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f"[startup][gpu:{gpu_id}][{self.model_variant}] Preload failed: {exc}", flush=True)
                    exceptions.append(exc)
        if exceptions:
            raise RuntimeError(
                f"{self.model_variant} preload failed on {len(exceptions)} GPU(s). Please check the terminal logs."
            ) from exceptions[0]
        print(
            f"[startup][{self.model_variant}] GPU preload finished. Ready to handle {self.size} concurrent request(s).",
            flush=True,
        )

    def acquire(self) -> LanceT2VV2TPipeline:
        with self._condition:
            while not self._available:
                self._condition.wait()
            return self._available.popleft()

    def release(self, pipeline: LanceT2VV2TPipeline) -> None:
        with self._condition:
            self._available.append(pipeline)
            self._condition.notify()

    def unload_all(self) -> None:
        print(f"[runtime][{self.model_variant}] Unloading model pool from GPU(s): {self.gpu_ids}", flush=True)
        with self._condition:
            while len(self._available) != len(self.pipelines):
                self._condition.wait()

        for pipeline in self.pipelines:
            pipeline.unload()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        print(f"[runtime][{self.model_variant}] Model pool unloaded.", flush=True)

    def generate(
        self,
        task: str,
        prompt: str,
        system_prompt: Optional[str],
        input_video: Optional[str],
        input_image: Optional[str],
        height: int,
        width: int,
        num_frames: int,
        seed: int,
        resolution: str,
        validation_num_timesteps: int,
        validation_timestep_shift: float,
        cfg_text_scale: float,
    ):
        pipeline = self.acquire()
        try:
            return pipeline.generate(
                task=task,
                prompt=prompt,
                system_prompt=system_prompt,
                input_video=input_video,
                input_image=input_image,
                height=height,
                width=width,
                num_frames=num_frames,
                seed=seed,
                resolution=resolution,
                validation_num_timesteps=validation_num_timesteps,
                validation_timestep_shift=validation_timestep_shift,
                cfg_text_scale=cfg_text_scale,
            )
        finally:
            self.release(pipeline)

ACTIVE_PIPELINE_POOL: Optional[PipelinePool] = None
ACTIVE_POOL_LOCK = threading.Lock()

def get_task_model_variant(task: str) -> str:
    internal_task = normalize_task(task)
    return MODEL_VARIANT_IMAGE if internal_task in IMAGE_TASKS else MODEL_VARIANT_VIDEO

def get_pipeline_pool(task: str) -> PipelinePool:
    global ACTIVE_PIPELINE_POOL
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Lance inference requires a GPU. The Gradio UI can start on CPU, but generation is disabled "
            "until GPU hardware is attached."
        )
    model_variant = get_task_model_variant(task)
    gpu_ids = parse_gpu_ids(os.getenv("LANCE_GPUS", DEFAULT_GPUS))
    with ACTIVE_POOL_LOCK:
        if ACTIVE_PIPELINE_POOL is not None and ACTIVE_PIPELINE_POOL.model_variant == model_variant:
            if not ACTIVE_PIPELINE_POOL.is_initialized:
                ACTIVE_PIPELINE_POOL.initialize_all()
            return ACTIVE_PIPELINE_POOL

        if ACTIVE_PIPELINE_POOL is not None:
            previous_variant = ACTIVE_PIPELINE_POOL.model_variant
            print(
                f"[runtime] Switching Lance model from {previous_variant} to {model_variant}.",
                flush=True,
            )
            ACTIVE_PIPELINE_POOL.unload_all()
            ACTIVE_PIPELINE_POOL = None

        ACTIVE_PIPELINE_POOL = PipelinePool(gpu_ids, model_variant=model_variant)
        ACTIVE_PIPELINE_POOL.initialize_all()
        return ACTIVE_PIPELINE_POOL

def run_task(
    task: str,
    prompt: str,
    system_prompt: Optional[str],
    input_video: Optional[str],
    input_image: Optional[str],
    height: int,
    width: int,
    num_frames: int,
    seed: int,
    resolution: str,
    validation_num_timesteps: int,
    validation_timestep_shift: float,
    cfg_text_scale: float,
):
    internal_task = normalize_task(task)
    if internal_task in UNDERSTANDING_TASKS:
        system_prompt = normalize_understanding_system_prompt(internal_task, system_prompt)

    if internal_task in UNDERSTANDING_TASKS and not prompt:
        return None, None, "", "Please enter a question."
    if internal_task in {TASK_VIDEO_EDIT, TASK_X2T_VIDEO} and not input_video:
        return None, None, "", "Please upload an input video."
    if internal_task in {TASK_IMAGE_EDIT, TASK_X2T_IMAGE} and not input_image:
        return None, None, "", "Please upload an input image."
    if height <= 0 or width <= 0:
        return None, None, "", "Height and width must be greater than 0."
    if num_frames <= 0:
        return None, None, "", "The number of frames must be greater than 0."

    num_frames_ui = int(num_frames)
    normalized_resolution = normalize_resolution_for_backend(str(resolution), internal_task)

    if internal_task == TASK_T2V:
        num_frames = video_seconds_to_num_frames(num_frames_ui)

    return run_task_gpu(
        task=task,
        prompt=prompt,
        system_prompt=system_prompt,
        input_video=input_video,
        input_image=input_image,
        height=height,
        width=width,
        num_frames=num_frames,
        seed=seed,
        resolution=normalized_resolution,
        validation_num_timesteps=validation_num_timesteps,
        validation_timestep_shift=validation_timestep_shift,
        cfg_text_scale=cfg_text_scale,
    )

def run_task_gpu(
    task: str,
    prompt: str,
    system_prompt: Optional[str],
    input_video: Optional[str],
    input_image: Optional[str],
    height: int,
    width: int,
    num_frames: int,
    seed: int,
    resolution: str,
    validation_num_timesteps: int,
    validation_timestep_shift: float,
    cfg_text_scale: float,
):
    pipeline_pool = get_pipeline_pool(task)
    return pipeline_pool.generate(
        task=task,
        prompt=prompt,
        system_prompt=system_prompt,
        input_video=input_video,
        input_image=input_image,
        height=height,
        width=width,
        num_frames=num_frames,
        seed=seed,
        resolution=resolution,
        validation_num_timesteps=validation_num_timesteps,
        validation_timestep_shift=validation_timestep_shift,
        cfg_text_scale=cfg_text_scale,
    )

if __name__ == "__main__":
    args = parse_args()
    os.environ["LANCE_GPUS"] = args.gpus
    print(
        "[startup] Local-only mode. UI will launch first; model weights are loaded from local paths during inference.",
        flush=True,
    )
    concurrency_limit = 1
    demo = build_demo(run_task)
    demo.queue(
        max_size=args.queue_size,
        default_concurrency_limit=concurrency_limit,
    ).launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=False,
        allowed_paths=[str(REPO_ROOT.resolve()), str(GRADIO_TMP_ROOT.resolve())],
        ssr_mode=False,
    )
