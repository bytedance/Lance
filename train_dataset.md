# Lance Training Dataset Construction

## 1.1 Local Dataset Parquet Construction

Training ultimately reads parquet files. Below is the recommended local parquet format for current local training. Download example datasets from [Hugging Face](https://huggingface.co/datasets/bytedance-research/Lance_example_dataset) and place them under local `./datasets`.

### 1.1.1 t2i

- Corresponding `dataset_type`: `text2image_general`

| Field | Type | Description |
| --- | --- | --- |
| `caption` | `string` | Text condition |
| `image_bytes` | `binary` | Target image |


### 1.1.2 t2v

- Corresponding `dataset_type`: `text2video_general`

| Field | Type | Description |
| --- | --- | --- |
| `caption` | `string` | Text condition |
| `video_bytes` | `binary` | Target video |


### 1.1.3 i2i

- Corresponding `dataset_type`: `image2image`

| Field | Type | Description |
| --- | --- | --- |
| `caption` | `string` | Editing instruction |
| `input_image_bytes` | `binary` | Conditional image |
| `output_image_bytes` | `binary` | Target image |


### 1.1.4 v2v

- Corresponding `dataset_type`: `video2video`

| Field | Type | Description |
| --- | --- | --- |
| `caption` | `string` | Editing instruction |
| `input_video_bytes` | `binary` | Conditional video |
| `output_video_bytes` | `binary` | Target video |


### 1.1.5 i2t

- Corresponding `dataset_type`: `x2t_general`

| Field | Type | Description |
| --- | --- | --- |
| `image_bytes` | `binary` | Conditional image |
| `caption_i` | `string` | Instruction / system prompt |
| `caption_q` | `string` | Question, can be empty `""` |
| `caption_a` | `string` | Target answer |


### 1.1.6 v2t

- Corresponding `dataset_type`: `x2t_general`

| Field | Type | Description |
| --- | --- | --- |
| `video_bytes` | `binary` | Conditional video |
| `caption_i` | `string` | Instruction / system prompt |
| `caption_q` | `string` | Question, can be empty `""` |
| `caption_a` | `string` | Target answer |


### 1.1.7 About `caption_i / caption_q / caption_a`

These three fields form an `I / Q / A` tuple in `x2t` tasks:

| Field | Meaning |
| --- | --- |
| `caption_i` | instruction |
| `caption_q` | question |
| `caption_a` | answer |

The training code packs them into a triplet and then feeds them into prompt construction logic.


## 1.2 Datasets and YAML

### 1.2.1 Dataset Registry

The current datasets are registered in `data/dataset_info.py`. The two core local entries are:

- `x2v_interleave_local`
- `x2t_interleave_local`

Meaning:

- `x2v_interleave_local`: local parquet image/video generation tasks, where media is stored as bytes
- `x2t_interleave_local`: local parquet image/video understanding-to-text tasks, where media is stored as bytes


### 1.2.2 Common Parameters

| Parameter | Explanation | Options | Option meanings |
| --- | --- | --- | --- |
| `dataset_type` | Determines how one parquet row is parsed into `interleave_array` and `element_dtype_array` | Common values are explained at right | `text2image_general`: text-to-image, fields are usually `caption` + `image_bytes`; `text2video_general`: text-to-video, fields are usually `caption` + `video_bytes`; `x2t_general`: visual understanding to text, fields are usually `image_bytes` or `video_bytes`, plus `caption_i/q/a`; `image2image`: image editing, fields are usually `caption` + `input_image_bytes` + `output_image_bytes`; `video2video`: video editing, fields are usually `caption` + `input_video_bytes` + `output_video_bytes` |
| `dataset_names.data_dir` | Specifies the parquet path to read | Local path | For example `datasets/video2text/local_256.parquet` |
| `raw_bytes_input` | Controls whether visual media is read directly as bytes from parquet | `true` / `false` | `true`: directly read `image_bytes` / `video_bytes` from parquet; `false`: the input is usually a URL or path and media is fetched online during training |
| `is_image` | Controls whether the visual input is treated as image or video | `true` / `false` | `true`: treat as image, commonly used for `t2i`, `i2i`, `i2t`; `false`: treat as video, commonly used for `t2v`, `v2v`, `v2t` |
| `target_modality` | Specifies the target modality | `image` / `video` / `text` | `image`: target is image; `video`: target is video; `text`: target is text |
| `task_type` | Defines task semantics and how condition / target are split inside one sample | Common values are explained at right | `t2v`: text-to-video generation; `ff2v`: first-frame-to-video; `flf2v`: first-and-last-frame-to-video; `tv2v`: text + visual conditioned video editing/generation; `vt2v`: visual + text conditioned video editing/generation; `v2t`: pure visual-to-text understanding; `tv2t`: text + visual conditioned text understanding; `vt2t`: visual + text conditioned text understanding; `t2t`: pure text-to-text |
| `text_template` | Controls whether chat/prompt template construction is enabled | `true` / `false` | `true`: use a QwenVL-style template and structure the prompt as system / user / assistant; `false`: skip the template and tokenize raw text directly |
| `vision_cond_type` | Controls how conditional visual branches are organized in editing tasks | Common values are explained at right | `["vit"]`: use only the VIT conditional branch; `["vae_Nsplit"]`: use only the VAE conditional branch; `["vit", "vae_Nsplit"]`: use both VIT and VAE conditional branches, commonly used in editing tasks |
| `debug_parquet_repeat` | Repeats parquet files in small debug datasets to avoid very short epochs | Positive integer | Repeats the local parquet file list, commonly used for smoke tests |
| `video_transform_args` | Controls transforms for the VAE or main visual branch | Config dict | Includes resolution, bucket mode, normalization, etc. |
| `vit_video_transform_args` | Controls transforms for the VIT branch | Config dict | Usually only needed when a VIT conditional branch is used |
| `video_frame_sampler_args` | Controls the video frame sampling strategy | Config dict | Includes sampler type, fps, max duration, whether to truncate, etc. |


### 1.2.3 t2i

Reference config: `config/train_local/t2i_local.yaml`

| Item | Value |
| --- | --- |
| `dataset_type` | `text2image_general` |
| `is_image` | `true` |
| `target_modality` | `image` |
| Common `task_type` | `t2v` |
| Required extra config | `video_transform_args` |
| Usually not needed | `vit_video_transform_args`, `video_frame_sampler_args` |


### 1.2.4 t2v

Reference config: `config/train_local/t2v_local.yaml`

| Item | Value |
| --- | --- |
| `dataset_type` | `text2video_general` |
| `is_image` | `false` |
| `target_modality` | `video` |
| Common `task_type` | `t2v`, `ff2v` |
| Required extra config | `video_transform_args`, `video_frame_sampler_args` |
| Usually not needed | `vit_video_transform_args` |


### 1.2.5 i2i

Reference config: `config/train_local/i2i_local.yaml`

| Item | Value |
| --- | --- |
| `dataset_type` | `image2image` |
| `is_image` | `true` |
| `target_modality` | `image` |
| Common `task_type` | `tv2v` |
| `sample_task` | `edit` |
| Common `vision_cond_type` | `["vit", "vae_Nsplit"]` |
| Required extra config | `video_transform_args`, `vit_video_transform_args` |
| Usually not needed | `video_frame_sampler_args` |


### 1.2.6 v2v

Reference config: `config/train_local/v2v_local.yaml`

| Item | Value |
| --- | --- |
| `dataset_type` | `video2video` |
| `is_image` | `false` |
| `target_modality` | `video` |
| Common `task_type` | `tv2v` |
| `sample_task` | `edit` |
| Common `vision_cond_type` | `["vit", "vae_Nsplit"]` |
| Required extra config | `video_transform_args`, `vit_video_transform_args`, `video_frame_sampler_args` |
| Usually not needed | None |


### 1.2.7 i2t

Reference config: `config/train_local/i2t_local.yaml`

| Item | Value |
| --- | --- |
| `dataset_type` | `x2t_general` |
| `is_image` | `true` |
| Target type | Text (fixed by the `x2t` dataset class) |
| Common `task_type` | `tv2t` |
| Required extra config | `video_transform_args` |
| Usually not needed | `vit_video_transform_args`, `video_frame_sampler_args` |


### 1.2.8 v2t

Reference config: `config/train_local/v2t_local.yaml`

| Item | Value |
| --- | --- |
| `dataset_type` | `x2t_general` |
| `is_image` | `false` |
| Target type | Text (fixed by the `x2t` dataset class) |
| Common `task_type` | `tv2t` |
| Required extra config | `video_transform_args`, `video_frame_sampler_args` |
| Usually not needed | `vit_video_transform_args` |
