# Lance训练数据集构建

## 1.1 本地数据集 parquet 构建

训练最终读取的是 parquet。下面给出当前本地训练推荐的 parquet 格式。从[Hugging Face](https://huggingface.co/datasets/bytedance-research/Lance_example_dataset)中下载example dataset并放置在本地`./datasets`下

### 1.1.1 t2i

- 对应 `dataset_type`：`text2image_general`

| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| `caption` | `string` | 文本条件 |
| `image_bytes` | `binary` | 目标图像 |


### 1.1.2 t2v

- 对应 `dataset_type`：`text2video_general`

| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| `caption` | `string` | 文本条件 |
| `video_bytes` | `binary` | 目标视频 |


### 1.1.3 i2i

- 对应 `dataset_type`：`image2image`

| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| `caption` | `string` | 编辑指令 |
| `input_image_bytes` | `binary` | 条件图 |
| `output_image_bytes` | `binary` | 目标图 |


### 1.1.4 v2v

- 对应 `dataset_type`：`video2video`

| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| `caption` | `string` | 编辑指令 |
| `input_video_bytes` | `binary` | 条件视频 |
| `output_video_bytes` | `binary` | 目标视频 |


### 1.1.5 i2t

- 对应 `dataset_type`：`x2t_general`

| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| `image_bytes` | `binary` | 条件图像 |
| `caption_i` | `string` | instruction / system prompt |
| `caption_q` | `string` | question，可为空串 `""` |
| `caption_a` | `string` | target answer |


### 1.1.6 v2t

- 对应 `dataset_type`：`x2t_general`

| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| `video_bytes` | `binary` | 条件视频 |
| `caption_i` | `string` | instruction / system prompt |
| `caption_q` | `string` | question，可为空串 `""` |
| `caption_a` | `string` | target answer |


### 1.1.7 关于 `caption_i / caption_q / caption_a`

这三列在 `x2t` 任务里对应一组 `I / Q / A`：

| 字段名 | 含义 |
| --- | --- |
| `caption_i` | instruction |
| `caption_q` | question |
| `caption_a` | answer |

训练代码会把它们组装成一个三元列表，再进入 prompt 构造逻辑。

## 1.2 dataset 和对应 yaml

### 1.2.1 dataset 注册项

当前数据集注册在 `data/dataset_info.py` 中，核心有 2 个：

- `x2v_interleave_local`
- `x2t_interleave_local`

含义：

- `x2v_interleave_local`: 本地 parquet 的图像或视频生成任务（图像视频以byte格式保存）
- `x2t_interleave_local`: 本地 parquet 的图文或视频到文本任务（图像视频以byte格式保存）

### 1.2.2 公共参数

| 参数名 | 解释 | 选项 | 选项含义 |
| --- | --- | --- | --- |
| `dataset_type` | 决定 parquet 单行数据如何被解析成 `interleave_array` 和 `element_dtype_array` | 常用取值见右侧 | `text2image_general`：文本到图像，字段通常为 `caption` + `image_bytes`；`text2video_general`：文本到视频，字段通常为 `caption` + `video_bytes`；`x2t_general`：视觉理解到文本，字段通常为 `image_bytes` 或 `video_bytes`，以及 `caption_i/q/a`；`image2image`：图像编辑，字段通常为 `caption` + `input_image_bytes` + `output_image_bytes`；`video2video`：视频编辑，字段通常为 `caption` + `input_video_bytes` + `output_video_bytes` |
| `dataset_names.data_dir` | 指定实际读取的 parquet 路径 | 本地路径 | 例如 `datasets/video2text/local_256.parquet` |
| `raw_bytes_input` | 决定视觉媒体是否直接按 parquet 中的 byte 数据读取 | `true` / `false` | `true`：直接读取 parquet 中的 `image_bytes` / `video_bytes`；`false`：输入通常是 URL 或路径，训练时再在线拉取媒体 |
| `is_image` | 指定视觉输入按图像还是视频处理 | `true` / `false` | `true`：按图像处理，常用于 `t2i`、`i2i`、`i2t`；`false`：按视频处理，常用于 `t2v`、`v2v`、`v2t` |
| `target_modality` | 指定目标模态 | `image` / `video` / `text` | `image`：目标是图像；`video`：目标是视频；`text`：目标是文本 |
| `task_type` | 决定任务语义，以及一条样本中 condition / target 如何拆分 | 常用取值见右侧 | `t2v`：文本到视频生成；`ff2v`：首帧生成视频；`flf2v`：首尾帧生成视频；`tv2v`：文本 + 视觉条件的视频编辑/生成；`vt2v`：视觉 + 文本条件的视频编辑/生成；`v2t`：纯视觉到文本理解；`tv2t`：文本 + 视觉条件的文本理解；`vt2t`：视觉 + 文本条件的文本理解；`t2t`：纯文本到文本 |
| `text_template` | 决定是否走 chat/prompt 模板拼接 | `true` / `false` | `true`：使用 QwenVL 风格模板，把 system / user / assistant 结构化拼接后再训练；`false`：不走模板，直接对原始文本 token 化 |
| `vision_cond_type` | 决定编辑任务里条件视觉分支如何组织 | 常用取值见右侧 | `["vit"]`：只使用 VIT 条件分支；`["vae_Nsplit"]`：只使用 VAE 条件分支；`["vit", "vae_Nsplit"]`：同时使用 VIT 和 VAE 条件分支，常用于编辑任务 |
| `debug_parquet_repeat` | 小数据集调试时重复 parquet，避免 epoch 太短 | 正整数 | 重复本地 parquet 文件列表，常用于 smoke test |
| `video_transform_args` | 控制 VAE 或主视觉分支的媒体变换 | 配置字典 | 包括分辨率、bucket、归一化等 |
| `vit_video_transform_args` | 控制 VIT 分支的媒体变换 | 配置字典 | 通常只在需要 VIT 条件时设置 |
| `video_frame_sampler_args` | 控制视频采样策略 | 配置字典 | 包括采样类型、fps、最大时长、是否截断等 |


### 1.2.3 t2i

参考配置：`config/train_local/t2i_local.yaml`

| 项 | 值 |
| --- | --- |
| `dataset_type` | `text2image_general` |
| `is_image` | `true` |
| `target_modality` | `image` |
| 常用 `task_type` | `t2v` |
| 必要附加配置 | `video_transform_args` |
| 一般不需要 | `vit_video_transform_args`、`video_frame_sampler_args` |


### 1.2.4 t2v

参考配置：`config/train_local/t2v_local.yaml`

| 项 | 值 |
| --- | --- |
| `dataset_type` | `text2video_general` |
| `is_image` | `false` |
| `target_modality` | `video` |
| 常用 `task_type` | `t2v`、`ff2v` |
| 必要附加配置 | `video_transform_args`、`video_frame_sampler_args` |
| 一般不需要 | `vit_video_transform_args` |


### 1.2.5 i2i

参考配置：`config/train_local/i2i_local.yaml`

| 项 | 值 |
| --- | --- |
| `dataset_type` | `image2image` |
| `is_image` | `true` |
| `target_modality` | `image` |
| 常用 `task_type` | `tv2v` |
| `sample_task` | `edit` |
| 常用 `vision_cond_type` | `["vit", "vae_Nsplit"]` |
| 必要附加配置 | `video_transform_args`、`vit_video_transform_args` |
| 一般不需要 | `video_frame_sampler_args` |


### 1.2.6 v2v

参考配置：`config/train_local/v2v_local.yaml`

| 项 | 值 |
| --- | --- |
| `dataset_type` | `video2video` |
| `is_image` | `false` |
| `target_modality` | `video` |
| 常用 `task_type` | `tv2v` |
| `sample_task` | `edit` |
| 常用 `vision_cond_type` | `["vit", "vae_Nsplit"]` |
| 必要附加配置 | `video_transform_args`、`vit_video_transform_args`、`video_frame_sampler_args` |
| 一般不需要 | 无 |


### 1.2.7 i2t

参考配置：`config/train_local/i2t_local.yaml`

| 项 | 值 |
| --- | --- |
| `dataset_type` | `x2t_general` |
| `is_image` | `true` |
| 目标类型 | 文本（由 `x2t` 数据集固定） |
| 常用 `task_type` | `tv2t` |
| 必要附加配置 | `video_transform_args` |
| 一般不需要 | `vit_video_transform_args`、`video_frame_sampler_args` |


### 1.2.8 v2t

参考配置：`config/train_local/v2t_local.yaml`

| 项 | 值 |
| --- | --- |
| `dataset_type` | `x2t_general` |
| `is_image` | `false` |
| 目标类型 | 文本（由 `x2t` 数据集固定） |
| 常用 `task_type` | `tv2t` |
| 必要附加配置 | `video_transform_args`、`video_frame_sampler_args` |
| 一般不需要 | `vit_video_transform_args` |
