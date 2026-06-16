# Lance 训练脚本使用说明

本文档说明以下训练脚本的用途和常见用法：

- `scripts/sft_lance_unified.sh`
- `scripts/sft_lance_generation.sh`
- `scripts/sft_lance_understand.sh`

这三个脚本最终都会通过 `accelerate launch` 启动 `train/unified_train.py`。主要区别在于默认数据配置和 `VISUAL_GEN` 开关。


## 1. 环境安装

本次训练代码发布在原推理环境的基础上新增了一些依赖。如果你在微调代码发布前已经安装过环境，请基于更新后的 `requirements.txt` 重新安装依赖：

```bash
pip install -r requirements.txt
```


## 2. 数据准备

训练读取本地 parquet 文件。请从 [Hugging Face](https://huggingface.co/datasets/bytedance-research/Lance_example_dataset) 下载示例数据集，并放到本地 `./datasets` 目录下。

期望的本地目录结构如下：

```text
datasets/
├── image2image/
├── image2text/
├── text2image/
├── text2video/
├── video2text/
└── video2video/
```

可直接使用的本地配置位于 `config/train_local/`，例如：

| 任务 | 配置 |
| --- | --- |
| `t2i` | `config/train_local/t2i_local.yaml` |
| `t2v` | `config/train_local/t2v_local.yaml` |
| `i2i` | `config/train_local/i2i_local.yaml` |
| `v2v` | `config/train_local/v2v_local.yaml` |
| `i2t` | `config/train_local/i2t_local.yaml` |
| `v2t` | `config/train_local/v2t_local.yaml` |

关于 parquet schema、支持的任务类型和自定义数据集构建方式，请参考 [train_dataset_zh.md](train_dataset_zh.md)。

## 3. 训练脚本

准备好环境、模型权重和示例数据后，可以启动下面的训练脚本。

默认情况下，脚本会使用当前机器上 `nvidia-smi` 可见的全部 GPU。如果需要少卡启动，可以在命令前设置 `ARNOLD_WORKER_GPU`，例如 `ARNOLD_WORKER_GPU=1 bash scripts/sft_lance_unified.sh`。
对于资源较少的本地机器，也可以降低 dataloader worker 数，例如 `NUM_WORKERS=2 ARNOLD_WORKER_GPU=1 bash scripts/sft_lance_unified.sh`。

### 3.1 统一混合训练

```bash
bash scripts/sft_lance_unified.sh
```

### 3.2 生成任务训练

```bash
bash scripts/sft_lance_generation.sh
```

本地 `t2v` 训练可以通过环境变量覆盖数据配置和实验名：

```bash
DATASET_CONFIG_FILE=config/train_local/t2v_local.yaml \
VAL_DATASET_CONFIG_FILE=config/train_local/t2v_local.yaml \
WANDB_NAME=t2v_local_debug \
bash scripts/sft_lance_generation.sh
```

### 3.3 理解任务训练

```bash
bash scripts/sft_lance_understand.sh
```

本地 `v2t` 训练可以通过环境变量覆盖数据配置和实验名：

```bash
DATASET_CONFIG_FILE=config/train_local/v2t_local.yaml \
VAL_DATASET_CONFIG_FILE=config/train_local/v2t_local.yaml \
WANDB_NAME=v2t_local_debug \
bash scripts/sft_lance_understand.sh
```

**注意**：训练脚本默认使用 `MODEL_PATH=./downloads/Lance_3B_Video`，即支持视频任务的统一 checkpoint。若只微调图像类任务，例如 `t2i`、`i2i` 或 `i2t`，可以在启动前将 `MODEL_PATH` 切换为 `./downloads/Lance_3B`。


## 4. 训练脚本选择

这些脚本会将 shell 变量展开为命令行参数，并传给 `train/unified_train.py`。实际使用时，应先确定这次要训练的任务类型。

| 脚本 | 默认配置 | 默认开关 | 适用场景 | 常见需要修改的字段 |
| --- | --- | --- | --- | --- |
| `scripts/sft_lance_unified.sh` | `config/train_local/unified.yaml` | `VISUAL_UND=True`, `VISUAL_GEN=True` | 理解 + 生成混合训练 | `DATASET_CONFIG_FILE`, `VAL_DATASET_CONFIG_FILE`, `WANDB_NAME` |
| `scripts/sft_lance_generation.sh` | `config/train_local/multi_gen.yaml` | `VISUAL_UND=True`, `VISUAL_GEN=True` | `t2i`、`t2v`、`i2i`、`v2v` 等生成任务 | `DATASET_CONFIG_FILE`, `VAL_DATASET_CONFIG_FILE`, `WANDB_NAME` |
| `scripts/sft_lance_understand.sh` | `config/train_local/multi_und.yaml` | `VISUAL_UND=True`, `VISUAL_GEN=False` | `i2t`、`v2t` 等理解任务 | `DATASET_CONFIG_FILE`, `VAL_DATASET_CONFIG_FILE`, `WANDB_NAME` |


## 5. 优先确认的关键参数

下面这些参数是大多数训练前都需要优先确认的内容，可以理解为最常需要调整的一层配置。

| 参数 | 作用 | 什么时候修改 | 建议 |
| --- | --- | --- | --- |
| `DATASET_CONFIG_FILE` | 指定训练集 yaml | 几乎每次训练都需要确认 | 指向本次真正要训练的数据配置 |
| `VAL_DATASET_CONFIG_FILE` | 指定验证集 yaml | 当前验证暂不支持 | 保持默认值 |
| `WANDB_NAME` | 设置实验名 | 几乎每次训练都需要修改 | 包含任务名、数据集名和日期 |
| `VISUAL_UND` | 启用视觉理解分支 | 通常不常修改 | 理解任务和大多数生成任务都保持 `True` |
| `VISUAL_GEN` | 启用视觉生成分支 | 在理解/生成任务切换时必须确认 | 理解任务设为 `False`，生成任务设为 `True` |
| `SAVE_EVERY` | checkpoint 保存间隔 | 调试和正式训练都常修改 | 调试时可设小，长训时建议设大 |
| `CKPT_DEBUG_STEPS` | 早期 debug checkpoint | 调试时常修改 | 不需要早期 debug checkpoint 时设为 `-1` |
| `VALIDATION_STEP` | 验证间隔 | 当前验证暂不支持 | 保持 `-1`，不要设为正整数 |
| `NUM_SHARD` | FSDP shard 数 | 调整并行策略时修改 | 结合 GPU 数量和显存预算一起调整 |
| `NUM_REPLICATE` | replica 数 | 通常随 `NUM_SHARD` 变化 | 由 `TOTAL_RANK / NUM_SHARD` 计算得到 |


## 6. 容易配错的两个开关

### 6.1 `VISUAL_GEN`

`VISUAL_GEN` 控制是否启用视觉生成分支，包括 VAE latent、flow matching 和 MSE 路径。

- 生成任务常见设置：
  - `VISUAL_UND=True`
  - `VISUAL_GEN=True`

- 理解任务常见设置：
  - `VISUAL_UND=True`
  - `VISUAL_GEN=False`

如果理解任务误设为 `VISUAL_GEN=True`，但 batch 中没有生成分支需要的 latent 字段，`Lance.forward(...)` 可能会进入错误分支并失败。

### 6.2 `VALIDATION_STEP`

三个脚本默认都是：

```bash
VALIDATION_STEP=-1
```

这表示：

- 不准备固定验证集
- 训练循环中不会触发 `validate_on_fixed_batch(...)`

训练脚本中的 validation 逻辑目前尚未完整检查。当前不支持设置正数来启用验证，因此不要设置 `VALIDATION_STEP=100` 这类值，保持 `-1` 即可。


## 7. 使用建议

1. 纯理解任务优先使用 `sft_lance_understand.sh`。
2. 纯生成任务优先使用 `sft_lance_generation.sh`。
3. 确实需要混合任务训练时再使用 `sft_lance_unified.sh`。
4. 调试阶段优先修改：
   - `DATASET_CONFIG_FILE`
   - `WANDB_NAME`
   - `VISUAL_GEN`
   - `SAVE_EVERY`
   - `CKPT_DEBUG_STEPS`
   - `VALIDATION_STEP`
5. 本地 parquet 训练时，优先确认：
   - yaml 确实指向本地 parquet 文件
   - `_local` 数据集类和 parquet schema 对齐
   - 理解任务没有误设为 `VISUAL_GEN=True`
