# lance训练说明

这份文档说明下面三个训练脚本的用途和常见用法：

- `scripts/sft_lance_unified.sh`
- `scripts/sft_lance_generation.sh`
- `scripts/sft_lance_understand.sh`

这三个脚本本质上都是通过 `accelerate launch` 启动 `train/unified_train.py`，差异在默认的数据配置和 `VISUAL_GEN` 开关。

## 1. 训练脚本选择

这三个脚本最终都会把 shell 变量展开成命令行参数，传给 `train/unified_train.py`。实际使用时，先决定你要跑哪一类任务。

| 脚本 | 默认配置 | 默认开关 | 适用场景 | 常见需要修改的项 |
| --- | --- | --- | --- | --- |
| `scripts/sft_lance_unified.sh` | `config/train_local/unified.yaml` | `VISUAL_UND=True`，`VISUAL_GEN=True` | 理解 + 生成混合训练 | `DATASET_CONFIG_FILE`、`VAL_DATASET_CONFIG_FILE`、`WANDB_NAME` |
| `scripts/sft_lance_generation.sh` | `config/train_local/multi_gen.yaml` | `VISUAL_UND=True`，`VISUAL_GEN=True` | 生成任务，如 `t2i`、`t2v`、`i2i`、`v2v` | `DATASET_CONFIG_FILE`、`VAL_DATASET_CONFIG_FILE`、`WANDB_NAME` |
| `scripts/sft_lance_understand.sh` | `config/train_local/multi_und.yaml` | `VISUAL_UND=True`，`VISUAL_GEN=False` | 理解任务，如 `i2t`、`v2t` | `DATASET_CONFIG_FILE`、`VAL_DATASET_CONFIG_FILE`、`WANDB_NAME` |


## 2. 需要优先修改的关键参数

下面这些参数是训练前最应该先确认的。可以把它们理解成“最常动”的一层。

| 参数 | 作用 | 什么时候改 | 建议 |
| --- | --- | --- | --- |
| `DATASET_CONFIG_FILE` | 指定训练集 yaml | 基本每次都会改 | 改成你这次真正要训练的数据配置 |
| `VAL_DATASET_CONFIG_FILE` | 指定验证集 yaml | 当前验证暂不支持 | 保留默认值即可 |
| `WANDB_NAME` | 区分实验名 | 基本每次都会改 | 带上任务名、数据集名、日期 |
| `VISUAL_UND` | 是否启用视觉理解分支 | 一般不常改 | 理解任务和大多数生成任务都保持 `True` |
| `VISUAL_GEN` | 是否启用视觉生成分支 | 理解/生成切换时必须确认 | 理解任务设 `False`，生成任务设 `True` |
| `SAVE_EVERY` | checkpoint 保存间隔 | 调试和正式训练都常改 | 调试时可小；正式训练建议调大 |
| `CKPT_DEBUG_STEPS` | 很早期的调试保存 | 调试时常改 | 不需要时建议设为 `-1` |
| `VALIDATION_STEP` | 验证间隔 | 当前验证暂不支持 | 保持 `-1`，不要设置为正整数 |
| `NUM_SHARD` | FSDP shard 数 | 改并行策略时改 | 和 GPU 数、显存规划一起看 |
| `NUM_REPLICATE` | replicate 数 | 一般随 `NUM_SHARD` 自动变化 | 由 `TOTAL_RANK / NUM_SHARD` 得到 |


## 3. 最容易配错的两个开关

### 3.1 `VISUAL_GEN`

`VISUAL_GEN` 控制是否启用 VAE latent / flow matching / MSE 这套视觉生成分支。

- 生成任务常见设置：
  - `VISUAL_UND=True`
  - `VISUAL_GEN=True`

- 理解任务常见设置：
  - `VISUAL_UND=True`
  - `VISUAL_GEN=False`

如果理解任务把 `VISUAL_GEN` 错误地开成 `True`，而 batch 里又没有视觉生成分支需要的 latent 字段，就容易在 `Lance.forward(...)` 里进入错误分支并报错。

### 3.2 `VALIDATION_STEP`

三个脚本默认都是：

```bash
VALIDATION_STEP=-1
```

这表示：

- 不准备固定验证集
- 训练循环里不触发 `validate_on_fixed_batch(...)`

当前训练脚本的 validation 逻辑还没有完成检查，暂不支持通过正整数启用验证。因此不要设置 `VALIDATION_STEP=100` 这类值，保持 `-1`。


## 4. 最常见的启动方式

### 4.1 统一混合训练

```bash
bash scripts/sft_lance_unified.sh
```

### 4.2 本地 `v2t` 理解训练

```bash
DATASET_CONFIG_FILE=config/train_local/v2t_local.yaml \
VAL_DATASET_CONFIG_FILE=config/train_local/v2t_local.yaml \
WANDB_NAME=v2t_local_debug \
bash scripts/sft_lance_understand.sh
```

### 4.3 本地 `t2v` 生成训练

```bash
DATASET_CONFIG_FILE=config/train_local/t2v_local.yaml \
VAL_DATASET_CONFIG_FILE=config/train_local/t2v_local.yaml \
WANDB_NAME=t2v_local_debug \
bash scripts/sft_lance_generation.sh
```


## 5. 使用建议

1. 纯理解任务优先用 `sft_lance_understand.sh`
2. 纯生成任务优先用 `sft_lance_generation.sh`
3. 混合任务再用 `sft_lance_unified.sh`
4. 调试阶段优先先改：
   - `DATASET_CONFIG_FILE`
   - `WANDB_NAME`
   - `VISUAL_GEN`
   - `SAVE_EVERY`
   - `CKPT_DEBUG_STEPS`
   - `VALIDATION_STEP`
5. 本地 parquet 训练时，优先确认：
   - yaml 是否真的指向本地 parquet
   - `_local` 数据集是否和 parquet schema 对齐
   - 理解任务是否误开了 `VISUAL_GEN=True`
