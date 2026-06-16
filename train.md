# Lance Training Script Usage

This document explains the purpose and common usage of the following training scripts:

- `scripts/sft_lance_unified.sh`
- `scripts/sft_lance_generation.sh`
- `scripts/sft_lance_understand.sh`

All three scripts ultimately launch `train/unified_train.py` via `accelerate launch`. The main differences are the default dataset config and the `VISUAL_GEN` switch.


## 1. Environment Setup

The training release adds extra dependencies on top of the inference-only setup. If you installed the environment before the fine-tuning code was released, reinstall the dependencies from the updated `requirements.txt`:

```bash
pip install -r requirements.txt
```


## 2. Data Preparation

Training reads local parquet files. Download example datasets from [Hugging Face](https://huggingface.co/datasets/bytedance-research/Lance_example_dataset) and place them under local `./datasets`.

Expected local layout:

```text
datasets/
├── image2image/
├── image2text/
├── text2image/
├── text2video/
├── video2text/
└── video2video/
```

The ready-to-use local configs live in `config/train_local/`, for example:

| Task | Config |
| --- | --- |
| `t2i` | `config/train_local/t2i_local.yaml` |
| `t2v` | `config/train_local/t2v_local.yaml` |
| `i2i` | `config/train_local/i2i_local.yaml` |
| `v2v` | `config/train_local/v2v_local.yaml` |
| `i2t` | `config/train_local/i2t_local.yaml` |
| `v2t` | `config/train_local/v2t_local.yaml` |

For parquet schemas, supported task types, and custom dataset construction, see [train_dataset.md](train_dataset.md).

## 3. Launch Patterns

After preparing the environment, model weights, and example datasets, launch one of the training scripts below.

By default, the scripts use all GPUs visible to `nvidia-smi` on the current machine. To run on fewer GPUs, set `ARNOLD_WORKER_GPU` before launching, for example `ARNOLD_WORKER_GPU=1 bash scripts/sft_lance_unified.sh`.
For smaller local machines, you can also lower dataloader workers, for example `NUM_WORKERS=2 ARNOLD_WORKER_GPU=1 bash scripts/sft_lance_unified.sh`.

### 3.1 Unified mixed training

```bash
bash scripts/sft_lance_unified.sh
```

### 3.2 Generation-task training

```bash
bash scripts/sft_lance_generation.sh
```

For a local `t2v` run, override the dataset config and experiment name:

```bash
DATASET_CONFIG_FILE=config/train_local/t2v_local.yaml \
VAL_DATASET_CONFIG_FILE=config/train_local/t2v_local.yaml \
WANDB_NAME=t2v_local_debug \
bash scripts/sft_lance_generation.sh
```

### 3.3 Understanding-task training

```bash
bash scripts/sft_lance_understand.sh
```

For a local `v2t` run, override the dataset config and experiment name:

```bash
DATASET_CONFIG_FILE=config/train_local/v2t_local.yaml \
VAL_DATASET_CONFIG_FILE=config/train_local/v2t_local.yaml \
WANDB_NAME=v2t_local_debug \
bash scripts/sft_lance_understand.sh
```

**NOTE**: The scripts default to `MODEL_PATH=./downloads/Lance_3B_Video`, the unified video-capable checkpoint. For image-only fine-tuning, such as `t2i`, `i2i`, or `i2t`, you can switch `MODEL_PATH` to `./downloads/Lance_3B` before launch.


## 4. Training Script Selection

These scripts expand shell variables into command-line arguments and pass them to `train/unified_train.py`. In practice, you should first decide which class of task you want to train.

| Script | Default config | Default switches | Suitable scenarios | Common fields to modify |
| --- | --- | --- | --- | --- |
| `scripts/sft_lance_unified.sh` | `config/train_local/unified.yaml` | `VISUAL_UND=True`, `VISUAL_GEN=True` | Mixed understanding + generation training | `DATASET_CONFIG_FILE`, `VAL_DATASET_CONFIG_FILE`, `WANDB_NAME` |
| `scripts/sft_lance_generation.sh` | `config/train_local/multi_gen.yaml` | `VISUAL_UND=True`, `VISUAL_GEN=True` | Generation tasks such as `t2i`, `t2v`, `i2i`, `v2v` | `DATASET_CONFIG_FILE`, `VAL_DATASET_CONFIG_FILE`, `WANDB_NAME` |
| `scripts/sft_lance_understand.sh` | `config/train_local/multi_und.yaml` | `VISUAL_UND=True`, `VISUAL_GEN=False` | Understanding tasks such as `i2t`, `v2t` | `DATASET_CONFIG_FILE`, `VAL_DATASET_CONFIG_FILE`, `WANDB_NAME` |


## 5. Key Parameters to Modify First

These are the parameters you should verify before most runs. Think of them as the first layer of knobs that usually need to be changed.

| Parameter | Purpose | When to change | Recommendation |
| --- | --- | --- | --- |
| `DATASET_CONFIG_FILE` | Specifies the training dataset yaml | Almost every run | Point it to the dataset config you actually want to train |
| `VAL_DATASET_CONFIG_FILE` | Specifies the validation dataset yaml | Validation is currently not supported | Keep the default value |
| `WANDB_NAME` | Names the experiment | Almost every run | Include task name, dataset name, and date |
| `VISUAL_UND` | Enables the visual understanding branch | Usually not changed often | Keep `True` for understanding tasks and most generation tasks |
| `VISUAL_GEN` | Enables the visual generation branch | Must be checked when switching between understanding and generation | Set `False` for understanding tasks, `True` for generation tasks |
| `SAVE_EVERY` | Checkpoint save interval | Commonly changed in both debug and formal runs | Smaller for debugging, larger for long runs |
| `CKPT_DEBUG_STEPS` | Very early debug checkpoint | Commonly changed during debugging | Set to `-1` if you do not need early debug checkpoints |
| `VALIDATION_STEP` | Validation interval | Validation is currently not supported | Keep `-1`; do not set it to a positive integer |
| `NUM_SHARD` | Number of FSDP shards | When changing the parallelism strategy | Tune together with GPU count and memory budget |
| `NUM_REPLICATE` | Number of replicas | Usually changes with `NUM_SHARD` | Computed as `TOTAL_RANK / NUM_SHARD` |


## 6. Two Switches That Are Easy to Misconfigure

### 6.1 `VISUAL_GEN`

`VISUAL_GEN` controls whether the visual generation branch is enabled, including the VAE latent / flow matching / MSE path.

- Common settings for generation tasks:
  - `VISUAL_UND=True`
  - `VISUAL_GEN=True`

- Common settings for understanding tasks:
  - `VISUAL_UND=True`
  - `VISUAL_GEN=False`

If you accidentally set `VISUAL_GEN=True` for an understanding task, but the batch does not contain the latent fields required by the generation branch, `Lance.forward(...)` may enter the wrong branch and fail.

### 6.2 `VALIDATION_STEP`

All three scripts default to:

```bash
VALIDATION_STEP=-1
```

This means:

- no fixed validation dataset is prepared
- `validate_on_fixed_batch(...)` is never triggered in the training loop

The validation logic in the training script has not been fully checked yet. Enabling validation with a positive value is currently not supported, so do not set values such as `VALIDATION_STEP=100`; keep it as `-1`.


## 7. Practical Recommendations

1. Use `sft_lance_understand.sh` first for pure understanding tasks.
2. Use `sft_lance_generation.sh` first for pure generation tasks.
3. Use `sft_lance_unified.sh` when you really want mixed-task training.
4. During debugging, prioritize changing:
   - `DATASET_CONFIG_FILE`
   - `WANDB_NAME`
   - `VISUAL_GEN`
   - `SAVE_EVERY`
   - `CKPT_DEBUG_STEPS`
   - `VALIDATION_STEP`
5. For local parquet training, verify first:
   - the yaml really points to a local parquet file
   - the `_local` dataset class matches the parquet schema
   - understanding tasks do not accidentally run with `VISUAL_GEN=True`
