# Lance Training Script Usage

This document explains the purpose and common usage of the following training scripts:

- `scripts/sft_lance_unified.sh`
- `scripts/sft_lance_generation.sh`
- `scripts/sft_lance_understand.sh`

All three scripts ultimately launch `train/unified_train.py` via `accelerate launch`. The main differences are the default dataset config and the `VISUAL_GEN` switch.

Important: the default scripts are configured for one 8-GPU machine:

```bash
ARNOLD_WORKER_NUM=1
ARNOLD_WORKER_GPU=8
NUM_SHARD=4
```

If you run on a local or single-machine environment with fewer than 8 visible GPUs, do not directly copy the launch commands below. First edit the selected script and set `ARNOLD_WORKER_GPU` to your actual GPU count. Also set `NUM_SHARD` to a value that is no larger than `TOTAL_RANK` and can divide `TOTAL_RANK`; otherwise `accelerate launch --num_processes $TOTAL_RANK` may wait for unavailable processes or fail during distributed initialization.


## 1. Training Script Selection

These scripts expand shell variables into command-line arguments and pass them to `train/unified_train.py`. In practice, you should first decide which class of task you want to train.

| Script | Default config | Default switches | Suitable scenarios | Common fields to modify |
| --- | --- | --- | --- | --- |
| `scripts/sft_lance_unified.sh` | `config/train_local/unified.yaml` | `VISUAL_UND=True`, `VISUAL_GEN=True` | Mixed understanding + generation training | `DATASET_CONFIG_FILE`, `VAL_DATASET_CONFIG_FILE`, `WANDB_NAME` |
| `scripts/sft_lance_generation.sh` | `config/train_local/multi_gen.yaml` | `VISUAL_UND=True`, `VISUAL_GEN=True` | Generation tasks such as `t2i`, `t2v`, `i2i`, `v2v` | `DATASET_CONFIG_FILE`, `VAL_DATASET_CONFIG_FILE`, `WANDB_NAME` |
| `scripts/sft_lance_understand.sh` | `config/train_local/multi_und.yaml` | `VISUAL_UND=True`, `VISUAL_GEN=False` | Understanding tasks such as `i2t`, `v2t` | `DATASET_CONFIG_FILE`, `VAL_DATASET_CONFIG_FILE`, `WANDB_NAME` |


## 2. Key Parameters to Modify First

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


## 3. Two Switches That Are Easy to Misconfigure

### 3.1 `VISUAL_GEN`

`VISUAL_GEN` controls whether the visual generation branch is enabled, including the VAE latent / flow matching / MSE path.

- Common settings for generation tasks:
  - `VISUAL_UND=True`
  - `VISUAL_GEN=True`

- Common settings for understanding tasks:
  - `VISUAL_UND=True`
  - `VISUAL_GEN=False`

If you accidentally set `VISUAL_GEN=True` for an understanding task, but the batch does not contain the latent fields required by the generation branch, `Lance.forward(...)` may enter the wrong branch and fail.

### 3.2 `VALIDATION_STEP`

All three scripts default to:

```bash
VALIDATION_STEP=-1
```

This means:

- no fixed validation dataset is prepared
- `validate_on_fixed_batch(...)` is never triggered in the training loop

The validation logic in the training script has not been fully checked yet. Enabling validation with a positive value is currently not supported, so do not set values such as `VALIDATION_STEP=100`; keep it as `-1`.


## 4. Common Launch Patterns

### 4.1 Unified mixed training

```bash
bash scripts/sft_lance_unified.sh
```

### 4.2 Local `v2t` understanding training

```bash
DATASET_CONFIG_FILE=config/train_local/v2t_local.yaml \
VAL_DATASET_CONFIG_FILE=config/train_local/v2t_local.yaml \
WANDB_NAME=v2t_local_debug \
bash scripts/sft_lance_understand.sh
```

### 4.3 Local `t2v` generation training

```bash
DATASET_CONFIG_FILE=config/train_local/t2v_local.yaml \
VAL_DATASET_CONFIG_FILE=config/train_local/t2v_local.yaml \
WANDB_NAME=t2v_local_debug \
bash scripts/sft_lance_generation.sh
```


## 5. Practical Recommendations

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
