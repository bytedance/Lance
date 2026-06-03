<div align="center">
  <img src="assets/logo/lance-logo.webp" alt="Lance logo" width="300">

  <h1 align="center"><sup>Lance: Unified Multimodal Modeling by Multi-Task Synergy</sup></h1>
  <p>
    <strong>
    <a href="https://scholar.google.com.hk/citations?user=FXxoQlsAAAAJ&hl=zh-CN&oi=ao" style="text-decoration: none; color: inherit;">Fengyi Fu</a><sup>*</sup>,
    <a href="https://corleone-huang.github.io/" style="text-decoration: none; color: inherit;">Mengqi Huang</a><sup>*,✉</sup>,
    <a href="https://scholar.google.com.hk/citations?user=9ER6nVkAAAAJ&hl=zh-CN&oi=ao" style="text-decoration: none; color: inherit;">Shaojin Wu</a><sup>*</sup>,
    Yunsheng Jiang<sup>*</sup>,
    Yufei Huo,
    <a href="https://guojianzhu.com/" style="text-decoration: none; color: inherit;">Jianzhu Guo</a><sup>✉,§</sup>
    </strong><br>
    Hao Li,
    Yinghang Song,
    Fei Ding,
    Qian He,
    Zheren Fu,
    Zhendong Mao,
    Yongdong Zhang
    <br>
    <em>ByteDance</em>
    <br>
    <sup>*</sup> 共同一作 &nbsp;&nbsp; <sup>✉</sup> 通讯作者 &nbsp;&nbsp; <sup>§</sup> Project lead
  </p>
  <p>
    <a href="https://lance-project.github.io/" style="text-decoration: none; margin: 0 8px;"><img src="https://img.shields.io/badge/Homepage-Lance-blue?style=flat" alt="Homepage"></a>
    <a href="http://arxiv.org/abs/2605.18678" style="text-decoration: none; margin: 0 8px;"><img src="https://img.shields.io/badge/Paper-arXiv-red?style=flat&logo=arxiv" alt="arXiv"></a>
    <a href="https://huggingface.co/bytedance-research/Lance" style="text-decoration: none; margin: 0 8px;"><img src="https://img.shields.io/badge/Model-HuggingFace-yellow?style=flat&logo=huggingface" alt="Model"></a>
    <a href="https://huggingface.co/spaces/bytedance-research/Lance" style="text-decoration: none; margin: 0 8px;"><img src="https://img.shields.io/badge/Demo-HuggingFace-40bfe6?style=flat&logo=huggingface" alt="Demo"></a>
    <br>
    <a href="./README.md"><ins>English</ins></a> | 简体中文
  </p>
</div>

> **注意：** Lance 是一个研究项目，而不是经过充分产品化打磨的模型。当前开源 checkpoint 使用不超过 128 张 A100 GPU 训练，训练阶段覆盖到 768x768 图像生成和 480p、12 FPS 视频生成。我们希望将 Lance 作为一个研究参考，分享在较小模型规模和相对有限算力下统一图像/视频理解、生成和编辑的建模思路、训练流程和推理代码。模型效果可能会随 prompt、分辨率、时长、运动复杂度和编辑场景而波动，post-training recipe 仍有进一步改进空间。我们欢迎社区提供建设性反馈，帮助项目持续改进。

## 🔥 更新

- **`2026/06/03`**: 🚀 Lance 现已被 [vLLM-Omni](https://github.com/vllm-project/vllm-omni) 支持。查看 [recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/ByteDance/Lance.md)！
- **`2026/05/29`**: 🔥 增加 Image-to-Video generation 支持。[查看更多](assets/docs/changelog/2026-05-29.md)！
- **`2026/05/26`**: 🎨 Gradio 界面现已支持图像和视频生成、编辑与理解任务。[欢迎体验](assets/docs/changelog/2026-05-26.md)！
- **`2026/05/25`**: ✨ [Hugging Face Space](https://huggingface.co/spaces/bytedance-research/Lance) 已上线，感谢 HF 团队的支持！
- **`2026/05/19`**: 🤗 技术报告现已发布于 [arXiv](http://arxiv.org/abs/2605.18678)。
- **`2026/05/18`**: 🔥 我们发布了 [项目主页](https://lance-project.github.io/)，并在 [GitHub](https://github.com/bytedance/Lance/) 和 [Hugging Face](https://huggingface.co/bytedance-research/Lance) 上开源了初版推理代码和模型权重。

## 🌟 亮点

**Lance** 是一个3B参数、原生统一的多模态模型，在单一框架下同时支持 **图像与视频的理解、生成和编辑**。

- **3B 规模高效。** 仅使用 **3B active parameters**，Lance 即可在图像生成、图像编辑和视频生成等基准上取得有竞争力的表现。
- **从零训练。** Lance 采用分阶段多任务训练配方从零训练，并在 **不超过 128 张 A100 GPU** 的预算内完成训练。

我们正在持续更新和改进本仓库。如果你发现任何问题或有改进建议，欢迎提出 issue 或提交 pull request（PR）💖。

<div align="center">
  <img src="assets/benchmarks/benchmark-overview.png" alt="Lance benchmark overview across image generation, image editing, video generation, and video understanding" width="980">
</div>

## 📅 路线图

- [ ] 发布 fine-tuning 代码。

## 🎨 演示

<details>
<summary><strong>展开查看演示结果</strong></summary>

<div align="center">
  <strong>🔥 建议浏览我们的 <a href="https://lance-project.github.io/">主页</a> 查看更多效果。🔥</strong>
</div>

<h3 align="center">文生视频</h3>

<table align="center">
  <tr>
    <td><a href="assets/text-to-video/videos/text-to-video-demo-01.mp4"><img src="assets/text-to-video/previews/text-to-video-demo-01.gif" width="100%"></a></td>
    <td><a href="assets/text-to-video/videos/text-to-video-demo-02.mp4"><img src="assets/text-to-video/previews/text-to-video-demo-02.gif" width="100%"></a></td>
    <td><a href="assets/text-to-video/videos/text-to-video-demo-03.mp4"><img src="assets/text-to-video/previews/text-to-video-demo-03.gif" width="100%"></a></td>
    <td><a href="assets/text-to-video/videos/text-to-video-demo-04.mp4"><img src="assets/text-to-video/previews/text-to-video-demo-04.gif" width="100%"></a></td>
  </tr>
  <tr>
    <td><a href="assets/text-to-video/videos/text-to-video-demo-05.mp4"><img src="assets/text-to-video/previews/text-to-video-demo-05.gif" width="100%"></a></td>
    <td><a href="assets/text-to-video/videos/text-to-video-demo-06.mp4"><img src="assets/text-to-video/previews/text-to-video-demo-06.gif" width="100%"></a></td>
    <td><a href="assets/text-to-video/videos/text-to-video-demo-07.mp4"><img src="assets/text-to-video/previews/text-to-video-demo-07.gif" width="100%"></a></td>
    <td><a href="assets/text-to-video/videos/text-to-video-demo-08.mp4"><img src="assets/text-to-video/previews/text-to-video-demo-08.gif" width="100%"></a></td>
  </tr>
</table>

<h3 align="center">视频编辑</h3>

<table align="center">
  <tr>
    <td><a href="assets/video-editing/videos/video-editing-demo-01.mp4"><img src="assets/video-editing/previews/video-editing-demo-01.gif" width="100%"></a></td>
    <td><a href="assets/video-editing/videos/video-editing-demo-02.mp4"><img src="assets/video-editing/previews/video-editing-demo-02.gif" width="100%"></a></td>
    <td><a href="assets/video-editing/videos/video-editing-demo-03.mp4"><img src="assets/video-editing/previews/video-editing-demo-03.gif" width="100%"></a></td>
    <td><a href="assets/video-editing/videos/video-editing-demo-04.mp4"><img src="assets/video-editing/previews/video-editing-demo-04.gif" width="100%"></a></td>
  </tr>
  <tr>
    <td><a href="assets/video-editing/videos/video-editing-demo-05.mp4"><img src="assets/video-editing/previews/video-editing-demo-05.gif" width="100%"></a></td>
    <td><a href="assets/video-editing/videos/video-editing-demo-06.mp4"><img src="assets/video-editing/previews/video-editing-demo-06.gif" width="100%"></a></td>
    <td><a href="assets/video-editing/videos/video-editing-demo-07.mp4"><img src="assets/video-editing/previews/video-editing-demo-07.gif" width="100%"></a></td>
    <td><a href="assets/video-editing/videos/video-editing-demo-08.mp4"><img src="assets/video-editing/previews/video-editing-demo-08.gif" width="100%"></a></td>
  </tr>
</table>

<h3 align="center">多轮一致性编辑</h3>

<div align="center">
  <a href="assets/multi-turn-editing/videos/multi-turn-editing-demo-01.mp4">
    <img src="assets/multi-turn-editing/previews/multi-turn-editing-demo-01.gif" width="100%">
  </a>
</div>

<h3 align="center">智能视频生成</h3>

<table align="center">
  <tr>
    <td><a href="assets/intelligent-video/videos/intelligent-video-demo-01.mp4"><img src="assets/intelligent-video/previews/intelligent-video-demo-01.gif" width="100%"></a></td>
    <td><a href="assets/intelligent-video/videos/intelligent-video-demo-02.mp4"><img src="assets/intelligent-video/previews/intelligent-video-demo-02.gif" width="100%"></a></td>
    <td><a href="assets/intelligent-video/videos/intelligent-video-demo-03.mp4"><img src="assets/intelligent-video/previews/intelligent-video-demo-03.gif" width="100%"></a></td>
    <td><a href="assets/intelligent-video/videos/intelligent-video-demo-04.mp4"><img src="assets/intelligent-video/previews/intelligent-video-demo-04.gif" width="100%"></a></td>
  </tr>
</table>

</details>

## 🚀 安装

### 推荐环境

- **软件环境：** Python 3.10+，CUDA 12.4+（必需）
- **硬件环境：** 推理至少需要一张显存不低于 40GB 的 GPU

我们在 NVIDIA A100 上测试通过了以下依赖组合：

- PyTorch 2.8.0 + cu126 + flash-attn 2.8.3
- PyTorch 2.5.1 + cu124 + flash-attn 2.6.3

默认安装命令使用 PyTorch 2.8.0 + cu126 环境。对于其他 GPU 型号，请根据驱动版本、CUDA runtime、Python 版本和 GPU 架构自行选择并验证匹配的 PyTorch 与 `flash-attn` 版本组合。


### 安装步骤

首先，克隆代码仓库：

```bash
git clone https://github.com/bytedance/Lance.git
cd Lance
```

然后，配置环境：

```bash
conda create -n Lance python=3.11 -y
conda activate Lance
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install flash-attn==2.8.3 --no-build-isolation
```

> **注意：** 如果从源码安装 `flash-attn` 失败，可以改为安装预编译 wheel。下面的 wheelhouse 来自第三方仓库，仅作为**参考提供**；请在安装前确认 wheel 与当前 Python、PyTorch 和 CUDA 版本匹配：
>
> ```bash
> pip install --no-cache-dir --no-deps --force-reinstall \
> "https://huggingface.co/strangertoolshf/flash_attention_2_wheelhouse/resolve/main/wheelhouse-flash_attn-2.8.3/linux_x86_64/torch2.8/cu12/abiTRUE/cp311/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
> ```

然后，从 [Hugging Face 上的 Lance-3B](https://huggingface.co/bytedance-research/Lance) 下载所需的全部模型权重，并放置到 `downloads/` 目录下：

```bash
from huggingface_hub import snapshot_download

save_dir = "./downloads/"
repo_id = "bytedance-research/Lance"
cache_dir = save_dir + "/cache"

snapshot_download(cache_dir=cache_dir,
  local_dir=save_dir,
  repo_id=repo_id,
  local_dir_use_symlinks=False,
  resume_download=True,
  allow_patterns=["*.json", "*.safetensors", "*.bin", "*.py", "*.md", "*.txt","*.pth",],
)
```

## 📚 使用方法

### 推理

#### 基本用法

```bash
bash inference_lance.sh
```

- 运行前，请先在 `inference_lance.sh` 顶部配置推理参数。
- **支持任务：** `t2i`、`t2v`、`i2v`、`image_edit`、`video_edit`、`x2t_image` 和 `x2t_video`。你也可以在 `inference_lance.py` 中修改 `TASK_DEFAULT_CONFIGS`，自定义每个任务默认使用的数据样例。
- **注意：** 对于所有任务，建议在编写输入 prompt 时参考提供示例中的 `prompt` 格式，这通常有助于获得更好的生成效果。


#### 任务示例

##### 文生视频

```bash
bash inference_lance.sh \
  --TASK_NAME t2v \
  --MODEL_PATH downloads/Lance_3B_Video \
  --RESOLUTION video_480p \
  --NUM_FRAMES 121 \
  --VIDEO_HEIGHT 480 \
  --VIDEO_WIDTH 848 \
  --SAVE_PATH_GEN results/t2v
```

##### 图生视频

```bash
bash inference_lance.sh \
  --TASK_NAME i2v \
  --MODEL_PATH downloads/Lance_3B_Video \
  --RESOLUTION video_480p \
  --NUM_FRAMES 61 \
  --VIDEO_HEIGHT 480 \
  --VIDEO_WIDTH 848 \
  --SAVE_PATH_GEN results/i2v
```

生成任务可选参数：

- `--ENHANCE_PROMPT true`：启用 T2V/I2V prompt rewrite。prompt enhance rewrite 通常能提升生成效果。该选项需要 `openai==2.26.0`，已写入 `requirements.txt`；如果没有通过 `requirements.txt` 安装依赖，请先执行 `pip install openai==2.26.0`。启用前请先在 `common/utils/caption_rewrite.py` 中配置 `API_KEY`、`MODEL_NAME` 和 `BASE_URL`；如果没有配置有效 rewrite 参数，会自动跳过 prompt rewrite，此时建议尽量参考提供示例中的 prompt 风格手写输入。

##### 文生图

```bash
bash inference_lance.sh \
  --TASK_NAME t2i \
  --MODEL_PATH downloads/Lance_3B \
  --RESOLUTION image_768res \
  --VIDEO_HEIGHT 768 \
  --VIDEO_WIDTH 768 \
  --SAVE_PATH_GEN results/t2i
```

##### 视频编辑

```bash
bash inference_lance.sh \
  --TASK_NAME video_edit \
  --MODEL_PATH downloads/Lance_3B_Video \
  --RESOLUTION video_480p \
  --SAVE_PATH_GEN results/video_edit
```

##### 图像编辑

```bash
bash inference_lance.sh \
  --TASK_NAME image_edit \
  --MODEL_PATH downloads/Lance_3B \
  --RESOLUTION image_768res \
  --SAVE_PATH_GEN results/image_edit
```

##### 视频理解

```bash
bash inference_lance.sh \
  --TASK_NAME x2t_video \
  --MODEL_PATH downloads/Lance_3B_Video \
  --RESOLUTION video_480p \
  --NUM_FRAMES 50 \
  --SAVE_PATH_GEN results/x2t_video
```

##### 图像理解

```bash
bash inference_lance.sh \
  --TASK_NAME x2t_image \
  --MODEL_PATH downloads/Lance_3B \
  --RESOLUTION image_768res \
  --SAVE_PATH_GEN results/x2t_image
```

所有任务示例可选参数：

- `--CONFIG_PATH path/to/config.json`：使用自定义验证 JSON/JSONL 文件，而不是任务默认示例配置。

<details>
<summary><strong>展开任务和参数参考</strong></summary>

#### 可用任务

| 任务名 | 说明 | 示例 JSON |
|------------------------|--------------------------------------------------|----------------------------------------------|
| `t2v` | 文生视频 | `config/examples/t2v_example.json` |
| `t2i` | 文生图 | `config/examples/t2i_example.json` |
| `i2v` | 图生视频 | `config/examples/i2v_example.json` |
| `image_edit` | 图像编辑 | `config/examples/image_edit_example.json` |
| `video_edit` | 视频编辑 | `config/examples/video_edit_example.json` |
| `x2t_image` | 图像理解 | `config/examples/x2t_image_example.json` |
| `x2t_video` | 视频理解 | `config/examples/x2t_video_example.json` |

关于理解任务的示例文件：

- `config/examples/x2t_image_example.json`：用于图像理解示例，包括视觉问答、基于图像的推理和图像描述。
- `config/examples/x2t_video_example.json`：用于视频理解示例，包括视频问答和视频描述。

#### 参数说明

你可以在 `inference_lance.sh` 顶部配置以下超参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `MODEL_PATH` | `"downloads/Lance_3B"` | 下载后的 Lance 模型权重路径（如 `Lance_3B` 或 `Lance_3B_Video`）。 |
| `NUM_GPUS` | `1` | 用于推理的 GPU 数量。 |
| `VALIDATION_NUM_TIMESTEPS` | `30` | 去噪步数（例如 30 或 50）。 |
| `VALIDATION_TIMESTEP_SHIFT` | `3.5` | Flow matching 调度中的 timestep shift 参数。 |
| `CFG_TEXT_SCALE` | `4.0` | 文本条件的 CFG（Classifier-Free Guidance）系数。 |
| `VALIDATION_DATA_SEED` | `42` | 用于复现实验的随机种子。 |
| `NUM_FRAMES` | `50` | 视频生成帧数（最大 121）。*图像任务不使用该参数。* |
| `VIDEO_HEIGHT` / `VIDEO_WIDTH`| `768` | 空间分辨率。*编辑任务不使用该参数（由输入图像/视频决定）。* |
| `RESOLUTION` | `"video_480p"` | 基础分辨率预设（如 `image_768res` 或 `video_480p`）。 |
| `CONFIG_PATH` | `""` | 可选的自定义验证 JSON/JSONL 文件路径。为空时使用任务默认示例配置。 |
| `ENHANCE_PROMPT` | `false` | 可选的 T2V/I2V prompt rewrite 开关。T2V 使用纯文本 rewrite，I2V 使用文本加输入图像 rewrite。prompt enhance rewrite 通常能提升生成效果。该选项需要 `openai==2.26.0`，已写入 `requirements.txt`；也可以手动执行 `pip install openai==2.26.0`。启用前请先在 `common/utils/caption_rewrite.py` 中配置 `API_KEY`、`MODEL_NAME` 和 `BASE_URL`；如果没有有效 rewrite 参数，建议尽量参考提供示例中的 prompt 风格手写输入。 |

</details>

### 🖥️ Gradio

你可以启动本地 Gradio demo，体验视频/图像生成、编辑和理解：

```bash
python lance_gradio.py --server-name 0.0.0.0 --server-port 7860
```

### 基准评测

<details>
<summary><strong>DPG-Bench 评测</strong></summary>

<div align="center">
<table align="center">
  <thead>
    <tr>
      <th align="left">模型</th>
      <th align="center">#&nbsp;Params.</th>
      <th align="center">Global</th>
      <th align="center">Entity</th>
      <th align="center">Attribute</th>
      <th align="center">Relation</th>
      <th align="center">Other</th>
      <th align="center">Overall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center" colspan="8"><i>仅生成模型</i></td>
    </tr>
    <tr>
      <td align="left">SDXL</td><td align="center">3.5B</td><td align="center">83.27</td><td align="center">82.43</td><td align="center">80.91</td><td align="center">86.76</td><td align="center">80.41</td><td align="center">74.65</td>
    </tr>
    <tr>
      <td align="left">DALL-E 3</td><td align="center">-</td><td align="center">90.97</td><td align="center">89.61</td><td align="center">88.39</td><td align="center">90.58</td><td align="center">89.83</td><td align="center">83.50</td>
    </tr>
    <tr>
      <td align="left">SD3-Medium</td><td align="center">2B</td><td align="center">87.90</td><td align="center">91.01</td><td align="center">88.83</td><td align="center">80.70</td><td align="center">88.68</td><td align="center">84.08</td>
    </tr>
    <tr>
      <td align="left">FLUX.1-dev</td><td align="center">12B</td><td align="center">74.35</td><td align="center">90.00</td><td align="center">88.96</td><td align="center">90.87</td><td align="center">88.33</td><td align="center">83.84</td>
    </tr>
    <tr>
      <td align="left">Qwen-Image</td><td align="center">20B</td><td align="center">91.32</td><td align="center">91.56</td><td align="center">92.02</td><td align="center">94.31</td><td align="center">92.73</td><td align="center">88.32</td>
    </tr>
    <tr>
      <td align="center" colspan="8"><i>统一模型</i></td>
    </tr>
    <tr>
      <td align="left">Janus-Pro-7B</td><td align="center">7B</td><td align="center">86.90</td><td align="center">88.90</td><td align="center">89.40</td><td align="center">89.32</td><td align="center">89.48</td><td align="center">84.19</td>
    </tr>
    <tr>
      <td align="left">OmniGen2</td><td align="center">4B</td><td align="center">88.81</td><td align="center">88.83</td><td align="center">90.18</td><td align="center">89.37</td><td align="center">90.27</td><td align="center">83.57</td>
    </tr>
    <tr>
      <td align="left">Show-o2</td><td align="center">7B</td><td align="center">89.00</td><td align="center"><b>91.78</b></td><td align="center">89.96</td><td align="center">91.81</td><td align="center"><b>91.64</b></td><td align="center">86.14</td>
    </tr>
    <tr>
      <td align="left">BAGEL<sup>†</sup></td><td align="center">7B</td><td align="center">88.94</td><td align="center">90.37</td><td align="center"><u>91.29</u></td><td align="center">90.82</td><td align="center">88.67</td><td align="center">85.07</td>
    </tr>
    <tr>
      <td align="left">InternVL-U</td><td align="center">1.7B</td><td align="center"><u>90.39</u></td><td align="center">90.78</td><td align="center">90.68</td><td align="center">90.29</td><td align="center">88.77</td><td align="center">85.18</td>
    </tr>
    <tr>
      <td align="left">TUNA</td><td align="center">7B</td><td align="center"><b>90.42</b></td><td align="center"><u>91.68</u></td><td align="center">90.94</td><td align="center"><u>91.87</u></td><td align="center"><u>90.73</u></td><td align="center"><b>86.76</b></td>
    </tr>
    <tr>
      <td align="left">TUNA-2</td><td align="center">7B</td><td align="center">89.50</td><td align="center">91.40</td><td align="center"><b>92.07</b></td><td align="center">91.91</td><td align="center">88.81</td><td align="center"><u>86.54</u></td>
    </tr>
    <tr>
      <td align="left">🌟 <b>Lance (Ours)</b></td><td align="center"><b>3B</b></td><td align="center"><b>83.89</b></td><td align="center"><b>91.07</b></td><td align="center"><b>89.36</b></td><td align="center"><b>93.38</b></td><td align="center"><b>80.80</b></td><td align="center"><b>84.67</b></td>
    </tr>
  </tbody>
</table>
</div>

<p align="center"><em><sup>†</sup> 表示该方法在生成前使用 LLM rewriter 进行提示词改写。</em></p>

</details>

<details>
<summary><strong>GenEval 评测</strong></summary>

<div align="center">
<table align="center">
  <thead>
    <tr>
      <th align="left">模型</th>
      <th align="center">#&nbsp;Params.</th>
      <th align="center">1-Obj.</th>
      <th align="center">2-Obj.</th>
      <th align="center">Count</th>
      <th align="center">Colors</th>
      <th align="center">Position</th>
      <th align="center">Attr.</th>
      <th align="center">Overall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center" colspan="9"><i>仅生成模型</i></td>
    </tr>
    <tr>
      <td align="left">SDXL</td><td align="center">3.5B</td><td align="center">0.98</td><td align="center">0.74</td><td align="center">0.39</td><td align="center">0.85</td><td align="center">0.15</td><td align="center">0.23</td><td align="center">0.55</td>
    </tr>
    <tr>
      <td align="left">DALL-E 3</td><td align="center">-</td><td align="center">0.96</td><td align="center">0.87</td><td align="center">0.47</td><td align="center">0.83</td><td align="center">0.43</td><td align="center">0.45</td><td align="center">0.67</td>
    </tr>
    <tr>
      <td align="left">SD3-Medium</td><td align="center">2B</td><td align="center">0.99</td><td align="center">0.94</td><td align="center">0.72</td><td align="center">0.89</td><td align="center">0.33</td><td align="center">0.60</td><td align="center">0.74</td>
    </tr>
    <tr>
      <td align="left">FLUX.1-dev</td><td align="center">12B</td><td align="center">0.98</td><td align="center">0.93</td><td align="center">0.75</td><td align="center">0.93</td><td align="center">0.68</td><td align="center">0.65</td><td align="center">0.82</td>
    </tr>
    <tr>
      <td align="left">Qwen-Image</td><td align="center">20B</td><td align="center">0.99</td><td align="center">0.92</td><td align="center">0.89</td><td align="center">0.88</td><td align="center">0.76</td><td align="center">0.77</td><td align="center">0.87</td>
    </tr>
    <tr>
      <td align="center" colspan="9"><i>统一模型</i></td>
    </tr>
    <tr>
      <td align="left">Janus-Pro-7B</td><td align="center">7B</td><td align="center"><u>0.99</u></td><td align="center">0.89</td><td align="center">0.59</td><td align="center">0.90</td><td align="center">0.79</td><td align="center">0.66</td><td align="center">0.80</td>
    </tr>
    <tr>
      <td align="left">OmniGen2</td><td align="center">4B</td><td align="center"><b>1.00</b></td><td align="center">0.95</td><td align="center">0.64</td><td align="center">0.88</td><td align="center">0.55</td><td align="center">0.76</td><td align="center">0.80</td>
    </tr>
    <tr>
      <td align="left">Show-o2</td><td align="center">7B</td><td align="center"><b>1.00</b></td><td align="center">0.87</td><td align="center">0.58</td><td align="center">0.92</td><td align="center">0.52</td><td align="center">0.62</td><td align="center">0.76</td>
    </tr>
    <tr>
      <td align="left">BAGEL<sup>†</sup></td><td align="center">7B</td><td align="center">0.98</td><td align="center">0.95</td><td align="center"><b>0.84</b></td><td align="center"><u>0.95</u></td><td align="center">0.78</td><td align="center">0.77</td><td align="center">0.88</td>
    </tr>
    <tr>
      <td align="left">Mogao</td><td align="center">7B</td><td align="center"><b>1.00</b></td><td align="center"><b>0.97</b></td><td align="center"><u>0.83</u></td><td align="center">0.93</td><td align="center">0.84</td><td align="center">0.80</td><td align="center"><u>0.89</u></td>
    </tr>
    <tr>
      <td align="left">InternVL-U</td><td align="center">1.7B</td><td align="center"><u>0.99</u></td><td align="center">0.94</td><td align="center">0.74</td><td align="center">0.91</td><td align="center">0.77</td><td align="center">0.74</td><td align="center">0.85</td>
    </tr>
    <tr>
      <td align="left">TUNA</td><td align="center">7B</td><td align="center"><b>1.00</b></td><td align="center"><b>0.97</b></td><td align="center">0.81</td><td align="center">0.91</td><td align="center"><b>0.88</b></td><td align="center"><b>0.83</b></td><td align="center"><b>0.90</b></td>
    </tr>
    <tr>
      <td align="left">TUNA-2</td><td align="center">7B</td><td align="center"><u>0.99</u></td><td align="center"><u>0.96</u></td><td align="center">0.80</td><td align="center">0.91</td><td align="center">0.84</td><td align="center">0.76</td><td align="center">0.87</td>
    </tr>
    <tr>
      <td align="left">🌟 <b>Lance (Ours)</b></td><td align="center"><b>3B</b></td><td align="center"><b>1.00</b></td><td align="center"><b>0.94</b></td><td align="center"><b>0.84</b></td><td align="center"><b>0.97</b></td><td align="center"><b>0.87</b></td><td align="center"><b>0.81</b></td><td align="center"><b>0.90</b></td>
    </tr>
  </tbody>
</table>
</div>

<p align="center"><em><sup>†</sup> 表示该方法在生成前使用 LLM rewriter 进行提示词改写。</em></p>

</details>

<details>
<summary><strong>GEdit-Bench 评测</strong></summary>

<div align="center">
<table align="center">
  <thead>
    <tr>
      <th align="left">模型</th>
      <th align="center">#&nbsp;Params.</th>
      <th align="center">BC</th>
      <th align="center">CA</th>
      <th align="center">MM</th>
      <th align="center">MC</th>
      <th align="center">PB</th>
      <th align="center">ST</th>
      <th align="center">SA</th>
      <th align="center">SR</th>
      <th align="center">SRp</th>
      <th align="center">TM</th>
      <th align="center">TT</th>
      <th align="center">Avg/G_O</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center" colspan="14"><i>仅生成模型</i></td>
    </tr>
    <tr>
      <td align="left">Gemini 2.0</td><td align="center">-</td><td align="center">-</td><td align="center">-</td><td align="center">-</td><td align="center">-</td><td align="center">-</td><td align="center">-</td><td align="center">-</td><td align="center">-</td><td align="center">-</td><td align="center">-</td><td align="center">-</td><td align="center">6.32</td>
    </tr>
    <tr>
      <td align="left">GPT Image 1</td><td align="center">-</td><td align="center">6.96</td><td align="center">6.85</td><td align="center">7.10</td><td align="center">5.41</td><td align="center">6.74</td><td align="center">7.44</td><td align="center">7.51</td><td align="center">8.73</td><td align="center">8.55</td><td align="center">8.45</td><td align="center">8.69</td><td align="center">7.49</td>
    </tr>
    <tr>
      <td align="left">Qwen-Image-Edit</td><td align="center">20B</td><td align="center">8.23</td><td align="center">8.30</td><td align="center">7.33</td><td align="center">8.05</td><td align="center">7.49</td><td align="center">6.74</td><td align="center">8.57</td><td align="center">8.09</td><td align="center">8.29</td><td align="center">8.48</td><td align="center">8.50</td><td align="center">8.01</td>
    </tr>
    <tr>
      <td align="center" colspan="14"><i>统一模型</i></td>
    </tr>
    <tr>
      <td align="left">Lumina-DiMOO</td><td align="center">8B</td><td align="center">3.43</td><td align="center">4.27</td><td align="center">3.08</td><td align="center">2.77</td><td align="center">4.74</td><td align="center">5.19</td><td align="center">4.44</td><td align="center">3.80</td><td align="center">4.38</td><td align="center">2.68</td><td align="center">4.20</td><td align="center">3.91</td>
    </tr>
    <tr>
      <td align="left">Ovis-U1</td><td align="center">1.2B</td><td align="center"><u>7.49</u></td><td align="center">6.88</td><td align="center">6.21</td><td align="center">4.79</td><td align="center">5.98</td><td align="center"><u>6.46</u></td><td align="center">7.49</td><td align="center"><u>7.25</u></td><td align="center"><u>7.27</u></td><td align="center">4.48</td><td align="center">6.31</td><td align="center">6.42</td>
    </tr>
    <tr>
      <td align="left">BAGEL</td><td align="center">7B</td><td align="center">7.32</td><td align="center">6.91</td><td align="center">6.38</td><td align="center">4.75</td><td align="center">4.57</td><td align="center">6.15</td><td align="center"><b>7.90</b></td><td align="center">7.16</td><td align="center">7.02</td><td align="center"><u>7.32</u></td><td align="center">6.22</td><td align="center">6.52</td>
    </tr>
    <tr>
      <td align="left">InternVL-U</td><td align="center">1.7B</td><td align="center">7.08</td><td align="center">7.05</td><td align="center">6.38</td><td align="center"><u>7.02</u></td><td align="center"><u>6.03</u></td><td align="center">6.27</td><td align="center">7.13</td><td align="center">6.55</td><td align="center">6.33</td><td align="center">6.59</td><td align="center"><u>6.85</u></td><td align="center">6.66</td>
    </tr>
    <tr>
      <td align="left">InternVL-U (w/ CoT)</td><td align="center">1.7B</td><td align="center">7.05</td><td align="center"><b>7.87</b></td><td align="center"><u>6.50</u></td><td align="center">6.99</td><td align="center">5.77</td><td align="center">6.10</td><td align="center">7.33</td><td align="center">7.16</td><td align="center">7.12</td><td align="center"><b>7.36</b></td><td align="center">6.46</td><td align="center"><u>6.88</u></td>
    </tr>
    <tr>
      <td align="left">🌟 <b>Lance (Ours)</b></td><td align="center"><b>3B</b></td><td align="center"><b>7.73</b></td><td align="center"><u>7.74</u></td><td align="center"><b>7.28</b></td><td align="center"><b>7.83</b></td><td align="center"><b>7.50</b></td><td align="center"><b>7.03</b></td><td align="center"><u>7.64</u></td><td align="center"><b>7.85</b></td><td align="center"><b>7.71</b></td><td align="center">4.46</td><td align="center"><b>7.57</b></td><td align="center"><b>7.30</b></td>
    </tr>
  </tbody>
</table>
</div>

</details>

<details>
<summary><strong>VBench 评测（视频生成）</strong></summary>

<div align="center">
<table align="center">
  <thead>
    <tr>
      <th align="left">类型</th>
      <th align="left">Model</th>
      <th align="center">#&nbsp;Params.</th>
      <th align="center">Total Score ↑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center" rowspan="12"><i>Gen. Only</i></td>
      <td align="left">ModelScope</td><td align="center">1.7B</td><td align="center">75.75</td>
    </tr>
    <tr>
      <td align="left">LaVie</td><td align="center">3B</td><td align="center">77.08</td>
    </tr>
    <tr>
      <td align="left">Show-1</td><td align="center">6B</td><td align="center">78.93</td>
    </tr>
    <tr>
      <td align="left">AnimateDiff-V2</td><td align="center">-</td><td align="center">80.27</td>
    </tr>
    <tr>
      <td align="left">VideoCrafter-2.0</td><td align="center">-</td><td align="center">80.44</td>
    </tr>
    <tr>
      <td align="left">CogVideoX</td><td align="center">5B</td><td align="center">81.61</td>
    </tr>
    <tr>
      <td align="left">Kling</td><td align="center">-</td><td align="center">81.85</td>
    </tr>
    <tr>
      <td align="left">Open-Sora-2.0</td><td align="center">-</td><td align="center">81.71</td>
    </tr>
    <tr>
      <td align="left">Gen-3</td><td align="center">-</td><td align="center">82.32</td>
    </tr>
    <tr>
      <td align="left">Step-Video-T2V</td><td align="center">30B</td><td align="center">81.83</td>
    </tr>
    <tr>
      <td align="left">Hunyuan Video</td><td align="center">-</td><td align="center">83.43</td>
    </tr>
    <tr>
      <td align="left">Wan2.1-T2V</td><td align="center">14B</td><td align="center">83.69</td>
    </tr>
    <tr>
      <td align="center" rowspan="6"><i>Unified</i></td>
      <td align="left">HaproOmni</td><td align="center">7B</td><td align="center">78.10</td>
    </tr>
    <tr>
      <td align="left">Emu3</td><td align="center">8B</td><td align="center">80.96</td>
    </tr>
    <tr>
      <td align="left">VILA-U</td><td align="center">7B</td><td align="center">74.01</td>
    </tr>
    <tr>
      <td align="left">Show-o2</td><td align="center">2B</td><td align="center">81.34</td>
    </tr>
    <tr>
      <td align="left">TUNA</td><td align="center">1.5B</td><td align="center"><u>84.06</u></td>
    </tr>
    <tr>
      <td align="left">🌟 <b>Lance (Ours)</b></td><td align="center"><b>3B</b></td><td align="center"><b>85.11</b></td>
    </tr>
  </tbody>
</table>
</div>

</details>

#### 运行基准评测

`benchmarks/` 目录下提供了可直接运行的基准评测脚本：

| 基准 | 模态 | 脚本 |
|------------------------|----------|---------------------------------------------------------------|
| GenEVAL（图像生成） | 图像 | `benchmarks/image_gen/GenEVAL/sample_GenEVAL.sh` |
| DPG（图像生成） | 图像 | `benchmarks/image_gen/DPG/sample_DPG.sh` |
| GEdit（图像编辑） | 图像 | `benchmarks/image_gen/GEdit/sample_GEdit.sh` |
| VBench（视频生成） | 视频 | `benchmarks/video_gen/Vbench/sample_vbench.sh` |


## 📄 许可证

Copyright 2025 ByteDance Ltd. and/or its affiliates.

## 🙏 致谢

我们感谢 [BAGEL](https://github.com/ByteDance-Seed/bagel)、[Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) 和 [Wan2.2](https://github.com/Wan-Video/Wan2.2) 的贡献者，感谢他们开放的研究与社区贡献。

## 💖 引用

如果 **Lance** 对您的项目或研究有帮助，欢迎 🌟 本仓库，并使用以下 BibTeX 引用我们的工作：

```bibtex
@misc{fu2026lanceunifiedmultimodalmodeling,
      title         = {Lance: Unified Multimodal Modeling by Multi-Task Synergy},
      author        = {Fengyi Fu and Mengqi Huang and Shaojin Wu and Yunsheng Jiang and Yufei Huo and Hao Li and Yinghang Song and Fei Ding and Jianzhu Guo and Qian He and Zheren Fu and Zhendong Mao and Yongdong Zhang},
      year          = {2026},
      eprint        = {2605.18678},
      archivePrefix = {arXiv},
      primaryClass  = {cs.CV},
      url           = {https://arxiv.org/abs/2605.18678},
}
```

## 📞 联系方式

如有问题、反馈或合作需求，请联系 [Mengqi Huang](https://corleone-huang.github.io/) 和 [Jianzhu Guo](https://guojianzhu.com/)。
