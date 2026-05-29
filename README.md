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
    <sup>*</sup> Equal contribution &nbsp;&nbsp; <sup>✉</sup> Corresponding authors &nbsp;&nbsp; <sup>§</sup> Project lead
  </p>
  <p>
    <a href="https://lance-project.github.io/" style="text-decoration: none; margin: 0 8px;"><img src="https://img.shields.io/badge/Homepage-Lance-blue?style=flat" alt="Homepage"></a>
    <a href="http://arxiv.org/abs/2605.18678" style="text-decoration: none; margin: 0 8px;"><img src="https://img.shields.io/badge/Paper-arXiv-red?style=flat&logo=arxiv" alt="arXiv"></a>
    <a href="https://huggingface.co/bytedance-research/Lance" style="text-decoration: none; margin: 0 8px;"><img src="https://img.shields.io/badge/Model-HuggingFace-yellow?style=flat&logo=huggingface" alt="Model"></a>
    <a href="https://huggingface.co/spaces/bytedance-research/Lance" style="text-decoration: none; margin: 0 8px;"><img src="https://img.shields.io/badge/Demo-HuggingFace-40bfe6?style=flat&logo=huggingface" alt="Demo"></a>
    <br>
    English | <a href="./README_zh.md"><ins>简体中文</ins></a>
  </p>
</div>

> **Note:** Lance is a research project rather than a polished product model. The released checkpoint was trained with up to 128 A100 GPUs, with training conducted up to 768x768 image generation and 480p, 12 FPS video generation. Our goal is to share a research artifact for studying unified image/video understanding, generation, and editing under a relatively small model and limited compute budget. Output quality may vary across prompts, resolutions, duration, motion complexity, and editing scenarios, and we see further opportunities to improve the post-training recipe. We appreciate constructive feedback from the community as we continue improving the project.

## 🔥 Updates

- **`2026/05/29`**: 💪 Added code support for Text-Image-to-Video (First-Frame-to-Video). [More to see](assets/docs/changelog/2026-05-29.md)!
- **`2026/05/26`**: 🎨 The Gradio interface now supports image and video generation, editing, and understanding. [Try it out](assets/docs/changelog/2026-05-26.md)!
- **`2026/05/25`**: ✨ The [Hugging Face Space](https://huggingface.co/spaces/bytedance-research/Lance) is now live, thanks to the HF team!
- **`2026/05/19`**: 🤗 The technical report is now available on [arXiv](http://arxiv.org/abs/2605.18678).
- **`2026/05/18`**: 🔥 We launched the [project homepage](https://lance-project.github.io/) and released the initial inference code and model weights on [GitHub](https://github.com/bytedance/Lance/) and [Hugging Face](https://huggingface.co/bytedance-research/Lance).

## 🌟 Highlights

**Lance** is a 3B native unified multimodal model that supports **image and video understanding, generation, and editing** within a single framework.

- **Efficient at 3B scale.** With only **3B active parameters**, Lance achieves competitive performance across image generation, image editing, and video generation benchmarks.
- **Training from scratch.** Lance is trained from scratch with a staged multi-task recipe and within a budget of **up to 128 A100 GPUs**.

We are actively updating and improving this repository. If you find any bugs or have suggestions, please feel free to open an issue or submit a pull request (PR) 💖.

<div align="center">
  <img src="assets/benchmarks/benchmark-overview.png" alt="Lance benchmark overview across image generation, image editing, video generation, and video understanding" width="980">
</div>

## 📅 Roadmap

- [ ] Release the fine-tuning code.

## 🎨 Demo

<details>
<summary><strong>Show demo results</strong></summary>

<div align="center">
  <strong>🔥 We recommend visiting our <a href="https://lance-project.github.io/">homepage</a> for more visual results. 🔥</strong>
</div>

<h3 align="center">Text-to-Video</h3>

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

<h3 align="center">Video Editing</h3>

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

<h3 align="center">Multi-turn Consistency Editing</h3>

<div align="center">
  <a href="assets/multi-turn-editing/videos/multi-turn-editing-demo-01.mp4">
    <img src="assets/multi-turn-editing/previews/multi-turn-editing-demo-01.gif" width="100%">
  </a>
</div>

<h3 align="center">Intelligent Video Generation</h3>

<table align="center">
  <tr>
    <td><a href="assets/intelligent-video/videos/intelligent-video-demo-01.mp4"><img src="assets/intelligent-video/previews/intelligent-video-demo-01.gif" width="100%"></a></td>
    <td><a href="assets/intelligent-video/videos/intelligent-video-demo-02.mp4"><img src="assets/intelligent-video/previews/intelligent-video-demo-02.gif" width="100%"></a></td>
    <td><a href="assets/intelligent-video/videos/intelligent-video-demo-03.mp4"><img src="assets/intelligent-video/previews/intelligent-video-demo-03.gif" width="100%"></a></td>
    <td><a href="assets/intelligent-video/videos/intelligent-video-demo-04.mp4"><img src="assets/intelligent-video/previews/intelligent-video-demo-04.gif" width="100%"></a></td>
  </tr>
</table>

</details>

## 🚀 Installation

### Recommended Environment

- **Software:** Python 3.10+, CUDA 12.4+ (required)
- **Hardware:** A GPU with at least 40GB VRAM is required for inference

We have tested the following dependency combinations on NVIDIA A100:

- PyTorch 2.8.0 + cu126 + flash-attn 2.8.3
- PyTorch 2.5.1 + cu124 + flash-attn 2.6.3

The default installation commands use the PyTorch 2.8.0 + cu126 setup. For other GPU models, please choose and validate a PyTorch build and a matching `flash-attn` version according to your driver, CUDA runtime, Python version, and GPU architecture.

### Installation Steps

First, clone the repository:

```bash
git clone https://github.com/bytedance/Lance.git
cd Lance
```

Then, set up the environment:

```bash
conda create -n Lance python=3.11 -y
conda activate Lance
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install flash-attn==2.8.3 --no-build-isolation
```

> **Note:** If installing `flash-attn` from source fails, you can install a prebuilt wheel instead. The wheelhouse below is from a third-party repository and is provided for **reference only**; please verify that any wheel you install matches your Python, PyTorch and CUDA versions.

> ```bash
> pip install --no-cache-dir --no-deps --force-reinstall \
> "https://huggingface.co/strangertoolshf/flash_attention_2_wheelhouse/resolve/main/wheelhouse-flash_attn-2.8.3/linux_x86_64/torch2.8/cu12/abiTRUE/cp311/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
> ```


Then, download the model weights from [Lance-3B on Hugging Face](https://huggingface.co/bytedance-research/Lance) and place them in the `downloads/` directory:

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


## 📚 Usage

### Inference

#### Basic Usage

```bash
bash inference_lance.sh
```

- Before running, please configure the inference parameters at the top of `inference_lance.sh`.
- **Supported tasks:** `t2i`, `t2v`, `i2v`, `image_edit`, `video_edit`, `x2t_image`, and `x2t_video`. You can modify `TASK_DEFAULT_CONFIGS` in `inference_lance.py` to customize the default data samples for each task.
- **Note:** For all tasks, we recommend following the `prompt` format used in the provided examples when writing input prompts, as this typically leads to better generation quality.

#### Task Examples

##### Text-to-Video

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

##### Text-Image-to-Video

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

Optional parameters for video generation task examples:

- `--ENHANCE_PROMPT true`: enable prompt rewrite for T2V/I2V. Prompt enhancement generally improves generation quality. Before enabling it, set `API_KEY`, `MODEL_NAME`, and `client` in `common/utils/caption_rewrite.py`. If no API key is configured there, prompt rewrite is skipped; in that case, we recommend **writing prompts in the style of the provided examples**.

##### Text-to-Image

```bash
bash inference_lance.sh \
  --TASK_NAME t2i \
  --MODEL_PATH downloads/Lance_3B \
  --RESOLUTION image_768res \
  --VIDEO_HEIGHT 768 \
  --VIDEO_WIDTH 768 \
  --SAVE_PATH_GEN results/t2i
```

##### Video Editing

```bash
bash inference_lance.sh \
  --TASK_NAME video_edit \
  --MODEL_PATH downloads/Lance_3B_Video \
  --RESOLUTION video_480p \
  --SAVE_PATH_GEN results/video_edit
```

##### Image Editing

```bash
bash inference_lance.sh \
  --TASK_NAME image_edit \
  --MODEL_PATH downloads/Lance_3B \
  --RESOLUTION image_768res \
  --SAVE_PATH_GEN results/image_edit
```

##### Video Understanding

```bash
bash inference_lance.sh \
  --TASK_NAME x2t_video \
  --MODEL_PATH downloads/Lance_3B_Video \
  --RESOLUTION video_480p \
  --NUM_FRAMES 50 \
  --SAVE_PATH_GEN results/x2t_video
```

##### Image Understanding

```bash
bash inference_lance.sh \
  --TASK_NAME x2t_image \
  --MODEL_PATH downloads/Lance_3B \
  --RESOLUTION image_768res \
  --SAVE_PATH_GEN results/x2t_image
```

Optional parameters for all task examples:

- `--CONFIG_PATH path/to/config.json`: use a custom validation JSON/JSONL file instead of the task default example config.


<details>
<summary><strong>Show task and parameter reference</strong></summary>

#### Available Tasks

| Task Name              | Description                                      | Example JSON                                 |
|------------------------|--------------------------------------------------|----------------------------------------------|
| `t2v`                  | Text-to-Video generation                         | `config/examples/t2v_example.json`           |
| `t2i`                  | Text-to-Image generation                         | `config/examples/t2i_example.json`           |
| `i2v`                 | First-Frame-to-Video generation                  | `config/examples/i2v_example.json`          |
| `image_edit`           | Image editing                                    | `config/examples/image_edit_example.json`    |
| `video_edit`           | Video editing                                    | `config/examples/video_edit_example.json`    |
| `x2t_image`            | Image understanding            | `config/examples/x2t_image_example.json`    |
| `x2t_video`            | Video understanding            | `config/examples/x2t_video_example.json`    |

For understanding examples:

- `config/examples/x2t_image_example.json`: image understanding examples for visual question answering, reasoning and image captioning.
- `config/examples/x2t_video_example.json`: video understanding examples for video question answering and video captioning.

#### Parameters

You can configure the following hyperparameters at the top of the `inference_lance.sh` script:

| Parameter | Default Value | Description |
| --- | --- | --- |
| `MODEL_PATH` | `"downloads/Lance_3B"` | Path to the downloaded Lance model weights  (`Lance_3B` or `Lance_3B_Video`). |
| `NUM_GPUS` | `1` | Number of GPUs to use for inference. |
| `VALIDATION_NUM_TIMESTEPS` | `30` | Number of denoising steps (e.g., 30 or 50). |
| `VALIDATION_TIMESTEP_SHIFT` | `3.5` | Timestep shift parameter for flow matching scheduling. |
| `CFG_TEXT_SCALE` | `4.0` | Classifier-Free Guidance (CFG) scale for text conditioning. |
| `VALIDATION_DATA_SEED` | `42` | Random seed for generation reproducibility. |
| `NUM_FRAMES` | `50` | Number of frames for video generation (Max: 121). *Unused for image tasks.* |
| `VIDEO_HEIGHT` / `VIDEO_WIDTH`| `768` | Spatial resolution. *Unused for editing tasks (determined by input image/video).* |
| `RESOLUTION` | `"video_480p"` | Base resolution preset (`image_768res` or `video_480p`). |
| `CONFIG_PATH` | `""` | Optional path to a custom validation JSON/JSONL file. When empty, the task default example config is used. |
| `ENHANCE_PROMPT` | `false` | Optional T2V/I2V prompt rewrite switch. T2V uses text-only rewrite; I2V uses text plus the input image. Prompt enhancement generally improves generation quality. Configure the rewrite API key and client in `common/utils/caption_rewrite.py` before setting this to `true`; without a key, we recommend writing prompts in the style of the provided examples. |

</details>

### 🖥️ Gradio

You can launch the local Gradio demo for video/image generation, editing, and understanding:

```bash
python lance_gradio.py --server-name 0.0.0.0 --server-port 7860
```

### Benchmarks

<details>
<summary><strong>DPG-Bench Evaluation</strong></summary>

<div align="center">
<table align="center">
  <thead>
    <tr>
      <th align="left">Models</th>
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
      <td align="center" colspan="8"><i>Generation-only Models</i></td>
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
      <td align="center" colspan="8"><i>Unified Models</i></td>
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

<p align="center"><em><sup>†</sup> indicates methods that use LLM rewriters for prompt rewriting before generation.</em></p>

</details>

<details>
<summary><strong>GenEval Evaluation</strong></summary>

<div align="center">
<table align="center">
  <thead>
    <tr>
      <th align="left">Models</th>
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
      <td align="center" colspan="9"><i>Generation-only Models</i></td>
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
      <td align="center" colspan="9"><i>Unified Models</i></td>
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

<p align="center"><em><sup>†</sup> indicates methods that use LLM rewriters for prompt rewriting before generation.</em></p>

</details>

<details>
<summary><strong>GEdit-Bench Evaluation</strong></summary>

<div align="center">
<table align="center">
  <thead>
    <tr>
      <th align="left">Models</th>
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
      <td align="center" colspan="14"><i>Generation-only Models</i></td>
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
      <td align="center" colspan="14"><i>Unified Models</i></td>
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
<summary><strong>VBench Evaluation (Video Generation)</strong></summary>

<div align="center">
<table align="center">
  <thead>
    <tr>
      <th align="left">Type</th>
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

#### Running Benchmarks

Ready-to-run benchmark scripts are provided under `benchmarks/`:

| Benchmark              | Modality | Script                                                        |
|------------------------|----------|---------------------------------------------------------------|
| GenEVAL (image gen)    | Image    | `benchmarks/image_gen/GenEVAL/sample_GenEVAL.sh`              |
| DPG (image gen)        | Image    | `benchmarks/image_gen/DPG/sample_DPG.sh`                      |
| GEdit (image edit)     | Image    | `benchmarks/image_gen/GEdit/sample_GEdit.sh`                  |
| VBench (video gen)     | Video    | `benchmarks/video_gen/Vbench/sample_vbench.sh`                |


## 📄 License

Copyright 2025 ByteDance Ltd. and/or its affiliates.

## 🙏 Acknowledgements

We would like to thank the contributors of [BAGEL](https://github.com/ByteDance-Seed/bagel), [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct), and [Wan2.2](https://github.com/Wan-Video/Wan2.2) for their open research and contributions.

## 💖 Citation

If you find **Lance** useful for your project or research, welcome to 🌟 this repo and cite our work using the following BibTeX:

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

## 📞 Contact

For questions, issues, or collaborations, please contact [Mengqi Huang](https://corleone-huang.github.io/) and [Jianzhu Guo](https://guojianzhu.com/).
