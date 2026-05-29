# coding: utf-8
import argparse
import base64
import json
import mimetypes
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Iterable, List

if __package__ in (None, ""):
    # Avoid shadowing stdlib logging with common/utils/logging.py when this file
    # is executed directly as `python common/utils/caption_rewrite.py`.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path = [path for path in sys.path if os.path.abspath(path or os.getcwd()) != script_dir]

import openai

# NOTE: Replace the following few lines for the model you want to use.
API_KEY = "YOUR_API_KEY"
MODEL_NAME = "YOUR_MODEL_NAME"
BASE_URL = "https://api.openai.com/v1"
MAX_TOKENS = 2048
THINKING_ENABLED = False
THINKING_BUDGET_TOKENS = 2000


# Configure the client here.
def create_client(api_key: str | None = None):
    return openai.OpenAI(
        api_key=api_key or API_KEY,
        base_url=BASE_URL,
    )


# Default values for caption rewrite.
TEMPERATURE = 0.3
DEFAULT_STYLE_EXAMPLE_PATH = Path("config/examples/t2v_example.json")
DEFAULT_I2V_STYLE_EXAMPLE_PATH = Path("config/examples/i2v_example.json")
DEFAULT_NUM_STYLE_EXAMPLES = 6


def get_rewrite_config_error(api_key: str | None = None) -> str | None:
    key = api_key or API_KEY
    key = key.strip() if isinstance(key, str) else key

    model_name = MODEL_NAME.strip() if isinstance(MODEL_NAME, str) else MODEL_NAME
    base_url = BASE_URL.strip() if isinstance(BASE_URL, str) else BASE_URL

    if not key or key.startswith("YOUR_"):
        return "API_KEY is not configured."
    if not model_name or model_name.startswith("YOUR_"):
        return "MODEL_NAME is not configured."
    if not base_url or base_url.startswith("YOUR_"):
        return "BASE_URL is not configured."
    if not (base_url.startswith("http://") or base_url.startswith("https://")):
        return f"BASE_URL should start with http:// or https://, got: {base_url}"

    return None


def has_valid_rewrite_config(api_key: str | None = None) -> bool:
    error = get_rewrite_config_error(api_key)
    if error:
        warnings.warn(
            f"Prompt rewrite is disabled: {error} "
            "Please configure API_KEY, MODEL_NAME, and BASE_URL before using --ENHANCE_PROMPT true.",
            RuntimeWarning,
        )
        return False
    return True


def has_rewrite_api_key(api_key: str | None = None) -> bool:
    """Backward-compatible alias. It now checks the full rewrite config."""
    return has_valid_rewrite_config(api_key)


def load_style_examples(
    example_path: str | Path = DEFAULT_STYLE_EXAMPLE_PATH,
    max_examples: int = DEFAULT_NUM_STYLE_EXAMPLES,
) -> List[str]:
    """Load T2V captions from a JSON file as rewrite style references."""
    path = Path(example_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    values: Iterable[Any] = data.values() if isinstance(data, dict) else data
    examples: List[str] = []
    for value in values:
        if isinstance(value, str):
            examples.append(value.strip())
        elif isinstance(value, dict):
            prompt = value.get("prompt") or value.get("caption") or value.get("data")
            if isinstance(prompt, str):
                examples.append(prompt.strip())
            elif isinstance(value.get("interleave_array"), list) and value["interleave_array"]:
                interleave_prompt = value["interleave_array"][0]
                if isinstance(interleave_prompt, str):
                    examples.append(interleave_prompt.strip())
        if len(examples) >= max_examples:
            break
    return [example for example in examples if example]


def encode_image_as_data_url(image_path: str | Path) -> str:
    """Encode a local image file as a data URL for multimodal chat input."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    mime_type = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    image_b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{image_b64}"


def build_rewrite_instruction(prompt: str, style_examples: List[str]) -> str:
    """Build the shared caption rewrite instruction."""
    examples_text = "\n\n".join(
        f"Style example {idx + 1}:\n{example}"
        for idx, example in enumerate(style_examples)
    )
    return f"""You are a professional prompt rewriter for text-to-video generation.

Rewrite the input prompt into a polished English video caption that matches the style of the reference examples.

Requirements:
- Preserve the user's original subject, action, setting, and important visual details.
- Match the reference style: cinematic, concrete, visually rich, with clear subject framing, environment, lighting, camera motion, and readable action.
- In general, describe the video scene details first, then describe the camera movement.
- Prefer one cohesive paragraph. Do not use bullets, markdown, labels, or quotation marks.
- Do not invent unrelated subjects, props, locations, or story events.
- If the input is not English, translate it naturally into English while rewriting.
- Generate a detailed text prompt with rich visual specifics, clear motion, and enough concrete information for video generation.

Reference examples:
{examples_text}

Input prompt:
{prompt}

Rewritten caption:"""


def build_i2v_rewrite_instruction(prompt: str, style_examples: List[str]) -> str:
    """Build the image-conditioned I2V rewrite instruction."""
    examples_text = "\n\n".join(
        f"Style example {idx + 1}:\n{example}"
        for idx, example in enumerate(style_examples)
    )
    return f"""You are a professional prompt rewriter for first-frame-to-video generation.

You are given an input text prompt and a reference image. Rewrite them into one polished English video prompt that matches the style of the reference examples.

Requirements:
- Use the reference examples from config/examples/i2v_example.json as the target writing style.
- Preserve the user's intended action and motion from the input text prompt.
- Fully describe the visible content of the input image, including the main subject, appearance, pose, environment, lighting, materials, colors, spatial layout, and important background details.
- The rewritten prompt must be grounded in the input image. Do not invent unrelated subjects, locations, or props that are not supported by either the image or the text prompt.
- In general, describe the video scene details first, then describe the camera movement.
- Prefer one cohesive paragraph. Do not use bullets, markdown, labels, or quotation marks.
- Generate a detailed text prompt with rich visual specifics, clear motion, and enough concrete information for video generation.

Reference examples:
{examples_text}

Input text prompt:
{prompt}

Rewritten caption:"""


def run_chat_completion(
    instruction: str,
    content=None,
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
    llm_client=None,
    api_key: str | None = None,
) -> str:
    """Run the shared `client.chat.completions.create` inference body."""
    llm_client = llm_client or create_client(api_key=api_key)
    if content is None:
        content = [{"type": "text", "text": instruction}]
    request_kwargs = {}
    if THINKING_ENABLED:
        request_kwargs["extra_body"] = {
            "thinking": {
                "include_thoughts": True,
                "budget_tokens": THINKING_BUDGET_TOKENS,
            }
        }
    response = llm_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        **request_kwargs,
    )
    content = response.choices[0].message.content
    if not content or not content.strip():
        raise RuntimeError(f"Empty rewrite result: {response.model_dump_json(indent=2)}")
    return content.strip()


def rewrite_caption(
    prompt: str,
    style_example_path: str | Path = DEFAULT_STYLE_EXAMPLE_PATH,
    num_style_examples: int = DEFAULT_NUM_STYLE_EXAMPLES,
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
    llm_client=None,
    api_key: str | None = None,
) -> str:
    """Rewrite an input prompt in the style of config/examples/t2v_example.json."""
    if not prompt or not prompt.strip():
        raise ValueError("prompt must be a non-empty string.")

    style_examples = load_style_examples(style_example_path, num_style_examples)
    instruction = build_rewrite_instruction(prompt.strip(), style_examples)
    return run_chat_completion(
        instruction=instruction,
        max_tokens=max_tokens,
        temperature=temperature,
        llm_client=llm_client,
        api_key=api_key,
    )


def rewrite_i2v_prompt(
    prompt: str,
    image_path: str | Path,
    style_example_path: str | Path = DEFAULT_I2V_STYLE_EXAMPLE_PATH,
    num_style_examples: int = DEFAULT_NUM_STYLE_EXAMPLES,
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
    llm_client=None,
    api_key: str | None = None,
) -> str:
    """Rewrite an I2V text prompt using the input image and TI2V style examples."""
    if not prompt or not prompt.strip():
        raise ValueError("prompt must be a non-empty string.")
    style_examples = load_style_examples(style_example_path, num_style_examples)
    instruction = build_i2v_rewrite_instruction(prompt.strip(), style_examples)
    content = [
        {"type": "text", "text": instruction},
        {"type": "image_url", "image_url": {"url": encode_image_as_data_url(image_path)}},
    ]
    return run_chat_completion(
        instruction=instruction,
        content=content,
        max_tokens=max_tokens,
        temperature=temperature,
        llm_client=llm_client,
        api_key=api_key,
    )


def rewrite_prompt(prompt: str, **kwargs) -> str:
    """Alias kept for shell/inference integration readability."""
    return rewrite_caption(prompt, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite a caption with the configured chat model.")
    parser.add_argument("prompt", nargs="?", help="Input prompt to rewrite.")
    parser.add_argument("--mode", choices=["t2v", "i2v"], default="t2v", help="Rewrite mode.")
    parser.add_argument("--image-path", default=None, help="Input image path for i2v rewrite.")
    parser.add_argument("--prompt-file", default=None, help="Read the input prompt from a text file.")
    parser.add_argument("--style-example-path", default=str(DEFAULT_STYLE_EXAMPLE_PATH), help="Path to style reference JSON.")
    parser.add_argument("--num-style-examples", type=int, default=DEFAULT_NUM_STYLE_EXAMPLES, help="Number of style examples.")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS, help="Maximum output tokens.")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE, help="Sampling temperature.")
    args = parser.parse_args()

    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    elif args.prompt:
        prompt = args.prompt
    else:
        parser.error("Either prompt or --prompt-file is required.")

    config_error = get_rewrite_config_error()
    if config_error:
        parser.error(
            f"{config_error} Set API_KEY, MODEL_NAME, and BASE_URL at the top of this file "
            "or configure them before using --ENHANCE_PROMPT true."
        )

    if args.mode == "i2v":
        if not args.image_path:
            parser.error("--image-path is required when --mode i2v.")
        style_example_path = (
            args.style_example_path
            if args.style_example_path != str(DEFAULT_STYLE_EXAMPLE_PATH)
            else DEFAULT_I2V_STYLE_EXAMPLE_PATH
        )
        rewritten = rewrite_i2v_prompt(
            prompt=prompt,
            image_path=args.image_path,
            style_example_path=style_example_path,
            num_style_examples=args.num_style_examples,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    else:
        rewritten = rewrite_caption(
            prompt=prompt,
            style_example_path=args.style_example_path,
            num_style_examples=args.num_style_examples,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    print(rewritten)


if __name__ == "__main__":
    main()