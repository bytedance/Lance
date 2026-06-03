from __future__ import annotations

import base64
from typing import Callable, Optional

import gradio as gr

from .helpers import *
from .settings import *
from .styles import APP_CSS

def build_running_status_markdown() -> str:
    return "Running..."

def get_logo_data_uri() -> str:
    if not LANCE_LOGO_PATH.exists():
        return ""
    encoded_logo = base64.b64encode(LANCE_LOGO_PATH.read_bytes()).decode("ascii")
    return f"data:image/webp;base64,{encoded_logo}"

def build_header_html() -> str:
    logo_data_uri = get_logo_data_uri()
    logo_html = (
        f'<img class="lance-logo" src="{logo_data_uri}" alt="Lance logo">'
        if logo_data_uri
        else ""
    )
    return f"""
    <div class="lance-hero">
        {logo_html}
        <h1 class="lance-title">Lance: Unified Multimodal Modeling by Multi-Task Synergy</h1>
        <div class="lance-badges">
            <a href="{LANCE_HOMEPAGE_URL}" target="_blank" rel="noopener noreferrer">
                <img alt="Homepage" src="https://img.shields.io/badge/Homepage-Lance-2563eb?style=flat&labelColor=475569">
            </a>
            <a href="{LANCE_PAPER_URL}" target="_blank" rel="noopener noreferrer">
                <img alt="Paper" src="https://img.shields.io/badge/Paper-arXiv-2563eb?style=flat&labelColor=475569&logo=arxiv">
            </a>
            <a href="{LANCE_HUGGING_FACE_URL}" target="_blank" rel="noopener noreferrer">
                <img alt="Hugging Face" src="https://img.shields.io/badge/Model-HuggingFace-2563eb?style=flat&labelColor=475569&logo=huggingface">
            </a>
            <a href="{LANCE_GITHUB_URL}" target="_blank" rel="noopener noreferrer">
                <img alt="GitHub" src="https://img.shields.io/badge/Code-GitHub-2563eb?style=flat&labelColor=475569&logo=github">
            </a>
        </div>
    </div>
    """

def update_task_ui(task: str):
    internal_task = normalize_task(task)
    is_image_task = internal_task in IMAGE_TASKS
    is_edit_task = internal_task in EDIT_TASKS
    is_understanding_task = internal_task in UNDERSTANDING_TASKS
    is_text_to_visual_task = internal_task in {TASK_T2V, TASK_T2I, TASK_I2V}
    resolution_choices = get_resolution_choices_for_task(internal_task)
    resolution_value = get_default_resolution_for_task(internal_task)
    video_duration_value = get_default_video_duration_seconds(internal_task)
    aspect_ratio_value = DEFAULT_IMAGE_ASPECT_RATIO if is_image_task else DEFAULT_VIDEO_ASPECT_RATIO
    width_value, height_value = get_size_for_aspect_ratio(internal_task, aspect_ratio_value, resolution_value)
    size_markdown = format_size_markdown(internal_task, width_value, height_value)
    system_prompt_choices = get_understanding_system_prompt_choices(internal_task)

    if is_text_to_visual_task:
        text_label = "Prompt"
        text_placeholder = "Describe what you want to generate..."
    elif is_edit_task:
        text_label = "Instruction"
        text_placeholder = "Describe the edit you want..."
    else:
        text_label = "Question"
        text_placeholder = "Ask a question about the input..."

    if internal_task in {TASK_T2V, TASK_I2V, TASK_VIDEO_EDIT}:
        output_label = "Output Video"
    elif internal_task in {TASK_T2I, TASK_IMAGE_EDIT}:
        output_label = "Output Image"
    else:
        output_label = "Output Text"

    output_icon = "video" if output_label == "Output Video" else "image" if output_label == "Output Image" else "text"
    show_aspect_ratio = internal_task in {TASK_T2V, TASK_T2I}
    show_input_video = internal_task in {TASK_VIDEO_EDIT, TASK_X2T_VIDEO}
    show_input_image = internal_task in {TASK_I2V, TASK_IMAGE_EDIT, TASK_X2T_IMAGE}
    show_video_resolution_settings = internal_task in {TASK_T2V, TASK_I2V}

    return (
        gr.update(value=build_lance_label_html(text_label, "lance-prompt-label")),
        gr.update(
            label=text_label,
            placeholder=text_placeholder,
            visible=True,
            value="",
        ),
        gr.update(
            choices=system_prompt_choices,
            value=system_prompt_choices[0],
            visible=False,
        ),
        # Switching task pages should always start from a clean input state.
        # Clear both visual input boxes even if one of them stays visible across tasks.
        gr.update(label="Input Video", visible=show_input_video, value=None),
        gr.update(label="Input Image", visible=show_input_image, value=None),
        gr.update(visible=show_aspect_ratio),
        gr.update(visible=False),
        gr.update(visible=internal_task in {TASK_T2V, TASK_I2V}),
        gr.update(visible=show_video_resolution_settings),
        gr.update(choices=get_aspect_ratio_choices_for_task(internal_task), value=aspect_ratio_value, visible=show_aspect_ratio),
        gr.update(value=height_value),
        gr.update(value=width_value),
        gr.update(choices=get_output_resolution_choices_for_task(internal_task, resolution_value), value=size_markdown, visible=False),
        gr.update(visible=internal_task in {TASK_T2V, TASK_I2V}, value=video_duration_value),
        gr.update(choices=resolution_choices, value=resolution_value, visible=show_video_resolution_settings),
        gr.update(value=build_lance_icon_label_html(output_label, output_icon, "lance-output-label")),
        gr.update(visible=internal_task in {TASK_T2V, TASK_I2V, TASK_VIDEO_EDIT}),
        gr.update(visible=internal_task in {TASK_T2I, TASK_IMAGE_EDIT}),
        gr.update(visible=is_understanding_task, value=""),
        gr.update(visible=internal_task == TASK_T2V),
        gr.update(visible=internal_task == TASK_I2V),
        gr.update(visible=internal_task == TASK_VIDEO_EDIT),
        gr.update(visible=internal_task == TASK_X2T_VIDEO),
        gr.update(visible=internal_task == TASK_T2I),
        gr.update(visible=internal_task == TASK_IMAGE_EDIT),
        gr.update(visible=internal_task == TASK_X2T_IMAGE),
    )

def build_demo(run_task_fn: Callable) -> gr.Blocks:
    with gr.Blocks(title="Lance", css=APP_CSS) as demo:
        gr.HTML(build_header_html())

        with gr.Column(elem_classes=["lance-taskbar-wrap"]):
            task = gr.Radio(
                label="Task",
                show_label=False,
                choices=TASK_CHOICES,
                value=TASK_LABEL_VIDEO_GENERATION,
                elem_classes=["task-selector"],
            )

        with gr.Row(elem_classes=["lance-main-row"]):
            with gr.Column(scale=1, elem_classes=["lance-main-column", "lance-input-column"]):
                with gr.Column(elem_classes=["lance-panel", "lance-task-prompt-panel"]):
                    prompt_label = gr.HTML(build_lance_label_html("Prompt", "lance-prompt-label"), elem_classes=["lance-label-html"])
                    prompt = gr.Textbox(
                        label="Prompt",
                        show_label=False,
                        lines=6,
                        placeholder="Describe the video you want to generate...",
                        elem_classes=["main-prompt-control"],
                    )
                    with gr.Row(elem_classes=["prompt-options"]):
                        with gr.Group(elem_classes=["prompt-chip", "video-resolution-row"]) as video_resolution_row:
                            resolution = gr.Dropdown(
                                label="Video Resolution",
                                show_label=False,
                                choices=VIDEO_RESOLUTION_DISPLAY_CHOICES,
                                value=DEFAULT_RESOLUTION,
                                allow_custom_value=True,
                                elem_classes=["generation-control"],
                            )
                        with gr.Group(elem_classes=["prompt-chip", "aspect-ratio-row"]) as aspect_ratio_row:
                            aspect_ratio = gr.Dropdown(
                                label="Aspect Ratio",
                                show_label=False,
                                choices=get_aspect_ratio_choices_for_task(TASK_T2V),
                                value=DEFAULT_VIDEO_ASPECT_RATIO,
                                elem_classes=["generation-control"],
                            )
                        with gr.Group(elem_classes=["prompt-chip", "video-duration-row"]) as video_duration_row:
                            num_frames = gr.Dropdown(
                                label="Video Duration",
                                show_label=False,
                                choices=get_video_duration_choices(),
                                value=DEFAULT_VIDEO_DURATION_SECONDS,
                                elem_classes=["generation-control"],
                            )
                        with gr.Group(visible=False, elem_classes=["prompt-chip", "output-resolution-row"]) as output_resolution_row:
                            real_size = gr.Dropdown(
                                label="Output Resolution",
                                show_label=False,
                                choices=get_output_resolution_choices_for_task(TASK_T2V),
                                value=format_size_markdown(TASK_T2V, DEFAULT_WIDTH, DEFAULT_HEIGHT),
                                interactive=False,
                                visible=False,
                                allow_custom_value=True,
                                elem_classes=["generation-control"],
                            )

                system_prompt = gr.Dropdown(
                    label="System Prompt",
                    choices=get_understanding_system_prompt_choices(TASK_X2T_VIDEO),
                    value=V2T_QA_SYSTEM_PROMPT,
                    visible=False,
                    allow_custom_value=True,
                )
                input_video = gr.Video(label="Input Video", visible=False, elem_classes=["lance-display-frame"])
                input_image = gr.Image(label="Input Image", type="filepath", visible=False, elem_classes=["lance-display-frame"])
                height = gr.Number(value=DEFAULT_HEIGHT, precision=0, visible=False)
                width = gr.Number(value=DEFAULT_WIDTH, precision=0, visible=False)

                with gr.Accordion("Advanced Parameters", open=False, elem_classes=["lance-advanced-accordion"]):
                    seed = gr.Number(label="Seed (-1 for random seed)", value=DEFAULT_BASIC_SEED, precision=0)
                    validation_num_timesteps = gr.Slider(
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=DEFAULT_TIMESTEPS,
                        label="Validation Num Timesteps",
                    )
                    with gr.Row():
                        validation_timestep_shift = gr.Number(label="Validation Timestep Shift", value=DEFAULT_TIMESTEP_SHIFT)
                        cfg_text_scale = gr.Number(label="CFG Text Scale", value=DEFAULT_CFG_TEXT_SCALE)

            with gr.Column(scale=1, elem_classes=["lance-main-column", "lance-output-column"]):
                with gr.Column(elem_classes=["lance-panel", "lance-output-panel"]):
                    output_label = gr.HTML(
                        build_lance_icon_label_html("Output Video", "video", "lance-output-label"),
                        elem_classes=["lance-label-html"],
                    )
                    output_video = gr.Video(label="Output Video", show_label=False, elem_classes=["lance-display-frame", "output-media-control"])
                    output_image = gr.Image(label="Output Image", show_label=False, type="filepath", visible=False, elem_classes=["lance-display-frame", "output-media-control"])
                    output_text = gr.Textbox(label="Output Text", show_label=False, lines=3, visible=False, elem_classes=["lance-display-frame", "output-text-control"])
                status = gr.Markdown("", elem_classes=["lance-run-status"])

        run_button = gr.Button("🚀 Generate", variant="primary", elem_classes=["lance-run-button"])
        def build_prompt_example_table(examples: list[list], media_type: Optional[str] = None):
            """Recommended example list with complete-fit reference media previews."""
            example_buttons = []
            with gr.Column(elem_classes=["prompt-example-full-table"]):
                for row in examples:
                    example_prompt = str(row[0]) if row else ""

                    preview_video_path = input_video_path = None
                    preview_image_path = input_image_path = None
                    if media_type == "video":
                        preview_video_path = str(row[1]) if len(row) > 1 and row[1] else None
                        input_video_path = str(row[2]) if len(row) > 2 and row[2] else preview_video_path
                    elif media_type == "image":
                        preview_image_path = str(row[3]) if len(row) > 3 and row[3] else (str(row[2]) if len(row) > 2 and row[2] else None)
                        input_image_path = str(row[4]) if len(row) > 4 and row[4] else preview_image_path

                    button_label = example_prompt if len(example_prompt) <= 360 else f"{example_prompt[:357]}..."

                    if media_type in {"video", "image"}:
                        with gr.Row(elem_classes=["prompt-example-multimodal-row"]):
                            with gr.Column(elem_classes=["prompt-example-prompt-cell"]):
                                example_button = gr.Button(
                                    button_label,
                                    variant="secondary",
                                    elem_classes=["prompt-example-row-button"],
                                )
                            with gr.Column(elem_classes=["prompt-example-media-cell"]):
                                if media_type == "video":
                                    gr.Video(
                                        value=preview_video_path,
                                        show_label=False,
                                        interactive=False,
                                        elem_classes=["prompt-example-media-html", "prompt-example-video"],
                                    )
                                else:
                                    gr.HTML(
                                        build_example_media_html(preview_image_path, "image"),
                                        elem_classes=["prompt-example-media-html"],
                                    )
                    else:
                        example_button = gr.Button(
                            button_label,
                            variant="secondary",
                            elem_classes=["prompt-example-row-button"],
                        )

                    example_buttons.append((example_button, example_prompt, input_video_path, input_image_path))
            return example_buttons

        def examples_section(title: str, examples: list[list], media_type: Optional[str] = None, visible: bool = False):
            with gr.Column(visible=visible, elem_classes=["lance-recommended-section"]) as group:
                gr.HTML(build_lance_label_html(title, "lance-section-label"), elem_classes=["lance-label-html"])
                with gr.Group(elem_classes=["example-panel", "prompt-examples"]):
                    buttons = build_prompt_example_table(examples, media_type=media_type)
            return group, buttons

        video_generation_examples_group, video_generation_example_buttons = examples_section(
            "Text-to-Video recommended cases", VIDEO_GENERATION_EXAMPLES, visible=True
        )
        image_to_video_examples_group, image_to_video_example_buttons = examples_section(
            "Image-to-Video recommended cases", IMAGE_TO_VIDEO_EXAMPLES, media_type="image"
        )
        video_edit_examples_group, video_edit_example_buttons = examples_section(
            "Video edit recommended cases", VIDEO_EDIT_EXAMPLES, media_type="video"
        )
        video_understanding_examples_group, video_understanding_example_buttons = examples_section(
            "Video understanding recommended cases", VIDEO_UNDERSTANDING_EXAMPLES, media_type="video"
        )
        image_generation_examples_group, image_generation_example_buttons = examples_section(
            "Text-to-Image recommended cases", IMAGE_GENERATION_EXAMPLES
        )
        image_edit_examples_group, image_edit_example_buttons = examples_section(
            "Image edit recommended cases", IMAGE_EDIT_EXAMPLES, media_type="image"
        )
        image_understanding_examples_group, image_understanding_example_buttons = examples_section(
            "Image understanding recommended cases", IMAGE_UNDERSTANDING_EXAMPLES, media_type="image"
        )

        task.change(
            fn=update_task_ui,
            inputs=[task],
            outputs=[
                prompt_label,
                prompt,
                system_prompt,
                input_video,
                input_image,
                aspect_ratio_row,
                output_resolution_row,
                video_duration_row,
                video_resolution_row,
                aspect_ratio,
                height,
                width,
                real_size,
                num_frames,
                resolution,
                output_label,
                output_video,
                output_image,
                output_text,
                video_generation_examples_group,
                image_to_video_examples_group,
                video_edit_examples_group,
                video_understanding_examples_group,
                image_generation_examples_group,
                image_edit_examples_group,
                image_understanding_examples_group,
            ],
        )

        aspect_ratio.change(
            fn=update_size_from_aspect_ratio,
            inputs=[task, aspect_ratio, resolution],
            outputs=[height, width, real_size],
            queue=False,
            show_api=False,
        )
        # real_size is hidden and derived from task/resolution/aspect_ratio.
        # Do not attach a .change handler here: dynamic Dropdown choices can briefly
        # contain 360p values while the selected value is 480p (or vice versa),
        # which makes Gradio reject the stale value during preprocessing.
        resolution.change(
            fn=update_output_resolution_from_video_profile,
            inputs=[task, aspect_ratio, resolution],
            outputs=[real_size, height, width],
            queue=False,
            show_api=False,
        )

        for example_button, example_prompt, _, _ in video_generation_example_buttons + image_generation_example_buttons:
            example_button.click(
                fn=make_prompt_example_click_handler(example_prompt),
                inputs=[task],
                outputs=[prompt, system_prompt, aspect_ratio, height, width, num_frames, resolution, real_size],
                queue=False,
                show_api=False,
            )

        for example_button, example_prompt, example_video, example_image in (
            image_to_video_example_buttons
            + video_edit_example_buttons
            + video_understanding_example_buttons
            + image_edit_example_buttons
            + image_understanding_example_buttons
        ):
            example_button.click(
                fn=make_media_prompt_example_click_handler(example_prompt, example_video, example_image),
                inputs=[task],
                outputs=[prompt, input_video, input_image, system_prompt, aspect_ratio, height, width, num_frames, resolution, real_size],
                queue=False,
                show_api=False,
            )

        run_button.click(
            fn=build_running_status_markdown,
            inputs=[],
            outputs=[status],
            queue=False,
            show_api=False,
        ).then(
            fn=run_task_fn,
            inputs=[
                task,
                prompt,
                system_prompt,
                input_video,
                input_image,
                height,
                width,
                num_frames,
                seed,
                resolution,
                validation_num_timesteps,
                validation_timestep_shift,
                cfg_text_scale,
            ],
            outputs=[output_video, output_image, output_text, status],
            show_progress="minimal",
        )

    return demo
