# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
from typing import Any, Dict, List, Tuple
from jinja2 import Environment, BaseLoader


JINJA_PROMPT_TMPL = (
    "<|im_start|>system\n"
    "{{ system_prompt }}<|im_end|>\n"
    "{% for m in msgs -%}"
    "<|im_start|>{{ m.role }}\n"
    "{% if not (m.role == 'assistant' and not include_assistant_content) -%}"
    "{{ m.content | render_mm_list }}"
    "{% endif -%}"
    "{% if (not (loop.last and m.role == 'assistant')) or include_assistant_content -%}"
    "<|im_end|>\n"
    "{% endif -%}"
    "{% endfor -%}"
)

VS, VE = "<|vision_start|>", "<|vision_end|>"
VP, IP = "<|video_pad|>", "<|image_pad|>"

def expand_and_index_by_token_ids_new(
    rendered_text: str,
    tokens: List[int],
    tokenizer,
    target_text: str = "",
    search_text: str = "",
) -> Tuple[str, List[int], List[List[int]], List[int]]:
    """
    Returns:
      new_rendered_text: expanded text
      all_token_id     : token ids of new_rendered_text
      spans_index      : indexes of each pad block in all_token_id, in occurrence order, e.g. [[100..199], [350..549], ...]
      tgt_index        : indexes of target_text in all_token_id, or [] if not found
    """
    vs_ids = tokenizer(VS, add_special_tokens=False)["input_ids"]
    ve_ids = tokenizer(VE, add_special_tokens=False)["input_ids"]
    vp_ids = tokenizer(VP, add_special_tokens=False)["input_ids"]
    ip_ids = tokenizer(IP, add_special_tokens=False)["input_ids"]

    enc = tokenizer(rendered_text, add_special_tokens=False)
    base_ids = enc["input_ids"]

    # ---------- 1) Scan VP/IP in occurrence order and expand each to K copies, recording pad block metadata ----------
    # find all VS positions and pair them with nearest VE after each VS

    all_ids: List[int] = []
    spans_index: List[List[int]] = []

    i = 0               # Scan pointer for base_ids.
    tk_ptr = 0          # Pointer for tokens(K).

    while True:
        try:
            vs_positions_ = base_ids[i:].index(vs_ids[0]) + i
        except:
            all_ids.extend(base_ids[i:])
            break
        all_ids.extend(base_ids[i: vs_positions_])
        i = vs_positions_ + 3

        # Expand the sequence and insert placeholder ids into pad_ids.
        pad_ids = base_ids[vs_positions_ + 1:vs_positions_ + 2]
        K = int(tokens[tk_ptr])
        start, end = len(all_ids) + 1, len(all_ids) + 1 + K
        all_ids.extend(vs_ids + pad_ids * K + ve_ids)
        tk_ptr += 1

        # Collect indexes of each pad token block in all_token_id, in occurrence order, e.g. [[100..199], [350..549], ...].
        #start, end = vs_positions_ + 1, vs_positions_ + 1 + K
        spans_index.append(list(range(start, end)))

    tgt_index: List[int] = []
    if target_text:
        tgt_ids_identify = tokenizer(target_text, add_special_tokens=False)["input_ids"]
        i = 0               # Scan pointer for base_ids.

        while i < len(all_ids):
            tgt_positions_ = all_ids[i:].index(tgt_ids_identify[0]) + i
            if all_ids[tgt_positions_+len(tgt_ids_identify)-1] == tgt_ids_identify[-1]:
                tgt_index = list(range(tgt_positions_+len(tgt_ids_identify), len(all_ids)))
                break
            else:
                i = tgt_positions_ + 1

    search_index: List[int] = []
    if search_text:
        search_ids_identify = tokenizer(search_text, add_special_tokens=False)["input_ids"]
        i = 0               # Scan pointer for base_ids.

        while i < len(all_ids):
            search_positions_ = all_ids[i:].index(search_ids_identify[0]) + i
            if all_ids[search_positions_:search_positions_+len(search_ids_identify)] == search_ids_identify:
                search_index = list(range(search_positions_, search_positions_+len(search_ids_identify)))
                break
            else:
                i = search_positions_ + 1


    return all_ids, spans_index, tgt_index, search_index

def _extract_system_prompt(messages: List[Dict[str, Any]], default_system: str) -> str:
    for m in messages:
        if m.get("role") == "system":
            c = m.get("content", "")
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                texts = [it.get("text", "") for it in c if isinstance(it, dict) and it.get("type") == "text"]
                if texts:
                    return "".join(texts)
    return default_system


def _normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    norm: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            continue
        c = m.get("content", "")
        if isinstance(c, str):
            items = [{"type": "text", "text": c}]
        elif isinstance(c, list):
            items = c
        else:
            items = []
        norm.append({"role": role, "content": items})
    return norm


def render_qwenvl_prompt(
    messages: List[Dict[str, Any]],
    default_system: str = "You are a helpful assistant.",
    include_assistant_content: bool = False,  # Key option: whether to render assistant text.
    force_video_pad: bool = False,
) -> str:
    system_prompt = _extract_system_prompt(messages, default_system)
    msgs = _normalize_messages(messages)

    def _render_mm_list(items: Any) -> str:
        if isinstance(items, str):
            return items
        if not isinstance(items, list):
            return ""
        parts: List[str] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            t = it.get("type")
            if t == "text":
                parts.append(it.get("text", ""))
            elif t == "image":
                if force_video_pad:
                    parts.append("<|vision_start|><|image_pad|><|vision_end|>")
                else:
                    parts.append("<|vision_start|><|video_pad|><|vision_end|>")
            elif t == "video":
                parts.append("<|vision_start|><|video_pad|><|vision_end|>")
            # Other modalities can be added here.
        return "".join(parts)

    env = Environment(
        loader=BaseLoader(),
        autoescape=False,
        trim_blocks=True,  # Remove newlines after block endings.
        lstrip_blocks=True,  # Remove whitespace before block starts.
        newline_sequence="\n",
        keep_trailing_newline=False,
    )
    env.filters["render_mm_list"] = _render_mm_list
    template = env.from_string(JINJA_PROMPT_TMPL)

    return template.render(
        system_prompt=system_prompt,
        msgs=msgs,
        include_assistant_content=include_assistant_content,
    )
