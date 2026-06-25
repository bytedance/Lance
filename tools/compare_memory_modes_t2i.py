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

"""Compare text-to-image outputs from parallel and relay memory modes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path


RESOLUTION_SIZES = {
    "image_256res": (256, 256),
    "image_512res": (512, 512),
    "image_768res": (768, 768),
}
DEFAULT_PROMPT = 'A cat holds a poster with rainbow text "STOP"'
MODES = ("parallel", "relay")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="downloads/Lance_3B", help="Path to Lance T2I model weights.")
    parser.add_argument(
        "--resolutions",
        nargs="+",
        default=["image_256res"],
        choices=sorted(RESOLUTION_SIZES),
        help="T2I resolution presets to compare.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt used for both memory modes.")
    parser.add_argument("--output-root", default="results/memory_mode_compare_t2i", help="Where outputs and logs are written.")
    parser.add_argument("--timesteps", type=int, default=30, help="Validation denoising steps.")
    parser.add_argument("--timestep-shift", type=float, default=3.5, help="Validation timestep shift.")
    parser.add_argument("--seed", type=int, default=42, help="Validation data seed.")
    parser.add_argument("--cfg-text-scale", type=float, default=4.0, help="Text CFG scale.")
    parser.add_argument("--shell", default="bash", help="Shell executable used to run inference_lance.sh.")
    parser.add_argument("--relay-memory-log", action="store_true", help="Forward --RELAY_MEMORY_LOG true to inference.")
    parser.add_argument("--reuse-existing", action="store_true", help="Hash existing outputs instead of rerunning finished modes.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running inference.")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_prompt_config(output_root: Path, prompt: str) -> Path:
    config_dir = output_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "t2i_prompt.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump({"000000.png": prompt}, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return config_path


def output_png(save_dir: Path) -> Path:
    candidate = save_dir / "000000.png"
    if candidate.exists():
        return candidate
    pngs = sorted(save_dir.glob("*.png"))
    if pngs:
        return pngs[0]
    raise FileNotFoundError(f"No PNG output found in {save_dir}")


def build_command(args: argparse.Namespace, root: Path, config_path: Path, resolution: str, mode: str, save_dir: Path) -> list[str]:
    height, width = RESOLUTION_SIZES[resolution]
    return [
        args.shell,
        str(root / "inference_lance.sh"),
        "--TASK_NAME",
        "t2i",
        "--MODEL_PATH",
        args.model_path,
        "--CONFIG_PATH",
        str(config_path),
        "--RESOLUTION",
        resolution,
        "--VIDEO_HEIGHT",
        str(height),
        "--VIDEO_WIDTH",
        str(width),
        "--NUM_FRAMES",
        "1",
        "--VALIDATION_NUM_TIMESTEPS",
        str(args.timesteps),
        "--VALIDATION_TIMESTEP_SHIFT",
        str(args.timestep_shift),
        "--VALIDATION_DATA_SEED",
        str(args.seed),
        "--CFG_TEXT_SCALE",
        str(args.cfg_text_scale),
        "--USE_KVCACHE",
        "true",
        "--MEMORY_MODE",
        mode,
        "--RELAY_MEMORY_LOG",
        str(args.relay_memory_log).lower(),
        "--SAVE_PATH_GEN",
        str(save_dir),
    ]


def run_mode(args: argparse.Namespace, root: Path, config_path: Path, output_root: Path, resolution: str, mode: str) -> dict:
    save_dir = output_root / f"{resolution}_{mode}"
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{resolution}_{mode}.log"
    cmd = build_command(args, root, config_path, resolution, mode, save_dir)
    printable_cmd = " ".join(shlex.quote(part) for part in cmd)

    if args.dry_run:
        print(printable_cmd)
        return {
            "resolution": resolution,
            "mode": mode,
            "exit_code": "",
            "seconds": "",
            "sha256": "",
            "path": str(save_dir / "000000.png"),
            "log": str(log_path),
        }

    start = time.perf_counter()
    if args.reuse_existing and (save_dir / "000000.png").exists():
        exit_code = 0
    else:
        save_dir.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write(f"$ {printable_cmd}\n\n")
            log_file.flush()
            result = subprocess.run(cmd, cwd=root, stdout=log_file, stderr=subprocess.STDOUT, text=True, check=False)
        exit_code = result.returncode

    seconds = time.perf_counter() - start
    image_path = output_png(save_dir) if exit_code == 0 else save_dir / "000000.png"
    image_hash = sha256_file(image_path) if image_path.exists() else ""

    return {
        "resolution": resolution,
        "mode": mode,
        "exit_code": str(exit_code),
        "seconds": f"{seconds:.2f}",
        "sha256": image_hash,
        "path": str(image_path),
        "log": str(log_path),
    }


def write_summary(output_root: Path, rows: list[dict]) -> None:
    csv_path = output_root / "summary.csv"
    fieldnames = ["resolution", "mode", "exit_code", "seconds", "sha256", "path", "log"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_path = output_root / "summary.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("| Resolution | Mode | Exit code | Seconds | SHA256 |\n")
        handle.write("| --- | --- | ---: | ---: | --- |\n")
        for row in rows:
            handle.write(
                f"| {row['resolution']} | {row['mode']} | {row['exit_code']} | {row['seconds']} | {row['sha256']} |\n"
            )


def compare_rows(rows: list[dict]) -> int:
    exit_code = 0
    by_resolution: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_resolution.setdefault(row["resolution"], {})[row["mode"]] = row

    for resolution, modes in by_resolution.items():
        parallel = modes.get("parallel")
        relay = modes.get("relay")
        if parallel is None or relay is None:
            print(f"{resolution}: missing comparison row", file=sys.stderr)
            exit_code = 2
            continue
        if parallel["exit_code"] != "0" or relay["exit_code"] != "0":
            print(f"{resolution}: inference failed; see logs", file=sys.stderr)
            exit_code = 1
            continue
        if not parallel["sha256"] or not relay["sha256"]:
            print(f"{resolution}: missing output hash", file=sys.stderr)
            exit_code = 2
            continue
        if parallel["sha256"] != relay["sha256"]:
            print(f"{resolution}: hash mismatch", file=sys.stderr)
            exit_code = 2
            continue
        print(f"{resolution}: bit-identical")

    return exit_code


def main() -> int:
    args = parse_args()
    root = repo_root()
    output_root = (root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config_path = write_prompt_config(output_root, args.prompt)

    rows = []
    for resolution in args.resolutions:
        for mode in MODES:
            rows.append(run_mode(args, root, config_path, output_root, resolution, mode))

    write_summary(output_root, rows)
    if args.dry_run:
        print(f"Dry run complete. Summary scaffold: {output_root}")
        return 0
    return compare_rows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
