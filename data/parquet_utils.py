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
# coding: utf-8

"""
Generic parquet/data file helpers that do not depend on cluster-specific settings.
"""

import os
from typing import Callable, Iterable, List, Optional

import pyarrow.fs as pf


SUPPORTED_DATA_EXTENSIONS = (".parquet", ".json", ".jsonl")


def select_supported_files(file_paths: Iterable[str]) -> List[str]:
    file_paths = list(file_paths)
    parquet_files = [path for path in file_paths if path.endswith(".parquet")]
    if parquet_files:
        return parquet_files
    return [path for path in file_paths if path.endswith((".json", ".jsonl"))]


def list_local_supported_data_files(data_dir: str) -> List[str]:
    if os.path.isfile(data_dir):
        if not data_dir.endswith(SUPPORTED_DATA_EXTENSIONS):
            raise ValueError(f"Unsupported data file: {data_dir}")
        return [data_dir]

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data path does not exist: {data_dir}")

    files = [
        os.path.join(data_dir, name)
        for name in os.listdir(data_dir)
        if name.endswith(SUPPORTED_DATA_EXTENSIONS)
    ]
    return select_supported_files(files)


def init_arrow_fs(parquet_file_path: str, remote_fs_factory: Optional[Callable[[], pf.FileSystem]] = None) -> pf.FileSystem:
    if parquet_file_path.startswith("hdfs://"):
        if remote_fs_factory is None:
            raise ValueError(f"Remote parquet path requires a filesystem factory: {parquet_file_path}")
        return remote_fs_factory()
    return pf.LocalFileSystem()


def init_arrow_pf_fs(parquet_file_path: str) -> pf.FileSystem:
    from data.internal_data_utils.parquet_utils import init_arrow_pf_fs as _init_arrow_pf_fs

    return _init_arrow_pf_fs(parquet_file_path)


def get_parquet_data_paths_balanced(data_dir_list, rank=0, world_size=1, num_repeat=1):
    from data.internal_data_utils.parquet_utils import get_parquet_data_paths_balanced as _get_paths

    return _get_paths(data_dir_list=data_dir_list, rank=rank, world_size=world_size, num_repeat=num_repeat)


def read_parquet_rows(parquet_file, row_group_id: int, columns: Optional[List[str]] = None):
    return parquet_file.read_row_group(row_group_id, columns=columns).to_pylist()
