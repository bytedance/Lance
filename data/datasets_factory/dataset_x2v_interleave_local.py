# coding: utf-8

import os
import pyarrow.parquet as pq
import torch

from data.parquet_utils import init_arrow_fs, read_parquet_rows
from data.datasets_factory.dataset_x2v_interleave import X2VInterleaveIterableDataset


class X2VInterleaveLocalIterableDataset(X2VInterleaveIterableDataset):
    def __init__(self, *args, **kwargs):
        # Debug/local smoke-test dataset: read local parquet rows with raw image/video bytes.
        kwargs.setdefault("raw_bytes_input", True)
        super().__init__(*args, **kwargs)
        # Debug-only option for quick smoke tests on tiny local datasets:
        # repeat the same parquet file list to avoid very short epochs / empty shards.
        self.debug_parquet_repeat = max(1, int(kwargs.get("debug_parquet_repeat", 1)))
        self.data_paths = (kwargs.get("all_data_paths") or self.data_dir_list) * self.debug_parquet_repeat
        self._validate_local_data_paths()

    def _validate_local_data_paths(self):
        for path in self.data_paths:
            if str(path).startswith(("hdfs://", "http://", "https://")):
                raise ValueError(
                    f"{self.__class__.__name__} only supports local parquet files with embedded bytes, got: {path}"
                )
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Local parquet file not found: {path}")

    def _validate_local_row(self, row):
        dataset_type_to_required_keys = {
            "text2image_general": {"caption", "image_bytes"},
            "text2video_general": {"caption", "video_bytes"},
            "image2image": {"caption", "input_image_bytes", "output_image_bytes"},
            "video2video": {"caption", "input_video_bytes", "output_video_bytes"},
        }
        required_keys = dataset_type_to_required_keys.get(self.dataset_type)
        if required_keys is None:
            raise ValueError(
                f"{self.__class__.__name__} only supports local byte dataset types "
                f"{sorted(dataset_type_to_required_keys)}, got {self.dataset_type}"
            )

        missing = sorted(key for key in required_keys if key not in row)
        if missing:
            raise ValueError(
                f"{self.__class__.__name__} expects dataset_type={self.dataset_type} rows to contain keys {sorted(required_keys)}, "
                f"missing {missing}"
            )

    def set_epoch(self, seed=42):
        if not self.data_paths:
            return

        data_paths = sorted(self.data_paths)
        self.rng.seed(seed)
        self.rng.shuffle(data_paths)
        self.data_paths_per_rank = data_paths
        self.num_files_per_rank = len(data_paths)

    def __iter__(self):
        if not hasattr(self, "data_paths_per_rank"):
            self.set_epoch(self.seed)

        info = torch.utils.data.get_worker_info()
        worker_id = 0 if info is None else info.id
        num_workers = 1 if info is None else info.num_workers
        global_worker_id = self.local_rank * num_workers + worker_id
        global_num_workers = max(1, self.world_size * num_workers)

        while True:
            for parquet_idx, parquet_file_path in enumerate(self.data_paths_per_rank):
                fs = init_arrow_fs(parquet_file_path)
                with fs.open_input_file(parquet_file_path) as f:
                    parquet_file = pq.ParquetFile(f)
                    for row_group_id in range(parquet_file.num_row_groups):
                        rows = read_parquet_rows(parquet_file, row_group_id)
                        for row_idx, row in enumerate(rows):
                            if row_idx % global_num_workers != global_worker_id:
                                continue
                            try:
                                self._validate_local_row(row)
                            except Exception as e:
                                self.logger.warning(
                                    f"Invalid local x2v row in rg#{row_group_id} row#{row_idx} {parquet_file_path}: {e}"
                                )
                                continue
                            sample = self._process_row(
                                row=row,
                                parquet_idx=parquet_idx,
                                row_group_id=row_group_id,
                                row_idx=row_idx,
                                worker_id=worker_id,
                                parquet_file_path=parquet_file_path,
                            )
                            if sample:
                                yield sample

            self.logger.info(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")
