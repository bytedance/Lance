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

"""
Logging utility functions.
"""

import logging
import sys
from typing import Optional

from .distributed import get_global_rank, get_local_rank, get_world_size


def get_logger(name: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Avoid duplicate records from parent loggers.

    if not logger.handlers:  # Only inspect this logger, not ancestors.
        h = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "[%(asctime)s] "
            + (f"[Rank:{get_global_rank()}]" if get_world_size() > 1 else "")
            + (f"[LocalRank:{get_local_rank()}]" if get_world_size() > 1 else "")
            + "[%(pathname)s:%(lineno)d][%(threadName).12s][%(name)s][%(levelname).5s] %(message)s"
        )
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


