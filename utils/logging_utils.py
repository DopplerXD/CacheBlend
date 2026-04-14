"""日志工具：统一项目日志格式与级别。"""

import logging
import os
from typing import Optional


_DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """创建并返回一个带统一格式的 logger。

    说明：
    1. 只在首次创建时绑定 StreamHandler，避免重复打印。
    2. 级别默认读取环境变量 LOG_LEVEL，否则使用 INFO。
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT,
                                               datefmt=_DEFAULT_DATEFMT))
        logger.addHandler(handler)

    resolved_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    logger.setLevel(getattr(logging, resolved_level, logging.INFO))
    logger.propagate = False
    return logger
