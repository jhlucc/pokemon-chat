#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import logging
import contextvars
from typing import Optional
from src.core.settings import settings


request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover
        record.request_id = request_id_var.get()
        return True


class HourlyFileHandler(logging.Handler):
    """
    自定义 Handler，将日志按 日期/小时.log 的结构存储
    """
    def __init__(self, log_directory: str):
        super().__init__()
        self.log_directory = log_directory
        self.current_key = None
        self.file_handler: Optional[logging.FileHandler] = None
        # 定义 Formatter
        self.formatter = logging.Formatter(
            "[%(asctime)s] [req:%(request_id)s] [%(filename)s|%(funcName)s] [line:%(lineno)d] %(levelname)-8s: %(message)s",
            datefmt="%Y-%m-%d %H:%M"
        )
        self.setFormatter(self.formatter)

    def emit(self, record):
        try:
            # 根据当前时间决定写入哪个文件
            ts = time.time()
            date_str = time.strftime("%Y-%m-%d", time.localtime(ts))
            hour_str = time.strftime("%H", time.localtime(ts))
            
            key = f"{date_str}_{hour_str}"
            
            if self.current_key != key or self.file_handler is None:
                self._rotate(date_str, hour_str, key)
            
            if self.file_handler:
                self.file_handler.emit(record)
        except Exception:
            self.handleError(record)

    def _rotate(self, date_str, hour_str, key):
        """切换日志文件"""
        if self.file_handler:
            self.file_handler.close()
            
        date_folder = os.path.join(self.log_directory, date_str)
        os.makedirs(date_folder, exist_ok=True)
        
        log_filename = f"{date_str}_{hour_str}.log"
        log_file = os.path.join(date_folder, log_filename)
        
        self.file_handler = logging.FileHandler(log_file, encoding="utf-8")
        self.file_handler.setFormatter(self.formatter)
        self.current_key = key

    def close(self):
        if self.file_handler:
            self.file_handler.close()
        super().close()


_setup_done = False

def setup_global_logging():
    """初始化全局日志配置"""
    global _setup_done
    if _setup_done:
        return
        
    root_logger = logging.getLogger()
    # 默认级别
    root_logger.setLevel(logging.DEBUG)
    
    # 清除旧的 handlers (避免重复)
    if root_logger.handlers:
        for h in root_logger.handlers[:]:
            root_logger.removeHandler(h)
    
    log_dir = str(settings.paths.log_dir)
    if not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir)
        except Exception:
            pass
            
    # 1. 所有的日志都输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "[%(asctime)s] [req:%(request_id)s] [%(filename)s|%(funcName)s] [line:%(lineno)d] %(levelname)-8s: %(message)s",
        datefmt="%Y-%m-%d %H:%M"
    )
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 2. 文件日志 (按小时轮转)
    file_handler = HourlyFileHandler(log_dir)
    file_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)

    # Ensure every record has request_id (even when not in a request context).
    root_logger.addFilter(RequestIdFilter())
    
    # 第三方库日志降噪
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("neo4j").setLevel(logging.WARNING)
    
    _setup_done = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    获取 Logger 实例
    推荐使用: logger = get_logger(__name__)
    """
    setup_global_logging()
    return logging.getLogger(name)


class LogManager:
    """
    [Deprecated] 兼容旧代码的 LogManager。
    请改用 get_logger(__name__)。
    """
    def __init__(self, log_directory: str = None):
        setup_global_logging()
        # 保持旧行为，使用 "LogManager" 这个名字，
        # 或者使用调用者的名字？旧代码没法自动获取调用者名字，除非 inspect。
        # 这里统一用 LogManager，虽然会丢失 info，但兼容性最重要。
        self.logger = logging.getLogger("LogManager")
        self.logger.setLevel(logging.DEBUG)

    def debug(self, message: str) -> None:
        self.logger.debug(message, stacklevel=2)

    def info(self, message: str) -> None:
        self.logger.info(message, stacklevel=2)

    def warning(self, message: str) -> None:
        self.logger.warning(message, stacklevel=2)

    def error(self, message: str) -> None:
        self.logger.error(message, stacklevel=2)


if __name__ == "__main__":
    # Test new factory
    log = get_logger("TestModule")
    log.info("Testing get_logger")
    
    # Test legacy
    legacy = LogManager()
    legacy.info("Testing legacy LogManager")
