#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import logging
from configs.settings import LOG_DIR


class LogManager:
    """
    日志文件会被存放到指定的日志根目录下，根据当前日期创建文件夹，
    每个日志文件以当前小时为命名单位，而日志记录的时间格式只显示到分钟。
    """

    def __init__(self, log_directory: str = LOG_DIR):
        """
        :param log_directory: 日志存放的根目录
        """
        self.log_directory = log_directory
        if not os.path.exists(self.log_directory):
            os.makedirs(self.log_directory)
        self.logger = logging.getLogger("LogManager")
        self.logger.setLevel(logging.DEBUG)
        # 将日志记录时间格式调整为到分钟级别
        self.formatter = logging.Formatter(
            "[%(asctime)s] [%(filename)s|%(funcName)s] [line:%(lineno)d] %(levelname)-8s: %(message)s",
            datefmt="%Y-%m-%d %H:%M"  # 调整为到分钟精度
        )

    def _create_file_handler(self) -> logging.FileHandler:
        """
        根据当前日期和小时创建日志文件处理器。
        """
        # 当前日期
        date_str = time.strftime("%Y-%m-%d")
        # 当前小时（两位数）
        hour_str = time.strftime("%H")
        # 创建日期文件夹
        date_folder = os.path.join(self.log_directory, date_str)
        os.makedirs(date_folder, exist_ok=True)
        # 文件名为 日期+小时.log，例如 2025-04-09_10.log
        log_filename = f"{date_str}_{hour_str}.log"
        log_file = os.path.join(date_folder, log_filename)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(self.formatter)
        return file_handler

    def _create_console_handler(self) -> logging.StreamHandler:
        """
        创建控制台日志处理器。
        """
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(self.formatter)
        return console_handler

    def _log(self, level: str, message: str) -> None:
        """
        :param level: 日志级别 ("debug"、"info"、"warning"、"error")
        :param message: 要记录的日志消息
        """
        file_handler = self._create_file_handler()
        console_handler = self._create_console_handler()

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        if level == "debug":
            self.logger.debug(message)
        elif level == "info":
            self.logger.info(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)

        # 记录完后移除处理器
        self.logger.removeHandler(file_handler)
        self.logger.removeHandler(console_handler)
        file_handler.close()

    def debug(self, message: str) -> None:
        """记录 DEBUG 日志"""
        self._log("debug", message)

    def info(self, message: str) -> None:
        """记录 INFO 日志"""
        self._log("info", message)

    def warning(self, message: str) -> None:
        """记录 WARNING 日志"""
        self._log("warning", message)

    def error(self, message: str) -> None:
        """记录 ERROR 日志"""
        self._log("error", message)


if __name__ == "__main__":
    logger_instance = LogManager()
    logger_instance.debug("这是一条调试信息")
    logger_instance.info("这是一条普通信息")
    logger_instance.warning("这是一条警告信息")
    logger_instance.error("这是一条错误信息")
