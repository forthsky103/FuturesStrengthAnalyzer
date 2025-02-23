# src/logging_utils.py
import logging
import os

def setup_logging(log_file_path: str = "results/app.log", level: int = logging.INFO):
    """
    配置日志，输出到文件和控制台。
    
    Args:
        log_file_path (str): 日志文件路径，默认为 "results/app.log"。
        level (int): 日志级别，默认为 INFO。
    """
    # 确保日志目录存在
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # 创建日志格式
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)

    # 创建文件处理器
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # 配置全局日志
    logging.basicConfig(level=level, handlers=[file_handler, console_handler])

    logging.info("日志配置完成")