import os
import logging
from typing import Optional
from openai import OpenAI

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FinReasoning")

def get_openai_client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> OpenAI:
    """
    获取统一的OpenAI客户端
    """
    # 优先使用传入的参数，否则使用环境变量
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.zhizengzeng.com/v1/")
    
    if not api_key:
        logger.warning("未检测到OPENAI_API_KEY，请确保已设置环境变量或通过参数传入")
    
    return OpenAI(api_key=api_key, base_url=base_url)

def setup_logger(name: str, log_file: Optional[str] = None):
    """
    设置logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 防止重复添加handler
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # 控制台Handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 文件Handler
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
    return logger
