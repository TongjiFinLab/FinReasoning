import argparse
import os
import sys
import logging
from datetime import datetime


sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

try:
    from src import Evaluate_Alignment, Evaluate_Consistency, Evaluate_Depth
    from src.utils import setup_logger, get_openai_client
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure you are running this script from the project root and 'src' directory exists.")
    sys.exit(1)

def check_llm_connection(config, logger):
    """测试 LLM 连接是否正常"""
    logger.info(f"正在测试 LLM 连接 (Model: {config['model']})...")
    try:
        client = get_openai_client(api_key=config['api_key'], base_url=config['base_url'])
        # 尝试发送一个极简请求
        client.chat.completions.create(
            model=config['model'],
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=1
        )
        logger.info("✓ LLM 连接测试成功")
        return True
    except Exception as e:
        logger.error(f"✗ LLM 连接测试失败: {e}")
        logger.error("请检查 api_key, base_url 和 model 配置是否正确。")
        return False

def main():
    parser = argparse.ArgumentParser(description="FinReasoning 金融推理评测框架")
    
    # 通用参数
    parser.add_argument('--task', type=str, required=True, 
                        choices=['alignment', 'consistency', 'depth', 'all'],
                        help="选择要运行的评测任务 (alignment, consistency, depth, all)")
    parser.add_argument('--input-path', type=str, default=None,
                        help="输入JSON/JSONL文件或目录路径。如果未提供，将搜索默认目录 (data/Alignment/, data/Consistency/, data/Depth/)")
    parser.add_argument('--output-dir', type=str, default='eval_results',
                        help="结果保存目录 (默认: eval_results)")
    # 测试模式参数
    parser.add_argument('--test-run', action='store_true',
                        help="测试运行模式：每个任务只处理第一条数据，用于快速验证环境和代码")    

    args = parser.parse_args()
    
    # 设置全局日志
    logger = setup_logger("FinReasoning_Main")

    # ================= LLM 配置 =================
    # 请在此处直接修改 LLM 相关配置
    LLM_SETTINGS = {
        'model': '', #评测模型
        'api_key': '', # 建议填写您的 API Key
        'base_url': '', # 建议填写您的 Base URL
        'judge_model': '' # 裁判模型
    }
    # ===============================================
    
    # 准备配置字典
    config = {
        'input_path': args.input_path,
        'output_dir': args.output_dir,
        'model': LLM_SETTINGS['model'],
        'api_key': LLM_SETTINGS['api_key'],
        'base_url': LLM_SETTINGS['base_url'],
        'judge_model': LLM_SETTINGS['judge_model']
    }

    # 执行 LLM 连接测试
    if not check_llm_connection(config, logger):
        sys.exit(1)

    # 如果是测试运行模式，设置相关参数限制数据量
    if args.test_run:
        logger.info("启动测试运行模式 (--test-run)")
        logger.info("将仅处理每个任务的第1条数据...")
        config['max_qa'] = 1  # 限制处理数量为1
    
    tasks_to_run = []
    if args.task == 'all':
        tasks_to_run = ['alignment', 'consistency', 'depth']
    else:
        tasks_to_run = [args.task]

    # 生成本次运行的时间戳
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for task in tasks_to_run:
        logger.info(f"开始执行任务: {task.upper()}")
        
        # 构建层次化输出路径: output_dir / timestamp / task / model
        sanitized_model = config['model'].replace('/', '_').replace('\\', '_')
        task_output_dir = os.path.join(config['output_dir'], run_timestamp, task, sanitized_model)
        
        # 创建任务专用配置
        task_config = config.copy()
        task_config['output_dir'] = task_output_dir
        
        try:
            if task == 'alignment':
                Evaluate_Alignment.run(task_config)
            elif task == 'consistency':
                Evaluate_Consistency.run(task_config)
            elif task == 'depth':
                Evaluate_Depth.run(task_config)
        except Exception as e:
            logger.error(f"任务 {task} 执行失败: {e}", exc_info=True)
            
    logger.info("所有请求的任务已完成。")

if __name__ == "__main__":
    main()