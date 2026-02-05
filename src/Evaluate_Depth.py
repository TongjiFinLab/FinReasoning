import os
import json
import time
from tqdm import tqdm
from .utils import get_openai_client, setup_logger

logger = setup_logger("Depth_Evaluator")

# 各维度的评分标准
SCORING_RUBRICS = {
    "论证合理性与因果深度": "逻辑链条(40分):A→B→C闭环; 因果深度(30分):解释必然性; 专业性(30分):CFA规范。",
    "事实准确性与情境化": "事实密度(40分):引用数值; 计算严谨性(30分):勾稽关系; 情境分析(30分):行业背景含义。",
    "完整性与比较分析": "批判性思维(40分):识别边界/风险; 比较视角(30分):对冲分析; 论点平衡性(30分):客观中立。",
    "结构丰富度与严谨性": "框架完整性(40分):分析维度覆盖; 逻辑层次(30分):总分总/嵌套; 表达颗粒度(30分):概念严谨。"
}

class DepthEvaluator:
    def __init__(self, api_key, base_url, target_model="gpt-4o", judge_model="deepseek-chat"):
        self.client = get_openai_client(api_key, base_url)
        self.target_model = target_model
        # 深度评测通常需要更强的模型作为裁判，这里默认deepseek-chat或由外部指定
        self.judge_model = judge_model
        
    def get_model_response(self, model_name, evidence, question):
        """调用被测模型"""
        prompt = f"【背景事实】\n{evidence}\n\n【分析问题】\n{question}"
        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "你是一名专业的金融分析师。请根据提供的背景事实回答问题。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling target model: {e}")
            return f"Error: {e}"

    def judge_answer(self, category, question, gold_answer, model_answer):
        """裁判模型评分逻辑"""
        rubric = SCORING_RUBRICS.get(category, "综合逻辑、准确性和专业性评分。")
        
        judge_prompt = f"""
你是一名严苛的金融评测专家。请根据提供的评分标准，对比【标准答案】对【待测回答】进行评分。

【评分标准】
{rubric}

【题目信息】
问题：{question}
标准答案：{gold_answer}

【待测回答】
{model_answer}

请直接输出JSON格式：
{{
  "score": 0-100的数字,
  "reason": "简短的扣分理由"
}}
"""
        try:
            response = self.client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": judge_prompt}],
                response_format={"type": "json_object"},
                temperature=0
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Error calling judge model: {e}")
            return {"score": 0, "reason": "评分调用失败"}

    def run_evaluation(self, input_dir, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 支持目录或单个文件
        if os.path.isdir(input_dir):
            all_categories_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
            input_base = input_dir
        else:
            if not os.path.exists(input_dir):
                logger.error(f"Input path not found: {input_dir}")
                return
            input_base = os.path.dirname(input_dir)
            all_categories_files = [os.path.basename(input_dir)]

        if not all_categories_files:
            logger.warning(f"No JSON files found in {input_dir}")
            return

        for filename in all_categories_files:
            logger.info(f"正在评测维度: {filename}")
            input_path = os.path.join(input_base, filename)
            
            try:
                with open(input_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                logger.error(f"无法读取文件 {input_path}: {e}")
                continue

            results = []
            category_total_score = 0
            
            for item in tqdm(data, desc=f"Evaluating {filename}"):
                # 1. 模型推理
                model_out = self.get_model_response(self.target_model, item.get('evidence', ''), item['question'])
                
                # 2. 裁判评分
                # 某些数据可能没有category字段，使用文件名作为fallback
                category = item.get('category', filename.replace('.json', ''))
                eval_res = self.judge_answer(category, item['question'], item['answer'], model_out)
                
                # 3. 记录结果
                res_item = {
                    "qa_id": item.get('qa_id', 'unknown'),
                    "source": item.get('source', 'unknown'),
                    "question": item['question'],
                    "gold_answer": item['answer'],
                    "model_answer": model_out,
                    "score": eval_res.get("score", 0),
                    "reason": eval_res.get("reason", "")
                }
                results.append(res_item)
                category_total_score += float(eval_res.get("score", 0))

            # 保存该维度的评测结果
            avg_score = category_total_score / len(data) if data else 0
            output_filename = f"result_{self.target_model}_{filename}"
            output_path = os.path.join(output_dir, output_filename)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "model": self.target_model,
                    "dimension": filename,
                    "average_score": avg_score,
                    "details": results
                }, f, ensure_ascii=False, indent=4)
            
            logger.info(f"维度 {filename} 评测完成，平均分: {avg_score:.2f}, 结果保存至 {output_path}")

def run(config):
    """
    统一入口函数
    """
    evaluator = DepthEvaluator(
        api_key=config.get('api_key'),
        base_url=config.get('base_url'),
        target_model=config.get('model', 'gpt-4o'),
        judge_model=config.get('judge_model', 'deepseek-chat')
    )
    
    input_dir = config.get('input_path')
    # 如果没有指定输入路径，尝试使用默认路径
    if not input_dir:
        # 尝试 dataset_step4 或 Depth
        possible_paths = ["dataset_step4", "Depth", os.path.join("data", "Depth")]
        for path in possible_paths:
            if os.path.exists(path):
                input_dir = path
                break
        
        if not input_dir:
            logger.warning("未找到默认输入目录 (dataset_step4, Depth, 或 data/Depth)，请指定 --input-dir")
            return

    output_dir = config.get('output_dir', 'eval_results/depth')
    
    evaluator.run_evaluation(input_dir, output_dir)
