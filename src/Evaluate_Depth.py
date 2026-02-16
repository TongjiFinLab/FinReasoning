import os
import json
import time
from tqdm import tqdm
from .utils import get_openai_client, setup_logger
from .Evaluation_prompt import prompt_jcd, prompt_fc, prompt_ca, prompt_ars

logger = setup_logger("Depth_Evaluator")

KEY_ALIAS_MAPPING = {
    # JCD
    "logical_chain": "逻辑链条", "logic_chain": "逻辑链条", 
    "causal_depth": "因果深度",
    "professionalism": "专业性", "expertise": "专业性",
    
    # FC
    "fact_density": "事实密度", "factual_density": "事实密度",
    "calculation_rigor": "计算严谨性", "computational_rigor": "计算严谨性",
    "contextual_analysis": "情境分析", "situational_analysis": "情境分析",
    
    # CA
    "critical_thinking": "批判性思维",
    "comparative_perspective": "比较视角", "comparison_perspective": "比较视角",
    "argument_balance": "论点平衡性", "balance": "论点平衡性",
    
    # ARS
    "framework_completeness": "框架完整性", "framework_integrity": "框架完整性", "framework完整性": "框架完整性",
    "logical_hierarchy": "逻辑层次", "logic_structure": "逻辑层次", "logical_structure": "逻辑层次",
    "expression_granularity": "表达颗粒度", "granularity": "表达颗粒度"
}

# 各维度的评分标准
SCORING_RUBRICS = {
    "JCD_Justification_Causal_Depth": {
        "prompt": prompt_jcd,
        "subdimensions": {
            "逻辑链条": 0.4,      # 40分
            "因果深度": 0.3,      # 30分
            "专业性": 0.3          # 30分
        }
    },
    "FC_Fact_Context": {
        "prompt": prompt_fc,
        "subdimensions": {
            "事实密度": 0.4,      # 40分
            "计算严谨性": 0.3,    # 30分
            "情境分析": 0.3       # 30分
        }
    },
    "CA_Completeness_Analysis": {
        "prompt": prompt_ca,
        "subdimensions": {
            "批判性思维": 0.4,    # 40分
            "比较视角": 0.3,      # 30分
            "论点平衡性": 0.3     # 30分
        }
    },
    "ARS_Argument_Richness_Structure": {
        "prompt": prompt_ars,
        "subdimensions": {
            "框架完整性": 0.4,    # 40分
            "逻辑层次": 0.3,      # 30分
            "表达颗粒度": 0.3     # 30分
        }
    }
}

class DepthEvaluator:
    def __init__(self, api_key, base_url, target_model="gpt-4o", judge_model="deepseek-chat"):
        self.client = get_openai_client(api_key, base_url)
        self.target_model = target_model
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
            logger.error(f"调用被测模型出错: {e}")
            return f"Error: {e}"

    def judge_answer(self, category, question, gold_answer, model_answer):
        """裁判模型评分逻辑"""
        rubric_config = SCORING_RUBRICS.get(category)
        
        if not rubric_config:
            for key, val in SCORING_RUBRICS.items():
                if key in category or category in key:
                    rubric_config = val
                    break
        
        if not rubric_config:
            logger.warning(f"警告: 未找到类别 '{category}' 的特定评分标准")
            return {"score": 0, "reason": "未找到评分标准", "details": {}}

        rubric_prompt_template = rubric_config["prompt"]
        subdims = rubric_config["subdimensions"]
        
        judge_prompt = f"""
你是一名严苛的金融评测专家。对比【标准答案】对【待测回答】进行评分。

【评分标准】
{rubric_prompt_template}

【题目信息】
问题：{question}
标准答案：{gold_answer}

【待测回答】
{model_answer}

请务必按上述JSON格式输出。
"""
        try:
            response = self.client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": judge_prompt}],
                response_format={"type": "json_object"},
                temperature=0
            )
            
            result_json = json.loads(response.choices[0].message.content)

            normalized_result_json = {}
            for k, v in result_json.items():
                mapped_key = k
                k_lower = k.lower()
                if k_lower in KEY_ALIAS_MAPPING:
                    mapped_key = KEY_ALIAS_MAPPING[k_lower]
                else:
                    for eng_alias, cn_name in KEY_ALIAS_MAPPING.items():
                        if eng_alias in k_lower:
                            mapped_key = cn_name
                            break
                            
                normalized_result_json[mapped_key] = v
            
            result_json = normalized_result_json
            
            # 计算加权总分
            # 原始分是0-5分，转换成0-100分制
            # 权重之和为1.0。例如 4 * 0.4 + 3 * 0.3 + 5 * 0.3 = 1.6 + 0.9 + 1.5 = 4.0
            # 最终得分 = 4.0 / 5.0 * 100 = 80
            
            total_weighted_points = 0.0
            combined_reason = []
            
            for subdim_name, weight in subdims.items():
                if subdim_name in result_json:
                    sub_score = float(result_json[subdim_name].get("score", 0))
                    explanation = result_json[subdim_name].get("explanation", "")
                else:
                    found = False
                    for key in result_json:
                        if subdim_name in key:
                            sub_score = float(result_json[key].get("score", 0))
                            explanation = result_json[key].get("explanation", "")
                            found = True
                            break
                    if not found:
                        sub_score = 0
                        explanation = "裁判模型缺失评价"
                
                total_weighted_points += sub_score * weight
                combined_reason.append(f"{subdim_name}({sub_score}): {explanation}")
                
            # 假设满分为5分，将其标准化为100分
            final_score = (total_weighted_points / 5.0) * 100
            
            return {
                "score": round(final_score, 2),
                "reason": " | ".join(combined_reason),
                "details": result_json # 保存原始的子维度评分详情
            }

        except Exception as e:
            logger.error(f"调用裁判模型出错: {e}")
            return {"score": 0, "reason": f"评分调用失败: {e}", "details": {}}

    def run_evaluation(self, input_dir, output_dir, max_qa=None):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 支持目录或单个文件
        if os.path.isdir(input_dir):
            all_categories_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
            input_base = input_dir
        else:
            if not os.path.exists(input_dir):
                logger.error(f"输入路径未找到: {input_dir}")
                return
            input_base = os.path.dirname(input_dir)
            all_categories_files = [os.path.basename(input_dir)]

        if not all_categories_files:
            logger.warning(f"在 {input_dir} 中未找到 JSON 文件")
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

            if max_qa:
                data = data[:max_qa]
                logger.info(f"限制评测数量: {max_qa} 条")

            results = []
            category_total_score = 0
            
            for item in tqdm(data, desc=f"正在评测 {filename}"):
                # 1. 模型推理
                model_out = self.get_model_response(self.target_model, item.get('evidence', ''), item['question'])
                
                # 2. 裁判评分
                current_category_guess = filename.replace('.json', '')
                category = item.get('category', current_category_guess)
                
                eval_res = self.judge_answer(category, item['question'], item['answer'], model_out)
                
                # 3. 记录结果
                res_item = {
                    "qa_id": item.get('qa_id', 'unknown'),
                    "source": item.get('source', 'unknown'),
                    "question": item['question'],
                    "gold_answer": item['answer'],
                    "model_answer": model_out,
                    "score": eval_res.get("score", 0),
                    "reason": eval_res.get("reason", ""),
                    "judge_details": eval_res.get("details", {})
                }
                results.append(res_item)
                category_total_score += float(eval_res.get("score", 0))

            # 保存该维度的评测结果
            avg_score = category_total_score / len(data) if data else 0
            safe_model_name = self.target_model.replace('/', '_').replace('\\', '_')
            output_filename = f"result_{safe_model_name}_{filename}"
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
        possible_paths = ["dataset_step4", "Depth", os.path.join("data", "Depth")]
        for path in possible_paths:
            if os.path.exists(path):
                input_dir = path
                break
        
        if not input_dir:
            logger.warning("未找到默认输入目录 (dataset_step4, Depth, 或 data/Depth)，请指定 --input-dir")
            return


    output_dir = config.get('output_dir', 'eval_results/depth')
    max_qa = config.get('max_qa')
    
    evaluator.run_evaluation(input_dir, output_dir, max_qa=max_qa)
