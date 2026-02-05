"""
通用评估脚本：调用LLM测试dataset文件夹下的jsonl文件
通过修改文件名来测试不同的样本
"""

import json
import re
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from .utils import get_openai_client, setup_logger

logger = setup_logger("Consistency_Evaluator")

try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    logger.warning("bert-score 未安装，BERTScore指标将无法计算。请运行: pip install bert-score")

try:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn 未安装，分类指标将无法计算。请运行: pip install scikit-learn")

try:
    from sentence_transformers import SentenceTransformer, util
    SIMCSE_AVAILABLE = True
except ImportError:
    SIMCSE_AVAILABLE = False
    logger.warning("sentence-transformers 未安装，SimCSE指标将无法计算。请运行: pip install sentence-transformers")


class ErrorDetectionEvaluator:
    """错误识别能力评估器"""
    
    def __init__(self, api_key: str, base_url: str, model: str = "deepseek-chat"):
        """初始化评估器
        
        Args:
            api_key: API密钥
            base_url: API基础URL
            model: 模型名称
        """
        self.client = get_openai_client(api_key, base_url)
        self.model = model
        
        # SimCSE模型
        if SIMCSE_AVAILABLE:
            try:
                self.simcse_model = SentenceTransformer("BAAI/bge-base-zh-v1.5")
            except Exception as e:
                logger.warning(f"加载SimCSE模型失败: {e}")
                self.simcse_model = None
        else:
            self.simcse_model = None
    
    def load_qa_dataset(self, qa_file: str) -> List[Dict[str, Any]]:
        """加载QA数据集"""
        qa_dataset = []
        with open(qa_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    qa_dataset.append(json.loads(line.strip()))
        return qa_dataset
    
    def call_model(self, prompt: str, max_retries: int = 3) -> Optional[Dict[str, Any]]:
        """调用模型API"""
        for retry in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            'role': 'system',
                            'content': '你是一个专业的金融事实核查专家，擅长识别文本中的事实错误。请严格按照JSON格式返回结果。'
                        },
                        {
                            'role': 'user',
                            'content': prompt
                        }
                    ],
                    temperature=0.3,
                    max_tokens=8000
                )
                
                full_response = response.choices[0].message.content
                response_text = full_response.strip()
                
                # 去除可能的markdown代码块标记
                if response_text.startswith('```json'):
                    response_text = response_text[7:]
                if response_text.startswith('```'):
                    response_text = response_text[3:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]
                
                result = json.loads(response_text.strip())
                return result
                
            except json.JSONDecodeError as e:
                if retry < max_retries - 1:
                    print(f"  ⚠️ JSON解析失败，重试中... ({retry + 2}/{max_retries})")
                else:
                    print(f"  ✗ JSON解析失败: {e}")
                    print(f"  原始响应: {full_response[:500]}...")
            except Exception as e:
                if retry < max_retries - 1:
                    print(f"  ⚠️ API调用失败，重试中... ({retry + 2}/{max_retries})")
                else:
                    print(f"  ✗ API调用失败: {e}")
        
        return None
    
    def evaluate_error_list(self, ground_truth_errors: List[Dict], predicted_errors: List[Dict]) -> Dict[str, float]:
        """评估错误列举（仅保留句子级指标，删除字符级指标）"""
        def normalize_text(text: str) -> str:
            if not text:
                return ''
            return re.sub(r'[\s，。、；：！？""\'（）【】]', '', text)

        # 句子级
        gt_sentences, pred_sentences = [], []
        for err in ground_truth_errors:
            loc = err.get('error_location', '') or err.get('context', '')
            if loc and loc.strip():
                gt_sentences.append(loc.strip())
        for err in predicted_errors:
            loc = err.get('error_location', '') or err.get('context', '')
            if loc and loc.strip():
                pred_sentences.append(loc.strip())

        if gt_sentences or pred_sentences:
            matched_gt, matched_pred = set(), set()
            for i, gt_sent in enumerate(gt_sentences):
                gt_norm = normalize_text(gt_sent)
                best_idx, best_sim = -1, 0.0
                for j, pred_sent in enumerate(pred_sentences):
                    if j in matched_pred:
                        continue
                    pred_norm = normalize_text(pred_sent)
                    is_contained = gt_norm in pred_norm or pred_norm in gt_norm
                    if gt_norm == pred_norm:
                        sim = 1.0
                    elif is_contained:
                        sim = 1.0
                    else:
                        gt_chars, pred_chars = set(gt_norm), set(pred_norm)
                        if gt_chars or pred_chars:
                            inter = len(gt_chars & pred_chars)
                            union = len(gt_chars | pred_chars)
                            sim = inter / union if union > 0 else 0.0
                        else:
                            sim = 0.0
                    if (is_contained or sim > 0.8) and sim > best_sim:
                        best_sim, best_idx = sim, j
                if best_idx >= 0:
                    matched_gt.add(i)
                    matched_pred.add(best_idx)
            sentence_precision = len(matched_pred) / len(pred_sentences) if pred_sentences else 0.0
            sentence_recall = len(matched_gt) / len(gt_sentences) if gt_sentences else 0.0
            sentence_f1 = (
                2 * sentence_precision * sentence_recall / (sentence_precision + sentence_recall)
                if (sentence_precision + sentence_recall) > 0
                else 0.0
            )
        else:
            sentence_f1 = sentence_precision = sentence_recall = 1.0

        return {
            'sentence_f1': sentence_f1,
            'sentence_precision': sentence_precision,
            'sentence_recall': sentence_recall,
            'num_gt_errors': len(ground_truth_errors),
            'num_pred_errors': len(predicted_errors),
        }

    def evaluate_explanation(self, ground_truth: str, prediction: str) -> Dict[str, float]:
        """评估错误解释"""
        metrics: Dict[str, float] = {}

        if BERTSCORE_AVAILABLE:
            try:
                P, R, F1 = bert_score([prediction], [ground_truth], lang='zh', verbose=False)
                metrics['bertscore_f1'] = float(F1[0].item())
                metrics['bertscore_precision'] = float(P[0].item())
                metrics['bertscore_recall'] = float(R[0].item())
            except Exception as e:
                print(f"  ⚠️ BERTScore计算失败: {e}")
                metrics['bertscore_f1'] = metrics['bertscore_precision'] = metrics['bertscore_recall'] = 0.0
        else:
            metrics['bertscore_f1'] = metrics['bertscore_precision'] = metrics['bertscore_recall'] = 0.0

        if SIMCSE_AVAILABLE and self.simcse_model is not None:
            try:
                embs = self.simcse_model.encode([ground_truth, prediction], convert_to_tensor=True)
                sim = util.cos_sim(embs[0], embs[1]).item()
                metrics['simcse'] = float(sim)
            except Exception as e:
                print(f"  ⚠️ SimCSE计算失败: {e}")
                metrics['simcse'] = 0.0
        else:
            metrics['simcse'] = 0.0

        return metrics

    def evaluate_corrected_text(self, gt_errors: List[Dict[str, Any]],
                                pred_errors: List[Dict[str, Any]]) -> Dict[str, float]:
        """评估修正文本"""
        if not BERTSCORE_AVAILABLE and not (SIMCSE_AVAILABLE and self.simcse_model is not None):
            return {
                'bertscore_f1': 0.0,
                'bertscore_precision': 0.0,
                'bertscore_recall': 0.0,
                'simcse': 0.0,
            }
        if not gt_errors:
            return {
                'bertscore_f1': 1.0,
                'bertscore_precision': 1.0,
                'bertscore_recall': 1.0,
                'simcse': 1.0,
            }

        def normalize_text(text: str) -> str:
            if not text:
                return ''
            return re.sub(r'[\s，。、；：！？""\'（）【】]', '', text)

        gt_sentences, pred_sentences = [], []
        for err in gt_errors:
            loc = err.get('error_location', '') or err.get('context', '')
            if loc and loc.strip():
                gt_sentences.append(loc.strip())
        for err in pred_errors:
            loc = err.get('error_location', '') or err.get('context', '')
            if loc and loc.strip():
                pred_sentences.append(loc.strip())

        matched_pairs: List[tuple[int, int]] = []
        matched_pred = set()
        for i, gt_sent in enumerate(gt_sentences):
            gt_norm = normalize_text(gt_sent)
            best_idx, best_sim = -1, 0.0
            for j, pred_sent in enumerate(pred_sentences):
                if j in matched_pred:
                    continue
                pred_norm = normalize_text(pred_sent)
                is_contained = gt_norm in pred_norm or pred_norm in gt_norm
                if gt_norm == pred_norm:
                    sim = 1.0
                elif is_contained:
                    sim = 1.0
                else:
                    gt_chars, pred_chars = set(gt_norm), set(pred_norm)
                    if gt_chars or pred_chars:
                        inter = len(gt_chars & pred_chars)
                        union = len(gt_chars | pred_chars)
                        sim = inter / union if union > 0 else 0.0
                    else:
                        sim = 0.0
                if (is_contained or sim > 0.8) and sim > best_sim:
                    best_sim, best_idx = sim, j
            if best_idx >= 0:
                matched_pairs.append((i, best_idx))
                matched_pred.add(best_idx)

        all_precisions: List[float] = []
        all_recalls: List[float] = []
        all_f1s: List[float] = []
        all_simcse: List[float] = []

        for gt_idx, pred_idx in matched_pairs:
            gt_correct = gt_errors[gt_idx].get('correct_context', '') or gt_errors[gt_idx].get('correct_content', '')
            pred_correct = pred_errors[pred_idx].get('correct_content', '') or pred_errors[pred_idx].get('correct_context', '')
            if not gt_correct or not pred_correct:
                continue

            if BERTSCORE_AVAILABLE:
                try:
                    P, R, F1 = bert_score([pred_correct], [gt_correct], lang='zh', verbose=False)
                    all_precisions.append(float(P[0].item()))
                    all_recalls.append(float(R[0].item()))
                    all_f1s.append(float(F1[0].item()))
                except Exception as e:
                    print(f"  ⚠️ BERTScore计算失败（修正文本）: {e}")

            if SIMCSE_AVAILABLE and self.simcse_model is not None:
                try:
                    embs = self.simcse_model.encode([gt_correct, pred_correct], convert_to_tensor=True)
                    sim = util.cos_sim(embs[0], embs[1]).item()
                    all_simcse.append(float(sim))
                except Exception as e:
                    print(f"  ⚠️ SimCSE计算失败（修正文本）: {e}")

        if not all_f1s and not all_simcse:
            return {
                'bertscore_f1': 0.0,
                'bertscore_precision': 0.0,
                'bertscore_recall': 0.0,
                'simcse': 0.0,
            }

        bert_f1 = sum(all_f1s) / len(all_f1s) if all_f1s else 0.0
        bert_p = sum(all_precisions) / len(all_precisions) if all_precisions else 0.0
        bert_r = sum(all_recalls) / len(all_recalls) if all_recalls else 0.0
        simcse_avg = sum(all_simcse) / len(all_simcse) if all_simcse else 0.0

        return {
            'bertscore_f1': bert_f1,
            'bertscore_precision': bert_p,
            'bertscore_recall': bert_r,
            'simcse': simcse_avg,
            'num_evaluated_errors': max(len(all_f1s), len(all_simcse)),
        }

    def print_statistics(self, evaluation_results: List[Dict[str, Any]]):
        """打印统计结果"""
        if not evaluation_results:
            print("  无评估结果")
            return

        n = len(evaluation_results)
        results_with_errors = [
            r for r in evaluation_results
            if r['error_list']['metrics']['num_gt_errors'] > 0
        ]
        n_with_errors = len(results_with_errors)

        sentence_f1s = [r['error_list']['metrics']['sentence_f1'] for r in results_with_errors]
        sentence_precisions = [r['error_list']['metrics']['sentence_precision'] for r in results_with_errors]
        sentence_recalls = [r['error_list']['metrics']['sentence_recall'] for r in results_with_errors]

        bertscore_f1s = [r['explanation']['metrics']['bertscore_f1'] for r in results_with_errors]
        simcse_explains = [r['explanation']['metrics']['simcse'] for r in results_with_errors]

        corrected_bertscore_f1s = [r['corrected_text']['metrics']['bertscore_f1'] for r in results_with_errors]
        simcse_corrected = [r['corrected_text']['metrics']['simcse'] for r in results_with_errors]

        print(f"\n{'='*80}")
        print("总体评估结果")
        print(f"{'='*80}")
        print(f"\n样本总数: {n}")
        print(f"有错误的样本数: {n_with_errors}")
        print(f"无错误的样本数: {n - n_with_errors}")

        if n_with_errors > 0:
            print(f"\n【1. 错误列举】（仅统计有错误的样本，共{n_with_errors}个）")
            print(f"  句子级 Precision: {sum(sentence_precisions)/n_with_errors:.4f}")
            print(f"  句子级 Recall: {sum(sentence_recalls)/n_with_errors:.4f}")
            print(f"  句子级 F1: {sum(sentence_f1s)/n_with_errors:.4f}")

            print(f"\n【2. 错误解释】（仅统计有错误的样本，共{n_with_errors}个）")
            print(f"  BERTScore-F1: {sum(bertscore_f1s)/n_with_errors:.4f}")
            print(f"  SimCSE: {sum(simcse_explains)/n_with_errors:.4f}")

            print(f"\n【3. 修正文本】（仅统计有错误的样本，共{n_with_errors}个）")
            print(f"  BERTScore-F1: {sum(corrected_bertscore_f1s)/n_with_errors:.4f}")
            print(f"  SimCSE: {sum(simcse_corrected)/n_with_errors:.4f}")
        else:
            print(f"\n【1-3. 错误检测相关指标】")
            print(f"  所有样本均无错误，跳过错误列举、错误解释、修正文本的评估")
        print(f"{'='*80}\n")

    def evaluate_dataset(self, qa_file: str, output_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """评估数据集"""
        print("=" * 80)
        print("开始评估数据集")
        print("=" * 80)
        print(f"\nQA 文件: {qa_file}")

        qa_dataset = self.load_qa_dataset(qa_file)
        print(f"✓ 加载 QA 数据集: {len(qa_dataset)} 条")

        # 读取已处理过的样本ID（用于断点续传）
        processed_qa_ids = set()
        if output_file and os.path.exists(output_file):
            os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            try:
                                saved_item = json.loads(line.strip())
                                qa_id = saved_item.get("qa_id") or saved_item.get("evaluation", {}).get("qa_id")
                                if qa_id:
                                    processed_qa_ids.add(qa_id)
                            except json.JSONDecodeError:
                                continue
                print(f"✓ 检测到已有结果文件，已处理 {len(processed_qa_ids)} 个样本，将跳过这些样本")
            except Exception as e:
                print(f"⚠️ 读取已有结果文件时出错: {e}，将从头开始处理")
        elif output_file:
            os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
        
        if output_file:
            print(f"结果将保存到: {output_file}")

        evaluation_results: List[Dict[str, Any]] = []
        skipped_count = 0

        for idx, qa_item in enumerate(qa_dataset, 1):
            qa_id = qa_item.get("qa_id", f"unknown_{idx}")
            question = qa_item.get("question", "")
            answer = qa_item.get("answer", {})
            
            # 检查是否已处理过
            if qa_id in processed_qa_ids:
                skipped_count += 1
                print(f"\n[{idx}/{len(qa_dataset)}] 跳过已处理样本: {qa_id}")
                continue
            
            print(f"\n[{idx}/{len(qa_dataset)}] 处理样本: {qa_id}")
            
            # 调用模型
            model_result = self.call_model(question)
            if model_result is None:
                print(f"  ✗ 模型调用失败，跳过该样本")
                continue
            
            # 提取ground truth和prediction
            gt_error_exists = answer.get("factual_error_exists", "") or answer.get(
                "semantic_inconsistency_exists", "否"
            )
            if not gt_error_exists:
                gt_error_exists = "否"
            gt_errors = answer.get("factual_errors", []) or answer.get("semantic_inconsistencies", [])
            gt_explanation = answer.get("error_explanation", "") or answer.get(
                "inconsistency_explanation", ""
            )

            pred_error_exists = model_result.get("error_exists", "否")
            pred_errors = model_result.get("errors", [])
            pred_explanation = model_result.get("explanation", "")

            has_errors = gt_error_exists in ['是', 'yes', 'Yes', 'YES', '1', 'true', 'True'] or len(gt_errors) > 0

            # 计算评估指标
            if has_errors:
                error_list_metrics = self.evaluate_error_list(gt_errors, pred_errors)
                explanation_metrics = self.evaluate_explanation(gt_explanation, pred_explanation)
                corrected_text_metrics = self.evaluate_corrected_text(gt_errors, pred_errors)
            else:
                error_list_metrics = {
                    'sentence_f1': 0.0,
                    'sentence_precision': 0.0,
                    'sentence_recall': 0.0,
                    'num_gt_errors': 0,
                    'num_pred_errors': len(pred_errors),
                }
                explanation_metrics = {
                    'bertscore_f1': 0.0,
                    'bertscore_precision': 0.0,
                    'bertscore_recall': 0.0,
                    'simcse': 0.0,
                }
                corrected_text_metrics = {
                    'bertscore_f1': 0.0,
                    'bertscore_precision': 0.0,
                    'bertscore_recall': 0.0,
                    'simcse': 0.0,
                    'num_evaluated_errors': 0,
                }

            evaluation_result = {
                "qa_id": qa_id,
                "has_errors": has_errors,
                "error_list": {
                    "ground_truth_count": len(gt_errors),
                    "prediction_count": len(pred_errors),
                    "metrics": error_list_metrics,
                },
                "explanation": {"metrics": explanation_metrics},
                "corrected_text": {"metrics": corrected_text_metrics},
            }
            
            evaluation_results.append(evaluation_result)
            
            # 每处理完一个样本，立即保存到文件
            if output_file:
                result_item = {
                    "qa_id": qa_id,
                    "model_result": model_result,
                    "evaluation": evaluation_result
                }
                # 以追加模式写入文件，每处理完一条就保存
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result_item, ensure_ascii=False) + "\n")
                print(f"  ✓ 完成评估并已保存")
            else:
                print(f"  ✓ 完成评估")

        # 打印处理摘要
        total_processed = len(evaluation_results)
        print(f"\n{'='*80}")
        print(f"处理完成摘要:")
        print(f"  总样本数: {len(qa_dataset)}")
        print(f"  跳过样本数: {skipped_count}")
        print(f"  新处理样本数: {total_processed}")
        print(f"{'='*80}")

        # 打印统计信息
        self.print_statistics(evaluation_results)
        
        return evaluation_results


def run(config):
    """
    统一入口函数
    """
    api_key = config.get('api_key')
    base_url = config.get('base_url')
    model = config.get('model', 'deepseek-chat')
    
    input_path = config.get('input_path')
    if not input_path:
        input_path = os.path.join("data", "Consistency")
    output_dir = config.get('output_dir', 'eval_results/consistency')
    
    files_to_process = []
    if os.path.isdir(input_path):
        for f in os.listdir(input_path):
            if f.endswith('.jsonl'):
                files_to_process.append(os.path.join(input_path, f))
    elif os.path.isfile(input_path):
        files_to_process.append(input_path)
    else:
        # 尝试默认路径
        if os.path.exists("Consistency"):
             for f in os.listdir("Consistency"):
                if f.endswith('.jsonl'):
                    files_to_process.append(os.path.join("Consistency", f))
        else:
            logger.error(f"Input path not found: {input_path}")
            return

    if not files_to_process:
        logger.warning(f"No jsonl files found to process in {input_path}")
        return

    # Ensure output dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    evaluator = ErrorDetectionEvaluator(api_key=api_key, base_url=base_url, model=model)
    
    for input_file in files_to_process:
        logger.info(f"Processing file: {input_file}")
        dataset_name = Path(input_file).stem
        output_file = os.path.join(output_dir, f"{dataset_name}_results.jsonl")
        evaluator.evaluate_dataset(qa_file=input_file, output_file=output_file)

