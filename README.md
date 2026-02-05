<div style="text-align:center">
<h2>From Generation to Decision: A Hierarchical Benchmark for Reliable Financial Reasoning with LLMs</h2>
</div>

<div align="center">

![license](https://img.shields.io/badge/License-Apache--2.0-blue.svg)

[简体中文](README-CN.md)

</div>

# Introduction

FinReasoning is a benchmark focused on evaluating the reasoning capabilities of Large Language Models (LLMs) in the Chinese financial domain. It aims to comprehensively assess the logical consistency, factual alignment, and depth of analysis of LLMs in financial scenarios:
* **Semantic Consistency Evaluation**: Evaluates potential hallucination risks, focusing on detecting logical errors, context contradictions, and reversed causality in financial texts.
* **Alignment and Fact Checking (Data Alignment)**: Focuses not only on text generation but also on evaluating numerical calculation and fact verification capabilities in specific financial scenarios, such as complex financial report analysis and multi-indicator comparison.
* **Deep Analysis**: Examines the model's ability to understand complex financial problems, provide rigorous argumentation, and perform self-consistent logical analysis.

In the future, FinReasoning will continue to deepen the financial reasoning evaluation system, incorporating more real-world business scenarios to provide more precise measurements of the capability boundaries of financial LLMs.

<div align="center">
  <img src="imgs/FinReasoning.png" width="100%"/>
  <br />
</div>

# Table of Contents

- [FinReasoning Benchmark Details](#finreasoning-benchmark-details)
- [Data Examples](#data-examples)
- [Quick Start](#quick-start)
- [Test Results](#test-results)

# FinReasoning Benchmark Details

## Benchmark Construction

FinReasoning contains 4800 high-quality financial evaluation data items, covering three main aspects: consistency, alignment, and deep analysis.

| Track                                 | Category             | Dimension       | Explantion                                                   | Number |
| ------------------------------------- | -------------------- | --------------- | ------------------------------------------------------------ | ------ |
| **Semantic Consistency**              | Terminology Assessment | Inconsistent Terminology Usage | Multiple expressions for the same concept, undermining rigor | 200    |
|                                       |                      | Improper Terminology | Use of colloquial or non-domain standard vocabulary        | 200    |
|                                       |                      | Terminology Confusion             | Use of words with similar semantics but different definitions | 200    |
|                                       | Factual Assessment   | Relation Error           | Errors in predicates, actions, or relationship descriptions of events | 200    |
|                                       |                      | Entity Error             | Wrong key entities or subject-object reversal                | 200    |
|                                       |                      | Context Error            | Errors in modifiers such as time, location, quantity, manner, etc. | 200    |
|                                       | Logic Assessment     | Reasoning Chain Error    | Logical gaps, skipped steps, or contradictory reasoning      | 200    |
|                                       |                      | Discourse Relation Error | Errors in chronological order or causal relationships        | 200    |
|                                       |                      | Context Inconsistency    | Conclusion conflicts with previous settings                  | 200    |
| **Data Alignment**                    | L1: Simple Alignment & Fact Checking | Value Modification       | Random addition, subtraction, multiplication, or division of original values | 150    |
|                                       |                      | Unit Modification        | e.g. Converting "Yuan" to "Ten Thousand Yuan/Hundred Million Yuan" | 150    |
|                                       |                      | Comparison Relation Modification | Tampering with logical words                                 | 150    |
|                                       |                      | Synonym Replacement      | Using financial synonym dictionary                           | 150    |
|                                       | L2: Complex Numerical Calculation & Analysis | Single Date Multi-Indicator | Composite calculation of multiple indicators for a single date | 200    |
|                                       |                      | Multi-Date Single Indicator | Cross-date comparison and calculation for a single indicator | 200    |
|                                       |                      | Multi-Date Multi-Indicator  | Cross-date comparison and calculation for multiple indicators | 200    |
|                                       |                      | Cross-Company/Industry Comparison | Ranking, Max/Min, Group comparison, etc.                     | 200    |
|                                       | L3: Rule-Driven Consistency Verification | /               | Comprehensive understanding of the quantitative relationship between rule logic and structured indicators | 400    |
| **Deep Analysis** | Justification & Causal Depth | /               | Whether the reasoning conforms to business logic, and whether the causal chain (A->B->C) is clear and complete | 300    |
|                                       | Factuality & Contextualization | /               | Whether the model accurately cites key facts (such as approved varieties, financial forecasts) from evidence to support arguments | 300    |
|                                       | Completeness & Comparative Analysis     | /               | Core assessment of critical thinking. Whether the model identifies potential risks, execution limitations, or counterbalancing factors, rather than being one-sidedly optimistic/pessimistic. | 300    |
|                                       | Architectural Richness & Strictness | /               | Whether the output logical framework is professional and the concept depth meets standards | 300    |

## Evaluation Metrics

We present a comprehensive  evaluation mechanism to score diverse dimensions of the results, and compute a composite score for each track by applying simple average weighting to the normalized values of the constituent metrics listed below.

| **Track**                    | **Evaluation Metrics**  | Method            | Obj./Subj. |
| ---------------------------- | ----------------------- | ----------------- | ---------- |
| **Semantic** **Consistency** | Error Location Accuracy | Sentence-F1       | Objective  |
|                              | Error Explanation       | BERTScore/SimCSE  | Objective  |
|                              |                         | LLM-as-a Judge    | Subjective |
|                              | Correct Content         | BERTScore/SimCSE  | Objective  |
|                              |                         | LLM-as-a Judge    | Subjective |
| **Data Alignment**           | Answer Accuracy         | Accuracy          | Objective  |
|                              | Retrieval Accuracy      | F1-Score(Data ID) | Objective  |
|                              |                         | F1-Score(Field）  | Objective  |
| **Deep Insight**             | JCD                     | LLM-as-a Judge    | Subjective |
|                              | F&C                     | LLM-as-a Judge    | Subjective |
|                              | C&A                     | LLM-as-a Judge    | Subjective |
|                              | ARS                     | LLM-as-a Judge    | Subjective |

# Data Examples

The data for each task is in JSON format, containing questions, answers, and relevant metadata. To save space, long text fields have been truncated, but the complete data structure is preserved.

### 1. Semantic Consistency

Evaluates the model's ability to identify entity errors (e.g., organization names, positions) in financial texts.

```json
{
  "qa_id": "fact_entity_error_qa_001",
  "source": "long_text_0001",
  "category": "Fact_Evaluation",
  "dimension": "Factual_Entity_Error",
  "question": "你是一名专业的金融事实核查专家...[Long Text Omitted]...",
  "answer": {
      "factual_error_exists": "是",
      "factual_errors": [
        {
          "error_location": "中国证券监督管理委员会维持了宽松的货币政策基调...",
          "wrong_entity": "中国证券监督管理委员会",
          "correct_entity": "中国人民银行（央行）",
          "error_type": "实体错误",
          "reason": "中国证券监督管理委员会（证监会）主要负责..."
        }
      ],
      "error_explanation": "文本包含3处实体错误...[Explanation Omitted]...",
      "corrected_text": "...[Corrected Text Omitted]..."
  },
  "benchmark_type": "Consistency",
  "metadata": {
      "error_type": "factual_entity_error",
      "modified_text": "...[Modified Text Omitted]...",
      "num_errors": 3,
      "original_metadata": {
          "original_text_type": "analysis",
          "num_source_chunks": 6
      }
  }
}
```

### 2. Data Alignment

Evaluates the model's capability to perform precise numerical queries and verification based on specific stock codes, dates, and indicators.

```json
{
  "qa_id": "qa_1_000001",
  "source": "",
  "category": "L1_Simple",
  "dimension": "Simple_Fact_Checking",
  "question": "XXX股份有限公司的2025年10月31日的daily_tafactor是否低于6.930547？",
  "answer": "否",
  "benchmark_type": "Alignment",
  "metadata": {
    "question_type": "comparison",
    "modification_type": null,
    "data_ids": [
      3472463
    ],
    "date": "2025-10-31",
    "indicator": "daily_tafactor",
    "stcode": "xxx",
    "calculation_results": [],
    "domain": ""
  }
}
```

### 3. Deep Analysis

Evaluates the model's ability to deduce and argue complex financial logic.

```json
{
  "qa_id": "334",
  "source": "790284965309",
  "category": "JCD_Justification_Causal_Depth",
  "dimension": "",
  "question": "请深入分析童装业务增速“环比提升”...[Long Text Omitted]...",
  "answer": "对361度童装业务的分析如下...[Long Text Omitted]...",
  "benchmark_type": "Depth",
  "metadata": {
      "evidence": "研报显示...[Evidence Omitted]..."
  }
}
```

# Quick Start

## Installation

We recommend using a Python 3.8+ environment.

```bash
# Clone repository
git clone https://github.com/YourOrg/FinReasoning.git
cd FinReasoning

# Install dependencies
pip install -r requirements.txt
```

> **Note**: The Alignment task relies on CSV data files (`stock_data.csv`, `database_sample.csv`) in the `data/Alignment/datebase` directory. Users can also expand and configure data according to their own needs.

## Evaluation

We have prepared a unified evaluation entry point `main.py` in the root directory.

### LLM Configuration Details

This framework supports two ways to configure the LLM (model, API Key, Base URL):

**Method 1: Direct Code Configuration (Recommended)**

Open `main.py`, find the `LLM_SETTINGS` configuration block and modify it.

**Method 2: Environment Variable Configuration**

If you wish to protect your Key from being committed to the code repository, you can set `api_key` and `base_url` to `None` in `main.py` and set the following environment variables:

- `OPENAI_API_KEY`: Your API Key
- `OPENAI_BASE_URL`: API Base URL

### Basic Usage

```bash
python main.py --task <task_name> [options]
```

### Parameter Description

- `--task`: **Required**. Evaluation task type (`alignment`, `consistency`, `depth`, `all`).
- `--input-path`: Input data path (defaults to automatic search in corresponding folders).
- `--output-dir`: Result output root directory (default `eval_results`).
- `--test-run`: Test mode, processes only one data item per task.

> **Note**: Results will be saved in a hierarchical structure: `Output Directory/Timestamp/Task/Model Name`.

### Running Examples

**Example 1: Run Consistency Evaluation**

```bash
python main.py --task consistency
```

**Example 2: Run Alignment Evaluation**
```bash
python main.py --task alignment
```

**Example 3: Run Depth Evaluation**
```bash
python main.py --task depth
```

**Example 4: Test Run Mode (Quick Verification)**
Use the `--test-run` parameter to process only 1 data item per task, used to check if the environment configuration is correct.
```bash
python main.py --task all --test-run
```

