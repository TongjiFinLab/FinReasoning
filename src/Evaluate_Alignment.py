import json
import os
import sys
import argparse
import time
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Set, Optional, Any
import pandas as pd
from .utils import get_openai_client, setup_logger

logger = setup_logger("Alignment_Evaluator")

# 定义数据文件路径
STOCK_DATA_PATH = os.path.join("data", "Alignment", "datebase", "stock_data.csv")
DATABASE_SAMPLE_PATH = os.path.join("data", "Alignment", "datebase", "database_sample.csv")

# 字段名到中文含义的映射字典
FIELD_MEANINGS = {
    'id': '数据行ID',
    'stcode': '股票代码',
    'dt': '交易日期',
    'fin_corp_high3mgr_reward': '前三高管薪酬',
    'fin_corp_mgr_avg_reward': '高管平均薪酬',
    'fin_corp_top10stockholder_prop_sum': '前十大股东持股比例合计',
    'fin_growth_q_netprofitgrowrate': '单季度净利润同比增长率',
    'fin_growth_q_torgrowrate': '单季度营业收入同比增长率',
    'fin_growth_q_operprofitgrowrate': '单季度营业利润同比增长率',
    'fin_growth_q_sue': '标准化预期外盈利',
    'fin_growth_q_sur': '标准化预期外营业收入',
    'fin_growth_q_roa_chg': '单季度ROA变动',
    'fin_growth_q_roe_chg': '单季度ROE变动',
    'fin_growth_q_sue_chg': '单季超预期幅度',
    'fin_growth_q_suroa': '标准化预期外ROA',
    'fin_growth_q_suroe': '标准化预期外ROE',
    'fin_growth_q_suecut': '标准化预期外扣非盈利',
    'fin_valuation_ep': '单季EP',
    'fin_valuation_sp': '单季SP',
    'fin_valuation_ep_ttm': '单季EPTTM',
    'fin_valuation_sp_ttm': '单季SPTTM',
    'fin_valuation_dividend_yield': '股息率',
    'fin_valuation_bp': '单季BP',
    'fin_valuation_ep_ttm_1y_quantile': '单季EP一年分位数',
    'fin_valuation_sp_ttm_1y_quantile': '单季SP一年分位数',
    'fin_valuation_bp_3y_quantile': '单季BP三年分位数',
    'fin_profit_roe': '单季ROE',
    'fin_profit_roa': '单季ROA',
    'fin_profit_delta_roe': 'DELTAROE',
    'fin_profit_delta_roa': 'DELTAROA',
    'fin_profit_gross_income_ratio': '单季度毛利率',
    'fin_profit_total_asset_rate': '季度资产周转率',
    'fin_category_size': '流通市值',
    'fin_category_nlsize': '非线性流通市值',
    's_val_pe_ttm': '市盈率(PE_TTM)，总市值/净利润（TTM)',
    's_val_ps_ttm': '市销率(PS_TTM)，市销率(PS,TTM)',
    's_val_pb_new': '市净率(PB)，总市值/净资产(LF）',
    's_price_div_dps': '股价/每股派息',
    's_dq_turn': '每日原始换手率',
    's_dq_mv': '当日流通市值',
    's_val_mv': '当日总市值',
    's_val_pcf_ncfttm': '市现率(PCF,现金净流量TTM)',
    'tot_cur_assets': '流动资产合计',
    'tot_cur_liab': '流动负债合计',
    'tot_liab': '负债合计',
    's_fa_debttoassets': '资产负债率',
    's_fa_roe': '净资产收益率',
    's_fa_roa': '总资产净利润',
    's_fa_grossprofitmargin': '销售毛利率',
    's_fa_netprofitmargin': '销售净利率',
    's_fa_assetsturn': '总资产周转率',
    's_fa_yoyeps_basic': '同比增长率-基本每股收益(%)',
    's_fa_yoy_or': '营业收入同比增长率(%)',
    's_fa_yoynetprofit_deducted': '同比增长率-归属母公',
    's_fa_yoy_equity': '净资产(同比增长率)',
    's_fa_yoyocf': '同比增长率-经营活动产生的现金流量净额(%)',
    's_fa_yoyroe': '同比增长率-净资产收益率(摊薄)(%)',
    's_fa_ocfps': '每股经营活动产生的现金流量净额',
    's_fa_deductedprofit': '扣除非经常性损益后的净利润',
    's_fa_tangibleasset': '有形资产',
    's_qfa_yoynetprofit': '单季度.归属母公司股东的净利润同比增长率(%)',
    'net_profit_excl_min_int_inc': '净利润(元)',
    'oper_rev': '营业收入(元)',
    'pe_cut': '扣除非经常性损益后的净利润/总市值',
    'tasset2mv': '有形资产/总市值',
    'nlsize': '对数流通市值',
    'debt2ncassets': '总资产/(流动资产合计-流动负债合计)',
    's_qfa_yoysales': '单季度.营业收入同比增长率(%)',
    'ln_s_dq_mv': '当日流通市值对数',
    'roe_ttm': '净资产收益率_TTM',
    'roa_ttm': '总资产净利率_TTM',
    'op_revenue_grow_rate': '营业收入同比增长',
    'netprofit_grow_rate': '净利润同比增长',
    'naor_yoy': '净资产收益率(摊薄)同比增长',
    'gross_income_ratio_ttm': '销售毛利率_TTM',
    'net_profit_ratio_ttm': '销售净利率_TTM',
    'current_ratio': '流动比率',
    'net_opercfto_to_assets': '全部资产现金回收率',
    'btop': '风格因子估值',
    'growth': '风格因子成长',
    'earnings_yield': '风格因子盈利',
    'leverage': '风格因子财务质量(杠杆)',
    'total_ashares': 'A股总股本',
    'free_float_shares': '自由流通股本',
    'float_ashares': '流通A股',
    'turn_freeshares': '自由流通股本换手率',
    'em_operate_opscore': '公司基本面评分：调整前基本面分',
    'em_operate_opadjscore': '公司基本面评分：最终基本面分',
    'valid_market_value': '自由流通市值因子',
    'free_float_value': '自由流通市值因子',
    'ln_free_float_value': '对数自由流通市值',
    'stock_name': '股票名称',
    'company_name': '公司名称',
    'gics': 'GICS行业分类',
    'industry': '行业',
    'daily_close': '收盘价',
    'daily_preclose': '前收盘价',
    'daily_open': '开盘价',
    'daily_high': '最高价',
    'daily_low': '最低价',
    'daily_volume': '成交量',
    'daily_amount': '成交额',
    'daily_turn': '换手率',
    'daily_tnum': '成交笔数',
    'daily_amplitude': '振幅',
    'daily_tafactor': '复权因子(后)',
}


class QALLMTester:
    """QA LLM测试器"""
    
    def __init__(self, model: str = "deepseek-chat", api_key: str = None, base_url: str = None):
        """
        初始化测试器
        
        Args:
            model: 使用的OpenAI模型
        """
        self.model = model
        self.client = get_openai_client(api_key, base_url)
        
        # 初始化SQLite内存数据库
        self.conn = sqlite3.connect(':memory:')
        self._load_data_to_sqlite()
        
    def _load_data_to_sqlite(self):
        """将CSV数据加载到SQLite内存数据库"""
        logger.info("Loading CSV data into in-memory SQLite database...")
        try:
            # 1. 加载 stock_data (只包含 stcode 和 basic info)
            # 假设 stock_data.csv 对应的表名是 stock_data
            if os.path.exists(STOCK_DATA_PATH):
                df_stock = pd.read_csv(STOCK_DATA_PATH)
                # 确保stcode是字符串格式，补零
                if 'stcode' in df_stock.columns:
                     df_stock['stcode'] = df_stock['stcode'].astype(str).str.zfill(6)
                if 'stock_code' in df_stock.columns:
                     df_stock['stock_code'] = df_stock['stock_code'].astype(str).str.zfill(6)
                     
                df_stock.to_sql('stock_data', self.conn, index=False, if_exists='replace')
                logger.info(f"Loaded stock_data: {len(df_stock)} rows")
            else:
                logger.warning(f"File not found: {STOCK_DATA_PATH}")

            # 2. 加载 database_sample (包含所有因子)
            # 假设 database_sample.csv 对应的表名是 stock_jbm_factors
            if os.path.exists(DATABASE_SAMPLE_PATH):
                df_factors = pd.read_csv(DATABASE_SAMPLE_PATH)
                if 'stcode' in df_factors.columns:
                    df_factors['stcode'] = df_factors['stcode'].astype(str).str.zfill(6)
                
                # 确保dt列是日期格式字符串
                if 'dt' in df_factors.columns:
                    df_factors['dt'] = pd.to_datetime(df_factors['dt']).dt.strftime('%Y-%m-%d')
                    
                df_factors.to_sql('stock_jbm_factors', self.conn, index=False, if_exists='replace')
                # 创建索引以加速查询
                self.conn.execute("CREATE INDEX idx_stcode ON stock_jbm_factors(stcode)")
                self.conn.execute("CREATE INDEX idx_dt ON stock_jbm_factors(dt)")
                logger.info(f"Loaded stock_jbm_factors: {len(df_factors)} rows")
            else:
                logger.warning(f"File not found: {DATABASE_SAMPLE_PATH}")
                
        except Exception as e:
            logger.error(f"Failed to load data to SQLite: {e}")
            raise e

    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()

    def get_company_info_from_qa(self, qa: Dict) -> Dict[str, str]:
        """
        从QA对象中提取公司信息
        
        Args:
            qa: QA对象字典
            
        Returns:
            Dict: 包含stcode和company_name的字典
        """
        company_info = {}
        metadata = qa.get('metadata', {})
    
        
        # 优先从company_group获取（多公司对比）
        if 'company_group' in metadata and isinstance(metadata['company_group'], list) and len(metadata['company_group']) > 0:
            company = metadata['company_group'][0]  # 先只处理第一个公司
            if isinstance(company, dict):
                if 'stock_code' in company:
                    stock_code = company['stock_code']
                    company_info['stcode'] = stock_code.split('.')[0] if '.' in stock_code else stock_code
                if 'company_name' in company:
                    company_info['company_name'] = company['company_name']
        
        # 如果没有company_group，从table_id或report_id获取（单公司）
        if 'stcode' not in company_info:
            table_id = metadata.get('table_id') or metadata.get('report_id')
            if table_id:
                company_info['stcode'] = table_id.split('.')[0] if '.' in table_id else table_id
        
        # 如果还没有stcode，尝试从data_ids查询
        if 'stcode' not in company_info:
            data_ids = metadata.get('data_ids', [])
            if data_ids:
                try:
                    df = self.query_data_by_ids(data_ids[:1])  # 只查询第一个
                    if not df.empty and 'stcode' in df.columns:
                        company_info['stcode'] = str(df['stcode'].iloc[0])
                    if not df.empty and 'company_name' in df.columns and pd.notna(df['company_name'].iloc[0]):
                        company_info['company_name'] = str(df['company_name'].iloc[0])
                except Exception:
                    pass
        
        return company_info
    
    def extract_indicators_from_qa(self, qa: Dict) -> Set[str]:
        """
        从QA对象中提取使用的指标字段
        
        Args:
            qa: QA对象字典
            
        Returns:
            Set[str]: 指标字段集合
        """
        indicators = set()
        metadata = qa.get('metadata', {})
        
        # 从metadata提取
        if 'indicator' in metadata and metadata['indicator']:
            indicators.add(metadata['indicator'])
        if 'indicators' in metadata and metadata['indicators']:
             val = metadata['indicators']
             if isinstance(val, list): indicators.update(val)
             elif isinstance(val, str): indicators.add(val)
        
        # 从indicator字段提取（单指标）
        if 'indicator' in qa and qa['indicator']:
            indicators.add(qa['indicator'])
        
        # 从indicators字段提取（多指标）
        if 'indicators' in qa and qa['indicators']:
            if isinstance(qa['indicators'], list):
                indicators.update(qa['indicators'])
            elif isinstance(qa['indicators'], str):
                indicators.add(qa['indicators'])
        
        # 从indicators_used字段提取（多指标，兼容旧格式）
        if 'indicators_used' in qa and qa['indicators_used']:
            if isinstance(qa['indicators_used'], list):
                indicators.update(qa['indicators_used'])
            elif isinstance(qa['indicators_used'], str):
                indicators.add(qa['indicators_used'])
        
        # 从calculation_results的data_used中提取
        if 'calculation_results' in qa and qa['calculation_results']:
            for calc_result in qa['calculation_results']:
                if 'data_used' in calc_result:
                    data_used = calc_result['data_used']
                    if isinstance(data_used, dict):
                        # 检查是否有indicator字段
                        if 'indicator' in data_used:
                            indicators.add(data_used['indicator'])
                        # 检查是否有其他指标字段（如s_fa_ocfps等）
                        for key, value in data_used.items():
                            if isinstance(value, dict) and 'name' in value:
                                # 这可能是指标字段
                                indicators.add(key)
        
        return indicators
    
    def build_query_plan_prompt(self, qa: Dict, company_info: Dict[str, str]) -> str:
        """
        构建第一阶段prompt：让LLM决定需要查询的数据（日期和字段）
        
        Args:
            qa: QA对象字典
            company_info: 公司信息字典（包含stcode和company_name）
            
        Returns:
            str: prompt文本
        """
        question = qa.get('question', '')
        stcode = company_info.get('stcode', '')
        company_name = company_info.get('company_name', '')
        
        # 构建字段含义说明
        field_descriptions = []
        for field_name, meaning in sorted(FIELD_MEANINGS.items()):
            field_descriptions.append(f"- {field_name}: {meaning}")
        
        field_descriptions_text = "\n".join(field_descriptions)
        
        # 构建公司信息说明
        company_info_text = f"股票代码(stcode): {stcode}"
        if company_name:
            company_info_text += f"\n公司名称: {company_name}"
        
        prompt = f"""你是一个金融数据分析助手。现在需要你决定为了回答一个问题，需要从数据库中查询哪些数据。

## 公司信息
{company_info_text}

## 数据库字段含义说明
以下是数据库中所有字段及其含义：
{field_descriptions_text}

## 数据库表结构说明
主表：stock_jbm_factors
- 主要字段：id (数据行ID), stcode (股票代码), dt (交易日期), 以及其他90+个财务和估值指标字段
- 每个公司有多条记录，每条记录对应一个交易日（dt字段）
- 查询时会自动使用当前公司的stcode：{stcode}

## 问题
{question}

## 任务
请分析这个问题，决定需要查询哪些数据来回答这个问题。请考虑：
1. 需要查询哪些字段（指标）？请列出字段名列表。
2. 需要查询哪些日期范围或具体的日期？

请严格按照以下JSON格式返回你的查询计划，不要添加任何其他文字说明：

```json
{{
  "fields_to_query": ["要查询的字段列表，例如：[\"fin_valuation_ep_ttm_1y_quantile\", \"fin_profit_roe\", \"s_fa_roe\"]"],
  "date_range": {{
    "start_date": "起始日期，格式YYYY-MM-DD，如果需要日期范围则填写，否则为null",
    "end_date": "结束日期，格式YYYY-MM-DD，如果需要日期范围则填写，否则为null",
    "specific_dates": ["具体日期列表，格式YYYY-MM-DD，如果需要查询特定日期则填写此列表，否则为null或空数组"]
  }},
  "reasoning": "解释为什么需要查询这些数据和字段，以及为什么选择这些日期范围或具体日期"
}}
```

**重要提示**：
- fields_to_query是一个字符串数组，包含需要查询的所有字段名
- 如果只需要特定日期，填写specific_dates数组（例如：["2020-01-15", "2020-06-30"]），start_date和end_date设为null
- 如果需要日期范围，填写start_date和end_date（例如：start_date: "2020-01-01", end_date: "2020-12-31"），specific_dates设为null或空数组
- 不能同时使用日期范围和具体日期列表
- 日期格式必须是YYYY-MM-DD
- 请确保返回的是有效的JSON格式，可以直接被解析"""
        return prompt
    
    def get_query_plan_from_llm(self, qa: Dict, company_info: Dict[str, str]) -> Dict[str, Any]:
        """
        第一阶段：让LLM决定需要查询的字段和日期
        
        Args:
            qa: QA对象字典
            company_info: 公司信息字典
            
        Returns:
            Dict: 包含查询计划的结果（字段列表和日期信息）
        """
        prompt = self.build_query_plan_prompt(qa, company_info)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            if not response or not response.choices or not response.choices[0].message:
                return {
                    'success': False,
                    'error': 'API响应无效'
                }
            
            message_content = response.choices[0].message.content
            if not message_content:
                return {
                    'success': False,
                    'error': 'API返回的content为空'
                }
            
            # 解析JSON响应
            import re
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', message_content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_match = re.search(r'\{.*\}', message_content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = message_content
            
            query_plan = json.loads(json_str)
            
            return {
                'success': True,
                'query_plan': query_plan,
                'raw_response': message_content
            }
            
        except json.JSONDecodeError as e:
            return {
                'success': False,
                'error': f'JSON解析失败: {str(e)}',
                'raw_response': message_content if 'message_content' in locals() else None
            }
        except Exception as e:
            import traceback
            return {
                'success': False,
                'error': str(e),
                'error_traceback': traceback.format_exc()
            }
    
    def build_sql_query(self, stcode: str, fields: List[str], 
                       date_range: Dict[str, Any] = None) -> tuple:
        """
        根据字段列表和日期信息构建SQL查询
        
        Args:
            stcode: 股票代码
            fields: 要查询的字段列表
            date_range: 日期信息字典，包含start_date、end_date或specific_dates
            
        Returns:
            tuple: (SQL查询语句, 参数列表)
        """
        # 确保包含必需的字段
        required_fields = ['id', 'stcode', 'dt']
        select_fields = []
        processed_fields = set()
        
        # 先添加必需字段
        for field in required_fields:
            select_fields.append(f'f.{field}')
            processed_fields.add(field)
        
        # 添加其他字段（避免重复）
        for field in fields:
            if field not in processed_fields:
                # 检查字段是否存在（可能在stock_data表中）
                if field in ['stock_name', 'company_name', 'gics', 'industry']:
                    select_fields.append(f's.{field}')
                else:
                    select_fields.append(f'f.{field}')
                processed_fields.add(field)
        
        # 构建SELECT子句
        select_clause = ', '.join(select_fields)
        
        # 构建WHERE子句
        where_conditions = ["f.stcode = ?"]
        params = [stcode]
        
        # 添加日期条件
        if date_range:
            start_date = date_range.get('start_date')
            end_date = date_range.get('end_date')
            specific_dates = date_range.get('specific_dates', [])
            
            # 如果有具体日期列表，使用IN子句
            if specific_dates and isinstance(specific_dates, list) and len(specific_dates) > 0:
                # 过滤掉None值
                specific_dates = [d for d in specific_dates if d is not None]
                if specific_dates:
                    placeholders = ','.join(['?'] * len(specific_dates))
                    where_conditions.append(f"f.dt IN ({placeholders})")
                    params.extend(specific_dates)
            # 如果有日期范围，使用BETWEEN或>= <=
            elif start_date and end_date:
                where_conditions.append("f.dt >= ? AND f.dt <= ?")
                params.extend([start_date, end_date])
            elif start_date:
                where_conditions.append("f.dt >= ?")
                params.append(start_date)
            elif end_date:
                where_conditions.append("f.dt <= ?")
                params.append(end_date)
        
        # 构建完整的SQL
        where_clause = ' AND '.join(where_conditions)
        sql_query = f"""
            SELECT {select_clause}
            FROM stock_jbm_factors f
            LEFT JOIN stock_data s ON s.stock_code = f.stcode
            WHERE {where_clause}
            ORDER BY f.dt
        """
        
        return sql_query.strip(), params
    
    def execute_sql_query(self, sql_query: str, params: List[Any]) -> pd.DataFrame:
        """
        安全执行SQL查询
        
        Args:
            sql_query: SQL查询语句（必须只包含SELECT语句）
            params: 查询参数列表
            
        Returns:
            pd.DataFrame: 查询结果
        """
        # 安全检查：只允许SELECT语句
        sql_query_upper = sql_query.strip().upper()
        if not sql_query_upper.startswith('SELECT'):
            logger.warning(f"Refused to execute non-SELECT query: {sql_query}")
            return pd.DataFrame()
            
        try:
            # 使用 pandas read_sql_query 直接从 SQLite 读取
            # params needs to be list/tuple
            df = pd.read_sql_query(sql_query, self.conn, params=params)
            
            # 转换日期列
            if 'dt' in df.columns:
                 df['dt'] = pd.to_datetime(df['dt']).dt.strftime('%Y-%m-%d')
            
            return df
            
        except Exception as e:
            logger.error(f"SQL execution error: {e}")
            logger.error(f"Query: {sql_query}")
            logger.error(f"Params: {params}")
            return pd.DataFrame()
    
    def get_dates_from_qa(self, qa: Dict) -> List[str]:
        """
        从QA对象中提取使用的日期
        
        Args:
            qa: QA对象字典
            
        Returns:
            List[str]: 日期列表（格式：YYYY-MM-DD）
        """
        dates = []
        metadata = qa.get('metadata', {})

        # 从metadata提取
        if 'date' in metadata and isinstance(metadata['date'], str):
            dates.append(metadata['date'])
        if 'dates' in metadata and isinstance(metadata['dates'], list):
            dates.extend([d for d in metadata['dates'] if isinstance(d, str)])
        
        # 从date字段提取（单日期）
        if 'date' in qa and qa['date']:
            date_str = qa['date']
            if isinstance(date_str, str):
                dates.append(date_str)
        
        # 从dates字段提取（多日期）
        if 'dates' in qa and qa['dates']:
            for date_val in qa['dates']:
                if isinstance(date_val, str):
                    dates.append(date_val)
        
        # 从calculation_results的data_used中提取日期
        if 'calculation_results' in qa and qa['calculation_results']:
            for calc_result in qa['calculation_results']:
                if 'data_used' in calc_result:
                    data_used = calc_result['data_used']
                    if isinstance(data_used, dict):
                        # 提取dates字段
                        if 'dates' in data_used and isinstance(data_used['dates'], list):
                            dates.extend([d for d in data_used['dates'] if isinstance(d, str)])
                        # 提取date字段
                        if 'date' in data_used and isinstance(data_used['date'], str):
                            dates.append(data_used['date'])
                        # 提取historical_dates字段
                        if 'historical_dates' in data_used and isinstance(data_used['historical_dates'], list):
                            dates.extend([d for d in data_used['historical_dates'] if isinstance(d, str)])
                        # 提取current_date字段
                        if 'current_date' in data_used and isinstance(data_used['current_date'], str):
                            dates.append(data_used['current_date'])
        
        # 去重并排序
        dates = sorted(list(set(dates)))
        return dates
    
    def query_data_by_ids(self, data_ids: List[int]) -> pd.DataFrame:
        """
        通过data_ids查询数据库中的数据行
        
        Args:
            data_ids: 数据ID列表
            
        Returns:
            pd.DataFrame: 查询结果
        """
        if not data_ids:
            return pd.DataFrame()
        
        try:
            placeholders = ','.join(['?'] * len(data_ids))
            query = f"""
                SELECT f.*, s.stock_name, s.company_name, s.gics, s.industry
                FROM stock_jbm_factors f
                LEFT JOIN stock_data s ON s.stock_code = f.stcode
                WHERE f.id IN ({placeholders})
                ORDER BY f.dt
            """
            
            df = pd.read_sql_query(query, self.conn, params=tuple(data_ids))
            return df
            
        except Exception as e:
            print(f"查询数据失败: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def get_adjacent_dates_data(self, qa: Dict, adjacent_days: int = 0) -> pd.DataFrame:
        """
        获取相邻日期的数据作为噪声数据
        
        Args:
            qa: QA对象字典
            adjacent_days: 相邻天数（前后各多少天）
            
        Returns:
            pd.DataFrame: 相邻日期的数据
        """
        if adjacent_days <= 0:
            return pd.DataFrame()
        
        dates = self.get_dates_from_qa(qa)
        if not dates:
            return pd.DataFrame()
        
        # 获取所有需要查询的公司代码（支持多公司对比）
        stcodes = []
        
        # 优先从company_group获取（多公司对比）
        if 'company_group' in qa and isinstance(qa['company_group'], list):
            for company in qa['company_group']:
                if isinstance(company, dict) and 'stock_code' in company:
                    stock_code = company['stock_code']
                    stcode = stock_code.split('.')[0] if '.' in stock_code else stock_code
                    if stcode and stcode not in stcodes:
                        stcodes.append(stcode)
        
        # 如果没有company_group，从table_id或report_id获取（单公司）
        if not stcodes:
            table_id = qa.get('table_id') or qa.get('report_id')
            if table_id:
                stcode = table_id.split('.')[0] if '.' in table_id else table_id
                if stcode:
                    stcodes.append(stcode)
        
        if not stcodes:
            return pd.DataFrame()
        
        try:
            # 获取QA使用的日期范围
            date_objs = [datetime.strptime(d, '%Y-%m-%d').date() for d in dates]
            qa_min_date = min(date_objs)
            qa_max_date = max(date_objs)
            
            # 计算扩展后的日期范围
            expanded_min_date = qa_min_date - timedelta(days=adjacent_days)
            expanded_max_date = qa_max_date + timedelta(days=adjacent_days)
            
            # 查询所有公司的相邻日期数据
            all_dfs = []
            
            for stcode in stcodes:
                # 首先查询该公司的数据库日期范围，确保不超出数据库的日期下限和上限
                date_range_query = """
                    SELECT MIN(f.dt) as min_dt, MAX(f.dt) as max_dt
                    FROM stock_jbm_factors f
                    WHERE f.stcode = ?
                """
                try:
                    range_df = pd.read_sql_query(date_range_query, self.conn, params=(stcode,))
                    if range_df.empty or pd.isna(range_df.iloc[0]['min_dt']):
                        continue
                     # 获取数据库中的日期范围
                    db_min_date = range_df.iloc[0]['min_dt']
                    db_max_date = range_df.iloc[0]['max_dt']

                except Exception:
                    continue
                
                # 确保db_min_date和db_max_date是date对象
                if isinstance(db_min_date, str):
                    db_min_date = datetime.strptime(db_min_date, '%Y-%m-%d').date()
                elif hasattr(db_min_date, 'date'):
                    db_min_date = db_min_date.date() if hasattr(db_min_date, 'date') else db_min_date
                
                if isinstance(db_max_date, str):
                    db_max_date = datetime.strptime(db_max_date, '%Y-%m-%d').date()
                elif hasattr(db_max_date, 'date'):
                    db_max_date = db_max_date.date() if hasattr(db_max_date, 'date') else db_max_date
                
                # 确保不超出数据库的日期范围
                query_min_date = max(expanded_min_date, db_min_date)
                query_max_date = min(expanded_max_date, db_max_date)
                
                # 如果查询范围无效，跳过该公司
                if query_min_date > query_max_date:
                    continue
                
                # 查询该公司的相邻日期数据
                query = """
                    SELECT f.*, s.stock_name, s.company_name, s.gics, s.industry
                    FROM stock_jbm_factors f
                    LEFT JOIN stock_data s ON s.stock_code = f.stcode
                    WHERE f.stcode = ?
                    AND f.dt >= ? AND f.dt <= ?
                    ORDER BY f.dt
                """
                df_company = pd.read_sql_query(query, self.conn, params=(stcode, str(query_min_date), str(query_max_date)))

                if not df_company.empty:
                     all_dfs.append(df_company)
            
            # 合并所有公司的数据（过滤掉空的DataFrame以避免警告）
            if all_dfs:
                # 过滤掉空的DataFrame
                non_empty_dfs = [df for df in all_dfs if not df.empty]
                if non_empty_dfs:
                    df_combined = pd.concat(non_empty_dfs, ignore_index=True)
                    return df_combined
                else:
                    return pd.DataFrame()
            else:
                return pd.DataFrame()
            
        except Exception as e:
            print(f"查询相邻日期数据失败: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def build_answer_prompt(self, qa: Dict, df_data: pd.DataFrame, 
                           company_info: Dict[str, str]) -> str:
        """
        构建第二阶段prompt：基于查询到的数据回答问题
        
        Args:
            qa: QA对象字典
            df_data: 查询到的数据DataFrame
            company_info: 公司信息字典
            
        Returns:
            str: prompt文本
        """
        question = qa.get('question', '')
        stcode = company_info.get('stcode', '')
        company_name = company_info.get('company_name', '')
        
        # 获取实际使用的列
        actual_columns = list(df_data.columns)
        
        # 排除不需要的字段
        exclude_from_table = {'gics', 'industry', 'company_name', 'stock_name'}
        columns_to_keep = [col for col in actual_columns if col not in exclude_from_table]
        df_filtered = df_data[columns_to_keep].copy()
        
        # 按日期排序
        if 'dt' in df_filtered.columns:
            df_filtered = df_filtered.sort_values('dt')
        
        # 构建字段含义说明（使用实际存在的列）
        field_descriptions = []
        for col in columns_to_keep:
            meaning = FIELD_MEANINGS.get(col, col)
            field_descriptions.append(f"- {col}: {meaning}")
        
        field_descriptions_text = "\n".join(field_descriptions)
        
        # 构建公司信息说明
        company_info_text = f"股票代码(stcode): {stcode}"
        if company_name:
            company_info_text += f"\n公司名称: {company_name}"
        
        # 转换为JSON格式
        data_json = df_filtered.to_json(orient='records', force_ascii=False, date_format='iso')
        data_records = json.loads(data_json)
        
        prompt = f"""你是一个金融数据分析助手。请根据提供的数据回答以下问题。

## 公司信息
{company_info_text}

## 字段含义说明
{field_descriptions_text}

## 数据表格
{json.dumps(data_records, ensure_ascii=False, indent=2)}

## 问题
{question}

## 要求
请严格按照以下JSON格式返回你的答案，不要添加任何其他文字说明：

```json
{{
  "answer": "是" 或 "否",
  "calculation_process": "详细的计算步骤说明，展示每一步的计算过程",
  "formulas_used": [
    {{
      "formula": "公式表达式",
      "description": "公式的含义说明"
    }}
  ],
  "data_used": [
    {{
      "id": 数据行ID,
      "field": "字段名",
      "value": 字段值,
      "stcode": "股票代码"
    }}
  ]
}}
```

**重要提示**：
- answer字段必须是"是"或"否"
- calculation_process字段应详细说明计算步骤
- formulas_used是一个数组，包含所有使用的公式
- data_used是一个数组，包含所有使用的数据，每个数据项必须包含id、field、value和stcode字段
- 请确保返回的是有效的JSON格式，可以直接被解析"""
        return prompt
    
    def answer_question_with_data(self, qa: Dict, df_data: pd.DataFrame, 
                                  company_info: Dict[str, str]) -> Dict[str, Any]:
        """
        第二阶段：让LLM基于数据回答问题
        
        Args:
            qa: QA对象字典
            df_data: 查询到的数据
            company_info: 公司信息字典
            
        Returns:
            Dict: 包含答案的结果
        """
        prompt = self.build_answer_prompt(qa, df_data, company_info)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            if not response or not response.choices or not response.choices[0].message:
                return {
                    'success': False,
                    'error': 'API响应无效'
                }
            
            message_content = response.choices[0].message.content
            if not message_content:
                return {
                    'success': False,
                    'error': 'API返回的content为空'
                }
            
            llm_answer_raw = message_content.strip()
            
            # 解析JSON响应
            import re
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', llm_answer_raw, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_match = re.search(r'\{.*\}', llm_answer_raw, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = llm_answer_raw
            
            llm_answer_parsed = json.loads(json_str)
            
            return {
                'success': True,
                'llm_answer_raw': llm_answer_raw,
                'llm_answer_parsed': llm_answer_parsed,
                'llm_answer_json': json_str
            }
            
        except json.JSONDecodeError as e:
            return {
                'success': False,
                'error': f'JSON解析失败: {str(e)}',
                'llm_answer_raw': message_content if 'message_content' in locals() else llm_answer_raw if 'llm_answer_raw' in locals() else None
            }
        except Exception as e:
            import traceback
            return {
                'success': False,
                'error': str(e),
                'error_traceback': traceback.format_exc()
            }
    
    def build_prompt(self, qa: Dict, df_required: pd.DataFrame, 
                    df_noise: pd.DataFrame = None,
                    extra_fields_ratio: float = 0.0,
                    indicators: Set[str] = None) -> str:
        """
        构建给LLM的prompt
        
        Args:
            qa: QA对象字典
            df_required: 必需的数据（从data_ids查询得到）
            df_noise: 噪声数据（可选）
            extra_fields_ratio: 额外字段的比例或数量。
                对于多字段问题：表示比例（0.0-1.0），在必需字段基础上增加该比例的字段。
                对于单字段问题（L1）：如果值<=1.0，则乘以10作为绝对数量（如0.5→5个字段）；
                如果值>1.0，则直接作为绝对数量。
            indicators: 使用的指标字段集合
            
        Returns:
            str: 构建的prompt
        """
        import random
        
        question = qa.get('question', '')
        
        # 确定要包含的字段（确保所有数据使用统一的字段集合）
        # 注意：id和stcode字段需要保留，以便LLM能够输出数据行ID和识别公司
        # 注意：在表格中不包含gics、industry、company_name字段，只保留stcode
        
        # 首先确定必需字段
        if indicators is None:
            indicators = self.extract_indicators_from_qa(qa)
        
        # 必需的元数据字段（只包含id、stcode、dt，不包含gics、industry、company_name）
        required_meta_fields = {'id', 'stcode', 'dt'}
        # 获取所有需要的字段
        required_fields = required_meta_fields.union(indicators)
        
        # 确定可用的字段（从df_required和df_noise的交集中选择）
        if df_noise is not None and not df_noise.empty:
            available_columns = set(df_required.columns) & set(df_noise.columns)
        else:
            available_columns = set(df_required.columns)
        
        # 排除不需要在表格中显示的字段
        exclude_from_table = {'gics', 'industry', 'company_name', 'stock_name'}
        available_columns = available_columns - exclude_from_table
        
        # 只保留存在的必需字段
        columns_to_include = [col for col in required_fields if col in available_columns]
        
        # 如果需要添加额外字段
        if extra_fields_ratio > 0.0:
            # 获取所有可用的额外字段（排除已包含的字段和不需要的字段）
            extra_available = available_columns - set(columns_to_include)
            # 排除元数据字段（这些已经包含在必需字段中）
            extra_available = extra_available - required_meta_fields
            
            if extra_available:
                # 计算指标字段数量（不包括元数据字段）
                num_indicator_fields = len([col for col in columns_to_include if col not in required_meta_fields])
                
                # 对于单字段问题（L1问题），使用不同的计算方式
                # 让extra_fields_ratio表示绝对数量或使用一个倍数
                if num_indicator_fields == 1:
                    # L1单字段问题：extra_fields_ratio表示要添加的字段数量
                    # 例如：extra_fields_ratio=5.0 表示添加5个额外字段
                    # 或者：extra_fields_ratio=0.5 表示添加 int(0.5 * 10) = 5 个额外字段
                    # 使用一个倍数（如10）来转换比例到绝对数量
                    if extra_fields_ratio <= 1.0:
                        # 如果ratio <= 1.0，认为是比例，乘以10转换为绝对数量
                        num_extra = max(1, int(extra_fields_ratio * 10))
                    else:
                        # 如果ratio > 1.0，认为是绝对数量
                        num_extra = int(extra_fields_ratio)
                else:
                    # 多字段问题：使用原来的比例计算方式
                    # 例如：如果已使用10个字段，extra_fields_ratio=0.5，则添加5个额外字段
                    num_required_fields = len(columns_to_include)
                    num_extra = max(1, int(num_required_fields * extra_fields_ratio))
                
                # 确保不超过所有可用额外字段的数量（上限是整个表格的可用字段）
                num_extra = min(num_extra, len(extra_available))
                
                # 随机选择额外字段
                if num_extra > 0:
                    extra_selected = random.sample(list(extra_available), num_extra)
                    columns_to_include.extend(extra_selected)
        
        # 确保columns_to_include不为空
        if not columns_to_include:
            columns_to_include = ['dt']  # 至少包含日期字段
        
        # 合并必需数据和噪声数据（使用统一的字段集合）
        if df_noise is not None and not df_noise.empty:
            # 确保两个DataFrame都只包含相同的列
            required_cols = [col for col in columns_to_include if col in df_required.columns]
            noise_cols = [col for col in columns_to_include if col in df_noise.columns]
            
            # 使用公共列
            common_cols = list(set(required_cols) & set(noise_cols))
            if not common_cols:
                common_cols = ['dt']  # 至少包含日期字段
            
            df_required_subset = df_required[common_cols].copy()
            df_noise_subset = df_noise[common_cols].copy()
            
            df_combined = pd.concat([df_required_subset, df_noise_subset], ignore_index=True)
            
            # 去重（基于id和dt，如果存在id字段）
            if 'id' in df_combined.columns and 'dt' in df_combined.columns:
                df_combined = df_combined.drop_duplicates(subset=['id', 'dt'], keep='first')
            elif 'dt' in df_combined.columns:
                # 如果没有id字段，只基于dt去重
                df_combined = df_combined.drop_duplicates(subset=['dt'], keep='first')
        else:
            # 只使用必需数据，确保只包含指定的列
            available_cols = [col for col in columns_to_include if col in df_required.columns]
            if not available_cols:
                available_cols = ['dt']
            df_combined = df_required[available_cols].copy()
        
        # 按日期排序
        if 'dt' in df_combined.columns:
            df_combined = df_combined.sort_values('dt')
        
        # 最终确保排除不需要的字段（即使之前包含了，也要在这里移除）
        exclude_from_table = {'gics', 'industry', 'company_name', 'stock_name'}
        columns_to_keep = [col for col in df_combined.columns if col not in exclude_from_table]
        df_combined = df_combined[columns_to_keep]
        
        # 获取实际使用的列（从合并后的DataFrame中获取）
        actual_columns = list(df_combined.columns)
        
        # 提取公司信息（优先从QA的company_group字段获取，支持多公司对比）
        # 注意：这些信息不会出现在数据表格中
        company_info_list = []
        
        # 优先从QA的company_group字段获取公司信息（多公司对比）
        if 'company_group' in qa and isinstance(qa['company_group'], list):
            for company in qa['company_group']:
                if isinstance(company, dict):
                    company_info_item = {}
                    if 'stock_code' in company:
                        stock_code = company['stock_code']
                        company_info_item['stcode'] = stock_code.split('.')[0] if '.' in stock_code else stock_code
                    if 'company_name' in company:
                        company_info_item['company_name'] = company['company_name']
                    if company_info_item:
                        company_info_list.append(company_info_item)
        
        # 如果没有company_group，从DataFrame中提取（单公司情况）
        if not company_info_list and 'stcode' in df_required.columns and not df_required.empty:
            # 获取所有唯一的公司信息
            if 'stcode' in df_required.columns and 'company_name' in df_required.columns:
                # 按stcode分组，获取每个公司的信息
                for stcode_val in df_required['stcode'].dropna().unique():
                    company_rows = df_required[df_required['stcode'] == stcode_val]
                    company_info_item = {
                        'stcode': str(stcode_val)
                    }
                    # 获取公司名称
                    company_name = company_rows['company_name'].dropna().iloc[0] if not company_rows['company_name'].dropna().empty else None
                    if company_name:
                        company_info_item['company_name'] = str(company_name)
                    elif 'stock_name' in company_rows.columns:
                        stock_name = company_rows['stock_name'].dropna().iloc[0] if not company_rows['stock_name'].dropna().empty else None
                        if stock_name:
                            company_info_item['stock_name'] = str(stock_name)
                    company_info_list.append(company_info_item)
        
        # 转换为JSON格式
        data_json = df_combined.to_json(orient='records', force_ascii=False, date_format='iso')
        data_records = json.loads(data_json)
        
        # 构建字段含义说明（使用实际存在的列）
        field_descriptions = []
        for col in actual_columns:
            meaning = FIELD_MEANINGS.get(col, col)
            field_descriptions.append(f"- {col}: {meaning}")
        
        field_descriptions_text = "\n".join(field_descriptions)
        
        # 构建公司信息说明（支持多公司）
        company_info_text = ""
        if company_info_list:
            company_info_lines = []
            if len(company_info_list) == 1:
                # 单公司情况
                company_info = company_info_list[0]
                if 'stcode' in company_info:
                    company_info_lines.append(f"股票代码(stcode): {company_info['stcode']}")
                if 'company_name' in company_info:
                    company_info_lines.append(f"公司名称: {company_info['company_name']}")
                elif 'stock_name' in company_info:
                    company_info_lines.append(f"股票名称: {company_info['stock_name']}")
            else:
                # 多公司对比情况
                company_info_lines.append("对比公司信息：")
                for i, company_info in enumerate(company_info_list, 1):
                    company_line = f"  公司{i}: "
                    if 'stcode' in company_info:
                        company_line += f"股票代码(stcode)={company_info['stcode']}"
                    if 'company_name' in company_info:
                        company_line += f", 公司名称={company_info['company_name']}"
                    elif 'stock_name' in company_info:
                        company_line += f", 股票名称={company_info['stock_name']}"
                    company_info_lines.append(company_line)
            
            if company_info_lines:
                company_info_text = "\n".join(company_info_lines)
        
        # 构建prompt
        prompt = f"""你是一个金融数据分析助手。请根据提供的数据回答以下问题。

## 公司信息
{company_info_text if company_info_text else "（未提供公司信息）"}

## 字段含义说明
{field_descriptions_text}

## 数据表格
{json.dumps(data_records, ensure_ascii=False, indent=2)}

## 问题
{question}

## 要求
请严格按照以下JSON格式返回你的答案，不要添加任何其他文字说明：

```json
{{
  "answer": "是" 或 "否",
  "calculation_process": "详细的计算步骤说明，展示每一步的计算过程",
  "formulas_used": [
    {{
      "formula": "公式表达式",
      "description": "公式的含义说明"
    }}
  ],
  "data_used": [
    {{
      "id": 数据行ID,
      "field": "字段名",
      "value": 字段值,
      "stcode": "股票代码"
    }}
  ]
}}
```

**重要提示**：
- answer字段必须是"是"或"否"
- calculation_process字段应详细说明计算步骤
- formulas_used是一个数组，包含所有使用的公式
- data_used是一个数组，包含所有使用的数据，每个数据项必须包含id、field、value和stcode字段
- 请确保返回的是有效的JSON格式，可以直接被解析"""
        return prompt
    
    def test_qa(self, qa: Dict, 
                adjacent_days: int = 0,
                extra_fields_ratio: float = 0.0) -> Dict[str, Any]:
        """
        测试单个QA对（两阶段流程）
        
        Args:
            qa: QA对象字典
            adjacent_days: 相邻天数（已废弃，保留以兼容接口）
            extra_fields_ratio: 额外字段的比例（已废弃，保留以兼容接口）
            
        Returns:
            Dict: 测试结果
        """
        # 记录开始时间
        start_time = time.time()
        
        # 提取标准答案信息（用于后续比对）
        expected_answer = qa.get('answer', '')
        expected_data_ids = set(qa.get('data_ids', []) or qa.get('metadata', {}).get('data_ids', []))
        expected_indicators = self.extract_indicators_from_qa(qa)
        
        # ========== 第一阶段：让LLM决定需要查询的数据 ==========
        print(f"    [阶段1] 让LLM决定需要查询的数据...")
        stage1_start_time = time.time()
        
        # 获取公司信息
        company_info = self.get_company_info_from_qa(qa)
        if 'stcode' not in company_info:
            return {
                'success': False,
                'error': '无法从QA中提取公司stcode信息'
            }
        
        # 获取LLM的查询计划
        query_plan_result = self.get_query_plan_from_llm(qa, company_info)
        
        stage1_end_time = time.time()
        stage1_duration = stage1_end_time - stage1_start_time
        print(f"    ⏱ 阶段1耗时: {stage1_duration:.2f}秒")
        
        if not query_plan_result.get('success'):
            return {
                'success': False,
                'error': f"阶段1失败: {query_plan_result.get('error', 'Unknown error')}",
                'stage1_raw_response': query_plan_result.get('raw_response')
            }
        
        query_plan = query_plan_result['query_plan']
        fields_to_query = query_plan.get('fields_to_query', [])
        date_range_info = query_plan.get('date_range', {})
        
        # 验证返回的数据
        if not fields_to_query or not isinstance(fields_to_query, list):
            return {
                'success': False,
                'error': 'LLM返回的查询计划中没有有效的fields_to_query字段',
                'query_plan': query_plan
            }
        
        stcode = company_info.get('stcode', '')
        if not stcode:
            return {
                'success': False,
                'error': '无法获取公司stcode信息'
            }
        
        print(f"    [阶段1] LLM决定的字段: {fields_to_query}")
        print(f"    [阶段1] LLM决定的日期范围: {date_range_info}")
        
        # 使用代码构建SQL查询
        sql_build_duration = 0.0
        try:
            sql_build_start_time = time.time()
            sql_query, query_params = self.build_sql_query(stcode, fields_to_query, date_range_info)
            sql_build_end_time = time.time()
            sql_build_duration = sql_build_end_time - sql_build_start_time
            print(f"    ⏱ SQL构建耗时: {sql_build_duration:.2f}秒")
            print(f"    [阶段1] 构建的SQL: {sql_query[:200]}...")
            print(f"    [阶段1] 查询参数: {query_params}")
        except Exception as e:
            import traceback
            return {
                'success': False,
                'error': f'构建SQL查询失败: {str(e)}',
                'error_traceback': traceback.format_exc(),
                'query_plan': query_plan
            }
        
        # 执行SQL查询
        try:
            sql_exec_start_time = time.time()
            df_queried = self.execute_sql_query(sql_query, query_params)
            sql_exec_end_time = time.time()
            sql_exec_duration = sql_exec_end_time - sql_exec_start_time
            print(f"    ⏱ SQL执行耗时: {sql_exec_duration:.2f}秒，查询到 {len(df_queried)} 行数据")
            
            if df_queried.empty:
                return {
                    'success': False,
                    'error': 'SQL查询返回空结果',
                    'sql_query': sql_query,
                    'query_params': query_params
                }
        except Exception as e:
            import traceback
            return {
                'success': False,
                'error': f'SQL执行失败: {str(e)}',
                'error_traceback': traceback.format_exc(),
                'sql_query': sql_query,
                'query_params': query_params
            }
        
        # ========== 第二阶段：让LLM基于查询到的数据回答问题 ==========
        print(f"    [阶段2] 让LLM基于数据回答问题...")
        stage2_start_time = time.time()
        
        # 获取LLM的答案
        answer_result = self.answer_question_with_data(qa, df_queried, company_info)
        
        stage2_end_time = time.time()
        stage2_duration = stage2_end_time - stage2_start_time
        print(f"    ⏱ 阶段2耗时: {stage2_duration:.2f}秒")
        
        if not answer_result.get('success'):
            return {
                'success': False,
                'error': f"阶段2失败: {answer_result.get('error', 'Unknown error')}",
                'stage2_raw_response': answer_result.get('llm_answer_raw'),
                'stage1_query_plan': query_plan,
                'num_queried_rows': len(df_queried)
            }
        
        # 解析LLM的答案
        llm_answer_parsed = answer_result.get('llm_answer_parsed')
        llm_answer_raw = answer_result.get('llm_answer_raw', '')
        llm_answer_json = answer_result.get('llm_answer_json', '')
        
        if not llm_answer_parsed:
            return {
                'success': False,
                'error': '阶段2返回的答案解析失败',
                'stage2_raw_response': llm_answer_raw,
                'stage1_query_plan': query_plan
            }
        
        # 提取答案和使用的数据
        llm_answer = llm_answer_parsed.get('answer', '')
        llm_data_used_ids = set()
        llm_data_used_fields = set()
        
        # 从data_used中提取id和field
        if 'data_used' in llm_answer_parsed and isinstance(llm_answer_parsed['data_used'], list):
            for data_item in llm_answer_parsed['data_used']:
                if isinstance(data_item, dict):
                    # 提取id
                    if 'id' in data_item:
                        try:
                            data_id = int(data_item['id'])
                            llm_data_used_ids.add(data_id)
                        except (ValueError, TypeError):
                            pass
                    # 提取field
                    if 'field' in data_item:
                        field_name = str(data_item['field'])
                        llm_data_used_fields.add(field_name)
        
        # 与ground truth进行比对
        answer_match = False
        if llm_answer and expected_answer:
            llm_answer_normalized = llm_answer.strip()
            expected_answer_normalized = expected_answer.strip()
            answer_match = (llm_answer_normalized == expected_answer_normalized)
        
        # 比对data_ids
        data_ids_match = False
        data_ids_precision = 0.0
        data_ids_recall = 0.0
        if llm_data_used_ids and expected_data_ids:
            intersection = llm_data_used_ids & expected_data_ids
            data_ids_match = (llm_data_used_ids == expected_data_ids)
            if len(llm_data_used_ids) > 0:
                data_ids_precision = len(intersection) / len(llm_data_used_ids)
            if len(expected_data_ids) > 0:
                data_ids_recall = len(intersection) / len(expected_data_ids)
        
        # 比对fields（indicators）
        fields_match = False
        fields_precision = 0.0
        fields_recall = 0.0
        if llm_data_used_fields and expected_indicators:
            expected_fields = set(expected_indicators)
            intersection = llm_data_used_fields & expected_fields
            fields_match = (llm_data_used_fields == expected_fields)
            if len(llm_data_used_fields) > 0:
                fields_precision = len(intersection) / len(llm_data_used_fields)
            if len(expected_fields) > 0:
                fields_recall = len(intersection) / len(expected_fields)
        
        total_duration = time.time() - start_time
        
        return {
            'success': True,
            'question': qa.get('question', ''),
            'expected_answer': expected_answer,
            'llm_answer': llm_answer,
            'answer_match': answer_match,
            'expected_data_ids': sorted(list(expected_data_ids)),
            'llm_data_used_ids': sorted(list(llm_data_used_ids)),
            'data_ids_match': data_ids_match,
            'data_ids_precision': data_ids_precision,
            'data_ids_recall': data_ids_recall,
            'expected_fields': sorted(list(expected_indicators)),
            'llm_data_used_fields': sorted(list(llm_data_used_fields)),
            'fields_match': fields_match,
            'fields_precision': fields_precision,
            'fields_recall': fields_recall,
            'llm_answer_raw': llm_answer_raw,
            'llm_answer_parsed': llm_answer_parsed,
            'llm_answer_json': llm_answer_json,
            'stage1_query_plan': query_plan,
            'stage1_raw_response': query_plan_result.get('raw_response'),
            'fields_to_query': fields_to_query,
            'date_range': date_range_info,
            'sql_query': sql_query,
            'query_params': query_params,
            'num_queried_rows': len(df_queried),
            'timing': {
                'stage1_duration': stage1_duration,
                'sql_build_duration': sql_build_duration,
                'sql_exec_duration': sql_exec_duration,
                'stage2_duration': stage2_duration,
                'total_duration': total_duration
            }
            }
    
    def test_qa_file(self, qa_file_path: str,
                    adjacent_days: int = 0,
                    extra_fields_ratio: float = 0.0,
                    max_qa: int = None,
                    start_idx: int = 0) -> List[Dict[str, Any]]:
        """
        测试QA文件中的所有QA对
        
        Args:
            qa_file_path: QA文件路径
            adjacent_days: 相邻天数
            extra_fields_ratio: 额外字段的比例（0.0-1.0）
            max_qa: 最多测试的QA数量
            start_idx: 起始索引
            
        Returns:
            List[Dict]: 测试结果列表
        """
        # 读取QA文件
        with open(qa_file_path, 'r', encoding='utf-8') as f:
            qa_list = json.load(f)
        
        if not isinstance(qa_list, list):
            print(f"错误: {qa_file_path} 不是有效的QA列表")
            return []
        
        # 应用start_idx和max_qa限制
        qa_list = qa_list[start_idx:]
        if max_qa:
            qa_list = qa_list[:max_qa]
        
        results = []
        total = len(qa_list)
        
        print(f"开始测试 {total} 个QA对...")
        
        for i, qa in enumerate(qa_list, 1):
            print(f"[{i}/{total}] 测试QA: {qa.get('question', '')[:50]}...")
            result = self.test_qa(qa, adjacent_days, extra_fields_ratio)
            result['qa_index'] = i - 1 + start_idx
            results.append(result)
            
            if not result['success']:
                error_msg = result.get('error', 'Unknown error')
                error_type = result.get('error_type', '')
                print(f"  ✗ 失败: {error_msg}")
                if error_type:
                    print(f"    错误类型: {error_type}")
                
                # 输出API响应信息（如果有）
                if 'api_response' in result:
                    api_resp = result.get('api_response')
                    if api_resp:
                        print(f"    API响应: {str(api_resp)[:200]}...")  # 只显示前200个字符
                
                if 'response_choices' in result:
                    choices = result.get('response_choices')
                    if choices:
                        print(f"    响应choices: {str(choices)[:200]}...")
                
                # 输出错误堆栈（如果有）
                if 'error_traceback' in result:
                    traceback_lines = result.get('error_traceback', '').split('\n')
                    print(f"    错误堆栈（最后5行）:")
                    for line in traceback_lines[-5:]:
                        if line.strip():
                            print(f"      {line}")
                
                # 输出原始响应（如果有）
                if 'llm_answer_raw' in result:
                    raw_answer = result.get('llm_answer_raw', '')
                    if raw_answer:
                        print(f"    模型原始返回: {raw_answer[:300]}...")  # 只显示前300个字符
            else:
                # 显示比对结果
                answer_status = "✓" if result.get('answer_match', False) else "✗"
                data_ids_status = "✓" if result.get('data_ids_match', False) else "✗"
                fields_status = "✓" if result.get('fields_match', False) else "✗"
                print(f"  ✓ 成功 | 答案:{answer_status} | 数据ID:{data_ids_status} | 字段:{fields_status}")
        
        return results


def run(config):
    """
    统一入口函数
    """
    # 提取配置
    model = config.get('model', 'deepseek-chat')
    qa_file = config.get('input_path')
    output_dir = config.get('output_dir', 'eval_results/alignment')
    
    # 自动查找默认 QA 文件
    files_to_process = []
    if not qa_file:
         if os.path.exists(os.path.join("data", "Alignment")):
              # 尝试找 alignment 下的 json，全部处理
              scan_dir = os.path.join("data", "Alignment")
              for f in os.listdir(scan_dir):
                   if f.endswith('.json'):
                        files_to_process.append(os.path.join(scan_dir, f))
    elif os.path.isdir(qa_file):
        for f in os.listdir(qa_file):
            if f.endswith('.json'):
                files_to_process.append(os.path.join(qa_file, f))
    else:
        files_to_process.append(qa_file)
    
    if not files_to_process:
        logger.error(f"No QA files found to process.")
        return

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 初始化测试器
    logger.info("="*70)
    logger.info("QA LLM测试系统 (Alignment)")
    logger.info("="*70)
    logger.info(f"[1/2] 初始化LLM测试器（模型：{model}）...")
    
    # 无需连接数据库，直接初始化
    tester = QALLMTester(
        model=model, 
        api_key=config.get('api_key'),
        base_url=config.get('base_url')
    )
    logger.info("✓ 测试器初始化成功 (Loading CSV to SQLite)")
    
    for qa_file_path in files_to_process:
        # 测试QA文件
        logger.info(f"\n[2/2] 处理QA文件: {qa_file_path}")
        
        results = tester.test_qa_file(
            qa_file_path,
            adjacent_days=config.get('adjacent_days', 10),
            extra_fields_ratio=config.get('extra_fields_ratio', 0.5),
            max_qa=config.get('max_qa'),
            start_idx=config.get('start_idx', 0)
        )
        
        # 保存结果
        logger.info(f"保存测试结果...")
        
        qa_file_name = Path(qa_file_path).stem
        model_name_clean = model.replace('/', '_').replace('-', '_')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(output_dir, f"test_results_{qa_file_name}_{model_name_clean}_{timestamp}.json")
        
        # 构建LLM输入描述
        llm_input_description = """两阶段测试流程：
    阶段1：让LLM决定需要查询的数据
      - 输入：公司信息（stcode、公司名称）、问题、数据库所有字段含义
      - 输出：LLM生成的SQL查询语句和查询参数
      - 执行：使用LLM生成的SQL查询数据库，获取数据
    阶段2：让LLM基于查询到的数据回答问题
      - 输入：查询到的数据表格、问题、字段含义说明
      - 输出：LLM的答案（是/否）及使用的数据和计算过程"""
        
        # 计算统计摘要
        success_results = [r for r in results if r.get('success', False)]
        failed_results = [r for r in results if not r.get('success', False)]
        
        summary = {
            'total': len(results),
            'success': len(success_results),
            'failed': len(failed_results),
            'answer_match': sum(1 for r in success_results if r.get('answer_match', False)),
            'data_ids_match': sum(1 for r in success_results if r.get('data_ids_match', False)),
            'fields_match': sum(1 for r in success_results if r.get('fields_match', False)),
            'avg_data_ids_precision': sum(r.get('data_ids_precision', 0) for r in success_results) / max(len(success_results), 1),
            'avg_data_ids_recall': sum(r.get('data_ids_recall', 0) for r in success_results) / max(len(success_results), 1),
            'avg_fields_precision': sum(r.get('fields_precision', 0) for r in success_results) / max(len(success_results), 1),
            'avg_fields_recall': sum(r.get('fields_recall', 0) for r in success_results) / max(len(success_results), 1)
        }
        
        # 计算率
        summary['answer_match_rate'] = summary['answer_match'] / max(summary['success'], 1)
        summary['data_ids_match_rate'] = summary['data_ids_match'] / max(summary['success'], 1)
        summary['fields_match_rate'] = summary['fields_match'] / max(summary['success'], 1)

        output_data = {
            'test_config': {
                'qa_file': qa_file_path,
                'model': model,
                'llm_input_description': llm_input_description
            },
            'results': results,
            'summary': summary
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✓ 结果已保存到: {output_path}")
        logger.info(f"  总计: {summary['total']}, 成功: {summary['success']}, 失败: {summary['failed']}")
        logger.info(f"  答案匹配率: {summary['answer_match_rate']*100:.2f}%")
    
    # 关闭连接
    tester.close()
    logger.info("✓ 评测完成")

