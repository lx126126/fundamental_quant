# 📈 Fundamental Quant Framework (`fqf`)

> **量化基本面投研框架**：集宏观/行业监测、个股标准化分析、策略构建与自动化回测于一体的深度投研基座。

---

## 🎯 项目愿景

本项目旨在通过工程化手段，将传统基本面投资中的"手动搜集"与"直觉决策"转化为**可编程、可验证、可重复**的自动化流水线。

## 🚀 快速开始

### 环境要求

- Python 3.10+

### 安装

```bash
git clone https://github.com/lx126126/fundamental_quant.git
cd fundamental_quant
pip install -e ".[dev]"
```

### 5 分钟跑通第一个分析

```python
from fqf.data import AKShareFetcher
from fqf.analyzer import FinancialCleaner, FinancialMetrics, ReportGenerator

# 1. 拉取贵州茅台财务数据
fetcher = AKShareFetcher()
bs_raw = fetcher.fetch_balance_sheet("600519")
is_raw = fetcher.fetch_income_statement("600519")
cf_raw = fetcher.fetch_cash_flow("600519")

# 2. 标准化清洗
cleaner = FinancialCleaner()
bs = cleaner.clean_balance_sheet(bs_raw)
is_ = cleaner.clean_income_statement(is_raw)
cf = cleaner.clean_cash_flow(cf_raw)

# 3. 计算 40+ 财务指标
metrics = FinancialMetrics()
all_metrics = metrics.compute_all(bs, is_, cf)
score = metrics.score(all_metrics)
print(f"综合评分: {score.get('composite_score', 0):.0f} / 100")

# 4. 生成 Value Line 一页式投研报告
report = ReportGenerator().generate(
    company_name="贵州茅台", symbol="600519",
    balance_sheet=bs, income_statement=is_, cash_flow=cf,
    valuation={"pe": 28.5, "pb": 9.2},
)
print(report)
```

> 完整流程演示见 [examples/demo_analyzer.ipynb](examples/demo_analyzer.ipynb)（含 mock 数据，无需网络即可运行）

## 🏗️ 系统架构

项目的核心逻辑遵循"数据驱动决策"的漏斗模型：

1. **宏观与行业监测 (Sentinel)**: 追踪流动性、利率、行业估值水位线，生成每日宏观简报。
2. **标准化投研逻辑 (Analyzer)**: 统一个股财务指标评价体系，生成 Value Line 风格的一页投研报告。
3. **策略构建与回测 (Strategy & Backtest)**: 将投资哲学转化为因子，通过历史数据验证逻辑。

```
Sentinel ──→ Analyzer ──→ Strategy & Backtest
(宏观/行业)    (个股投研)      (因子 + 绩效)
```

## 🛠️ 功能模块

### 1. 宏观/行业监测 (Sentinel) — 规划中

- **宏观因子**: 10Y 国债收益率、M2-M1 剪刀差、PMI 等 20+ 指标，基于规则引擎 + LLM 生成每日宏观简报。
- **行业脉搏**: 自动跟踪申万 31 个一级行业 ETF 的 PE/PB 历史分位点，生成行业观测报告。

### 2. 标准化投研 (Analyzer) — ✅ 已实现

- **财务三表自动化**: 自动清洗标准化资产负债表、利润表及现金流量表（80+ 字段映射、千分位/中文数值转换）。
- **核心指标评分**: ROE 杜邦分解、毛利率趋势、自由现金流覆盖率等 40+ 指标。
- **投研报告生成**: 生成 Value Line 一页式投研报告（Markdown 格式）。

### 3. 策略与回测 (Strategy & Backtest) — 规划中

- **因子化建模**: 将基本面逻辑（如：低估值 + 业绩反转）封装为可执行策略。
- **绩效评估**: 自动计算夏普比率、最大回撤、信息比率等指标。

## 📊 数据源

| 数据类型 | 来源 | 接口 | 缓存 |
|---------|------|------|------|
| 行情数据 | AKShare | `fetch_daily_hist` / `fetch_index_daily` | 1 天 |
| 财务数据 | AKShare | `fetch_balance_sheet` / `fetch_income_statement` / `fetch_cash_flow` | 5 天 |
| 估值数据 | AKShare | `fetch_valuation` / `fetch_stock_list` | 5 分钟 |
| 宏观数据 | AKShare | `fetch_macro_indicator` (PMI/M2/国债/CPI/PPI/LPR/社融) | 1 天 |
| 行业数据 | AKShare | `fetch_industry_list` / `fetch_industry_index` / `fetch_industry_constituents` | 1 天 |

## 📁 项目结构

```
fundamental_quant/
├── fqf/
│   ├── data/                     # 数据层（✅ 已实现）
│   │   ├── fetcher.py            #   DataFetcher ABC 基类 + 缓存
│   │   ├── akshare_fetcher.py    #   AKShareFetcher 完整实现
│   │   └── cleaner.py            #   DataCleaner（异常值 + 缺失值）
│   ├── analyzer/                 # 标准化投研（✅ 已实现）
│   │   ├── financial.py          #   FinancialCleaner（三表清洗）
│   │   ├── metrics.py            #   FinancialMetrics（40+ 指标 + 评分）
│   │   └── report.py             #   ReportGenerator（Value Line 报告）
│   ├── sentinel/                 # 宏观/行业监测（📋 骨架）
│   └── strategy/                 # 策略与回测（📋 骨架）
├── examples/
│   └── demo_analyzer.ipynb       # Analyzer 完整演示（含输出）
├── tests/
│   ├── test_data.py              # 16 项数据层测试
│   └── test_analyzer.py          # 35 项分析模块测试
├── DESIGN.md                     # 详细设计文档
└── pyproject.toml
```

## 🚧 开发状态

| Phase | 内容 | 状态 |
|-------|------|------|
| 详细设计文档 | DESIGN.md（50+ 页） | ✅ 完成 |
| Phase 1：数据层 | AKShare 全接口封装，parquet 缓存 | ✅ 完成 |
| Phase 1：Analyzer | 三表清洗 + 40+ 指标 + Value Line 报告 | ✅ 完成 |
| Phase 1：测试 | 51 项单元测试，覆盖率 > 90% | ✅ 完成 |
| Phase 1：示例 | demo_analyzer.ipynb 全流程演示 | ✅ 完成 |
| Phase 2 | Sentinel 宏观/行业监测 | 📋 待开发 |
| Phase 3 | Strategy + Backtest 策略与回测 | 📋 待开发 |
| Phase 4 | 打磨与文档 | 📋 待开发 |

## 📄 License

MIT
