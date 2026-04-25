# A-Share All-Weather Quant System — Technical Design (v1.0)

> **作者:** Vincent Liu
> **日期:** 2026-04-26
> **状态:** Draft → Phase 1 实施中
> **范围:** A 股全天候量化系统的双层架构设计、roadmap 与工程落地方案

---

## 0. 一句话摘要

我们不再造第三套系统，而是把 **`pysystemtrade`（成熟风险预算 / target-vol / 成本建模）** 与 **`online_system`（多因子 / regime / ML 友好）** 通过一个 **Signal Contract + Meta Allocator** 融合成 **Core + Active + State** 三层 A 股全天候系统，在 8–10 周内交付一个**风险可控、可进化、对家办可讲清楚**的产品。

---

## 1. 背景与定位

### 1.1 为什么是"全天候"

家办与稳健型投资人最关注三件事，恰好是全天候的卖点：

1. **稳健性**：极端市场也能活下来
2. **可扩展性 (capacity)**：能容纳大资金而不被自己反噬
3. **可进化 (evolving)**：技术壁垒持续累积，不是一锤子 alpha

我们对外讲的不是"我们做了一个策略"，而是：

> 我们在构建一个**以风险预算为核心、目标波动率为约束、宏观状态识别和多策略调度为增强**的 AI-driven all-weather allocation system。

### 1.2 现状盘点

| 系统 | 优势 | 短板 |
|------|------|------|
| `pysystemtrade` | 风险预算、EWMA 协方差、IDM/FDM 分散化、target vol、成本/容量建模、IBKR 生产链路 | 无 ML、纯规则、原生面向期货 |
| `online_system` | 多因子 scanner、regime filter、ML 友好、Web Dashboard、experiments 归档 | 风险模型偏简单、无目标波动率引擎、容量建模缺失 |
| **数据层 (已统一)** | 本地 PG 229,836 行 + Supabase 远程 DR；fetcher daemon；193 instruments + ETF | — |

**结论:** 两套系统在能力上几乎完美互补，融合而非重写是最优解。

---

## 2. 战略架构

### 2.1 三层定位

```
┌──────────────────────────────────────────────────────────────────────┐
│                         DATA LAYER (已就绪 ✅)                        │
│                                                                       │
│  本地 PG (astock_daily_prices 229k) ──sync──▶ Supabase (远程 DR)     │
│  fetcher.py (pm2) · xiximiao_client · ETF/宏观 (Phase 1 扩展)         │
└──────────────────────────────────────────────────────────────────────┘
            │                       │                       │
            ▼                       ▼                       ▼
┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│   CORE LAYER     │   │  ACTIVE LAYER    │   │   STATE LAYER    │
│  (pysystemtrade) │   │  (online_system) │   │  (新增 · 薄)     │
│                  │   │                  │   │                  │
│ AStockSimData    │   │ factor_scanner   │   │ RegimeHMM        │
│ → rawdata        │   │ → FactorStrategy │   │ MacroFeatures    │
│ → forecast scale │   │ → ActiveSignal   │   │ VolForecaster    │
│ → vol scalar     │   │   (contract)     │   │ StateEmbedding   │
│ → portfolio      │   │                  │   │                  │
│ → cost model     │   │ output:          │   │ output:          │
│                  │   │   list[Signal]   │   │   regime, σ̂,    │
│ output: core_w   │   │                  │   │   embedding      │
└──────────────────┘   └──────────────────┘   └──────────────────┘
            │                       │                       │
            └───────────────────────┼───────────────────────┘
                                    ▼
                       ┌──────────────────────────┐
                       │  META ALLOCATOR (新增)   │
                       │                          │
                       │  bucket_weights(         │
                       │    core, active, state   │
                       │  ) → target_weights      │
                       │                          │
                       │  约束:                   │
                       │   · target vol 6%        │
                       │   · risk parity 桶       │
                       │   · max ADV%             │
                       │   · turnover penalty     │
                       └──────────────────────────┘
                                    │
                                    ▼
                       ┌──────────────────────────┐
                       │   EXECUTION & RISK       │
                       │                          │
                       │  Rebalancer → RiskMgr →  │
                       │  Broker (Sim / IBKR)     │
                       │  Reconciler · KillSwitch │
                       └──────────────────────────┘
                                    │
                                    ▼
                       ┌──────────────────────────┐
                       │  REPORTING & DASHBOARD   │
                       │  online_system/web/      │
                       │  · equity / drawdown     │
                       │  · regime timeline       │
                       │  · bucket allocation     │
                       │  · factor attribution    │
                       └──────────────────────────┘
```

### 2.2 各层职责

#### Core Layer（pysystemtrade）— "Beta 做干净"
- **目标**：在 ETF（HS300 / 中证 500 / 红利 / 国债 / 黄金 / 现金）上做风险预算 + 目标波动率配置
- **关键能力**：EWMA 协方差、IDM/FDM 分散化乘数、buffering 抑制换手、A 股成本模型
- **输出**：稳健的 ETF 级 weights 与 `realized_vol_fcst`
- **容量**：极高（ETF），目标年化 vol 6%

#### Active Layer（online_system）— Smart Beta / Alpha
- **目标**：在个股上叠加 value / quality / momentum / low-vol 因子，输出方向性信号
- **关键能力**：`factor_scanner` 多因子打分、preset 切换、cross-sectional 排序
- **输出**：`list[ActiveSignal]`（symbol, score ∈ [-1,1], horizon, capacity）

#### State Layer（新增 · 薄）— Regime / Vol / Embedding
- **目标**：识别宏观状态 + 预测波动率 + 提供市场表征
- **关键能力**：HMM regime detection、GARCH/LightGBM vol forecast、Transformer encoder（Phase 3）
- **输出**：`RegimeState`, `vol_forecast`, `state_embedding`

#### Meta Allocator（新增）— 跨桶调度
- **输入**：Core 输出 + Active signals + State + 协方差矩阵 + 容量约束
- **输出**：最终 `target_weights`，满足 risk budget / target vol / capacity 约束
- **演进**：v1 risk parity 在桶级 → v2 ML state-conditional 桶权重（带 safe wrapper）

---

## 3. 关键接口契约

### 3.1 ActiveSignal（Core ↔ Active 解耦的关键）

```python
# online_system/core/signal_contract.py
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class ActiveSignal:
    symbol: str
    score: float            # ∈ [-1, 1]，注意不是收益预测，是相对强度
    horizon_days: int       # 信号有效期（如 5/20/60）
    confidence: float       # ∈ [0, 1]
    source: str             # "smart_beta.value" / "momentum.20d" / ...
    capacity_pct_adv: float # 建议最大占该 symbol ADV 比例（容量约束）

@dataclass(frozen=True)
class CoreAllocation:
    weights: dict[str, float]            # ETF 级别权重
    risk_contribution: dict[str, float]  # 各 ETF 的风险贡献
    realized_vol_fcst: float             # ex-ante 组合年化波动率预测
    method: str                          # "risk_parity" / "erc" / ...

@dataclass(frozen=True)
class RegimeState:
    label: str              # "bull_low_vol" / "bear_high_vol" / "transition" / ...
    confidence: float
    vol_forecast: float     # 未来 N 天预测年化波动率
    embedding: list[float] | None  # Phase 3 起非空

@dataclass(frozen=True)
class MetaInput:
    core: CoreAllocation
    active: list[ActiveSignal]
    state: RegimeState
    cov_matrix: "pd.DataFrame"
    capacity_constraints: dict
```

**为什么这个契约重要：**
- Core 与 Active 完全解耦，各自迭代不互相破坏
- Meta Allocator 是唯一聚合点，所有约束（vol / cap / turnover）在此统一施加
- ML 信号（Phase 3）只是新增 `source` 类型的 `ActiveSignal`，主干零改动

---

## 4. Roadmap（10 周）

### Phase 1 · Week 1–4 ｜ "讲得出的稳健"
**目标:** A 股 All-Weather Core 可回测、可复现、成本/容量可信

| Week | 交付 | 位置 |
|------|------|------|
| 1 | All-Weather ETF universe 定义（HS300/500/红利/国债/黄金/现金 6 桶） | `sysdata/astock/universe_allweather.py` |
| 1–2 | `AStockAllWeatherSimData`：扩展 `astock_sim_data.py`，补 ETF 日频接入 + carry-data 回退优雅 | `sysdata/sim/` |
| 2 | Risk Budget Engine wrapper：复用 `sysquant/estimators/covariance.py` (EWMA + shrinkage) | `sysquant/` 加薄 wrapper |
| 2–3 | Target Vol 配置（年化 6%）：复用 `positionsizing.py` + `portfolio.py`，写 A 股 config | `systems/provided/astock_allweather/` |
| 3 | A 股成本模型校准：印花税(千1)/佣金(万2.5)/滑点(bps)/冲击成本曲线 | `sysdata/sim/astock_sim_data.py` |
| 3–4 | 容量估算 (ADV%) + turnover penalty | `syscore/astock_capacity.py` |
| 4 | Core All-Weather 回测报告 (HTML) + 压力测试（2015 股灾 / 2018 / 2022） | `runs/core_allweather_v1/` |

**Milestone 1 产出:**
- `docs/results/core_allweather_v1.md` — 一页 tech memo
- 一个可 reproduce 的 run config + report

### Phase 2 · Week 5–7 ｜ "Active Alpha 层"

| Week | 交付 | 位置 |
|------|------|------|
| 5 | Signal Contract v1 实现 + Adapter (online_system → pysystemtrade) | `online_system/core/signal_contract.py` |
| 5–6 | Smart Beta 池：value / quality / low-vol / momentum（基于 `factor_strategy.py` preset） | 复用现有 `factor_scanner.py` |
| 6 | Regime Detection v1 升级：HMM on (HS300 vol, SHIBOR, CN10Y, breadth) | `online_system/portfolio/regime_hmm.py` |
| 6–7 | Meta Allocator v1：3 桶 risk parity (Core / Smart Beta / Momentum)，按 regime 加权 | `online_system/portfolio/meta_allocator.py` |
| 7 | Core+Active 对比回测 + capacity-adjusted Sharpe | `runs/active_v1/` |

### Phase 3 · Week 8–10 ｜ "Evolving 技术壁垒"

| Week | 交付 | 落点（**不预测收益**） |
|------|------|------|
| 8 | ML Vol Forecast：GARCH → LightGBM on vol features | 喂进 Target Vol Engine |
| 8–9 | Market State Embedding：1D-CNN / Transformer encoder | 喂进 Meta Allocator |
| 9 | Meta Allocator v2：state embedding 作为 feature → 桶权重，仍带 risk budget 约束 | safe wrapper 防止失控 |
| 10 | Research Copilot：LLM 自动跑因子候选回测、生成 stability report | 复用 `experiments/` |

**ML 落点原则:** 永远不让 ML 直接预测收益；只用于 **vol / state / strategy weighting / research efficiency** 四类风险可控的任务。

---

## 5. 工程约束与原则

### 5.1 仓库分工
- **主仓库 = `pysystemtrade`**（本仓库）
- 所有 Phase 1 工程改动落在本仓库
- `online_system` 作为 git submodule 或 PYTHONPATH 引用，仅修改其 signal output（Phase 2 起）

### 5.2 不重复造轮子
| 已有能力 | 直接复用，不重写 |
|---------|---------------|
| EWMA 协方差 | `sysquant/estimators/covariance.py` |
| Vol scaling | `systems/positionsizing.py` |
| Forecast combine | `systems/forecast_combine.py` |
| Cost model | `sysobjects/instruments.py:instrumentCosts` |
| Buffering | `systems/buffering.py` |
| IBKR 链路 | `sysbrokers/IB/` |
| A 股数据 | `sysdata/astock/*` + `sysdata/sim/astock_sim_data.py` |

### 5.3 投资人级风控（Phase 1 内必须）

- **目标 vol 锚定 6%**，但用保守 ex-ante estimator + buffering，避免机械追逐
- **单 ETF 最大权重 ≤ 35%**
- **单日成交 ≤ ADV 的 5%**（容量约束）
- **最大回撤阈值** 触发降仓（不 kill）
- **Kill switch + 对账**：复用 pysystemtrade 现成 reconciler 设计

### 5.4 Commit 节奏
- 每个 sub-task 完成即 commit + push
- Commit message 格式：`feat/fix/docs/refactor: <area>: <action>`
- Phase 完结打 git tag：`phase1-core-allweather-v1`

---

## 6. 投资人叙事映射

| ChatGPT Pitch 要点 | 我们的实现 |
|-------|-------|
| "我们不是单一策略团队，而是构建可长期进化的量化决策系统" | 三层架构 + Signal Contract，Phase 1→3 渐进 |
| "风险预算系统 + 目标波动率" | `sysquant/` + `positionsizing.py`，Phase 1 直接交付 |
| "Regime-aware all-weather allocation" | Phase 2 HMM → Phase 3 embedding |
| "Multi-signal decision layer" | Active Layer + Meta Allocator |
| "结构化融合，非简单 linear combination" | Meta Allocator v1 risk parity → v2 ML-conditional |
| "AI 用于 state / vol / weighting，不直接预测收益" | Phase 3 ML 落点原则（§4） |
| "成本和容量是约束，不是事后补丁" | Phase 1 cost calibration + ADV constraint 内生 |
| "可持续进化的研究系统" | Phase 3 Research Copilot + experiments/ 归档 |

可以直接对外讲：

> 我们理解的全天候，不是简单做股债金静态配置，而是构建一个以风险预算为核心、以目标波动率控制为约束、以宏观状态识别和多策略调度为增强的全天候配置系统。底层我们追求高容量、低换手、稳健的 risk premium 暴露；上层再通过机器学习驱动的状态识别、动态风控和策略分配，提升组合在不同市场环境下的适应性。我们的目标不是做一个固定组合，而是做一个**可持续进化的 AI-driven all-weather allocation system**。

---

## 7. 立即开工：Phase 1 Week 1 任务清单

- [ ] `sysdata/astock/universe_allweather.py` — 6 桶 ETF universe
- [ ] `experiments/astock_allweather/` 目录骨架 + README
- [ ] 数据完整性检查脚本（确保 6 桶 ETF 全在本地 PG 中且历史 ≥ 5 年）
- [ ] Commit

---

## 附录 A · 6 桶 All-Weather ETF 池（Phase 1 初稿）

| 桶 | 风险溢价 | 主代理 | 备代理 | 历史起点 |
|----|------|------|------|------|
| **股权 - 大盘** | Equity beta | `510300.SH` (HS300) | `510050.SH` (上证 50) | 2012 |
| **股权 - 中盘** | Equity / Size | `510500.SH` (中证 500) | `512100.SH` (中证 1000) | 2013 |
| **红利防御** | Quality / Yield | `510880.SH` (上证红利) | `512890.SH` (红利低波) | 2011 |
| **利率 - 国债** | Duration | `511010.SH` (国债 ETF) | `511260.SH` (10Y 国债 ETF) | 2013 |
| **黄金 - 抗通胀** | Real assets / Tail hedge | `518880.SH` (黄金 ETF) | `159934.SZ` (黄金 ETF) | 2013 |
| **现金 / 货币** | Carry / Liquidity | `511880.SH` (银华日利) | `511990.SH` (华宝添益) | 2013 |

> 备代理用于流动性/历史不足时的 fallback，所有代理需在 fetcher 中纳入并验证 history。

---

## 附录 B · 命名约定

- 所有 A 股全天候相关代码：`astock_allweather_*` 前缀
- 配置目录：`systems/provided/astock_allweather/`
- 运行结果目录：`runs/aw_<phase>_<version>_<timestamp>/`
- Git tag：`phase{N}-{milestone}-v{X}`
