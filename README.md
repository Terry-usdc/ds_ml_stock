# ds_ml_stock — 台股 ML 多空預測策略

以 **FT-Transformer**（Feature Tokenizer + Transformer）預測台股個股相對報酬，建立多空選股策略並以 FinLab 回測驗證。

---

## 專案結構

```
ds_ml_stock/
├── data_preprocessing.ipynb   # 特徵工程 & 前處理，輸出 parquet 至 Google Drive
├── training.ipynb             # 模型訓練，輸出 .pt 模型權重
└── backtest.ipynb             # 推論 + FinLab 回測
```

執行順序：`data_preprocessing` → `training` → `backtest`

所有 notebook 設計在 **Google Colab** 上執行，資料存放於 Google Drive（`/content/drive/MyDrive/機器學習/ds_ml_stock/`）。

---

## 資料來源

透過 [FinLab](https://finlab.tw) API 取得台股（TSE/OTC）歷史資料：

| 變數 | FinLab 欄位 | 說明 |
|------|------------|------|
| close | `price:收盤價` | 收盤價 |
| adj_close | `etl:adj_close` | 還原收盤價 |
| vol | `price:成交股數` | 成交量 |
| roe | `fundamental_features:ROE稅後` | ROE（%）|
| op_earn | `fundamental_features:營業利益率` | 營業利益率（%）|
| rev_m_yoy | `monthly_revenue:去年同月增減(%)` | 月營收年增率 |
| rev_q_yoy | `fundamental_features:營收成長率` | 季營收年增率 |
| yield_ratio | `price_earning_ratio:殖利率(%)` | 殖利率 |
| market_value | `etl:market_value` | 市值 |
| monthly_rev | `monthly_revenue:當月營收` | 月營收 |
| eps | `financial_statement:每股盈餘` | EPS |

基本面資料乘以 `(close > 0)` mask 濾掉下市/停牌股。

---

## 特徵工程

每個連續特徵同時保留兩種表示：
- **raw**：原始數值（橫斷面 Z-score 後）
- **_pct**：同日橫斷面百分位數排名（0~1）

部分特徵（`pe`, `roe`, `op_earn`, `yield_ratio`, `rev_m_yoy`）額外加 **_bins**（人工分箱類別特徵，走 Embedding）。

| 類型 | 特徵 |
|------|------|
| 乖離率 | close_5/10/20/60（收盤價 / N 日均價）|
| 動能偏態 | ret_skew_10/15/21/63 |
| 量能 | vol_5/10/20、vol_scale（成交值）|
| 期間報酬 | ret、ret_s_5/10/20 |
| 報酬波動 | ret_dev_5/10/20 |
| 基本面 | roe、op_earn、rev_m_yoy、rev_q_yoy |
| 估值 | pe（自算，允許負值）、yield_ratio |
| 規模 | market_value、vol_scale、monthly_rev |
| 學術動能 | mom1m（短反轉）、mom12m（11M動量）、chmom（動量加速）、maxret（彩票效應）、mom36m（長反轉）|

### 前處理 Pipeline

```
① FinLab data.get（季/月頻資料 × (close>0) mask）
② EDA：累積分布圖，偵測異常值（如 rev_m_yoy = 999999）
③ 特徵工程（衍生所有 raw / pct / bins 特徵）
④ build_panel（date × stock → MultiIndex panel）
⑤ 只保留月營收公布日（rev_dates）
⑥ dropna(how='any')
⑦ 計算 label_top / label_bottom（dropna 後，確保 1% 邊界在有效宇宙內）
⑧ 橫斷面 Z-score（只對連續特徵，bins / label 不動）
```

---

## Label 設計

- **forward return**：下一個 rev_date 的持有期間還原報酬
- **label_top**：同日橫斷面報酬排名 ≥ 99%（前 1%，做多訊號）
- **label_bottom**：同日橫斷面報酬排名 ≤ 1%（後 1%，放空訊號）

以 Z-score / 百分位定義 label 的目的：排除大盤系統性漲跌，聚焦個股相對強弱。

---

## 模型

### 架構：FT-Transformer（PyTorch 手刻）

```
x_cont ──→ Linear projection (W_j, b_j)  ──┐
x_cat  ──→ Embedding[i]                  ──┤ concat → [CLS] + tokens
                                             ↓
                              TransformerEncoder × n_layers
                              （Pre-LN, GELU, Multi-Head Attention）
                                             ↓
                                CLS token → LayerNorm → Linear → GELU → Linear
                                             ↓
                                          logit (B,)
```

### 超參數

| 參數 | 值 |
|------|----|
| d_token | 192 |
| n_heads | 8 |
| n_layers | 3 |
| dropout | 0.1 |
| optimizer | AdamW |
| lr | 1e-4 |
| weight_decay | 1e-5 |
| batch_size | 256 |
| epochs | 100 |
| scheduler | CosineAnnealingLR |

### 不平衡處理

正類僅占 1%（99:1 不平衡），各 branch 的處理方式不同（見 [Branches](#branches)）。

---

## 時間切分

| 集合 | 範圍 | 用途 |
|------|------|------|
| Train | ≤ 2022（所有可用年份）| 模型訓練 |
| Test | 2023, 2024 | 評估泛化能力 |

- 樣本點只在**月營收公布日**（rev_dates）存在，非每個交易日
- 訓練集為 SMOTE 後的平衡資料；測試集維持原始不平衡分布

---

## 評估指標

**Top-K Precision**：每個 rev_date 獨立計算，取預測機率最高的 K 支（K = 當日股票數 × 1%），計算其中真正屬於前/後 1% 的比例。

隨機基準 = 1%；模型目標是顯著高於此基準。

---

## 回測

- 工具：FinLab `backtest.sim()`
- 成交價：隔日開盤（避免收盤價 lookahead）
- 手續費：0.1425%、交易稅：0.3%
- 持倉：做多前 N 支 / 放空後 N 支（預設 N=10）
- 換倉：position 改變時才換倉（自然對齊 rev_dates）

回測結果分兩段顯示：
1. **訓練期（≤ 2022）**：做多 / 放空 / 多空合併
2. **測試期（2023–2024）**：做多 / 放空 / 多空合併

---

## Branches

| Branch | 不平衡處理 | bins 特徵 |
|--------|-----------|----------|
| `main` | SMOTE-NC + Tomek Links | ✅ 有 |
| `no-bins` | SMOTE + Tomek Links（無類別特徵，改用普通 SMOTE）| ❌ 無 |
| `no-smote` | 不使用 SMOTE，僅靠 `pos_weight` 補償 | ✅ 有 |

三個 branch 用來比較不同前處理策略對回測結果的影響。

---

## 環境需求

```
finlab
torch
pandas
numpy
matplotlib
imbalanced-learn   # no-smote branch 不需要
```

Colab 一鍵安裝：

```python
!pip install finlab imbalanced-learn -q
```

FinLab 登入（需申請 API Token）：

```python
import finlab
from google.colab import userdata
finlab.login(userdata.get("finlab"))
```
