---
name: project-stock-ml
description: 資料科學期末專案 — 台股機器學習預測股票報酬，多空操作策略
metadata: 
  node_type: memory
  type: project
  originSessionId: e9cbcade-39e5-4954-b26e-621fe15ded92
---

## 專案目標

建立機器學習模型預測台股（TSE/OTC）的相對報酬。

- **Label**: 下一個營收公布截止日期間的報酬 Z-score（橫斷面標準化），找出相對強勢股
- **策略**: 預測值最高的前 N 檔做多、最低的前 N 檔放空
- **資料宇宙**: `data.universe(market='TSE_OTC')`

**Why:** 以 Z-score 作為 label 是為了排除大盤系統性漲跌，聚焦於個股相對強弱。

**How to apply:** 評估 label 設計或模型輸出時，始終以橫斷面相對概念思考，而非絕對報酬。

---

## 模型架構

- **模型**: Tabular Transformer（FT-Transformer，Feature Tokenizer + Transformer）
- **輸入資料型態**: 橫斷面（Cross-sectional）— 同一天所有股票的特徵矩陣
- **任務**: 二元分類，訓練兩個獨立模型
  - **Top model**: 正類 = 當期 rev_date 橫斷面報酬前 1% 的股票（做多訊號）
  - **Bottom model**: 正類 = 當期 rev_date 橫斷面報酬後 1% 的股票（放空訊號）
- **Loss**: `BCEWithLogitsLoss(pos_weight=99)`（正類 weight=99 處理 99:1 不平衡）
- **評估指標**: Top-K Precision（預測分數最高的 K 支中，真正在前/後 1% 的比例）
- **類別特徵（bins）**: 走 `nn.Embedding`，連續特徵走 Linear projection

---

## 資料處理流程

### 1. 原始資料取得（FinLab）

| 變數 | FinLab 欄位 | 說明 |
|------|------------|------|
| close | `price:收盤價` | 收盤價 |
| adj_close | `etl:adj_close` | 還原收盤價 |
| vol | `price:成交股數` | 成交量 |
| roe | `fundamental_features:ROE稅後` | ROE |
| op_earn | `fundamental_features:營業利益率` | 營業利益率 |
| rev_m_yoy | `monthly_revenue:去年同月增減(%)` | 月營收年增率 |
| rev_q_yoy | `fundamental_features:營收成長率` | 季營收年增率 |
| yield_ratio | `price_earning_ratio:殖利率(%)` | 殖利率 |
| market_value | `etl:market_value` | 市值 |
| monthly_revenue | `monthly_revenue:當月營收` | 當月營收 |
| eps | `financial_statement:每股盈餘` | EPS |
| common_stock | `financial_statement:普通股股本` | 普通股股本 |

基本面特徵皆乘以 `(close>0)` mask 過濾掉下市/停牌股。

### 2. 衍生特徵

**乖離（price deviation）**
- close_5, close_10, close_20, close_60 = close / close.rolling(N).mean()

**動能偏態（momentum skew）**
- ret = adj_close.pct_change()
- ReturnSkew_10, 15, 21, 63 = ret.rolling(N).skew()

**業績動能**
- rev_m_yoy_clean, rev_q_yoy_clean：用 cap_extreme_zscores_robust 清洗極端值

**量能**
- vol_5, vol_10, vol_20 = vol / vol.rolling(N).mean()
- vol_scale = vol * close（成交值）

**期間報酬**
- ret_s_5, ret_s_10, ret_s_20 = ret.rolling(N).sum()

**報酬波動**
- ret_dev_5, ret_dev_10, ret_dev_20 = ret.rolling(N).std()

**學術動能因子**（月數以 21 個交易日為單位）

| 特徵 | 定義 | 預期方向 |
|------|------|---------|
| mom1m | adj_close / adj_close.shift(21) - 1 | 負（短期反轉） |
| mom12m | adj_close.shift(21) / adj_close.shift(252) - 1 | 正（跳過最近 1 個月的 11 個月動量） |
| chmom | (adj_close.shift(21)/adj_close.shift(126)-1) − (adj_close.shift(147)/adj_close.shift(252)-1) | 動量加速為正 |
| maxret | ret.rolling(21).max() | 負（彩票效應） |
| mom36m | adj_close.shift(273) / adj_close.shift(756) - 1 | 負（長期反轉） |

**估值**
- yield_ratio（殖利率）
- pe = (close / eps_4q).reindex(close.index)，其中 eps_4q = eps.rolling(4).sum()
  - 自己算的 PE：負 EPS 也會有值（負 PE），不會是 NaN
  - 透過 API 直接拿的 PE：負 EPS 才會是 NaN（因為 API 過濾掉負值）
  - 目前已重新納入特徵候選

**Label**
- label 對齊真實營收公布日，使用下列語法取得截止日：
  ```python
  all_trade_dates = close.index
  rev_dates = rev.index_str_to_date().index
  ```
- 代碼中的 `label_5d = adj_close.shift(-6) / adj_close.shift(-1)` 是暫時版（固定5日），正式 label 需動態對齊 rev_dates

### 3. 異常值清洗 — Robust Z-score（橫向）

`cap_extreme_zscores_robust(df, threshold=3, target_z=3)`

- 用 Median + MAD（×1.4826）計算 robust Z-score
- 超過 ±threshold 的值蓋帽至 ±target_z 對應的值
- 適用情境：同天多股票的橫斷面（主要用於 rev_m_yoy、rev_q_yoy）

### 4. Panel 資料建構

`build_panel_from_feature_dict(feature_label_dict)`

- 所有 DataFrame (date × stock) stack 成 MultiIndex (datetime, instrument) × features
- 強制統一股票代碼為 str、index 為 datetime64
- Outer join concat

### 時間切分

```
Train : 2012, 2015, 2018, 2021, 2022, 2023   （6 年）
Val   : 2013, 2016, 2019                      （3 年）
Test  : 2014, 2017, 2020, 2024                （4 年）
```
每個 split 跨越早、中、近三個時期，避免 train 全是舊資料、test 全是新資料。

### 5. 完整處理順序

```
① FinLab data.get（季頻/月頻資料 * (close>0).reindex(close.index)）
     └─ FinLab 會自動 ffill 低頻資料至日頻，不需另外處理
② EDA：輸出所有原始特徵的累積分布圖（前處理前，用於偵測 999999 等異常值）
③ 特徵工程（乖離、動能、業績、量能、估值、基本面、學術因子）
④ build_panel_from_feature_dict（含 label 原始 forward return）
⑤ 只保留 rev_dates（rev.index_str_to_date().index）
⑥ dropna(how='any')
⑦ 在清洗後的宇宙內計算 label_top / label_bottom（dropna 後才能確保 1% 是訓練宇宙內的 1%）
⑧ cross_sectional_zscore（只對連續特徵，bins 和 label 欄位不動）
```

**二元 label 計算（Step ⑦）：**
```python
label_top    = panel.groupby(level='datetime')['label'].transform(
                   lambda x: (x.rank(pct=True) >= 0.99).astype(float))
label_bottom = panel.groupby(level='datetime')['label'].transform(
                   lambda x: (x.rank(pct=True) <= 0.01).astype(float))
```

**FinLab 自動 ffill 機制**：FinLab DataFrame 在做四則運算（+、-、*、/）時，只要兩個 DataFrame 頻率對不上，就會自動將低頻資料 ffill 擴增至高頻的 index。例如 `data.get('xxx') * (close>0)` 或 `close / eps_4q` 都會觸發此行為。所有低頻特徵必須在最一開始資料取得階段就完成此轉換，不在後續步驟才補 ffill。

### 6. 橫斷面 Z-score 正規化

`cross_sectional_zscore(df)`

- 按 datetime（level=0）分組，對同天所有股票做 Z-score
- 在 dropna 之後執行，確保 Z-score 的基準宇宙與訓練資料完全一致

---

## 特殊特徵工程構想（待實作）

部分特徵（如 PE）同時保留三種表示：
1. **原始值** — 直接數值
2. **bins** — 人工分箱（反映非線性門檻效果），例如 PE: ≤0→0, 0~8→1, 8~15→2, 15~25→3, 25~35→4, >35→5
3. **rank_pct** — 同日在全市場的百分位數排名

目的：讓 Tabular Transformer 的交乘項能學到同一特徵在不同時空背景（估值高低環境）下的差異行為。

哪些特徵要做此處理、如何分箱，尚待討論確認。
