# ds_ml_stock — 台股 ML 多空預測策略

以 FT-Transformer（Feature Tokenizer + Transformer）預測台股個股相對報酬，建立多空選股策略並回測。

---

## 專案結構

```
ds_ml_stock/
├── data_preprocessing.ipynb   # 特徵工程 & 前處理，輸出 parquet
├── training.ipynb             # 模型訓練，輸出 .pt 模型權重
└── backtest.ipynb             # 載入模型推論，FinLab 回測
```

執行順序：`data_preprocessing` → `training` → `backtest`

所有 notebook 設計在 **Google Colab** 上執行，資料存放於 Google Drive。

---

## 資料來源

透過 [FinLab](https://finlab.tw) API 取得台股（TSE/OTC）歷史資料：

| 特徵類型 | 內容 |
|---------|------|
| 價格 | 收盤價、還原收盤價 |
| 量能 | 成交股數、成交值 |
| 基本面 | ROE、營業利益率、EPS、月/季營收年增率 |
| 估值 | 殖利率、自算 PE（允許負值）|
| 規模 | 市值、月營收 |
| 技術因子 | 乖離率、動能偏態、報酬動能、波動度 |
| 學術因子 | mom1m, mom12m, chmom, maxret, mom36m |

每個連續特徵同時保留 **raw 值** 與 **橫斷面 rank_pct（0~1）** 兩種表示。
部分特徵（pe / roe / op_earn / yield_ratio / rev_m_yoy）額外加 **bins**（類別，走 Embedding）。

---

## 模型

**FT-Transformer**（PyTorch 手刻）

```
連續特徵 → Linear projection  ┐
類別特徵 → Embedding          ├→ concat → prepend [CLS] → TransformerEncoder → CLS head → logit
```

- 任務：二元分類 × 2（Top model 做多 / Bottom model 放空）
- Label：同一 rev_date 橫斷面報酬前 1% 為正類（做多）/ 後 1% 為正類（放空）
- Loss：`BCEWithLogitsLoss(pos_weight)` 處理 99:1 不平衡

---

## 時間切分

| 集合 | 年份 |
|------|------|
| Train | ≤ 2022（所有可用年份）|
| Test  | 2023, 2024 |

Label 只對齊**月營收公布日（rev_dates）**，非每個交易日都有樣本。

---

## 回測

- 工具：FinLab `backtest.sim()`
- 成交價：隔日開盤（避免 lookahead）
- 費用：手續費 0.1425%、交易稅 0.3%
- 策略：做多前 N 支 / 放空後 N 支 / 多空合併（預設 N=10）

---

## Branches

| Branch | 說明 |
|--------|------|
| `main` | 基準版本：有 bins 特徵 + SMOTE-NC 處理不平衡 |
| `no-bins` | 移除 bins 特徵，改用普通 SMOTE |
| `no-smote` | 不使用 SMOTE，直接以 pos_weight 補償不平衡 |

---

## 環境需求

```
finlab
torch
pandas
numpy
imbalanced-learn   # no-smote branch 不需要
matplotlib
```

Colab 安裝：

```python
!pip install finlab imbalanced-learn -q
```
