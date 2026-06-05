# Antibot 风险识别与分层看板

## 1. 项目简介

Antibot 是一个基于用户访问行为特征的人机风险识别与分层分析看板。项目通过 IForest 与 XGBoost 两类模型结果，对用户日级行为进行风险分层，并在 Streamlit 中展示 DAU、直接拦截、审核池、观察池、地域分布、浏览器分布、C 段 IP 聚集、关系网策略模拟和红队压测结果。

项目目标：

- 识别异常机器流量和疑似黑产用户
- 区分强风险、审核风险、观察风险和正常流量
- 支持模型效果评估、风险分层解释和策略阈值调试
- 将离线模型结果转化为可交互分析看板

## 2. 核心能力

- 近 21 天神策 pageview 明细同步与本地 parquet 缓存
- IForest 模型异常识别结果缓存
- XGBoost mixed gray 模型风险概率缓存
- A-F 风险分层：
  - A：双模型一致高危
  - B：XGB 高置信新增
  - C：XGB 中高置信新增
  - D：XGB 边界风险
  - E：IForest 独有异常
  - F：双模型正常
- Streamlit 交互式阈值调整
- DuckDB 直接读取 parquet 特征缓存并 SQL 聚合
- DAU 趋势、直接拦截占比、人工审核池占比分析
- 省份、浏览器、首日登录、小时分布、C 段 IP 分布分析
- 红队压测结果展示，包括 cheap bot、夜间空降、C 段团伙、高度拟真人
- 关系网策略模拟，用于估算 C 段 IP + 软指纹连坐的新增收益

## 3. 数据与缓存结构

本地主要使用以下文件：

- `sensor_machine_2026_test.parquet`：原始 pageview 明细缓存，由数据同步逻辑生成
- `features_all_iforest_v2_exclude_eu.parquet`：IForest 模型预测后的用户日特征缓存
- `features_all_xgb_mixed_v2_exclude_eu_mixed_gray.parquet`：XGBoost 模型预测后的用户日特征缓存
- `antibot_pipeline_v2_exclude_eu.pkl`：IForest pipeline 模型文件，本仓库不包含
- `xgb_model_mixed_v2_exclude_eu_mixed_gray.pkl`：XGBoost 模型文件，本仓库不包含

这些数据、模型、缓存文件都不会提交到 GitHub，应通过 `.gitignore` 排除。README 中只描述文件用途，不提供真实文件内容。

## 4. 技术架构

```text
Impala / 神策 pageview 明细
→ 本地原始 parquet 缓存
→ IForest / XGBoost 模型打分
→ 用户日级特征缓存 parquet
→ DuckDB 创建 feature_view
→ Streamlit 看板展示
```

DuckDB 用于直接读取 parquet 并执行分析型 SQL，避免每次页面交互都用 pandas 重算大表，从而提升看板响应速度。

## 5. 技术栈

- Python
- Streamlit
- DuckDB
- Pandas / NumPy
- Plotly
- scikit-learn
- XGBoost
- joblib
- Impala / impyla
- Parquet / PyArrow

## 6. 本地运行方式

```bash
pip install -r requirements.txt
cp antibot.env.example antibot.env
# 编辑 antibot.env，填写本地数据目录、模型文件名和数据库连接信息
python -m streamlit run antibot.py
```

如果当前仓库使用的是 `requirement.txt` 文件名，请按本地实际文件名安装依赖：

```bash
pip install -r requirement.txt
```

如果只是查看脱敏代码，仓库可以直接阅读；如果要实际运行，需要本地准备数据缓存、模型文件和数据库连接配置。

## 7. 环境变量配置

真实配置放在 `antibot.env`，不要提交 GitHub。`antibot.env.example` 只提供字段模板。

主要配置项：

- `ANTIBOT_BASE_DIR`：本地数据和模型文件所在目录
- `ANTIBOT_DATA_FILE`：原始 pageview 缓存文件名
- `IFOREST_MODEL_FILE`：IForest 模型文件名
- `IFOREST_FEATURES_FILE`：IForest 特征缓存文件名
- `XGB_MODEL_FILE`：XGBoost 模型文件名
- `XGB_FEATURES_FILE`：XGBoost 特征缓存文件名
- `IMPALA_HOST`
- `IMPALA_PORT`
- `IMPALA_DATABASE`
- `IMPALA_USER`
- `IMPALA_PASSWORD`
- `IMPALA_AUTH_MECHANISM`

## 8. 缓存更新逻辑

- 点击侧边栏“获取昨日最新数据并重算大盘”后，会从数据仓库拉取增量数据
- 新数据会与本地原始 parquet 合并，并保留最近 21 天窗口
- 原始数据更新后，会重算可用模型的特征缓存
- 如果模型文件或原始数据比特征缓存更新，页面加载时会自动重建对应特征缓存
- 如果缓存有效，则直接复用 parquet 特征结果

## 9. 风险分层说明

- A：IForest 异常且 XGB 风险概率达到边界阈值，作为直接拦截核心风险
- B：IForest 正常但 XGB 高置信命中，默认进入高优先级审核
- C：XGB 中高置信新增风险，进入观察池或抽样审核
- D：XGB 边界风险，仅打标签不直接拦截
- E：IForest 独有异常，作为分歧样本复查
- F：双模型正常，默认放行

页面支持动态调整 XGB 边界、中高、高置信阈值，也支持选择是否将 B 类纳入直接拦截。

## 10. 红队压测

项目内置四类压测结果，用于观察模型边界：

- cheap bot：低质机器流量
- 夜间空降：夜间直接访问突增
- C 段团伙：同 C 段 IP 聚集访问
- 高度拟真人：更接近真实用户行为的灰产样本

这些压测结果用于解释模型召回、绕过率和正常池压力，不包含真实用户隐私数据。

## 11. 安全说明

本仓库不包含真实数据库账号、密码、内部数据、parquet 缓存或模型文件。真实 `antibot.env`、`.parquet`、`.pkl`、`.joblib`、`.csv`、`.xlsx`、notebook 输出等文件应通过 `.gitignore` 排除。

## 12. 项目状态

这是脱敏后的本地分析看板示例，用于展示人机识别、风险分层、模型评估、DuckDB 查询加速和 Streamlit 看板工程化能力。
