# Antibot 风险识别与分层看板

## 1. 项目简介

Antibot 是一个基于用户访问行为特征的人机风险识别与分层分析看板。项目通过 IForest 与 XGBoost 两类模型结果，对用户日级行为进行风险分层，并在 Streamlit 中展示 DAU、直接拦截、审核池、观察池、地域分布、浏览器分布、C 段 IP 聚集、关系网策略模拟和红队压测结果。

项目目标：

- 识别异常机器流量和疑似黑产用户
- 区分强风险、审核风险、观察风险和正常流量
- 支持模型效果评估、风险分层解释和策略阈值调试
- 将离线模型结果转化为可交互分析看板

## 2. 核心能力

- 近 45 天神策 pageview 明细同步与本地 parquet 缓存
- IForest 模型异常识别结果缓存
- XGBoost mixed gray 模型风险概率缓存
- 按日增量补数、历史单日回补与模型特征增量重算
- “寻求报道”访问用户独立分析，支持风险分类、缓存复用和 CSV/Excel 导出
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

- `sensor_machine_2026_test.parquet`：原始 pageview 明细缓存，包含 `$screen_width`、`$screen_height`，由数据同步逻辑生成
- `features_all_iforest_v3_resolution_zscore.parquet`：包含分辨率日占比正向 Z-score 的 IForest 用户日特征缓存
- `features_all_xgb_mixed_v3_resolution_zscore.parquet`：对应 13 特征 XGBoost 用户日预测缓存
- `sensor_machine_2026_test_parts/`：命令行同步过程中使用的 pageview 日分片目录，合并成功后默认清理
- `seek_report_joined.parquet`：“寻求报道”目标访问用户与本地模型结果的按日关联缓存
- `antibot_pipeline_v3_resolution_zscore.pkl`：12+1 IForest pipeline 模型文件，本仓库不包含
- `xgb_model_mixed_v3_resolution_zscore.pkl`：对应的 mixed-gray XGBoost 模型文件，本仓库不包含
- `redteam_results_resolution_v3.json`：四类红队场景的可重复压测结果，看板优先读取该文件

这些数据、模型、缓存文件都不会提交到 GitHub，应通过 `.gitignore` 排除。README 中只描述文件用途，不提供真实文件内容。

## 4. 技术架构

```text
Impala / 神策 pageview 明细
→ 按日分片落盘并合并为本地原始 parquet 缓存
→ 使用训练期正常用户基准计算主分辨率日占比正向 Z-score
→ IForest / XGBoost 模型打分
→ 用户日级特征缓存 parquet
→ DuckDB 创建 feature_view
→ Streamlit 看板展示

寻求报道 pageview
→ 按日聚合目标用户
→ 左连本地模型特征
→ seek_report_joined.parquet
→ 独立筛选、统计与导出
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

也可以用命令行脚本做定时同步，默认同步最近 45 天并重算模型缓存：

```bash
python antibot_daily_sync.py
# 如需同时更新“寻求报道”结果表缓存：
python antibot_daily_sync.py --update-seek-report
# 如需忽略缓存判断并强制全量刷新：
python antibot_daily_sync.py --update-seek-report --force-refresh
# 指定回补区间（默认结束日是昨天）：
python antibot_daily_sync.py --start-date 2026-07-01 --end-date 2026-07-09
# 长步骤默认每 30 秒输出一次等待心跳；如需调整：
python antibot_daily_sync.py --update-seek-report --progress-interval 60
# 大范围补数默认按 1 天一片执行，避免 Impala 大结果连接中断；如需调大片：
python antibot_daily_sync.py --update-seek-report --sql-chunk-days 3
# 调整单分片 SQL 的失败重试次数：
python antibot_daily_sync.py --sql-retries 2
# 如需关闭终端进度和等待心跳：
python antibot_daily_sync.py --no-progress
# 如需关闭 macOS 桌面通知：
python antibot_daily_sync.py --no-notify
```

脚本默认会先判断本地 parquet 是否已覆盖当前日期窗口且不早于模型文件；已是最新时会直接复用缓存，缺少个别日期时只补缺失日期，避免重复跑整段 SQL 和模型重算。

为了降低大表补数失败成本，命令行同步会先将原始 pageview 按日期写入本地日分片目录（默认 `sensor_machine_2026_test_parts/`），每个 SQL 分片成功后立即落盘；所有缺失日期补完后，再统一合并生成 `sensor_machine_2026_test.parquet` 供模型和看板复用。合并成功后默认清理本次窗口内的日分片；如需保留排查，可加 `--keep-raw-parts`。

同步脚本使用锁文件防止两个任务同时写缓存。如确认没有任务运行但遗留了锁，可用 `--force-lock`；也可用 `--lock-file` 指定其他锁路径。

### macOS 每日定时同步

```bash
# 安装 launchd 任务
zsh scripts/install_daily_sync_launchd.sh

# 手动运行与 launchd 相同的包装脚本
zsh scripts/run_antibot_daily_sync.sh

# 卸载 launchd 任务
zsh scripts/uninstall_daily_sync_launchd.sh
```

`launchd` 任务的日志默认写入 `~/.antibot_daily_sync_runtime/logs/`；在项目目录手动运行包装脚本时，默认写入项目的 `logs/`。运行结果会尝试发送 macOS 桌面通知。如 Python 不在默认路径，安装前设置 `ANTIBOT_PYTHON=/path/to/python`。

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
- 新数据会与本地原始 parquet 合并，并保留最近 45 天窗口
- 侧边栏支持临时回补某个历史单日，补齐后同步更新对应日期的模型结果
- 原始数据更新后，优先只重算新增或回补日期的模型特征；缓存结构不兼容或模型变更时才全量重建
- 如果模型文件或原始数据比特征缓存更新，页面加载时会自动重建对应特征缓存
- 如果缓存有效，则直接复用 parquet 特征结果
- “寻求报道”页签使用独立日期范围；已覆盖的日期直接复用 `seek_report_joined.parquet`，缺失或过期时再查询并替换对应日期

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

分辨率模型的压测可通过 `python scripts/run_redteam_resolution.py` 重跑，结果写入
`redteam_results_resolution_v3.json`，看板会优先读取该文件。对应的版本化 IForest
模型固化及 mixed-gray XGBoost 训练可通过 `python scripts/train_resolution_xgb.py` 重现。

## 11. 安全说明

本仓库不包含真实数据库账号、密码、内部数据、parquet 缓存或模型文件。真实 `antibot.env`、`.parquet`、`.pkl`、`.joblib`、`.csv`、`.xlsx`、notebook 输出等文件应通过 `.gitignore` 排除。

## 12. 项目状态

这是脱敏后的本地分析看板示例，用于展示人机识别、风险分层、模型评估、DuckDB 查询加速和 Streamlit 看板工程化能力。
