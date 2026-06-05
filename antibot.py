import os  # 逐行说明：导入运行所需模块
from datetime import datetime, timedelta  # 逐行说明：从模块中导入指定对象
from pathlib import Path  # 逐行说明：从模块中导入指定对象

import duckdb  # 逐行说明：导入运行所需模块
import joblib  # 逐行说明：导入运行所需模块
import pandas as pd  # 逐行说明：导入运行所需模块
import plotly.express as px  # 逐行说明：导入运行所需模块
import streamlit as st  # 逐行说明：导入运行所需模块
from impala.dbapi import connect  # 逐行说明：从模块中导入指定对象

from anti_bot_utils import UnifiedUserBehaviorCleaner  # 逐行说明：从模块中导入指定对象

st.set_page_config(page_title="Test Aitibot", layout="wide")  # 逐行说明：渲染或控制 Streamlit 界面

# ==========================================
# 1. 配置与路径定义
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent  # 逐行说明：设置 PROJECT_ROOT 的值
ENV_FILE = Path(  # 逐行说明：设置 ENV_FILE 的值
    os.getenv("ANTIBOT_ENV_FILE", str(PROJECT_ROOT / "antibot.env"))  # 逐行说明：读取环境变量配置
).expanduser().resolve()  # 逐行说明：展开并解析路径


def load_antibot_env_file(env_file: Path) -> None:  # 逐行说明：定义 load_antibot_env_file 函数
    if not env_file.exists():  # 逐行说明：判断条件是否成立
        return  # 逐行说明：返回当前函数结果

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():  # 逐行说明：开始循环遍历数据
        line = raw_line.strip()  # 逐行说明：设置 line 的值
        if not line or line.startswith("#") or "=" not in line:  # 逐行说明：判断条件是否成立
            continue  # 逐行说明：执行这一行逻辑

        key, value = line.split("=", 1)  # 逐行说明：设置 key, value 的值
        key = key.strip()  # 逐行说明：设置 key 的值
        value = value.strip().strip('"').strip("'")  # 逐行说明：设置 value 的值

        if key and key not in os.environ:  # 逐行说明：判断条件是否成立
            os.environ[key] = value  # 逐行说明：设置环境变量


def get_config_value(name: str, default: str | None = None) -> str | None:  # 逐行说明：定义 get_config_value 函数
    return os.getenv(name, default)  # 逐行说明：返回当前函数结果


load_antibot_env_file(ENV_FILE)  # 逐行说明：加载本地环境变量文件

BASE_DIR = Path(  # 逐行说明：设置 BASE_DIR 的值
    get_config_value("ANTIBOT_BASE_DIR", str(PROJECT_ROOT))  # 逐行说明：读取环境变量配置
).expanduser().resolve()  # 逐行说明：展开并解析路径
DATA_PATH = str(BASE_DIR / get_config_value("ANTIBOT_DATA_FILE", "sensor_machine_2026_test.parquet"))  # 逐行说明：设置 DATA_PATH 的值
MODEL_CONFIGS = {  # 逐行说明：设置 MODEL_CONFIGS 的值
    "原始模型": {  # 逐行说明：配置字典字段
        "model_path": str(BASE_DIR / get_config_value("IFOREST_MODEL_FILE", "antibot_pipeline_v2_exclude_eu.pkl")),  # 逐行说明：配置字典字段
        "features_path": str(BASE_DIR / get_config_value("IFOREST_FEATURES_FILE", "features_all_iforest_v2_exclude_eu.parquet")),  # 逐行说明：配置字典字段
        "prediction_mode": "anomaly",  # 逐行说明：配置字典字段
        "required_columns": ["model_anomaly", "final_time_risk"],  # 逐行说明：配置字典字段
    },  # 逐行说明：传入参数或列表项
    "XGB mixed 新模型": {  # 逐行说明：配置字典字段
        "model_path": str(BASE_DIR / get_config_value("XGB_MODEL_FILE", "xgb_model_mixed_v2_exclude_eu_mixed_gray.pkl")),  # 逐行说明：配置字典字段
        "features_path": str(BASE_DIR / get_config_value("XGB_FEATURES_FILE", "features_all_xgb_mixed_v2_exclude_eu_mixed_gray.parquet")),  # 逐行说明：配置字典字段
        "prediction_mode": "binary_bot",  # 逐行说明：配置字典字段
        "required_columns": ["model_anomaly", "xgb_bot_prob"],  # 逐行说明：配置字典字段
    },  # 逐行说明：传入参数或列表项
}  # 逐行说明：结束当前结构
IFOREST_MODEL_KEY = "原始模型"  # 逐行说明：设置 IFOREST_MODEL_KEY 的值
XGB_MODEL_KEY = "XGB mixed 新模型"  # 逐行说明：设置 XGB_MODEL_KEY 的值
DEFAULT_MODEL_KEY = IFOREST_MODEL_KEY  # 逐行说明：设置 DEFAULT_MODEL_KEY 的值
MODEL_PATH = MODEL_CONFIGS[DEFAULT_MODEL_KEY]["model_path"]  # 逐行说明：设置 MODEL_PATH 的值
FEATURES_ALL_PATH = MODEL_CONFIGS[DEFAULT_MODEL_KEY]["features_path"]  # 逐行说明：设置 FEATURES_ALL_PATH 的值
DIRECT_BLOCK_GROUPS = ("A_双模型一致高危",)  # 逐行说明：设置 DIRECT_BLOCK_GROUPS 的值
AUDIT_GROUPS = ("B_XGB高置信新增", "C_XGB中高置信新增")  # 逐行说明：设置 AUDIT_GROUPS 的值
OBSERVE_GROUPS = ("D_XGB边界风险", "E_IForest独有异常")  # 逐行说明：设置 OBSERVE_GROUPS 的值
XGB_RISK_THRESHOLDS = {  # 逐行说明：设置 XGB_RISK_THRESHOLDS 的值
    "edge": 0.50,  # 逐行说明：配置字典字段
    "mid": 0.90,  # 逐行说明：配置字典字段
    "high": 0.95,  # 逐行说明：配置字典字段
}  # 逐行说明：结束当前结构
CURRENT_MODEL_VERSION = "v2_exclude_eu_mixed_gray"  # 逐行说明：设置 CURRENT_MODEL_VERSION 的值
CURRENT_DATA_SCOPE = "剔除 eu.36kr.com"  # 逐行说明：设置 CURRENT_DATA_SCOPE 的值
RISK_ACTION_MAP = {  # 逐行说明：设置 RISK_ACTION_MAP 的值
    "A_双模型一致高危": "直接拦截/强风险",  # 逐行说明：配置字典字段
    "B_XGB高置信新增": "高优先级审核",  # 逐行说明：配置字典字段
    "C_XGB中高置信新增": "观察池/抽样审核",  # 逐行说明：配置字典字段
    "D_XGB边界风险": "仅打标签，不拦截",  # 逐行说明：配置字典字段
    "E_IForest独有异常": "分歧样本复查",  # 逐行说明：配置字典字段
    "F_双模型正常": "放行",  # 逐行说明：配置字典字段
}  # 逐行说明：结束当前结构
REDTEAM_ATTACK_RESULTS = [  # 逐行说明：设置 REDTEAM_ATTACK_RESULTS 的值
    {"scenario": "cheap bot", "attack_type": "cheap_bot", "user_days": 1000, "iforest_black": 1000, "xgb_05_black": 1000, "xgb_09_black": 1000, "xgb_095_black": 1000, "both_black": 1000, "xgb_high_new": 0, "xgb_mid_new": 0, "xgb_edge": 0, "verdict": "低质机器流量 100% 命中"},  # 逐行说明：传入参数或列表项
    {"scenario": "夜间空降", "attack_type": "night_direct_burst", "user_days": 1000, "iforest_black": 958, "xgb_05_black": 1000, "xgb_09_black": 958, "xgb_095_black": 958, "both_black": 958, "xgb_high_new": 0, "xgb_mid_new": 0, "xgb_edge": 42, "verdict": "A 类召回 95.8%，风险池覆盖 100%"},  # 逐行说明：传入参数或列表项
    {"scenario": "C 段团伙", "attack_type": "c_segment_gang", "user_days": 1000, "iforest_black": 642, "xgb_05_black": 902, "xgb_09_black": 639, "xgb_095_black": 543, "both_black": 642, "xgb_high_new": 3, "xgb_mid_new": 24, "xgb_edge": 233, "verdict": "A 类召回 64.2%，XGB@0.5 风险池覆盖 90.2%"},  # 逐行说明：传入参数或列表项
    {"scenario": "高度拟真人", "attack_type": "stealth_gray_bot", "user_days": 1000, "iforest_black": 0, "xgb_05_black": 1, "xgb_09_black": 0, "xgb_095_black": 0, "both_black": 0, "xgb_high_new": 0, "xgb_mid_new": 0, "xgb_edge": 1, "verdict": "基本绕过，属于当前字段体系下的模型盲区"},  # 逐行说明：传入参数或列表项
]  # 逐行说明：结束当前结构
REDTEAM_NORMAL_RESULTS = [  # 逐行说明：设置 REDTEAM_NORMAL_RESULTS 的值
    {"scenario": "cheap bot", "user_days": 7662, "iforest_black": 335, "xgb_05_black": 922, "xgb_09_black": 726, "xgb_095_black": 277, "both_black": 332, "xgb_high_new": 66, "xgb_mid_new": 361, "xgb_edge": 163},  # 逐行说明：传入参数或列表项
    {"scenario": "夜间空降", "user_days": 7662, "iforest_black": 333, "xgb_05_black": 667, "xgb_09_black": 246, "xgb_095_black": 133, "both_black": 333, "xgb_high_new": 4, "xgb_mid_new": 15, "xgb_edge": 315},  # 逐行说明：传入参数或列表项
    {"scenario": "C 段团伙", "user_days": 7662, "iforest_black": 338, "xgb_05_black": 666, "xgb_09_black": 252, "xgb_095_black": 144, "both_black": 338, "xgb_high_new": 4, "xgb_mid_new": 15, "xgb_edge": 309},  # 逐行说明：传入参数或列表项
    {"scenario": "高度拟真人", "user_days": 7662, "iforest_black": 331, "xgb_05_black": 665, "xgb_09_black": 246, "xgb_095_black": 132, "both_black": 331, "xgb_high_new": 4, "xgb_mid_new": 15, "xgb_edge": 315},  # 逐行说明：传入参数或列表项
]  # 逐行说明：结束当前结构


def build_redteam_tables():  # 逐行说明：定义 build_redteam_tables 函数
    attack_df = pd.DataFrame(REDTEAM_ATTACK_RESULTS)  # 逐行说明：设置 attack_df 的值
    normal_df = pd.DataFrame(REDTEAM_NORMAL_RESULTS)  # 逐行说明：设置 normal_df 的值

    for df in [attack_df, normal_df]:  # 逐行说明：开始循环遍历数据
        df["iforest_only"] = (df["iforest_black"] - df["both_black"]).clip(lower=0)  # 逐行说明：设置 df 的值
        df["risk_pool_hits"] = df["both_black"] + df["xgb_high_new"] + df["xgb_mid_new"] + df["xgb_edge"] + df["iforest_only"]  # 逐行说明：设置 df 的值
        df["risk_pool_rate"] = df["risk_pool_hits"] / df["user_days"]  # 逐行说明：设置 df 的值
        df["iforest_rate"] = df["iforest_black"] / df["user_days"]  # 逐行说明：设置 df 的值
        df["xgb_05_rate"] = df["xgb_05_black"] / df["user_days"]  # 逐行说明：设置 df 的值
        df["xgb_09_rate"] = df["xgb_09_black"] / df["user_days"]  # 逐行说明：设置 df 的值
        df["xgb_095_rate"] = df["xgb_095_black"] / df["user_days"]  # 逐行说明：设置 df 的值

    attack_df["bypass_rate"] = 1 - attack_df["risk_pool_rate"]  # 逐行说明：设置 attack_df 的值
    normal_df["normal_pool_direct_hit_rate"] = normal_df["both_black"] / normal_df["user_days"]  # 逐行说明：设置 normal_df 的值
    normal_df["normal_pool_xgb_added_hit_rate"] = (normal_df["xgb_high_new"] + normal_df["xgb_mid_new"] + normal_df["xgb_edge"]) / normal_df["user_days"]  # 逐行说明：设置 normal_df 的值

    summary_df = attack_df.merge(  # 逐行说明：设置 summary_df 的值
        normal_df[["scenario", "normal_pool_direct_hit_rate", "normal_pool_xgb_added_hit_rate", "risk_pool_rate"]],  # 逐行说明：传入参数或列表项
        on="scenario",  # 逐行说明：设置 on 的值
        suffixes=("", "_normal"),  # 逐行说明：设置 suffixes 的值
    )  # 逐行说明：结束当前结构
    summary_df = summary_df.rename(columns={"risk_pool_rate_normal": "normal_risk_pool_rate"})  # 逐行说明：设置 summary_df 的值
    return attack_df, normal_df, summary_df  # 逐行说明：返回当前函数结果


def _format_percent_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:  # 逐行说明：定义 _format_percent_columns 函数
    display_df = df.copy()  # 逐行说明：设置 display_df 的值
    for col in columns:  # 逐行说明：开始循环遍历数据
        if col in display_df.columns:  # 逐行说明：判断条件是否成立
            display_df[col] = display_df[col].map(lambda x: f"{x * 100:.2f}%")  # 逐行说明：设置 display_df 的值
    return display_df  # 逐行说明：返回当前函数结果


# ==========================================
# 2. 核心数据与离线/在线模型逻辑
# ==========================================
def validate_db_config() -> list[str]:  # 逐行说明：定义 validate_db_config 函数
    missing: list[str] = []  # 逐行说明：设置 missing 的值

    required_envs = [  # 逐行说明：设置 required_envs 的值
        "IMPALA_HOST",  # 逐行说明：传入参数或列表项
        "IMPALA_PORT",  # 逐行说明：传入参数或列表项
        "IMPALA_DATABASE",  # 逐行说明：传入参数或列表项
        "IMPALA_USER",  # 逐行说明：传入参数或列表项
        "IMPALA_AUTH_MECHANISM",  # 逐行说明：传入参数或列表项
    ]  # 逐行说明：结束当前结构

    for name in required_envs:  # 逐行说明：开始循环遍历数据
        if not get_config_value(name):  # 逐行说明：判断条件是否成立
            missing.append(name)  # 逐行说明：执行这一行逻辑

    if not (get_config_value("DB_PASSWORD") or get_config_value("IMPALA_PASSWORD")):  # 逐行说明：判断条件是否成立
        missing.append("DB_PASSWORD 或 IMPALA_PASSWORD")  # 逐行说明：执行这一行逻辑

    return missing  # 逐行说明：返回当前函数结果


@st.cache_resource  # 逐行说明：给下面的函数添加装饰器
def get_db_conn():  # 逐行说明：定义 get_db_conn 函数
    missing = validate_db_config()  # 逐行说明：设置 missing 的值
    if missing:  # 逐行说明：判断条件是否成立
        raise RuntimeError(f"缺少数据库配置：{', '.join(missing)}")  # 逐行说明：抛出异常提示调用方

    pw = get_config_value("DB_PASSWORD") or get_config_value("IMPALA_PASSWORD")  # 逐行说明：设置 pw 的值
    return connect(  # 逐行说明：返回当前函数结果
        host=get_config_value("IMPALA_HOST"),  # 逐行说明：设置 host 的值
        port=int(get_config_value("IMPALA_PORT", "21050")),  # 逐行说明：设置 port 的值
        database=get_config_value("IMPALA_DATABASE"),  # 逐行说明：设置 database 的值
        user=get_config_value("IMPALA_USER"),  # 逐行说明：设置 user 的值
        password=pw,  # 逐行说明：占位语句
        auth_mechanism=get_config_value("IMPALA_AUTH_MECHANISM", "NOSASL"),  # 逐行说明：设置 auth_mechanism 的值
    )  # 逐行说明：结束当前结构

@st.cache_resource  # 逐行说明：给下面的函数添加装饰器
def load_model(model_path: str):  # 逐行说明：定义 load_model 函数
    if not os.path.exists(model_path):  # 逐行说明：判断条件是否成立
        raise FileNotFoundError(f"模型文件不存在: {model_path}")  # 逐行说明：抛出异常提示调用方
    return joblib.load(model_path)  # 逐行说明：返回当前函数结果

def _resolve_scoring_parts(model_obj, model_path: str):  # 逐行说明：定义 _resolve_scoring_parts 函数
    if hasattr(model_obj, "named_steps"):  # 逐行说明：判断条件是否成立
        named_steps = model_obj.named_steps  # 逐行说明：设置 named_steps 的值
        if "cleaner" in named_steps and "model" in named_steps:  # 逐行说明：判断条件是否成立
            return named_steps["cleaner"], named_steps["model"]  # 逐行说明：返回当前函数结果

    original_path = MODEL_CONFIGS[DEFAULT_MODEL_KEY]["model_path"]  # 逐行说明：设置 original_path 的值
    if os.path.abspath(model_path) == os.path.abspath(original_path):  # 逐行说明：判断条件是否成立
        raise ValueError("原始模型文件需要包含 cleaner 和 model 两个步骤。")  # 逐行说明：抛出异常提示调用方

    original_pipeline = load_model(original_path)  # 逐行说明：设置 original_pipeline 的值
    if not hasattr(original_pipeline, "named_steps") or "cleaner" not in original_pipeline.named_steps:  # 逐行说明：判断条件是否成立
        raise ValueError("新模型不是 Pipeline，且原始模型里找不到 cleaner，无法生成特征。")  # 逐行说明：抛出异常提示调用方
    return original_pipeline.named_steps["cleaner"], model_obj  # 逐行说明：返回当前函数结果

def _model_feature_columns(model, features: pd.DataFrame) -> list[str]:  # 逐行说明：定义 _model_feature_columns 函数
    if hasattr(model, "feature_names_in_"):  # 逐行说明：判断条件是否成立
        return list(model.feature_names_in_)  # 逐行说明：返回当前函数结果

    get_booster = getattr(model, "get_booster", None)  # 逐行说明：设置 get_booster 的值
    if callable(get_booster):  # 逐行说明：判断条件是否成立
        booster = get_booster()  # 逐行说明：设置 booster 的值
        if getattr(booster, "feature_names", None):  # 逐行说明：判断条件是否成立
            return list(booster.feature_names)  # 逐行说明：返回当前函数结果

    return [col for col in features.columns if col != "final_time_risk"]  # 逐行说明：返回当前函数结果

def _predict_with_model(model, model_input: pd.DataFrame):  # 逐行说明：定义 _predict_with_model 函数
    try:  # 逐行说明：开始捕获可能出现的异常
        return model.predict(model_input)  # 逐行说明：返回当前函数结果
    except (TypeError, ValueError) as exc:  # 逐行说明：处理捕获到的异常
        try:  # 逐行说明：开始捕获可能出现的异常
            import xgboost as xgb  # 逐行说明：导入运行所需模块
            return model.predict(xgb.DMatrix(model_input))  # 逐行说明：返回当前函数结果
        except Exception:  # 逐行说明：处理捕获到的异常
            raise exc  # 逐行说明：抛出异常提示调用方

def _predict_bot_probability(model, model_input: pd.DataFrame, bot_label=1) -> pd.Series:  # 逐行说明：定义 _predict_bot_probability 函数
    predict_proba = getattr(model, "predict_proba", None)  # 逐行说明：设置 predict_proba 的值
    if callable(predict_proba):  # 逐行说明：判断条件是否成立
        try:  # 逐行说明：开始捕获可能出现的异常
            proba = predict_proba(model_input)  # 逐行说明：设置 proba 的值
            proba_frame = pd.DataFrame(proba)  # 逐行说明：设置 proba_frame 的值
            classes = list(getattr(model, "classes_", []))  # 逐行说明：设置 classes 的值
            bot_idx = classes.index(bot_label) if bot_label in classes else (1 if proba_frame.shape[1] > 1 else 0)  # 逐行说明：设置 bot_idx 的值
            return pd.to_numeric(proba_frame.iloc[:, bot_idx], errors="coerce").fillna(0.0).clip(0.0, 1.0)  # 逐行说明：返回当前函数结果
        except (TypeError, ValueError):  # 逐行说明：处理捕获到的异常
            pass  # 逐行说明：占位语句

    raw = _predict_with_model(model, model_input)  # 逐行说明：设置 raw 的值
    return pd.to_numeric(pd.Series(raw), errors="coerce").fillna(0.0).clip(0.0, 1.0)  # 逐行说明：返回当前函数结果

def _normalize_model_anomaly(raw_pred, prediction_mode: str) -> pd.Series:  # 逐行说明：定义 _normalize_model_anomaly 函数
    pred_frame = pd.DataFrame(raw_pred)  # 逐行说明：设置 pred_frame 的值
    if pred_frame.shape[1] > 1:  # 逐行说明：判断条件是否成立
        pred = pred_frame.iloc[:, 1] if prediction_mode == "binary_bot" else pred_frame.idxmax(axis=1)  # 逐行说明：设置 pred 的值
    else:  # 逐行说明：处理其他情况
        pred = pred_frame.iloc[:, 0]  # 逐行说明：设置 pred 的值
    numeric_pred = pd.to_numeric(pred, errors="coerce")  # 逐行说明：设置 numeric_pred 的值

    if prediction_mode == "binary_bot":  # 逐行说明：判断条件是否成立
        if numeric_pred.eq(-1).any():  # 逐行说明：判断条件是否成立
            return numeric_pred.apply(lambda x: -1 if x == -1 else 1)  # 逐行说明：返回当前函数结果
        return numeric_pred.ge(0.5).apply(lambda is_bot: -1 if is_bot else 1)  # 逐行说明：返回当前函数结果

    return numeric_pred.apply(lambda x: -1 if x == -1 else 1)  # 逐行说明：返回当前函数结果

def _core_predict_logic(df: pd.DataFrame, model_path: str, prediction_mode: str) -> pd.DataFrame:  # 逐行说明：定义 _core_predict_logic 函数
    """纯粹的模型打分层 (Scoring Layer)"""  # 逐行说明：执行这一行逻辑
    model_obj = load_model(model_path)  # 逐行说明：设置 model_obj 的值
    cleaner, model = _resolve_scoring_parts(model_obj, model_path)  # 逐行说明：设置 cleaner, model 的值

    features = cleaner.transform(df)  # 逐行说明：设置 features 的值
    features = features.copy()  # 逐行说明：设置 features 的值
    model_cols = _model_feature_columns(model, features)  # 逐行说明：设置 model_cols 的值
    for col in model_cols:  # 逐行说明：开始循环遍历数据
        if col not in features.columns:  # 逐行说明：判断条件是否成立
            features[col] = 0  # 逐行说明：设置 features 的值

    model_input = features[model_cols]  # 逐行说明：设置 model_input 的值
    raw_pred = _predict_with_model(model, model_input)  # 逐行说明：设置 raw_pred 的值
    features["model_anomaly"] = _normalize_model_anomaly(raw_pred, prediction_mode).values  # 逐行说明：设置 features 的值
    if prediction_mode == "binary_bot":  # 逐行说明：判断条件是否成立
        features["xgb_bot_prob"] = _predict_bot_probability(model, model_input).values  # 逐行说明：设置 features 的值
    return features.reset_index()  # 逐行说明：返回当前函数结果

def build_and_store_features(  # 逐行说明：定义 build_and_store_features 函数
    model_key: str,  # 逐行说明：传入参数或列表项
    model_path: str,  # 逐行说明：传入参数或列表项
    features_path: str,  # 逐行说明：传入参数或列表项
    prediction_mode: str,  # 逐行说明：传入参数或列表项
):  # 逐行说明：执行这一行逻辑
    if not os.path.exists(DATA_PATH):  # 逐行说明：判断条件是否成立
        raise FileNotFoundError(f"原始数据不存在: {DATA_PATH}")  # 逐行说明：抛出异常提示调用方
    df_all = pd.read_parquet(DATA_PATH)  # 逐行说明：设置 df_all 的值
    df_all = df_all.dropna(subset=["date"])  # 逐行说明：设置 df_all 的值

    features = _core_predict_logic(df_all, model_path, prediction_mode)  # 逐行说明：设置 features 的值
    features.to_parquet(features_path, index=False)  # 逐行说明：设置 features 的值
    return True  # 逐行说明：返回当前函数结果

def _file_mtime(path: str) -> float:  # 逐行说明：定义 _file_mtime 函数
    return os.path.getmtime(path) if os.path.exists(path) else 0.0  # 逐行说明：返回当前函数结果

def _feature_store_has_columns(features_path: str, required_columns: list[str]) -> bool:  # 逐行说明：定义 _feature_store_has_columns 函数
    if not required_columns:  # 逐行说明：判断条件是否成立
        return True  # 逐行说明：返回当前函数结果
    if not os.path.exists(features_path):  # 逐行说明：判断条件是否成立
        return False  # 逐行说明：返回当前函数结果
    try:  # 逐行说明：开始捕获可能出现的异常
        check_conn = duckdb.connect()  # 逐行说明：设置 check_conn 的值
        cols = check_conn.execute(f"DESCRIBE SELECT * FROM read_parquet('{features_path}')").df()["column_name"]  # 逐行说明：设置 cols 的值
        check_conn.close()  # 逐行说明：执行这一行逻辑
        return set(required_columns).issubset(set(cols))  # 逐行说明：返回当前函数结果
    except Exception:  # 逐行说明：处理捕获到的异常
        return False  # 逐行说明：返回当前函数结果

def ensure_feature_store(model_key: str) -> str:  # 逐行说明：定义 ensure_feature_store 函数
    config = MODEL_CONFIGS[model_key]  # 逐行说明：设置 config 的值
    model_path = config["model_path"]  # 逐行说明：设置 model_path 的值
    features_path = config["features_path"]  # 逐行说明：设置 features_path 的值

    if not os.path.exists(model_path):  # 逐行说明：判断条件是否成立
        raise FileNotFoundError(f"模型文件不存在: {model_path}")  # 逐行说明：抛出异常提示调用方
    if not os.path.exists(DATA_PATH):  # 逐行说明：判断条件是否成立
        raise FileNotFoundError(f"原始数据不存在: {DATA_PATH}")  # 逐行说明：抛出异常提示调用方

    needs_rebuild = (  # 逐行说明：设置 needs_rebuild 的值
        not os.path.exists(features_path)  # 逐行说明：执行这一行逻辑
        or _file_mtime(DATA_PATH) > _file_mtime(features_path)  # 逐行说明：执行这一行逻辑
        or _file_mtime(model_path) > _file_mtime(features_path)  # 逐行说明：执行这一行逻辑
        or not _feature_store_has_columns(features_path, config.get("required_columns", []))  # 逐行说明：执行这一行逻辑
    )  # 逐行说明：结束当前结构
    if needs_rebuild:  # 逐行说明：判断条件是否成立
        with st.spinner(f"正在使用【{model_key}】生成预测结果缓存..."):  # 逐行说明：进入上下文管理代码块
            build_and_store_features(  # 逐行说明：执行这一行逻辑
                model_key,  # 逐行说明：传入参数或列表项
                model_path,  # 逐行说明：传入参数或列表项
                features_path,  # 逐行说明：传入参数或列表项
                config["prediction_mode"],  # 逐行说明：传入参数或列表项
            )  # 逐行说明：结束当前结构
    return features_path  # 逐行说明：返回当前函数结果

def build_all_model_features():  # 逐行说明：定义 build_all_model_features 函数
    built_models = []  # 逐行说明：设置 built_models 的值
    skipped_models = []  # 逐行说明：设置 skipped_models 的值
    for model_key, config in MODEL_CONFIGS.items():  # 逐行说明：开始循环遍历数据
        if not os.path.exists(config["model_path"]):  # 逐行说明：判断条件是否成立
            skipped_models.append(model_key)  # 逐行说明：执行这一行逻辑
            continue  # 逐行说明：执行这一行逻辑

        with st.spinner(f"正在重算【{model_key}】特征缓存..."):  # 逐行说明：进入上下文管理代码块
            build_and_store_features(  # 逐行说明：执行这一行逻辑
                model_key,  # 逐行说明：传入参数或列表项
                config["model_path"],  # 逐行说明：传入参数或列表项
                config["features_path"],  # 逐行说明：传入参数或列表项
                config["prediction_mode"],  # 逐行说明：传入参数或列表项
            )  # 逐行说明：结束当前结构
        built_models.append(model_key)  # 逐行说明：执行这一行逻辑

    if not built_models:  # 逐行说明：判断条件是否成立
        raise FileNotFoundError("没有找到可用模型文件，请检查模型 pkl 是否放在脚本同目录。")  # 逐行说明：抛出异常提示调用方
    return built_models, skipped_models  # 逐行说明：返回当前函数结果

def build_event_sql(start_s, end_s):  # 逐行说明：定义 build_event_sql 函数
    return f"""
    /*SA(default)*/ SELECT date, distinct_id,
                   hour(TIME) AS hour_time,
                   $city, $os, $province, $browser, $ip,
                   $is_first_day, $is_first_time, $title,
                   $url, $referrer, $is_login_id, $manufacturer
       FROM EVENTS
       WHERE event = '$pageview'
         AND $url not like '%eu.36kr.com%'
         AND $bot_name is null
         AND date BETWEEN '{start_s}' AND '{end_s}'
         AND $lib = 'js'
    """  # 逐行说明：返回当前函数结果

@st.cache_data(ttl=86400, show_spinner="正在从数据仓库拉取近 21 天全量日志...")  # 逐行说明：给下面的函数添加装饰器
def fetch_big_data():  # 逐行说明：定义 fetch_big_data 函数
    conn = get_db_conn()  # 逐行说明：设置 conn 的值
    end_s = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")  # 逐行说明：设置 end_s 的值
    start_s = (datetime.now() - timedelta(days=22)).strftime("%Y-%m-%d")  # 逐行说明：设置 start_s 的值
    sql = build_event_sql(start_s, end_s)  # 逐行说明：设置 sql 的值
    df = pd.read_sql(sql, conn)  # 逐行说明：设置 df 的值
    df["date"] = pd.to_datetime(df["date"], errors="coerce")  # 逐行说明：设置 df 的值
    return df.dropna(subset=["date"])  # 逐行说明：返回当前函数结果

@st.cache_data(show_spinner=False)  # 逐行说明：给下面的函数添加装饰器
def convert_df_to_csv(df: pd.DataFrame) -> bytes:  # 逐行说明：定义 convert_df_to_csv 函数
    return df.to_csv(index=False).encode("utf-8")  # 逐行说明：返回当前函数结果

NETWORK_ROI_COLUMNS = ["date", "DAU", "模型拦截", "关系网可计算DAU", "关系网新增", "关系网影响比(%)"]  # 逐行说明：设置 NETWORK_ROI_COLUMNS 的值


def _empty_network_roi() -> pd.DataFrame:  # 逐行说明：定义 _empty_network_roi 函数
    return pd.DataFrame(columns=NETWORK_ROI_COLUMNS)  # 逐行说明：返回当前函数结果


def _blank_series(series: pd.Series) -> pd.Series:  # 逐行说明：定义 _blank_series 函数
    return series.fillna("").astype(str).str.strip()  # 逐行说明：返回当前函数结果


def _bool_flag(series: pd.Series) -> pd.Series:  # 逐行说明：定义 _bool_flag 函数
    return series.fillna(False).astype(str).str.strip().str.lower().isin(["true", "1", "t", "yes"])  # 逐行说明：返回当前函数结果


def _daily_network_totals(df: pd.DataFrame) -> pd.DataFrame:  # 逐行说明：定义 _daily_network_totals 函数
    if df.empty or not {"date", "distinct_id", "is_bot"}.issubset(df.columns):  # 逐行说明：判断条件是否成立
        return pd.DataFrame(columns=["date", "DAU", "模型拦截"])  # 逐行说明：返回当前函数结果

    daily_base = df.dropna(subset=["date", "distinct_id"]).copy()  # 逐行说明：设置 daily_base 的值
    daily_base["is_bot_flag"] = _bool_flag(daily_base["is_bot"])  # 逐行说明：设置 daily_base 的值
    daily = (  # 逐行说明：设置 daily 的值
        daily_base.groupby(["date", "distinct_id"])  # 逐行说明：执行这一行逻辑
        .agg(is_bot_flag=("is_bot_flag", "max"))  # 逐行说明：设置变量值
        .reset_index()  # 逐行说明：执行这一行逻辑
    )  # 逐行说明：结束当前结构
    return daily.groupby("date").agg(  # 逐行说明：返回当前函数结果
        DAU=("distinct_id", "nunique"),  # 逐行说明：设置变量值
        模型拦截=("is_bot_flag", "sum"),  # 逐行说明：设置变量值
    ).reset_index()  # 逐行说明：执行这一行逻辑


def _finish_network_roi(
    daily_total: pd.DataFrame,
    eligible_counts: pd.DataFrame | None = None,
    net_counts: pd.DataFrame | None = None,
) -> pd.DataFrame:  # 逐行说明：定义 _finish_network_roi 函数
    if daily_total.empty:  # 逐行说明：判断条件是否成立
        return _empty_network_roi()  # 逐行说明：返回当前函数结果

    res = daily_total.copy()  # 逐行说明：设置 res 的值
    if eligible_counts is not None and not eligible_counts.empty:  # 逐行说明：判断条件是否成立
        res = res.merge(eligible_counts, on="date", how="left")  # 逐行说明：设置 res 的值
    else:  # 逐行说明：处理其他情况
        res["关系网可计算DAU"] = 0  # 逐行说明：设置 res 的值

    if net_counts is not None and not net_counts.empty:  # 逐行说明：判断条件是否成立
        res = res.merge(net_counts, on="date", how="left")  # 逐行说明：设置 res 的值
    else:  # 逐行说明：处理其他情况
        res["关系网新增"] = 0  # 逐行说明：设置 res 的值

    res["关系网可计算DAU"] = res["关系网可计算DAU"].fillna(0).astype(int)  # 逐行说明：设置 res 的值
    res["关系网新增"] = res["关系网新增"].fillna(0).astype(int)  # 逐行说明：设置 res 的值
    res["关系网影响比(%)"] = (res["关系网新增"] / res["DAU"] * 100).fillna(0).round(2)  # 逐行说明：设置 res 的值
    return res[NETWORK_ROI_COLUMNS]  # 逐行说明：返回当前函数结果


def prepare_network_fields(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:  # 逐行说明：定义 prepare_network_fields 函数
    """补齐关系网计算所需字段，避免上游字段漂移时直接报错。"""  # 逐行说明：执行这一行逻辑
    if df.empty:  # 逐行说明：判断条件是否成立
        return df.copy(), None  # 逐行说明：返回当前函数结果

    base_df = df.copy()  # 逐行说明：设置 base_df 的值
    status_msgs = []  # 逐行说明：设置 status_msgs 的值

    if "ip_c_segment" not in base_df.columns or _blank_series(base_df["ip_c_segment"]).eq("").all():  # 逐行说明：判断条件是否成立
        if "$ip" not in base_df.columns:  # 逐行说明：判断条件是否成立
            return _empty_network_roi(), "当前数据缺少 ip_c_segment 和 $ip，关系网模拟不可用。"  # 逐行说明：返回当前函数结果
        base_df["ip_c_segment"] = _blank_series(base_df["$ip"]).str.extract(r"^(\d+\.\d+\.\d+)", expand=False).fillna("")  # 逐行说明：设置 base_df 的值
        status_msgs.append("C 段由 $ip 临时推导")  # 逐行说明：执行这一行逻辑

    if "soft_fp" not in base_df.columns or _blank_series(base_df["soft_fp"]).eq("").all():  # 逐行说明：判断条件是否成立
        weak_fp_cols = [  # 逐行说明：设置 weak_fp_cols 的值
            "$os", "$browser", "$manufacturer", "$province", "$is_first_day", "$is_login_id",
            "province_display", "browser_display",
        ]  # 逐行说明：结束当前结构
        available_cols = [col for col in weak_fp_cols if col in base_df.columns]  # 逐行说明：设置 available_cols 的值
        if not available_cols:  # 逐行说明：判断条件是否成立
            return _empty_network_roi(), "当前数据缺少 soft_fp，也缺少可构造弱软指纹的设备/地域字段，关系网模拟不可用。"  # 逐行说明：返回当前函数结果

        fp_parts = []  # 逐行说明：设置 fp_parts 的值
        for col in available_cols:  # 逐行说明：开始循环遍历数据
            cleaned = _blank_series(base_df[col])  # 逐行说明：设置 cleaned 的值
            fp_parts.append(cleaned.mask(cleaned.eq(""), f"未知{col}"))  # 逐行说明：执行这一行逻辑
        base_df["soft_fp"] = fp_parts[0]  # 逐行说明：设置 base_df 的值
        for fp_part in fp_parts[1:]:  # 逐行说明：开始循环遍历数据
            base_df["soft_fp"] = base_df["soft_fp"] + "|" + fp_part  # 逐行说明：设置 base_df 的值
        status_msgs.append("软指纹由设备、地域和浏览器字段临时构造")  # 逐行说明：执行这一行逻辑

    required_cols = ["date", "distinct_id", "is_bot", "ip_c_segment", "soft_fp"]  # 逐行说明：设置 required_cols 的值
    missing_cols = [col for col in required_cols if col not in base_df.columns]  # 逐行说明：设置 missing_cols 的值
    if missing_cols:  # 逐行说明：判断条件是否成立
        return _empty_network_roi(), f"当前数据缺少 {', '.join(missing_cols)}，关系网模拟不可用。"  # 逐行说明：返回当前函数结果

    base_df["ip_c_segment"] = _blank_series(base_df["ip_c_segment"])  # 逐行说明：设置 base_df 的值
    base_df["soft_fp"] = _blank_series(base_df["soft_fp"])  # 逐行说明：设置 base_df 的值
    base_df = base_df[(base_df["ip_c_segment"] != "") & (base_df["soft_fp"] != "")]  # 逐行说明：设置 base_df 的值
    return base_df, "；".join(status_msgs) if status_msgs else None  # 逐行说明：返回当前函数结果


@st.cache_data(show_spinner=False)  # 逐行说明：给下面的函数添加装饰器
def calculate_net_roi(df: pd.DataFrame, ratio: float) -> pd.DataFrame:  # 逐行说明：定义 calculate_net_roi 函数
    """计算基于 C 段 IP + 软指纹团伙的关系网新增拦截收益。"""  # 逐行说明：执行这一行逻辑
    if df.empty:  # 逐行说明：判断条件是否成立
        return _empty_network_roi()  # 逐行说明：返回当前函数结果

    daily_total = _daily_network_totals(df)  # 逐行说明：设置 daily_total 的值
    base_df, _ = prepare_network_fields(df)  # 逐行说明：设置 base_df 的值
    if base_df.empty:  # 逐行说明：判断条件是否成立
        return _finish_network_roi(daily_total)  # 逐行说明：返回当前函数结果
    base_df = base_df.copy()  # 逐行说明：设置 base_df 的值
    base_df["is_bot_flag"] = _bool_flag(base_df["is_bot"])  # 逐行说明：设置 base_df 的值
    eligible_daily = (  # 逐行说明：设置 eligible_daily 的值
        base_df.groupby(["date", "distinct_id"])  # 逐行说明：执行这一行逻辑
        .agg(is_bot_flag=("is_bot_flag", "max"))  # 逐行说明：设置变量值
        .reset_index()  # 逐行说明：执行这一行逻辑
    )  # 逐行说明：结束当前结构
    eligible_counts = eligible_daily.groupby("date").agg(关系网可计算DAU=("distinct_id", "nunique")).reset_index()  # 逐行说明：设置 eligible_counts 的值

    cluster_counts = (  # 逐行说明：设置 cluster_counts 的值
        base_df.groupby(["date", "ip_c_segment", "soft_fp"])["distinct_id"]  # 逐行说明：执行这一行逻辑
        .nunique()  # 逐行说明：执行这一行逻辑
        .reset_index(name="count")  # 逐行说明：设置变量值
    )  # 逐行说明：结束当前结构
    heavy_clusters = cluster_counts[cluster_counts["count"] >= 5]  # 逐行说明：设置 heavy_clusters 的值
    if heavy_clusters.empty:  # 逐行说明：判断条件是否成立
        return _finish_network_roi(daily_total, eligible_counts)  # 逐行说明：返回当前函数结果

    full_graph = base_df.merge(  # 逐行说明：设置 full_graph 的值
        heavy_clusters[["date", "ip_c_segment", "soft_fp"]],  # 逐行说明：传入参数或列表项
        on=["date", "ip_c_segment", "soft_fp"],  # 逐行说明：设置 on 的值
        how="inner",  # 逐行说明：设置 how 的值
    )  # 逐行说明：结束当前结构

    stats = (  # 逐行说明：设置 stats 的值
        full_graph.groupby(["date", "ip_c_segment", "soft_fp", "is_bot_flag"])["distinct_id"]  # 逐行说明：执行这一行逻辑
        .nunique()  # 逐行说明：执行这一行逻辑
        .unstack(fill_value=0)  # 逐行说明：设置变量值
    )  # 逐行说明：结束当前结构
    if True not in stats.columns:  # 逐行说明：判断条件是否成立
        stats[True] = 0  # 逐行说明：设置 stats 的值
    if False not in stats.columns:  # 逐行说明：判断条件是否成立
        stats[False] = 0  # 逐行说明：设置 stats 的值

    stats["total"] = stats[True] + stats[False]  # 逐行说明：设置 stats 的值
    stats["ratio"] = (stats[True] / stats["total"]).fillna(0)  # 逐行说明：设置 stats 的值

    dirty_indices = stats[stats["ratio"] >= ratio].index  # 逐行说明：设置 dirty_indices 的值
    missed_bot_pairs = full_graph[  # 逐行说明：设置 missed_bot_pairs 的值
        full_graph.set_index(["date", "ip_c_segment", "soft_fp"]).index.isin(dirty_indices)  # 逐行说明：执行这一行逻辑
        & (~full_graph["is_bot_flag"])  # 逐行说明：执行这一行逻辑
    ][["date", "distinct_id"]].drop_duplicates()  # 逐行说明：执行这一行逻辑

    base_df = base_df.merge(  # 逐行说明：设置 base_df 的值
        missed_bot_pairs.assign(is_net_bot=True),  # 逐行说明：设置 missed_bot_pairs 的值
        on=["date", "distinct_id"],  # 逐行说明：设置 on 的值
        how="left",  # 逐行说明：设置 how 的值
    )  # 逐行说明：结束当前结构
    base_df["is_net_bot"] = base_df["is_net_bot"].fillna(False) & (~base_df["is_bot_flag"])  # 逐行说明：设置 base_df 的值

    daily = (  # 逐行说明：设置 daily 的值
        base_df.groupby(["date", "distinct_id"])  # 逐行说明：执行这一行逻辑
        .agg(is_bot_flag=("is_bot_flag", "max"), is_net_bot=("is_net_bot", "max"))  # 逐行说明：设置变量值
        .reset_index()  # 逐行说明：执行这一行逻辑
    )  # 逐行说明：结束当前结构
    net_counts = daily.groupby("date").agg(  # 逐行说明：设置 net_counts 的值
        关系网新增=("is_net_bot", "sum"),  # 逐行说明：设置 关系网新增 的值
    ).reset_index()  # 逐行说明：执行这一行逻辑

    return _finish_network_roi(daily_total, eligible_counts, net_counts)  # 逐行说明：返回当前函数结果

def fetch_sql_delta(start_d, end_d):  # 逐行说明：定义 fetch_sql_delta 函数
    """只拉取 start_d 到 end_d 之间的数据"""  # 逐行说明：执行这一行逻辑
    print(f"📡 正在拉取增量数据: {start_d} 至 {end_d} ...")  # 逐行说明：执行这一行逻辑
    conn = get_db_conn()  # 逐行说明：设置 conn 的值
    sql = build_event_sql(start_d, end_d)  # 逐行说明：设置 sql 的值
    df = pd.read_sql(sql, conn)  # 逐行说明：设置 df 的值
    df["date"] = pd.to_datetime(df["date"], errors="coerce")  # 逐行说明：设置 df 的值
    return df.dropna(subset=["date"])  # 逐行说明：返回当前函数结果

# --- 增量更新控制台 ---
st.sidebar.markdown("---")  # 逐行说明：渲染或控制 Streamlit 界面
st.sidebar.subheader("🔄 数据同步中心")  # 逐行说明：渲染或控制 Streamlit 界面

def run_local_sync():  # 逐行说明：定义 run_local_sync 函数
    """本地一键增量更新逻辑"""  # 逐行说明：执行这一行逻辑
    missing = validate_db_config()  # 逐行说明：设置 missing 的值
    if missing:  # 逐行说明：判断条件是否成立
        return f"缺少数据库配置：{', '.join(missing)}，请检查 antibot.env"  # 逐行说明：返回当前函数结果

    # 1. 全部使用 Pandas 的时间戳标准进行计算
    today = pd.Timestamp.now().normalize()  # 逐行说明：设置 today 的值
    yesterday = today - pd.Timedelta(days=1)  # 逐行说明：设置 yesterday 的值
    cutoff = today - pd.Timedelta(days=21)  # 逐行说明：设置 cutoff 的值

    # 用于 SQL 查询的字符串
    yesterday_str = yesterday.strftime("%Y-%m-%d")  # 逐行说明：设置 yesterday_str 的值
    cutoff_str = cutoff.strftime("%Y-%m-%d")  # 逐行说明：设置 cutoff_str 的值

    # 2. 检查本地数据进度
    if os.path.exists(DATA_PATH):  # 逐行说明：判断条件是否成立
        df_old = pd.read_parquet(DATA_PATH)  # 逐行说明：设置 df_old 的值
        # 强转 Timestamp，彻底消灭类型冲突
        max_local_date = pd.to_datetime(df_old["date"].max())  # 逐行说明：设置 max_local_date 的值

        # 现在两边都是 Timestamp，比较绝对安全
        if max_local_date >= yesterday:  # 逐行说明：判断条件是否成立
            return "✅ 本地数据已是最新，无需同步。"  # 逐行说明：返回当前函数结果

        start_fetch_date = (max_local_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")  # 逐行说明：设置 start_fetch_date 的值
    else:  # 逐行说明：处理其他情况
        df_old = pd.DataFrame()  # 逐行说明：设置 df_old 的值
        start_fetch_date = cutoff_str  # 逐行说明：设置 start_fetch_date 的值

    # 3. 从数据库拉取缺口数据 (调用你的拉取逻辑)
    df_new = fetch_sql_delta(start_fetch_date, yesterday_str)  # 逐行说明：设置 df_new 的值

    if df_new.empty:  # 逐行说明：判断条件是否成立
        return "⚠️ 数据库中没有新数据。"  # 逐行说明：返回当前函数结果

    # 4. 缝合新老数据
    df_combined = pd.concat([df_old, df_new], ignore_index=True)  # 逐行说明：设置 df_combined 的值

    # 5. 剔除 21 天前的数据 (用强转防变异)
    df_combined["date_ts"] = pd.to_datetime(df_combined["date"])  # 逐行说明：设置 df_combined 的值
    df_combined = df_combined[df_combined["date_ts"] >= cutoff]  # 逐行说明：设置 df_combined 的值
    df_combined = df_combined.drop(columns=["date_ts"]) # 阅后即焚

    # 为了配合你后面的 DuckDB，确保最终存进去的 date 列是字符串
    df_combined["date"] = pd.to_datetime(df_combined["date"])  # 逐行说明：设置 df_combined 的值

    # 6. 覆盖保存原始 Parquet
    df_combined.to_parquet(DATA_PATH, index=False)  # 逐行说明：设置 df_combined 的值

    # 7. 重算所有可用模型的特征全量包
    built_models, skipped_models = build_all_model_features()  # 逐行说明：设置 built_models, skipped_models 的值

    # 8. 清空 Streamlit 缓存，强制刷新
    st.cache_data.clear()  # 逐行说明：渲染或控制 Streamlit 界面
    st.cache_resource.clear()  # 逐行说明：渲染或控制 Streamlit 界面

    msg = f"🎉 成功同步并处理了 {len(df_new)} 条新数据！已重算：{', '.join(built_models)}。"  # 逐行说明：设置 msg 的值
    if skipped_models:  # 逐行说明：判断条件是否成立
        msg += f" 未找到模型文件，已跳过：{', '.join(skipped_models)}。"  # 逐行说明：设置 msg + 的值
    return msg  # 逐行说明：返回当前函数结果

st.sidebar.subheader("🤖 风险模型")  # 逐行说明：渲染或控制 Streamlit 界面
st.sidebar.caption(f"IForest：{os.path.basename(MODEL_CONFIGS[IFOREST_MODEL_KEY]['model_path'])}")  # 逐行说明：渲染或控制 Streamlit 界面
st.sidebar.caption(f"XGB：{os.path.basename(MODEL_CONFIGS[XGB_MODEL_KEY]['model_path'])}")  # 逐行说明：渲染或控制 Streamlit 界面
st.sidebar.markdown("---")  # 逐行说明：渲染或控制 Streamlit 界面

if "sync_status_msg" in st.session_state:  # 逐行说明：判断是否有上一次同步后的提示
    if st.session_state.get("sync_status_level") == "warning":  # 逐行说明：判断提示是否属于警告类型
        st.sidebar.warning(st.session_state["sync_status_msg"])  # 逐行说明：在侧边栏展示同步警告
    else:  # 逐行说明：处理普通成功提示
        st.sidebar.success(st.session_state["sync_status_msg"])  # 逐行说明：在侧边栏展示同步成功信息

# 画一个同步按钮
if st.sidebar.button("🚀 获取昨日最新数据并重算大盘", type="primary"):  # 逐行说明：判断条件是否成立
    missing = validate_db_config()  # 逐行说明：设置 missing 的值
    if missing:  # 逐行说明：判断条件是否成立
        msg = f"缺少数据库配置：{', '.join(missing)}，请检查 antibot.env"  # 逐行说明：设置 msg 的值
        st.session_state.sync_status_msg = msg  # 逐行说明：保存同步结果，避免刷新后提示消失
        st.session_state.sync_status_level = "warning"  # 逐行说明：根据消息内容记录提示样式
        st.sidebar.warning(msg)  # 逐行说明：在侧边栏展示同步警告
    else:  # 逐行说明：处理其他情况
        with st.spinner("正在连接数据库拉取增量并计算特征，请稍候..."):  # 逐行说明：进入上下文管理代码块
            msg = run_local_sync()  # 逐行说明：设置 msg 的值
            st.session_state.sync_status_msg = msg  # 逐行说明：保存同步结果，避免刷新后提示消失
            st.session_state.sync_status_level = "warning" if msg.startswith("⚠️") or msg.startswith("缺少数据库配置") else "success"  # 逐行说明：根据消息内容记录提示样式
            st.rerun() # 瞬间刷新页面，新数据直接上盘
st.sidebar.markdown("---")  # 逐行说明：渲染或控制 Streamlit 界面

# 3. Streamlit UI与全局参数
# ==========================================
st.title("Antibot risk dashboard")  # 逐行说明：渲染或控制 Streamlit 界面
st.caption(f"当前模型版本：{CURRENT_MODEL_VERSION}；数据口径：{CURRENT_DATA_SCOPE}")  # 逐行说明：渲染或控制 Streamlit 界面
strategy_caption = st.empty()  # 逐行说明：设置 strategy_caption 的值
strategy_caption.caption("当前策略：IForest 异常信号 + XGB 黑产概率分层")  # 逐行说明：渲染或控制 Streamlit 界面
# 初始化 DuckDB 持久化内存连接
# 每次刷新页面都建立一个全新的、独立的内存连接，绝对不会多线程打架
con = duckdb.connect()  # 逐行说明：设置 con 的值

# 确定数据的全局日期边界
# ==========================================
# 3. 确定数据的全局日期边界 (纯本地极速模式)
# ==========================================
try:  # 逐行说明：开始捕获可能出现的异常
    IFOREST_FEATURES_PATH = ensure_feature_store(IFOREST_MODEL_KEY)  # 逐行说明：设置 IFOREST_FEATURES_PATH 的值
    XGB_FEATURES_PATH = ensure_feature_store(XGB_MODEL_KEY)  # 逐行说明：设置 XGB_FEATURES_PATH 的值

    # 直接用 DuckDB 从两个模型特征缓存的交集里秒级算出可分析日期范围
    date_range = con.execute(f"""
        WITH iforest_range AS (
            SELECT min(TRY_CAST(date AS DATE)) as min_d, max(TRY_CAST(date AS DATE)) as max_d
            FROM read_parquet('{IFOREST_FEATURES_PATH}')
        ),
        xgb_range AS (
            SELECT min(TRY_CAST(date AS DATE)) as min_d, max(TRY_CAST(date AS DATE)) as max_d
            FROM read_parquet('{XGB_FEATURES_PATH}')
        )
        SELECT GREATEST(i.min_d, x.min_d) as min_d, LEAST(i.max_d, x.max_d) as max_d
        FROM iforest_range i, xgb_range x
    """).df()  # 逐行说明：设置 date_range 的值

    min_date = pd.to_datetime(date_range['min_d'][0]).date()  # 逐行说明：设置 min_date 的值
    max_date = pd.to_datetime(date_range['max_d'][0]).date()  # 逐行说明：设置 max_date 的值

except Exception as exc:  # 逐行说明：处理捕获到的异常
    st.error(f"本地数据加载失败: {exc}")  # 逐行说明：渲染或控制 Streamlit 界面
    st.stop()  # 逐行说明：渲染或控制 Streamlit 界面

# ==========================================
# 4. 侧边栏 UI 构建
# ==========================================
st.sidebar.header("🗓️ 数据切片与规则引擎")  # 逐行说明：渲染或控制 Streamlit 界面

default_start = max(min_date, max_date - timedelta(days=3))  # 逐行说明：设置 default_start 的值
with st.sidebar.form("date_filter_form"):  # 逐行说明：进入上下文管理代码块
    st.write("### 1. 捞取数据范围")  # 逐行说明：渲染或控制 Streamlit 界面
    selected_dates = st.slider(  # 逐行说明：设置 selected_dates 的值
        "选择分析时段",  # 逐行说明：传入参数或列表项
        min_value=min_date, max_value=max_date,  # 逐行说明：设置 min_value 的值
        value=(default_start, max_date),  # 逐行说明：设置 value 的值
    )  # 逐行说明：结束当前结构
    st.form_submit_button("✅ 刷新时间切片", type="primary", use_container_width=True)  # 逐行说明：渲染或控制 Streamlit 界面

# 把日期变量提出来
start_s = selected_dates[0].strftime('%Y-%m-%d')  # 逐行说明：设置 start_s 的值
end_s = selected_dates[1].strftime('%Y-%m-%d')  # 逐行说明：设置 end_s 的值


# --- 独立模块 2：阈值调整区 ---
st.sidebar.write("### 2. 动态斩杀线调整")  # 逐行说明：渲染或控制 Streamlit 界面

# 1. “不要”的按钮：用开关控制是否启用规则
use_rule = st.sidebar.toggle("⚙️ 启用规则强杀", value=False)  # 逐行说明：设置 use_rule 的值

# 2. “重置”的按钮：点一下直接恢复 50
if st.sidebar.button("🔄 重置默认阈值 (50)"):  # 逐行说明：判断条件是否成立
    st.session_state.my_thresh_val = 50.0  # 逐行说明：渲染或控制 Streamlit 界面
    st.rerun() # 强制刷新页面

# 3. 滑块本体
if "my_thresh_val" not in st.session_state:  # 逐行说明：判断条件是否成立
    st.session_state.my_thresh_val = 50.0  # 逐行说明：渲染或控制 Streamlit 界面

manual_threshold = st.sidebar.slider(  # 逐行说明：设置 manual_threshold 的值
    "强规则危险得分",   # 逐行说明：传入参数或列表项
    min_value=10.0, max_value=100.0, step=5.0,  # 逐行说明：设置 min_value 的值
    key="my_thresh_val",  # 逐行说明：设置 key 的值
    disabled=not use_rule # 核心联动：如果上面的开关关了，这个滑块直接变灰不可点击
)  # 逐行说明：结束当前结构

# 4. 传递给大盘的真实阈值
# 如果开关关闭，就给一个永远达不到的极高阈值（比如 999.0），相当于停用规则
actual_threshold = manual_threshold if use_rule else 999.0  # 逐行说明：设置 actual_threshold 的值

st.sidebar.write("### 3. 风险分层处置")  # 逐行说明：渲染或控制 Streamlit 界面
xgb_edge_threshold = st.sidebar.number_input(  # 逐行说明：设置 xgb_edge_threshold 的值
    "XGB 边界风险阈值 (D 起点)",  # 逐行说明：传入参数或列表项
    min_value=0.0,  # 逐行说明：设置 min_value 的值
    max_value=1.0,  # 逐行说明：设置 max_value 的值
    value=float(XGB_RISK_THRESHOLDS["edge"]),  # 逐行说明：设置 value 的值
    step=0.01,  # 逐行说明：设置 step 的值
    format="%.2f",  # 逐行说明：设置 format 的值
    help="XGB 概率高于该值且 IForest 正常时，进入 D 类边界风险。",  # 逐行说明：设置 help 的值
)  # 逐行说明：结束当前结构
xgb_mid_threshold = st.sidebar.number_input(  # 逐行说明：设置 xgb_mid_threshold 的值
    "XGB 中高置信阈值 (C 起点)",  # 逐行说明：传入参数或列表项
    min_value=0.0,  # 逐行说明：设置 min_value 的值
    max_value=1.0,  # 逐行说明：设置 max_value 的值
    value=float(XGB_RISK_THRESHOLDS["mid"]),  # 逐行说明：设置 value 的值
    step=0.01,  # 逐行说明：设置 step 的值
    format="%.2f",  # 逐行说明：设置 format 的值
    help="XGB 概率高于该值且低于高置信阈值时，进入 C 类。",  # 逐行说明：设置 help 的值
)  # 逐行说明：结束当前结构
xgb_high_threshold = st.sidebar.number_input(  # 逐行说明：设置 xgb_high_threshold 的值
    "XGB 高置信阈值 (B 起点)",  # 逐行说明：传入参数或列表项
    min_value=0.0,  # 逐行说明：设置 min_value 的值
    max_value=1.0,  # 逐行说明：设置 max_value 的值
    value=float(XGB_RISK_THRESHOLDS["high"]),  # 逐行说明：设置 value 的值
    step=0.01,  # 逐行说明：设置 step 的值
    format="%.2f",  # 逐行说明：设置 format 的值
    help="IForest 正常但 XGB 概率高于该值时，进入 B 类高置信新增。",  # 逐行说明：设置 help 的值
)  # 逐行说明：结束当前结构
if not (xgb_edge_threshold <= xgb_mid_threshold <= xgb_high_threshold):  # 逐行说明：判断条件是否成立
    st.sidebar.error("XGB 阈值需要满足：边界 <= 中高 <= 高置信。")  # 逐行说明：渲染或控制 Streamlit 界面
    st.stop()  # 逐行说明：渲染或控制 Streamlit 界面
st.sidebar.caption(  # 逐行说明：渲染或控制 Streamlit 界面
    f"当前 XGB 分层：D≥{xgb_edge_threshold:.2f}，C≥{xgb_mid_threshold:.2f}，B≥{xgb_high_threshold:.2f}"
)  # 逐行说明：结束当前结构
block_b_high_conf = st.sidebar.toggle(  # 逐行说明：设置 block_b_high_conf 的值
    "将 B 类纳入直接拦截",  # 逐行说明：传入参数或列表项
    value=False,  # 逐行说明：设置 value 的值
    help="默认只直接拦截 A 类；业务确认能接受后，可把 B 类高置信新增也纳入拦截。",  # 逐行说明：设置 help 的值
)  # 逐行说明：结束当前结构
strategy_caption.caption(  # 逐行说明：渲染或控制 Streamlit 界面
    f"当前策略：IForest 异常信号 + XGB 黑产概率分层；D≥{xgb_edge_threshold:.2f}，C≥{xgb_mid_threshold:.2f}，B≥{xgb_high_threshold:.2f}。"
)  # 逐行说明：结束当前结构

st.sidebar.write("### 4. 关系网收益模拟")  # 逐行说明：渲染或控制 Streamlit 界面
use_network_rule = st.sidebar.toggle("🕸️ 模拟关系网连坐收益", value=False)  # 逐行说明：设置 use_network_rule 的值
st.sidebar.caption("仅用于 ROI/影响估算，不影响顶部拦截数或 final_label。")  # 逐行说明：渲染或控制 Streamlit 界面
poison_ratio = st.sidebar.slider(  # 逐行说明：设置 poison_ratio 的值
    "团伙污染率阈值 (高于此比例则计入模拟收益)",  # 逐行说明：传入参数或列表项
    min_value=0.05,  # 逐行说明：设置 min_value 的值
    max_value=1.0,  # 逐行说明：设置 max_value 的值
    value=0.20,  # 逐行说明：设置 value 的值
    step=0.05,  # 逐行说明：设置 step 的值
    help="仅用于估算关系网新增影响，不回写 final_label；设得越低模拟越激进。",  # 逐行说明：设置 help 的值
    disabled=not use_network_rule,  # 逐行说明：设置 disabled 的值
)  # 逐行说明：结束当前结构
actual_poison_ratio = poison_ratio if use_network_rule else 2.0  # 逐行说明：设置 actual_poison_ratio 的值
# ==========================================
# 4. 架构核心：构建统一特征视图 (Feature View Layer)
# ==========================================
con.execute("DROP VIEW IF EXISTS raw_source")  # 逐行说明：执行 DuckDB SQL 语句
con.execute("DROP TABLE IF EXISTS feature_view")  # 逐行说明：执行 DuckDB SQL 语句

con.execute(f"""
    CREATE VIEW raw_source AS
    SELECT
        r.*,
        i.model_anomaly as iforest_anomaly,
        i.model_anomaly as model_anomaly,
        i.final_time_risk,
        x.model_anomaly as xgb_model_anomaly,
        x.xgb_bot_prob as xgb_bot_prob_raw
    FROM read_parquet('{DATA_PATH}') r
    JOIN (
        -- 核心：在这里给 IForest 特征表加个保险，确保每个设备每天只有一行特征
        SELECT * FROM read_parquet('{IFOREST_FEATURES_PATH}')
        QUALIFY ROW_NUMBER() OVER(PARTITION BY date, distinct_id ORDER BY final_time_risk DESC) = 1
    ) i
      ON TRY_CAST(r.date AS DATE) = TRY_CAST(i.date AS DATE)
      AND r.distinct_id = i.distinct_id
    JOIN (
        -- XGB 特征缓存保存概率，用于风险分层
        SELECT * FROM read_parquet('{XGB_FEATURES_PATH}')
        QUALIFY ROW_NUMBER() OVER(PARTITION BY date, distinct_id ORDER BY xgb_bot_prob DESC) = 1
    ) x
      ON TRY_CAST(r.date AS DATE) = TRY_CAST(x.date AS DATE)
      AND r.distinct_id = x.distinct_id
    WHERE TRY_CAST(r.date AS DATE) BETWEEN CAST('{start_s}' AS DATE) AND CAST('{end_s}' AS DATE)
""")

block_b_sql = "true" if block_b_high_conf else "false"  # 逐行说明：设置 block_b_sql 的值

# 最终物化表
con.execute(f"""
    CREATE TEMP TABLE feature_view AS
    WITH manual_rules AS (
        SELECT DISTINCT date, distinct_id, true as is_manual_bot
        FROM raw_source
        WHERE "$os" = 'Android'
          AND "$browser" = 'Chrome Webview'
          AND ("$manufacturer" IS NULL OR "$manufacturer" = '')
    ),
    scored AS (
        SELECT
            f.*,
            COALESCE(m.is_manual_bot, false) as is_manual_bot,
            (iforest_anomaly = -1) as iforest_bot,
            COALESCE(TRY_CAST(xgb_bot_prob_raw AS DOUBLE), CASE WHEN xgb_model_anomaly = -1 THEN 1.0 ELSE 0.0 END) as xgb_bot_prob_safe,
            (final_time_risk >= {actual_threshold}) as is_rule_bot,
            COALESCE(NULLIF(TRIM(CAST("$province" AS VARCHAR)), ''), '未知') as province_display,
            COALESCE(NULLIF(TRIM(CAST("$browser" AS VARCHAR)), ''), '未知') as browser_display,
            CASE WHEN LOWER(TRIM(CAST("$is_first_day" AS VARCHAR))) IN ('1', 'true', 't', 'yes') THEN '首日登录' ELSE '非首日登录' END as first_day_display
        FROM raw_source f
        LEFT JOIN manual_rules m
          ON TRY_CAST(f.date AS DATE) = TRY_CAST(m.date AS DATE)
          AND f.distinct_id = m.distinct_id
    ),
    risk_labeled AS (
        SELECT
            *,
            CASE
                WHEN iforest_bot = true AND xgb_bot_prob_safe >= {xgb_edge_threshold} THEN 'A_双模型一致高危'
                WHEN iforest_bot = false AND xgb_bot_prob_safe >= {xgb_high_threshold} THEN 'B_XGB高置信新增'
                WHEN iforest_bot = false AND xgb_bot_prob_safe >= {xgb_mid_threshold} AND xgb_bot_prob_safe < {xgb_high_threshold} THEN 'C_XGB中高置信新增'
                WHEN iforest_bot = false AND xgb_bot_prob_safe >= {xgb_edge_threshold} AND xgb_bot_prob_safe < {xgb_mid_threshold} THEN 'D_XGB边界风险'
                WHEN iforest_bot = true AND xgb_bot_prob_safe < {xgb_edge_threshold} THEN 'E_IForest独有异常'
                WHEN iforest_bot = false AND xgb_bot_prob_safe < {xgb_edge_threshold} THEN 'F_双模型正常'
                ELSE 'UNKNOWN'
            END as risk_group
        FROM scored
    )
    SELECT
        *,
        xgb_bot_prob_safe as xgb_bot_prob,
        CASE
            WHEN risk_group = 'A_双模型一致高危' THEN '直接拦截/强风险'
            WHEN risk_group = 'B_XGB高置信新增' AND {block_b_sql} THEN '直接拦截/XGB高置信新增'
            WHEN risk_group = 'B_XGB高置信新增' THEN '高优先级审核'
            WHEN risk_group = 'C_XGB中高置信新增' THEN '观察池/抽样审核'
            WHEN risk_group = 'D_XGB边界风险' THEN '仅打标签，不拦截'
            WHEN risk_group = 'E_IForest独有异常' THEN '分歧样本复查'
            WHEN risk_group = 'F_双模型正常' THEN '放行'
            ELSE 'UNKNOWN'
        END as action,
        (risk_group = 'A_双模型一致高危' OR ({block_b_sql} AND risk_group = 'B_XGB高置信新增')) as is_model_block,
        (risk_group = 'A_双模型一致高危' OR ({block_b_sql} AND risk_group = 'B_XGB高置信新增') OR is_rule_bot) as is_auto_bot,
        (risk_group = 'A_双模型一致高危' OR ({block_b_sql} AND risk_group = 'B_XGB高置信新增') OR is_rule_bot OR is_manual_bot) as final_label
    FROM risk_labeled
""")

# 检查视图是否为空
row_count = con.execute("SELECT COUNT(*) FROM feature_view").fetchone()[0]  # 逐行说明：设置 row_count 的值
if row_count == 0:  # 逐行说明：判断条件是否成立
    st.warning("当前切片无法产出特征结果。")  # 逐行说明：渲染或控制 Streamlit 界面
    st.stop()  # 逐行说明：渲染或控制 Streamlit 界面

# 获取最新日期用于昨日统计
latest_date_str = con.execute("SELECT MAX(CAST(date AS DATE))::VARCHAR FROM feature_view").fetchone()[0]  # 逐行说明：设置 latest_date_str 的值
# ==========================================
# 5. 展现层 (Serving Layer) - 全 SQL 驱动
# ==========================================

# -- 顶部核心指标 --
metrics_sql = f"""
    SELECT
        COUNT(DISTINCT distinct_id) as total_dau,
        COUNT(DISTINCT CASE WHEN final_label THEN distinct_id END) as direct_block_dau,
        COUNT(DISTINCT CASE WHEN risk_group IN ('B_XGB高置信新增', 'C_XGB中高置信新增') AND NOT final_label THEN distinct_id END) as audit_dau,
        COUNT(DISTINCT CASE WHEN risk_group IN ('D_XGB边界风险', 'E_IForest独有异常') AND NOT final_label THEN distinct_id END) as observe_review_dau,
        COUNT(DISTINCT CASE WHEN risk_group = 'F_双模型正常' AND NOT final_label THEN distinct_id END) as pass_dau
    FROM feature_view
    WHERE date = '{latest_date_str}'
"""
m_data = con.execute(metrics_sql).fetchone()  # 逐行说明：设置 m_data 的值

header_container = st.container()  # 逐行说明：设置 header_container 的值
with header_container:  # 逐行说明：进入上下文管理代码块
    col1, col2, col3, col4, col5 = st.columns(5)  # 逐行说明：创建页面布局组件
    col1.metric("昨日总 DAU", f"{m_data[0]:,}")  # 逐行说明：创建页面布局组件
    col2.metric("直接拦截", f"{m_data[1]:,}")  # 逐行说明：创建页面布局组件
    col3.metric("人工审核池", f"{m_data[2]:,}")  # 逐行说明：创建页面布局组件
    col4.metric("观察/复查池", f"{m_data[3]:,}")  # 逐行说明：创建页面布局组件
    col5.metric("放行", f"{m_data[4]:,}")  # 逐行说明：创建页面布局组件
    st.divider()  # 逐行说明：渲染或控制 Streamlit 界面

tab_overview, tab_risk, tab_redteam, tab_geo, tab_detail, tab_network = st.tabs(["📊 核心概览", "🧭 风险分层", "🧪 红队压测", "🗺️ 地域分析", "🔍 特征明细", "🕸️ 关系网策略"])  # 逐行说明：创建页面布局组件

with tab_overview:  # 逐行说明：进入上下文管理代码块
    # -- 图表 1: 每日 DAU 趋势 --
    dau_trend_sql = """
        SELECT date,
               CASE
                   WHEN final_label THEN '直接拦截'
                   WHEN risk_group IN ('B_XGB高置信新增', 'C_XGB中高置信新增') THEN '人工审核池'
                   WHEN risk_group IN ('D_XGB边界风险', 'E_IForest独有异常') THEN '观察/复查池'
                   ELSE '放行'
               END as user_type,
               COUNT(DISTINCT distinct_id) as dau
        FROM feature_view
        GROUP BY 1, 2 ORDER BY date
    """
    fig_dau = px.line(con.execute(dau_trend_sql).df(), x="date", y="dau", color="user_type", markers=True, title="每日 DAU 趋势")  # 逐行说明：配置或生成图表
    fig_dau.update_layout(xaxis_title="日期", yaxis_title="DAU")  # 逐行说明：配置或生成图表
    st.plotly_chart(fig_dau, use_container_width=True)  # 逐行说明：渲染或控制 Streamlit 界面

    # -- 图表 2: 每日污染率与贡献率趋势 --
    daily_summary_sql = """
        SELECT date,
               COUNT(DISTINCT distinct_id) as total_dau,
               COUNT(DISTINCT CASE WHEN NOT final_label THEN distinct_id END) as kept_dau,
               COUNT(DISTINCT CASE WHEN final_label THEN distinct_id END) as removed_dau,
               COUNT(DISTINCT CASE WHEN is_manual_bot AND NOT is_auto_bot THEN distinct_id END) as manual_removed_dau,
               COUNT(DISTINCT CASE WHEN risk_group IN ('B_XGB高置信新增', 'C_XGB中高置信新增') AND NOT final_label THEN distinct_id END) as audit_dau,
               COUNT(DISTINCT CASE WHEN risk_group = 'D_XGB边界风险' AND NOT final_label THEN distinct_id END) as tag_dau,
               COUNT(DISTINCT CASE WHEN risk_group = 'E_IForest独有异常' AND NOT final_label THEN distinct_id END) as dispute_dau
        FROM feature_view GROUP BY date ORDER BY date
    """
    daily_summary_df = con.execute(daily_summary_sql).df()  # 逐行说明：设置 daily_summary_df 的值
    daily_summary_df["direct_block_rate"] = (daily_summary_df["removed_dau"] / daily_summary_df["total_dau"] * 100).fillna(0)  # 逐行说明：设置 daily_summary_df 的值
    daily_summary_df["audit_pool_rate"] = (daily_summary_df["audit_dau"] / daily_summary_df["total_dau"] * 100).fillna(0)  # 逐行说明：设置 daily_summary_df 的值

    trend_cols = st.columns(2)  # 逐行说明：创建页面布局组件
    with trend_cols[0]:  # 逐行说明：进入上下文管理代码块
        fig_pollution = px.line(daily_summary_df, x="date", y="direct_block_rate", markers=True, title="每日直接拦截占比")  # 逐行说明：配置或生成图表
        fig_pollution.update_layout(xaxis_title="日期", yaxis_title="直接拦截 / DAU (%)")  # 逐行说明：配置或生成图表
        st.plotly_chart(fig_pollution, use_container_width=True)  # 逐行说明：渲染或控制 Streamlit 界面
    with trend_cols[1]:  # 逐行说明：进入上下文管理代码块
        fig_manual_share = px.line(daily_summary_df, x="date", y="audit_pool_rate", markers=True, title="每日人工审核池占比")  # 逐行说明：配置或生成图表
        fig_manual_share.update_layout(xaxis_title="日期", yaxis_title="审核池 / DAU (%)")  # 逐行说明：配置或生成图表
        st.plotly_chart(fig_manual_share, use_container_width=True)  # 逐行说明：渲染或控制 Streamlit 界面

with tab_risk:  # 逐行说明：进入上下文管理代码块
    st.subheader("A-F 风险分层总览")  # 逐行说明：渲染或控制 Streamlit 界面
    st.caption(  # 逐行说明：渲染或控制 Streamlit 界面
        f"当前分层口径：D≥{xgb_edge_threshold:.2f}，C≥{xgb_mid_threshold:.2f}，B≥{xgb_high_threshold:.2f}；A 为 IForest 异常且 XGB≥{xgb_edge_threshold:.2f}。"
    )  # 逐行说明：结束当前结构
    risk_summary_sql = """
        SELECT
            risk_group,
            action,
            COUNT(*) as user_days,
            COUNT(DISTINCT distinct_id) as users,
            AVG(xgb_bot_prob) as avg_xgb_prob,
            QUANTILE_CONT(xgb_bot_prob, 0.50) as p50_xgb_prob,
            QUANTILE_CONT(xgb_bot_prob, 0.95) as p95_xgb_prob
        FROM (
            SELECT DISTINCT date, distinct_id, risk_group, action, xgb_bot_prob
            FROM feature_view
        )
        GROUP BY 1, 2
        ORDER BY risk_group
    """
    risk_summary_df = con.execute(risk_summary_sql).df()  # 逐行说明：设置 risk_summary_df 的值
    for prob_col in ["avg_xgb_prob", "p50_xgb_prob", "p95_xgb_prob"]:  # 逐行说明：开始循环遍历数据
        risk_summary_df[prob_col] = risk_summary_df[prob_col].round(4)  # 逐行说明：设置 risk_summary_df 的值
    st.dataframe(risk_summary_df, use_container_width=True, hide_index=True)  # 逐行说明：渲染或控制 Streamlit 界面

    risk_fig_cols = st.columns(2)  # 逐行说明：创建页面布局组件
    with risk_fig_cols[0]:  # 逐行说明：进入上下文管理代码块
        st.plotly_chart(px.bar(risk_summary_df, x="risk_group", y="user_days", color="action", title="风险分层规模（user-days）"), use_container_width=True)  # 逐行说明：渲染或控制 Streamlit 界面
    with risk_fig_cols[1]:  # 逐行说明：进入上下文管理代码块
        st.plotly_chart(px.bar(risk_summary_df, x="risk_group", y="avg_xgb_prob", color="risk_group", title="各层平均 XGB 风险概率"), use_container_width=True)  # 逐行说明：渲染或控制 Streamlit 界面

    st.subheader("分层明细样本")  # 逐行说明：渲染或控制 Streamlit 界面
    selected_risk_group = st.selectbox("选择风险分层", options=risk_summary_df["risk_group"].tolist())  # 逐行说明：设置 selected_risk_group 的值
    risk_sample_df = con.execute(f"""
        SELECT date, distinct_id, risk_group, action, iforest_bot, ROUND(xgb_bot_prob, 4) as xgb_bot_prob, final_time_risk, final_label
        FROM feature_view
        WHERE risk_group = '{selected_risk_group}'
        ORDER BY xgb_bot_prob DESC, final_time_risk DESC
        LIMIT 200
    """).df()  # 逐行说明：设置 risk_sample_df 的值
    st.dataframe(risk_sample_df, use_container_width=True, hide_index=True)  # 逐行说明：渲染或控制 Streamlit 界面

with tab_redteam:  # 逐行说明：进入上下文管理代码块
    attack_df, normal_df, redteam_summary_df = build_redteam_tables()  # 逐行说明：设置 attack_df, normal_df, redteam_summary_df 的值

    st.subheader("四类攻击压测总结")  # 逐行说明：渲染或控制 Streamlit 界面
    redteam_metrics = st.columns(4)  # 逐行说明：创建页面布局组件
    redteam_metrics[0].metric("最高召回", f"{attack_df['risk_pool_rate'].max() * 100:.1f}%")  # 逐行说明：创建页面布局组件
    redteam_metrics[1].metric("最低召回", f"{attack_df['risk_pool_rate'].min() * 100:.1f}%")  # 逐行说明：创建页面布局组件
    redteam_metrics[2].metric("最大绕过", f"{attack_df['bypass_rate'].max() * 100:.1f}%")  # 逐行说明：创建页面布局组件
    redteam_metrics[3].metric("正常池新增命中均值 proxy", f"{normal_df['normal_pool_xgb_added_hit_rate'].mean() * 100:.2f}%")  # 逐行说明：创建页面布局组件

    summary_display = redteam_summary_df[[  # 逐行说明：设置 summary_display 的值
        "scenario", "user_days", "risk_pool_rate", "bypass_rate", "iforest_rate", "xgb_09_rate", "xgb_095_rate",
        "normal_pool_direct_hit_rate", "normal_pool_xgb_added_hit_rate", "both_black", "xgb_high_new", "xgb_mid_new", "xgb_edge", "verdict"
    ]].rename(columns={  # 逐行说明：设置变量值
        "scenario": "攻击类型", "user_days": "注入用户日", "risk_pool_rate": "风险池召回率", "bypass_rate": "绕过率",
        "iforest_rate": "IForest召回率", "xgb_09_rate": "XGB>=0.9召回率", "xgb_095_rate": "XGB>=0.95召回率",
        "normal_pool_direct_hit_rate": "正常池A类命中率 proxy", "normal_pool_xgb_added_hit_rate": "正常池新增命中率 proxy(B/C/D)",
        "both_black": "A类", "xgb_high_new": "B类", "xgb_mid_new": "C类", "xgb_edge": "D类", "verdict": "结论",
    })  # 逐行说明：结束当前结构
    summary_display = _format_percent_columns(  # 逐行说明：设置 summary_display 的值
        summary_display,
        ["风险池召回率", "绕过率", "IForest召回率", "XGB>=0.9召回率", "XGB>=0.95召回率", "正常池A类命中率 proxy", "正常池新增命中率 proxy(B/C/D)"],
    )  # 逐行说明：结束当前结构
    st.dataframe(summary_display, use_container_width=True, hide_index=True)  # 逐行说明：渲染或控制 Streamlit 界面

    redteam_chart_df = redteam_summary_df.rename(columns={"risk_pool_rate": "风险池召回率", "bypass_rate": "绕过率"})  # 逐行说明：设置 redteam_chart_df 的值
    redteam_chart_cols = st.columns(2)  # 逐行说明：创建页面布局组件
    with redteam_chart_cols[0]:  # 逐行说明：进入上下文管理代码块
        fig_redteam_recall = px.bar(  # 逐行说明：配置或生成图表
            redteam_chart_df,
            x="scenario",
            y=["风险池召回率", "绕过率"],
            barmode="group",
            title="攻击召回率与绕过率",
            labels={"scenario": "攻击类型", "value": "比例", "variable": "指标"},
            text_auto=".1%",
        )  # 逐行说明：结束当前结构
        fig_redteam_recall.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1])  # 逐行说明：配置或生成图表
        st.plotly_chart(fig_redteam_recall, use_container_width=True)  # 逐行说明：渲染或控制 Streamlit 界面
    with redteam_chart_cols[1]:  # 逐行说明：进入上下文管理代码块
        normal_chart_df = normal_df.rename(columns={  # 逐行说明：设置 normal_chart_df 的值
            "normal_pool_direct_hit_rate": "A类直接命中率 proxy",
            "normal_pool_xgb_added_hit_rate": "B/C/D新增命中率 proxy",
        })  # 逐行说明：结束当前结构
        fig_redteam_fp = px.bar(  # 逐行说明：配置或生成图表
            normal_chart_df,
            x="scenario",
            y=["A类直接命中率 proxy", "B/C/D新增命中率 proxy"],
            barmode="group",
            title="正常池命中压力 proxy",
            labels={"scenario": "攻击类型", "value": "比例", "variable": "指标"},
            text_auto=".1%",
        )  # 逐行说明：结束当前结构
        fig_redteam_fp.update_layout(yaxis_tickformat=".0%")  # 逐行说明：配置或生成图表
        st.plotly_chart(fig_redteam_fp, use_container_width=True)  # 逐行说明：渲染或控制 Streamlit 界面

    st.subheader("压测明细")  # 逐行说明：渲染或控制 Streamlit 界面
    detail_left, detail_right = st.columns(2)  # 逐行说明：创建页面布局组件
    with detail_left:  # 逐行说明：进入上下文管理代码块
        attack_detail = attack_df[[  # 逐行说明：设置 attack_detail 的值
            "scenario", "iforest_black", "xgb_05_black", "xgb_09_black", "xgb_095_black",
            "both_black", "xgb_high_new", "xgb_mid_new", "xgb_edge", "iforest_only", "risk_pool_hits",
        ]].rename(columns={  # 逐行说明：设置变量值
            "scenario": "攻击类型", "iforest_black": "IForest命中", "xgb_05_black": "XGB>=0.5", "xgb_09_black": "XGB>=0.9",
            "xgb_095_black": "XGB>=0.95", "both_black": "A类", "xgb_high_new": "B类", "xgb_mid_new": "C类",
            "xgb_edge": "D类", "iforest_only": "E类", "risk_pool_hits": "入池总数",
        })  # 逐行说明：结束当前结构
        st.dataframe(attack_detail, use_container_width=True, hide_index=True)  # 逐行说明：渲染或控制 Streamlit 界面
    with detail_right:  # 逐行说明：进入上下文管理代码块
        normal_detail = normal_df[[  # 逐行说明：设置 normal_detail 的值
            "scenario", "user_days", "both_black", "xgb_high_new", "xgb_mid_new", "xgb_edge", "iforest_only", "risk_pool_rate",
            "normal_pool_direct_hit_rate", "normal_pool_xgb_added_hit_rate",
        ]].rename(columns={  # 逐行说明：设置变量值
            "scenario": "正常池对照", "user_days": "用户日", "both_black": "A类", "xgb_high_new": "B类", "xgb_mid_new": "C类",
            "xgb_edge": "D类", "iforest_only": "E类", "risk_pool_rate": "总入池率",
            "normal_pool_direct_hit_rate": "A类命中率 proxy", "normal_pool_xgb_added_hit_rate": "新增命中率 proxy(B/C/D)",
        })  # 逐行说明：结束当前结构
        normal_detail = _format_percent_columns(normal_detail, ["总入池率", "A类命中率 proxy", "新增命中率 proxy(B/C/D)"])  # 逐行说明：设置 normal_detail 的值
        st.dataframe(normal_detail, use_container_width=True, hide_index=True)  # 逐行说明：渲染或控制 Streamlit 界面

with tab_geo:  # 逐行说明：进入上下文管理代码块
    # -- 图表 3 & 4: 省份分布与剔除率 --
    province_sql = f"""
        SELECT province_display,
               COUNT(DISTINCT distinct_id) as before_count,
               COUNT(DISTINCT CASE WHEN NOT final_label THEN distinct_id END) as after_count,
               COUNT(DISTINCT CASE WHEN final_label THEN distinct_id END) as removed_count
        FROM feature_view WHERE date = '{latest_date_str}'
        GROUP BY 1 ORDER BY before_count DESC LIMIT 15
    """
    prov_df = con.execute(province_sql).df()  # 逐行说明：设置 prov_df 的值

    # 剔除前后对比图
    prov_melt = prov_df.melt(id_vars="province_display", value_vars=["before_count", "after_count"], var_name="stage", value_name="user_count")  # 逐行说明：设置 prov_melt 的值
    prov_melt["stage"] = prov_melt["stage"].map({"before_count": "剔除前", "after_count": "剔除后"})  # 逐行说明：设置 prov_melt 的值
    fig_prov_comp = px.bar(prov_melt, x="user_count", y="province_display", color="stage", barmode="group", orientation="h", title="昨日各省份剔除前后人数对比")  # 逐行说明：配置或生成图表

    # 各省份剔除率排行
    prov_df["removal_rate"] = (prov_df["removed_count"] / prov_df["before_count"] * 100).fillna(0)  # 逐行说明：设置 prov_df 的值
    prov_df_rate = prov_df.sort_values("removal_rate", ascending=False)  # 逐行说明：设置 prov_df_rate 的值
    fig_prov_rate = px.bar(prov_df_rate, x="removal_rate", y="province_display", orientation="h", title="昨日各省份剔除率排行", text=prov_df_rate["removal_rate"].map(lambda x: f"{x:.1f}%"))  # 逐行说明：配置或生成图表

    province_cols = st.columns(2)  # 逐行说明：创建页面布局组件
    with province_cols[0]:  # 逐行说明：进入上下文管理代码块
        st.plotly_chart(fig_prov_comp, use_container_width=True)  # 逐行说明：渲染或控制 Streamlit 界面
    with province_cols[1]:  # 逐行说明：进入上下文管理代码块
        st.plotly_chart(fig_prov_rate, use_container_width=True)  # 逐行说明：渲染或控制 Streamlit 界面

    # -- 图表 5: 剔除用户 Top 8 省份 --
    top_removed_sql = f"""
        SELECT province_display as province, COUNT(DISTINCT distinct_id) as user_count
        FROM feature_view WHERE date = '{latest_date_str}' AND final_label = true
        GROUP BY 1 ORDER BY user_count DESC LIMIT 8
    """
    fig_top_removed = px.bar(con.execute(top_removed_sql).df(), x="province", y="user_count", title="昨日直接拦截用户 Top 8 省份", text="user_count")  # 逐行说明：配置或生成图表
    st.plotly_chart(fig_top_removed, use_container_width=True)  # 逐行说明：渲染或控制 Streamlit 界面

    # 1. SQL 聚合：计算每个省份在剔除前后的占比
    # 我们用窗口函数直接在 SQL 里算出各自的总量和占比
    prov_comp_sql = f"""
        WITH stats AS (
            SELECT
                province_display,
                COUNT(DISTINCT distinct_id) as total_users,
                COUNT(DISTINCT CASE WHEN NOT final_label THEN distinct_id END) as human_users
            FROM feature_view
            WHERE date = '{latest_date_str}'
            GROUP BY 1
        ),
        totals AS (
            SELECT
                SUM(total_users) as grand_total_before,
                SUM(human_users) as grand_total_after
            FROM stats
        )
        SELECT
            s.province_display as province,
            s.total_users / t.grand_total_before * 100 as percent_before,
            s.human_users / t.grand_total_after * 100 as percent_after
        FROM stats s, totals t
        WHERE s.total_users > 0
        ORDER BY percent_before DESC
        LIMIT 15
    """
    df_prov_comp = con.execute(prov_comp_sql).df()  # 逐行说明：设置 df_prov_comp 的值

    # 2. 画哑铃图 (Dumbbell Plot)
    import plotly.graph_objects as go  # 逐行说明：导入运行所需模块

    fig_dumbbell = go.Figure()  # 逐行说明：配置或生成图表

    # 添加“剔除前”的散点
    fig_dumbbell.add_trace(go.Scatter(  # 逐行说明：配置或生成图表
        x=df_prov_comp["percent_before"],  # 逐行说明：设置 x 的值
        y=df_prov_comp["province"],  # 逐行说明：设置 y 的值
        mode='markers',  # 逐行说明：设置 mode 的值
        name='剔除前占比 (%)',  # 逐行说明：设置 name 的值
        marker=dict(color='#ff4b4b', size=12)  # 逐行说明：设置 marker 的值
    ))  # 逐行说明：执行这一行逻辑

    # 添加“剔除后”的散点
    fig_dumbbell.add_trace(go.Scatter(  # 逐行说明：配置或生成图表
        x=df_prov_comp["percent_after"],  # 逐行说明：设置 x 的值
        y=df_prov_comp["province"],  # 逐行说明：设置 y 的值
        mode='markers',  # 逐行说明：设置 mode 的值
        name='剔除后占比 (%)',  # 逐行说明：设置 name 的值
        marker=dict(color='#00cc96', size=12)  # 逐行说明：设置 marker 的值
    ))  # 逐行说明：执行这一行逻辑

    # 添加连接两点的线条
    for i, row in df_prov_comp.iterrows():  # 逐行说明：开始循环遍历数据
        fig_dumbbell.add_shape(  # 逐行说明：配置或生成图表
            type='line',  # 逐行说明：设置 type 的值
            x0=row["percent_before"], y0=row["province"],  # 逐行说明：设置 x0 的值
            x1=row["percent_after"], y1=row["province"],  # 逐行说明：设置 x1 的值
            line=dict(color='gray', width=2, dash='dot')  # 逐行说明：设置 line 的值
        )  # 逐行说明：结束当前结构

    fig_dumbbell.update_layout(  # 逐行说明：配置或生成图表
        title="各省份 DAU 占比结构偏移 (哑铃图)",  # 逐行说明：设置 title 的值
        xaxis_title="占总流量百分比 (%)",  # 逐行说明：设置 xaxis_title 的值
        yaxis_title="省份",  # 逐行说明：设置 yaxis_title 的值
        yaxis={'categoryorder':'total ascending'}, # 按比例排序
        height=600  # 逐行说明：设置 height 的值
    )  # 逐行说明：结束当前结构
    st.plotly_chart(fig_dumbbell, use_container_width=True)  # 逐行说明：渲染或控制 Streamlit 界面

    # 1. SQL 聚合：计算非黑产用户的省份占比
    clean_waterfall_sql = f"""
        WITH province_counts AS (
            SELECT
                province_display as province,
                COUNT(DISTINCT distinct_id) as human_count
            FROM feature_view
            WHERE date = '{latest_date_str}' AND final_label = false
            GROUP BY 1
        ),
        total_human AS (SELECT SUM(human_count) FROM province_counts),
        top_provinces AS (
            SELECT
                province,
                human_count,
                human_count * 100.0 / (SELECT * FROM total_human) as share
            FROM province_counts
            ORDER BY human_count DESC
            LIMIT 10
        ),
        others AS (
            SELECT
                '其他' as province,
                (SELECT * FROM total_human) - SUM(human_count) as human_count,
                100.0 - SUM(share) as share
            FROM top_provinces
        )
        SELECT * FROM top_provinces
        UNION ALL
        SELECT * FROM others
    """
    df_clean_wf = con.execute(clean_waterfall_sql).df()  # 逐行说明：设置 df_clean_wf 的值

    fig_clean_wf = go.Figure(go.Waterfall(  # 逐行说明：配置或生成图表
        name="真实用户占比",  # 逐行说明：设置 name 的值
        orientation="v",  # 逐行说明：设置 orientation 的值
        measure=["relative"] * len(df_clean_wf) + ["total"],  # 逐行说明：设置 measure 的值
        x=df_clean_wf["province"].tolist() + ["全量真实用户"],  # 逐行说明：设置 x 的值
        textposition="outside",  # 逐行说明：设置 textposition 的值
        text=[f"{v:.1f}%" for v in df_clean_wf["share"]] + ["100%"],  # 逐行说明：设置 text 的值
        y=df_clean_wf["share"].tolist() + [0],  # 逐行说明：设置 y 的值
        connector={"line": {"color": "rgb(63, 63, 63)"}},  # 逐行说明：设置 connector 的值
        increasing={"marker": {"color": "#00cc96"}}, # 真实用户用绿色表示，看着舒服
        totals={"marker": {"color": "#636efa"}}  # 逐行说明：设置 totals 的值
    ))  # 逐行说明：执行这一行逻辑

    fig_clean_wf.update_layout(  # 逐行说明：配置或生成图表
        title="昨日直接拦截后：各省份对放行 DAU 的占比贡献",  # 逐行说明：设置 title 的值
        yaxis_title="占比 (%)",  # 逐行说明：设置 yaxis_title 的值
        showlegend=False,  # 逐行说明：设置 showlegend 的值
        height=500  # 逐行说明：设置 height 的值
    )  # 逐行说明：结束当前结构

    # 1. SQL 聚合：计算黑产用户的省份占比
    bot_waterfall_sql = f"""
        WITH province_counts AS (
            SELECT
                province_display as province,
                COUNT(DISTINCT distinct_id) as bot_count
            FROM feature_view
            WHERE date = '{latest_date_str}' AND final_label = true
            GROUP BY 1
        ),
        total_bot AS (SELECT SUM(bot_count) FROM province_counts),
        top_provinces AS (
            SELECT
                province,
                bot_count,
                bot_count * 100.0 / (SELECT * FROM total_bot) as share
            FROM province_counts
            ORDER BY bot_count DESC
            LIMIT 10
        ),
        others AS (
            SELECT
                '其他' as province,
                (SELECT * FROM total_bot) - SUM(bot_count) as bot_count,
                100.0 - SUM(share) as share
            FROM top_provinces
        )
        SELECT * FROM top_provinces
        UNION ALL
        SELECT * FROM others
    """
    df_bot_wf = con.execute(bot_waterfall_sql).df()  # 逐行说明：设置 df_bot_wf 的值

    fig_bot_wf = go.Figure(go.Waterfall(  # 逐行说明：配置或生成图表
        name="黑产占比",  # 逐行说明：设置 name 的值
        orientation="v",  # 逐行说明：设置 orientation 的值
        measure=["relative"] * len(df_bot_wf) + ["total"],  # 逐行说明：设置 measure 的值
        x=df_bot_wf["province"].tolist() + ["总黑产流量"],  # 逐行说明：设置 x 的值
        textposition="outside",  # 逐行说明：设置 textposition 的值
        text=[f"{v:.1f}%" for v in df_bot_wf["share"]] + ["100%"],  # 逐行说明：设置 text 的值
        y=df_bot_wf["share"].tolist() + [0],  # 逐行说明：设置 y 的值
        connector={"line": {"color": "rgb(63, 63, 63)"}},  # 逐行说明：设置 connector 的值
        increasing={"marker": {"color": "#ef553b"}}, # 黑产用红色，警示色
        totals={"marker": {"color": "#3b3b3b"}}      # 汇总用深色
    ))  # 逐行说明：执行这一行逻辑

    fig_bot_wf.update_layout(  # 逐行说明：配置或生成图表
        title="昨日直接拦截结构：各省份对拦截量的占比贡献",  # 逐行说明：设置 title 的值
        yaxis_title="占比 (%)",  # 逐行说明：设置 yaxis_title 的值
        showlegend=False,  # 逐行说明：设置 showlegend 的值
        height=500  # 逐行说明：设置 height 的值
    )  # 逐行说明：结束当前结构

    waterfall_cols = st.columns(2)  # 逐行说明：创建页面布局组件
    with waterfall_cols[0]:  # 逐行说明：进入上下文管理代码块
        st.plotly_chart(fig_clean_wf, use_container_width=True)  # 逐行说明：渲染或控制 Streamlit 界面
    with waterfall_cols[1]:  # 逐行说明：进入上下文管理代码块
        st.plotly_chart(fig_bot_wf, use_container_width=True)  # 逐行说明：渲染或控制 Streamlit 界面

with tab_detail:  # 逐行说明：进入上下文管理代码块
    # -- 图表 6: 小时浏览时间分布 --
    hourly_sql = f"""
        SELECT hour_time, CASE WHEN final_label THEN '直接拦截' ELSE '未直接拦截' END as user_type, COUNT(DISTINCT distinct_id) as user_count
        FROM feature_view WHERE date = '{latest_date_str}'
        GROUP BY 1, 2 ORDER BY hour_time
    """
    fig_hourly = px.line(con.execute(hourly_sql).df(), x="hour_time", y="user_count", color="user_type", markers=True, title="直接拦截与未直接拦截用户浏览时间分布")  # 逐行说明：配置或生成图表
    st.plotly_chart(fig_hourly, use_container_width=True)  # 逐行说明：渲染或控制 Streamlit 界面

    # -- 图表 7: 浏览器饼图 --
    browser_cols = st.columns(2)  # 逐行说明：创建页面布局组件
    bot_browser_sql = f"SELECT browser_display, COUNT(DISTINCT distinct_id) as user_count FROM feature_view WHERE date = '{latest_date_str}' AND final_label = true GROUP BY 1 ORDER BY 2 DESC"  # 逐行说明：设置 bot_browser_sql 的值
    normal_browser_sql = f"SELECT browser_display, COUNT(DISTINCT distinct_id) as user_count FROM feature_view WHERE date = '{latest_date_str}' AND final_label = false GROUP BY 1 ORDER BY 2 DESC"  # 逐行说明：设置 normal_browser_sql 的值

    with browser_cols[0]:  # 逐行说明：进入上下文管理代码块
        st.plotly_chart(px.pie(con.execute(bot_browser_sql).df(), names="browser_display", values="user_count", hole=0.55, title="直接拦截用户浏览器分布"), use_container_width=True)  # 逐行说明：渲染或控制 Streamlit 界面
    with browser_cols[1]:  # 逐行说明：进入上下文管理代码块
        st.plotly_chart(px.pie(con.execute(normal_browser_sql).df(), names="browser_display", values="user_count", hole=0.55, title="未直接拦截用户浏览器分布"), use_container_width=True)  # 逐行说明：渲染或控制 Streamlit 界面

    # -- 图表 8: IP C 段分布直方图 --
    # 用 DuckDB 正则提取 C 段，极其高效
    ip_sql = f"""
        WITH ip_base AS (
            SELECT distinct_id, CASE WHEN final_label THEN '直接拦截' ELSE '未直接拦截' END as user_type,
                   REGEXP_EXTRACT(CAST("$ip" AS VARCHAR), '^(\\d+\\.\\d+\\.\\d+)', 1) as ip_c
            FROM feature_view WHERE date = '{latest_date_str}'
        ),
        ip_counts AS (
            SELECT ip_c, COUNT(DISTINCT distinct_id) as c_segment_user_count
            FROM ip_base GROUP BY ip_c
        )
        SELECT i.distinct_id, i.user_type, MAX(c.c_segment_user_count) as c_segment_user_count
        FROM ip_base i JOIN ip_counts c ON i.ip_c = c.ip_c
        GROUP BY 1, 2
    """
    ip_df = con.execute(ip_sql).df()  # 逐行说明：设置 ip_df 的值
    if not ip_df.empty:  # 逐行说明：判断条件是否成立
        fig_ip_hist = px.histogram(ip_df, x="c_segment_user_count", color="user_type", barmode="overlay", nbins=30, title="昨日直接拦截与未直接拦截 C 段 IP 用户数分布", opacity=0.65)  # 逐行说明：配置或生成图表
        fig_ip_hist.update_layout(xaxis_title="同 C 段用户数", yaxis_title="用户数")  # 逐行说明：配置或生成图表
        st.plotly_chart(fig_ip_hist, use_container_width=True)  # 逐行说明：渲染或控制 Streamlit 界面

    # -- 图表 9: 首日登录分布饼图 --
    first_day_cols = st.columns(2)  # 逐行说明：创建页面布局组件
    bot_fd_sql = f"SELECT first_day_display, COUNT(DISTINCT distinct_id) as user_count FROM feature_view WHERE date = '{latest_date_str}' AND final_label = true GROUP BY 1"  # 逐行说明：设置 bot_fd_sql 的值
    norm_fd_sql = f"SELECT first_day_display, COUNT(DISTINCT distinct_id) as user_count FROM feature_view WHERE date = '{latest_date_str}' AND final_label = false GROUP BY 1"  # 逐行说明：设置 norm_fd_sql 的值

    with first_day_cols[0]:  # 逐行说明：进入上下文管理代码块
        st.plotly_chart(px.pie(con.execute(bot_fd_sql).df(), names="first_day_display", values="user_count", hole=0.55, title="昨日直接拦截用户首日登录分布"), use_container_width=True)  # 逐行说明：渲染或控制 Streamlit 界面
    with first_day_cols[1]:  # 逐行说明：进入上下文管理代码块
        st.plotly_chart(px.pie(con.execute(norm_fd_sql).df(), names="first_day_display", values="user_count", hole=0.55, title="昨日未直接拦截用户首日登录分布"), use_container_width=True)  # 逐行说明：渲染或控制 Streamlit 界面

    # -- 表格 1: Title & URL UV-PV 对比 --
    title_url_cols = st.columns(2)  # 逐行说明：创建页面布局组件
    # 排除特定的通用首页 title，展示明细
    detail_sql_template = """
        SELECT "$title", "$url", COUNT(DISTINCT distinct_id) as uv, COUNT("$url") as pv
        FROM feature_view
        WHERE date = '{latest_date_str}' AND final_label = {label_bool}
          AND "$title" NOT LIKE '36氪_让一部分人先看到未来'
        GROUP BY 1, 2 ORDER BY pv DESC, uv DESC LIMIT 20
    """
    with title_url_cols[0]:  # 逐行说明：进入上下文管理代码块
        st.write("#### 昨日直接拦截用户标题/URL UV-PV")  # 逐行说明：渲染或控制 Streamlit 界面
        st.dataframe(con.execute(detail_sql_template.format(latest_date_str=latest_date_str, label_bool='true')).df(), use_container_width=True, hide_index=True)  # 逐行说明：渲染或控制 Streamlit 界面
    with title_url_cols[1]:  # 逐行说明：进入上下文管理代码块
        st.write("#### 昨日未直接拦截用户标题/URL UV-PV")  # 逐行说明：渲染或控制 Streamlit 界面
        st.dataframe(con.execute(detail_sql_template.format(latest_date_str=latest_date_str, label_bool='false')).df(), use_container_width=True, hide_index=True)  # 逐行说明：渲染或控制 Streamlit 界面

    # -- 表格 2: 每日明细总表 --
    st.write("#### 每日明细")  # 逐行说明：渲染或控制 Streamlit 界面
    daily_detail_df = daily_summary_df.rename(columns={  # 逐行说明：设置 daily_detail_df 的值
        "date": "日期", "total_dau": "dau", "kept_dau": "直接放行", "removed_dau": "直接拦截", "audit_dau": "人工审核池", "tag_dau": "仅打标签", "dispute_dau": "分歧复查"  # 逐行说明：配置字典字段
    })[["日期", "dau", "直接放行", "直接拦截", "人工审核池", "仅打标签", "分歧复查"]].sort_values("日期", ascending=False)  # 逐行说明：设置 }) 的值
    st.dataframe(daily_detail_df, use_container_width=True, hide_index=True)  # 逐行说明：渲染或控制 Streamlit 界面

    # -- 底部：名单导出 --
    bot_export_sql = "SELECT * FROM feature_view WHERE final_label = true OR risk_group != 'F_双模型正常'"  # 逐行说明：设置 bot_export_sql 的值
    bot_list_df = con.execute(bot_export_sql).df()  # 逐行说明：设置 bot_list_df 的值
    st.download_button(  # 逐行说明：渲染或控制 Streamlit 界面
        label="导出风险设备完整名单 (CSV)",  # 逐行说明：设置 label 的值
        data=convert_df_to_csv(bot_list_df),  # 逐行说明：设置 data 的值
        file_name=f"bot_list_{start_s}_to_{end_s}.csv",  # 逐行说明：设置 file_name 的值
        mime="text/csv",  # 逐行说明：设置 mime 的值
    )  # 逐行说明：结束当前结构

with tab_network:  # 逐行说明：进入上下文管理代码块
    st.subheader("关系网模拟新增占比趋势 (占总 DAU)")  # 逐行说明：渲染或控制 Streamlit 界面
    st.caption("当前模块仅模拟关系网连坐的新增影响，不回写 feature_view.final_label，也不改变顶部直接拦截指标。")  # 逐行说明：渲染或控制 Streamlit 界面

    feature_view_cols = set(con.execute("DESCRIBE feature_view").df()["column_name"].tolist())  # 逐行说明：设置 feature_view_cols 的值
    network_optional_cols = [  # 逐行说明：设置 network_optional_cols 的值
        "ip_c_segment", "soft_fp", "$ip", "$os", "$browser", "$manufacturer", "$province",
        "$is_first_day", "$is_login_id", "province_display", "browser_display",
    ]  # 逐行说明：结束当前结构
    network_select_cols = [col for col in network_optional_cols if col in feature_view_cols]  # 逐行说明：设置 network_select_cols 的值
    optional_select_sql = ""  # 逐行说明：设置 optional_select_sql 的值
    if network_select_cols:  # 逐行说明：判断条件是否成立
        optional_select_sql = ",\n            " + ",\n            ".join([f'"{col}"' for col in network_select_cols])  # 逐行说明：设置 optional_select_sql 的值

    plot_base_sql = f"""
        SELECT DISTINCT
            CAST(date AS DATE)::VARCHAR as date,
            distinct_id,
            final_label as is_bot{optional_select_sql}
        FROM feature_view
    """  # 逐行说明：设置 plot_base_sql 的值
    plot_base_df = con.execute(plot_base_sql).df()  # 逐行说明：设置 plot_base_df 的值
    _, network_field_msg = prepare_network_fields(plot_base_df)  # 逐行说明：设置 _, network_field_msg 的值
    if network_field_msg:  # 逐行说明：判断条件是否成立
        if "不可用" in network_field_msg:  # 逐行说明：判断条件是否成立
            st.warning(network_field_msg)  # 逐行说明：渲染或控制 Streamlit 界面
        else:  # 逐行说明：处理其他情况
            st.caption(f"关系网字段口径：{network_field_msg}；覆盖差异见表格里的“关系网可计算DAU”。")  # 逐行说明：渲染或控制 Streamlit 界面

    with st.spinner("正在穿透关系网计算全量数据..."):  # 逐行说明：进入上下文管理代码块
        roi_df = calculate_net_roi(plot_base_df, actual_poison_ratio)  # 逐行说明：设置 roi_df 的值

    if roi_df.empty:  # 逐行说明：判断条件是否成立
        st.info("当前时间切片没有可用于关系网分析的数据。")  # 逐行说明：渲染或控制 Streamlit 界面
    else:  # 逐行说明：处理其他情况
        fig_net_roi = px.line(  # 逐行说明：配置或生成图表
            roi_df,  # 逐行说明：传入参数或列表项
            x="date",  # 逐行说明：设置 x 的值
            y="关系网影响比(%)",  # 逐行说明：设置 y 的值
            markers=True,  # 逐行说明：设置 markers 的值
            title=f"当前污染率阈值: {poison_ratio * 100:.0f}%" if use_network_rule else "关系网收益模拟已关闭",  # 逐行说明：设置 title 的值
            labels={"关系网影响比(%)": "拦截人数 / DAU (%)"},  # 逐行说明：设置 labels 的值
            template="plotly_white",  # 逐行说明：设置 template 的值
        )  # 逐行说明：结束当前结构
        avg_val = roi_df["关系网影响比(%)"].mean()  # 逐行说明：设置 avg_val 的值
        fig_net_roi.add_hline(y=avg_val, line_dash="dash", annotation_text=f"平均占比: {avg_val:.2f}%")  # 逐行说明：配置或生成图表
        fig_net_roi.update_layout(xaxis_title="日期", yaxis_title="关系网模拟新增 / DAU (%)")  # 逐行说明：配置或生成图表
        st.plotly_chart(fig_net_roi, use_container_width=True)  # 逐行说明：渲染或控制 Streamlit 界面

        st.subheader("每日模拟收益统计")  # 逐行说明：渲染或控制 Streamlit 界面
        display_df = roi_df.copy()  # 逐行说明：设置 display_df 的值
        display_df["关系网影响比(%)"] = display_df["关系网影响比(%)"].map(lambda x: f"{x}%")  # 逐行说明：设置 display_df 的值
        st.dataframe(display_df.sort_values("date", ascending=False), use_container_width=True, hide_index=True)  # 逐行说明：渲染或控制 Streamlit 界面
