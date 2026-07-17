import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


RESOLUTION_Z_FEATURE = "resolution_share_positive_zscore"
OTHER_RESOLUTION = "__OTHER_RESOLUTION__"
MIN_SCREEN_PX = 100
MAX_SCREEN_PX = 10_000


def _normalize_screen_events(df):
    """Normalize raw Sensors Analytics screen fields without dropping events."""
    required = ["$screen_width", "$screen_height"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"分辨率特征缺少原始字段: {missing}")

    out = df.copy()
    width = pd.to_numeric(out["$screen_width"], errors="coerce")
    height = pd.to_numeric(out["$screen_height"], errors="coerce")
    valid = (
        width.notna()
        & height.notna()
        & width.between(MIN_SCREEN_PX, MAX_SCREEN_PX)
        & height.between(MIN_SCREEN_PX, MAX_SCREEN_PX)
    )
    width = width.where(valid).round().astype("Int64")
    height = height.where(valid).round().astype("Int64")
    short_side = pd.concat([width, height], axis=1).min(axis=1).astype("Int64")
    long_side = pd.concat([width, height], axis=1).max(axis=1).astype("Int64")
    out["resolution_canonical"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out.loc[valid, "resolution_canonical"] = (
        short_side.loc[valid].astype("string")
        + "x"
        + long_side.loc[valid].astype("string")
    )
    return out


def _stable_mode(series):
    values = series.dropna()
    if values.empty:
        return pd.NA
    return values.value_counts().index[0]


def _resolution_main_user_days(events):
    required = ["date", "distinct_id", "$screen_width", "$screen_height"]
    missing = [col for col in required if col not in events.columns]
    if missing:
        raise ValueError(f"分辨率特征输入缺少字段: {missing}")

    normalized = _normalize_screen_events(events)
    normalized["date"] = pd.to_datetime(normalized["date"], errors="raise").dt.normalize()
    return (
        normalized.groupby(["date", "distinct_id"], observed=True)["resolution_canonical"]
        .agg(_stable_mode)
        .rename("main_resolution_canonical")
        .reset_index()
    )


class FeatureColumnSelector(BaseEstimator, TransformerMixin):
    """Stable, pickle-safe selector used by versioned production pipelines."""

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        missing = [col for col in self.columns if col not in X.columns]
        if missing:
            raise ValueError(f"模型特征缺失: {missing}")
        return self

    def transform(self, X):
        missing = [col for col in self.columns if col not in X.columns]
        if missing:
            raise ValueError(f"模型特征缺失: {missing}")
        return X.loc[:, self.columns]

class UnifiedUserBehaviorCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, feature_cols):#创建初始参数，后续都要使用，所以在__init__中使用
        self.feature_cols = feature_cols#定义我们最终要输出的特征列顺序，确保每次 transform 都输出一致的列
        self.night_weights = {h: np.exp(-((h - 3.0)**2) / 6.0) for h in range(24)}#构造晚上用户活跃权重，峰值在凌晨3点，向两边平滑衰减
        self.evening_weights = {h: np.exp(-((h - 21.0)**2) / 4.0) for h in range(24)}#构造傍晚用户活跃权重，峰值在晚上9点，向两边平滑衰减

    def fit(self, X, y=None):
        df = X.copy()#复制一份新的数据
        
        
        # 学习真人基准分布
        daily_hour = df.groupby(["date", "hour_time"]).size().reset_index(name="pv")#按照日期、小时分组，size()统计会比 count()快
        daily_hour["ratio"] = daily_hour["pv"] / (daily_hour.groupby("date")["pv"].transform("sum") + 1e-12)#计算每个小时的访问占当天总访问的比例，避免除零错误
        standard_dist = daily_hour.groupby("hour_time")["ratio"].median().clip(lower=1e-6)#取每个小时的中位数作为标准分布，clip 确保没有零值，避免后续计算中的 log(0) 问题
        self.ref_hour_dist_ = (standard_dist / standard_dist.sum()).to_dict()#归一化成概率分布，并转换成字典形式，方便后续映射
        
        self.ref_night_mass_ = sum(self.ref_hour_dist_.get(h, 0.0) * self.night_weights[h] for h in range(24))#计算基准的夜晚活跃质量总和，作为后续风险计算的分母基准
        self.ref_evening_mass_ = sum(self.ref_hour_dist_.get(h, 0.0) * self.evening_weights[h] for h in range(24))#计算基准的傍晚活跃质量总和，作为后续风险计算的分母基准

        # 学习标准 C 段聚集度
        df['ip_c_segment'] = df['$ip'].astype(str).str.rsplit('.', n=1).str[0]#获取c段ip
        ip_counts = df.groupby(['date', 'ip_c_segment'])['distinct_id'].nunique().reset_index(name='cnt')#统计每天每个c段的独立用户数，得到一个包含 date、ip_c_segment 和 cnt（独立用户数）的 DataFrame
        self.ref_c_mean_ = ip_counts.groupby('date')['cnt'].median().mean()#计算c段独立用户数的中位数的均值
        self.ref_c_std_ = ip_counts.groupby('date')['cnt'].std().mean() or 1.0#计算c段独立用户数的标准差的均值，或默认值1.0
        
        return self

    def transform(self, X):
        df = X.copy()#复制一份新的数据，避免修改原始输入
        gcols = ['date', 'distinct_id']#定义用户级别的分组列，后续很多特征都是基于用户维度来计算的
        
        
        
        # --- 0. 基础特征提取 ---
        df['ip_c_segment'] = df['$ip'].astype(str).str.rsplit('.', n=1).str[0]#获取c段ip
        df['is_direct'] = (df['$referrer'].isna() | (df['$referrer'] == "")).astype(int)#判断是否是直接访问
        
        user_pv_map = df.groupby(gcols, observed=True).size().reset_index(name='user_total_pv')#统计每个用户每天的总访问次数，得到一个包含 date、distinct_id 和 user_total_pv 的 DataFrame
        df = df.merge(user_pv_map, on=gcols, how='left')#把用户总访问次数合并回原始数据，方便后续特征计算
        df['is_1pv_user'] = (df['user_total_pv'] == 1).astype(int)#统计该用户是否是当天的单次访问用户，单次访问往往风险更高

        # ==========================================
        # 🏆 塔一：流量块风险 (Traffic Block Tower)
        # ==========================================
        block_stats = df.groupby(['date', '$url', 'hour_time'], observed=True).agg(
            block_pv=('distinct_id', 'count'),#统计每个网址每小时的访问次数，得到 block_pv
            block_uv=('distinct_id', 'nunique'),#统计每个网址每小时的独立用户数，得到 block_uv
            block_1pv_uv=('is_1pv_user', 'sum'),#统计每个网址每小时的单次访问用户数，得到 block_1pv_uv
            block_direct_pv=('is_direct', 'sum')#统计每个网址每小时的直接访问次数，得到 block_direct_pv
        ).reset_index()

        block_stats = block_stats.sort_values(['date', '$url', 'hour_time'])#按照日期、网址和小时排序，确保后续计算增长率时的顺序正确
        block_stats['prev_pv'] = block_stats.groupby(['date', '$url'])['block_pv'].shift(1).fillna(0)#统计前一个小时的访问次数，作为计算增长率的分母，fillna(0)处理第一条记录的缺失值
        block_stats['growth_rate'] = (block_stats['block_pv'] / (block_stats['prev_pv'] + 1e-6)).clip(upper=50)#计算增长率，clip 限制上限为50
        block_stats['uv_pv_ratio'] = block_stats['block_uv'] / (block_stats['block_pv'] + 1e-6)#计算独立用户数与访问次数的比率，这个比率过高可能意味着异常流量,范围在0-1之间，不用归一化
        block_stats['target_1pv_ratio'] = block_stats['block_1pv_uv'] / (block_stats['block_uv'] + 1e-6)#计算单次访问用户数与独立用户数的比率，这个比率过高可能意味着大量新用户涌入,范围在0-1之间，不用归一化
        block_stats['target_direct_ratio'] = block_stats['block_direct_pv'] / (block_stats['block_pv'] + 1e-6)#计算直接访问次数与总访问次数的比率，这个比率过高可能意味着大量用户是通过直接访问进入,范围在0-1之间，不用归一化

        df = df.merge(
            block_stats[['date', '$url', 'hour_time', 'block_pv', 'growth_rate', 'uv_pv_ratio', 'target_1pv_ratio', 'target_direct_ratio']],
            on=['date', '$url', 'hour_time'], how='left'
        )#合并流量块特征回原始数据，方便后续用户级别的风险计算

        df['is_suspicious_block'] = (
            (df['growth_rate'] > 3.0) & 
            (df['uv_pv_ratio'] > 0.9) & 
            (df['target_1pv_ratio'] > 0.2)
        ).astype(int)#定义一个强规则：如果某个网址某小时的访问增长率超过3倍，且独立用户占比超过90%，且单次访问用户占比超过20%，就认为这个流量块是可疑的，标记为1，否则为0

        # ==========================================
        # 🏰 塔二：个体时间与环境风险 (User Tower)
        # ==========================================
        exact_ip_uv = df.groupby(['date', '$ip'], observed=True)['distinct_id'].nunique().reset_index(name='exact_ip_uv')#统计每个 IP 每天的独立用户数，得到一个包含 date、$ip 和 exact_ip_uv 的 DataFrame，exact_ip_uv 过高可能意味着多个用户共享同一个 IP，存在风险
        c_seg_counts = df.groupby(['date', 'ip_c_segment'], observed=True)['distinct_id'].nunique().reset_index(name='cnt')#统计每个 C 段每天的独立用户数，得到一个包含 date、ip_c_segment 和 cnt 的 DataFrame，cnt 过高可能意味着大量用户来自同一个 C 段，存在风险
        c_seg_counts['c_zscore'] = (c_seg_counts['cnt'] - self.ref_c_mean_) / self.ref_c_std_#计算每个 C 段独立用户数的 z-score，衡量其相对于基准的异常程度，z-score 越高可能意味着风险越大
        
        df = df.merge(exact_ip_uv, on=['date', '$ip'], how='left')#合并 exact_ip_uv 特征回原始数据，方便后续用户级别的风险计算
        df = df.merge(c_seg_counts[['date', 'ip_c_segment', 'c_zscore']], on=['date', 'ip_c_segment'], how='left')#合并 c_zscore 特征回原始数据，方便后续用户级别的风险计算

        uh = df.groupby(gcols + ['hour_time'], observed=True).size().reset_index(name='pv')#统计每个用户每小时的访问次数，得到一个包含 date、distinct_id、hour_time 和 pv 的 DataFrame
        uh = uh.merge(user_pv_map, on=gcols, how='left')#合并用户总访问次数回用户小时级别的 DataFrame，方便计算用户在每个小时的访问占比
        uh['user_share'] = uh['pv'] / (uh['user_total_pv'] + 1e-12)#计算用户在每个小时的访问占比，避免除零错误
        uh['std_ratio'] = uh['hour_time'].map(self.ref_hour_dist_).fillna(1e-6)#填充标准比率，避免除零错误
        
        uh['kl_part'] = uh['user_share'] * np.log((uh['user_share'] + 1e-8) / (uh['std_ratio'] + 1e-8))#这个人的作息，到底在多大程度上偏离了正常人类
        uh['w_night_share'] = uh['user_share'] * uh['hour_time'].map(self.night_weights)#这个人的作息中，有多少是落在我们定义的凌晨高危时段的，权重越高代表越接近凌晨高危时段
        uh['w_evening_share'] = uh['user_share'] * uh['hour_time'].map(self.evening_weights)#这个人的作息中，有多少是落在我们定义的傍晚高危时段的，权重越高代表越接近傍晚高危时段

        hour_feat = uh.groupby(gcols, observed=True).agg(
            time_kl_dist=('kl_part', 'sum'),#计算用户作息分布与正常人类基准分布的 KL 散度，越大代表作息越异常
            user_night_density=('w_night_share', 'sum'),#计算用户在凌晨高危时段的加权访问占比，越大代表用户作息越集中在凌晨高危时段
            user_evening_density=('w_evening_share', 'sum')#计算用户在傍晚高危时段的加权访问占比，越大代表用户作息越集中在傍晚高危时段
        )
        hour_feat['night_relative_risk'] = (np.log(hour_feat['user_night_density'] + 1e-7) - np.log(self.ref_night_mass_ + 1e-7)).clip(lower=0)#看用户比大盘平均水平高出了多少个数量级
        hour_feat['evening_relative_risk'] = (np.log(hour_feat['user_evening_density'] + 1e-7) - np.log(self.ref_evening_mass_ + 1e-7)).clip(lower=0)#看用户比大盘平均水平高出了多少个数量级

        # ==========================================
        # 👯‍♂️ 塔三：克隆人攻击风险 (Clone Attack Tower) 
        # ==========================================
        user_signature = df.groupby(gcols, observed=True).agg(
            hour=('hour_time', 'mean'),#用户的平均访问小时，虽然不一定有实际意义，但可以作为用户作息的一个粗略特征
            url_nunique=('$url', 'nunique'),#用户访问过的不同网址数量，过少可能意味着用户行为单一，存在风险
            is_direct=('is_direct', 'mean'),#用户访问中直接访问的比例，过高可能意味着用户行为异常，存在风险
            ip_c=('ip_c_segment', 'first')#用户的 C 段 IP，虽然不一定有实际意义，但可以作为用户环境的一个粗略特征
        ).reset_index()

        # 统计相同类型用户
        user_signature['hour_bin'] = user_signature['hour'].round()#把用户的平均访问小时四舍五入到整数，作为一个粗略的时间特征，方便后续分组统计
        user_signature['cluster_size'] = user_signature.groupby(
            ['date', 'hour_bin', 'url_nunique', 'is_direct'], observed=True
        )['distinct_id'].transform('count')#计算每个用户群体的大小，即具有相同特征的用户数量

        # 转为按 date 和 distinct_id 为索引的形式，方便下一步无缝 join
        cluster_feat = user_signature.set_index(gcols)[['cluster_size']]
        # ==========================================
        # ⚔️ 终极融合与群体收网
        # ==========================================
        features = df.groupby(gcols, observed=True).agg(
            total_pv=('user_total_pv', 'max'),#统计每个用户最大 pv
            is_direct_ratio=('is_direct', 'mean'),#统计每个用户的直接访问比例，越高可能意味着用户行为越异常，存在风险
            max_c_zscore=('c_zscore', 'max'),#统计每个用户所在 C 段独立用户数 z-score 的最大值，越高可能意味着用户所在的 C 段越异常，存在风险
            max_exact_ip_uv=('exact_ip_uv', 'max'),#统计每个用户ip段独立用户是的最大值，越高可能意味着用户所在的 IP 越异常，存在风险
            max_block_pv=('block_pv', 'max'),#统计每个用户访问过的 URL 在对应小时的最大访问次数，越高可能意味着用户访问了一个非常热门的 URL，存在风险   
            avg_block_pv=('block_pv', 'mean'),#统计每个用户访问过的 URL 在对应小时的平均访问次数，越高可能意味着用户访问了一个非常热门的 URL，存在风险          
            max_growth_rate=('growth_rate', 'max'), #统计每个用户访问过的 URL 在对应小时的最大增长率，越高可能意味着用户访问了一个正在被攻击的 URL，存在风险 
            max_uv_pv_ratio=('uv_pv_ratio', 'max'), #统计每个用户访问过的 URL 在对应小时的最大独立用户数与访问次数的比率，越高可能意味着用户访问了一个独立用户占比异常的 URL，存在风险      
            max_target_1pv_ratio=('target_1pv_ratio', 'max'), #统计每个用户访问过的 URL 在对应小时的只访问一次的独立用户数与独立用户数的比率，越高可能意味着用户访问了一个单次访问用户占比异常的 URL，存在风险
            max_target_direct_ratio=('target_direct_ratio', 'max'), #统计每个用户访问过的 URL 在对应小时的直接访问次数与总访问次数的比率，越高可能意味着用户访问了一个直接访问占比异常的 URL，存在风险
            attack_block_ratio=('is_suspicious_block', 'mean')#统计用户访问的URL中可疑流量块的比例，越高可能意味着用户更倾向于访问可疑流量块，存在风险
            
        ).join(hour_feat).join(cluster_feat) #把之前计算的时间特征和克隆攻击特征合并到用户级别的特征表中，方便后续风险计算
        
        features['max_c_zscore'] = features['max_c_zscore'].clip(lower=0)#保持底层特征的干净

        # 基础双塔风险
        features['time_risk'] = features['time_kl_dist'] * np.log1p(features['total_pv'])#用户作息异常程度乘以访问次数的对数，访问次数越多，风险越大
        features['env_risk'] = np.log1p(features['max_exact_ip_uv']) * 0.5 + np.log1p(features['max_c_zscore']) * 0.5#用户环境异常程度乘以访问次数的对数，访问次数越多，风险越大
        features['final_time_risk'] = features[['time_risk', 'night_relative_risk', 'evening_relative_risk', 'env_risk']].max(axis=1)
      
        # 🎯 强规则 1: 流量块攻击狙击
        is_block_attack = (
            (features['attack_block_ratio'] > 0.7) & 
            (features['total_pv'] <= 3) & 
            (features['is_direct_ratio'] > 0.7)
        )
        features.loc[is_block_attack, 'final_time_risk'] += 25.0 
        
        # 🎯 强规则 2: 克隆人攻击狙击 (截图逻辑)
        is_clone_attack = (
            (features['cluster_size'] > 50) & 
            (features['total_pv'] <= 2)
        )
        features.loc[is_clone_attack, 'final_time_risk'] += 20.0

        # 补充凌晨高危区的基础拦截
        is_night_attack = (features['total_pv'] == 1) & (features['is_direct_ratio'] > 0.6) & (features['user_night_density'] > 0.4)
        features.loc[is_night_attack, 'final_time_risk'] += 15.0
        #补充晚上高危区的基础拦截
        is_evening_stealth_attack = (
        (features['user_evening_density'] > 0.5) &  # 傍晚密度极高
        (features['total_pv'] == 1) &               # 依然是单点攻击
        (features['is_direct_ratio'] == 1.0) &      # 必须是100%纯净的空降
        (features['cluster_size'] > 20)             # 关键：必须伴随一定的克隆特征
        )
        features.loc[is_evening_stealth_attack, 'final_time_risk'] += 10.0

        output_list = list(self.feature_cols) 
        
        # 第二步：强行把“业务分”加进输出名单。
        # 这样 reindex 就不会删掉它，而是把它一起吐给 Streamlit
        if 'final_time_risk' not in output_list:
            output_list.append('final_time_risk')
            
        # 第三步：现在 output_list 是 13 个名字了，安检通过！
        return features.reindex(columns=output_list, fill_value=0.0)


class UnifiedUserBehaviorCleanerWithResolutionShareZ(UnifiedUserBehaviorCleaner):
    """Original behavior features plus a positive daily resolution-share Z-score.

    The reference distribution is learned only during ``fit``.  At prediction
    time each user's main canonical resolution is mapped to that day's share,
    and only an upward deviation from the fitted normal baseline is retained.
    Invalid or missing resolutions are neutral (zero) and never remove a user.
    """

    def __init__(
        self,
        feature_cols,
        min_baseline_days=7,
        min_baseline_user_days=100,
        z_clip=10.0,
    ):
        super().__init__(feature_cols=feature_cols)
        self.min_baseline_days = min_baseline_days
        self.min_baseline_user_days = min_baseline_user_days
        self.z_clip = z_clip

    def _bucket_user_days(self, X):
        user_days = _resolution_main_user_days(X)
        valid = user_days["main_resolution_canonical"].notna()
        user_days["resolution_bucket"] = pd.Series(
            pd.NA, index=user_days.index, dtype="string"
        )
        is_common = user_days["main_resolution_canonical"].isin(self.common_resolutions_)
        user_days.loc[valid & is_common, "resolution_bucket"] = user_days.loc[
            valid & is_common, "main_resolution_canonical"
        ].astype("string")
        user_days.loc[valid & ~is_common, "resolution_bucket"] = OTHER_RESOLUTION
        return user_days

    def _daily_share_grid(self, user_days, dates):
        dates = list(pd.to_datetime(pd.Index(dates), errors="raise").normalize().unique())
        categories = list(self.common_resolutions_) + [OTHER_RESOLUTION]
        valid = user_days.dropna(subset=["resolution_bucket"])
        counts = valid.groupby(
            ["date", "resolution_bucket"], observed=True
        ).size().rename("resolution_user_days")
        grid = pd.MultiIndex.from_product(
            [dates, categories], names=["date", "resolution_bucket"]
        )
        daily = counts.reindex(grid, fill_value=0).reset_index()
        totals = valid.groupby("date", observed=True).size().rename("valid_user_days")
        daily = daily.merge(totals, on="date", how="left")
        daily["valid_user_days"] = daily["valid_user_days"].fillna(0).astype(int)
        daily["daily_share"] = np.where(
            daily["valid_user_days"].gt(0),
            daily["resolution_user_days"] / daily["valid_user_days"],
            0.0,
        )
        return daily

    def _summarize_daily_shares(self, daily):
        rows = []
        for resolution, part in daily.groupby("resolution_bucket", observed=True):
            values = part["daily_share"].astype(float)
            center = float(values.median())
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            mad = float((values - center).abs().median())
            robust_std = 1.4826 * mad
            if np.isfinite(robust_std) and robust_std > self.share_scale_floor_:
                scale = robust_std
                scale_source = "MAD"
            elif np.isfinite(std) and std > self.share_scale_floor_:
                scale = std
                scale_source = "STD_FALLBACK"
            else:
                scale = self.share_scale_floor_
                scale_source = "FLOOR"
            rows.append({
                "resolution": str(resolution),
                "daily_share_mean": mean,
                "daily_share_median": center,
                "daily_share_std": std,
                "daily_share_mad": mad,
                "daily_share_scale": float(scale),
                "scale_source": scale_source,
                "baseline_days": int(part["date"].nunique()),
                "total_user_days": int(part["resolution_user_days"].sum()),
                "days_present": int(
                    part.loc[part["resolution_user_days"].gt(0), "date"].nunique()
                ),
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _score_daily_share(current, baseline, z_clip):
        scored = current.merge(
            baseline[["resolution", "daily_share_median", "daily_share_scale"]],
            left_on="resolution_bucket",
            right_on="resolution",
            how="left",
            validate="many_to_one",
        )
        scored[RESOLUTION_Z_FEATURE] = (
            (scored["daily_share"] - scored["daily_share_median"])
            / scored["daily_share_scale"]
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(
            lower=0.0, upper=z_clip
        )
        return scored[["date", "resolution_bucket", "daily_share", RESOLUTION_Z_FEATURE]]

    def fit(self, X, y=None):
        super().fit(X, y)
        raw_user_days = _resolution_main_user_days(X)
        valid = raw_user_days.dropna(subset=["main_resolution_canonical"])
        if valid.empty:
            raise ValueError("12+1 训练正常样本没有有效主分辨率。")

        resolution_stats = valid.groupby(
            "main_resolution_canonical", observed=True
        ).agg(
            total_user_days=("distinct_id", "size"),
            days_present=("date", "nunique"),
        ).reset_index()
        common = resolution_stats.loc[
            resolution_stats["total_user_days"].ge(self.min_baseline_user_days)
            & resolution_stats["days_present"].ge(self.min_baseline_days),
            "main_resolution_canonical",
        ]
        self.common_resolutions_ = tuple(sorted(common.astype(str).tolist()))
        self.training_dates_ = tuple(
            sorted(pd.to_datetime(valid["date"]).dt.normalize().unique())
        )
        if len(self.training_dates_) < 3:
            raise ValueError("训练期有效日期少于3天，无法构造稳定的分辨率日占比基准。")

        bucketed = self._bucket_user_days(X)
        daily_valid_totals = bucketed.dropna(subset=["resolution_bucket"]).groupby(
            "date", observed=True
        ).size()
        typical_daily_users = float(daily_valid_totals.median())
        self.share_scale_floor_ = max(0.5 / max(typical_daily_users, 1.0), 1e-6)
        self.training_daily_share_ = self._daily_share_grid(bucketed, self.training_dates_)
        self.resolution_daily_share_baseline_ = self._summarize_daily_shares(
            self.training_daily_share_
        )
        self.resolution_vocabulary_stats_ = resolution_stats
        return self

    def _resolution_z_feature(self, X, leave_one_day_out):
        user_days = self._bucket_user_days(X)
        if user_days.empty:
            empty_index = pd.MultiIndex.from_arrays(
                [[], []], names=["date", "distinct_id"]
            )
            return pd.DataFrame(columns=[RESOLUTION_Z_FEATURE], index=empty_index)

        current_dates = sorted(pd.to_datetime(user_days["date"]).dt.normalize().unique())
        current_daily = self._daily_share_grid(user_days, current_dates)
        if leave_one_day_out:
            score_parts = []
            for day in current_dates:
                reference = self.training_daily_share_.loc[
                    self.training_daily_share_["date"].ne(day)
                ]
                loo_baseline = self._summarize_daily_shares(reference)
                score_parts.append(self._score_daily_share(
                    current_daily.loc[current_daily["date"].eq(day)],
                    loo_baseline,
                    self.z_clip,
                ))
            daily_scores = pd.concat(score_parts, ignore_index=True)
        else:
            daily_scores = self._score_daily_share(
                current_daily, self.resolution_daily_share_baseline_, self.z_clip
            )

        mapped = user_days.merge(
            daily_scores[["date", "resolution_bucket", RESOLUTION_Z_FEATURE]],
            on=["date", "resolution_bucket"],
            how="left",
            validate="many_to_one",
        )
        mapped[RESOLUTION_Z_FEATURE] = mapped[RESOLUTION_Z_FEATURE].fillna(0.0)
        return mapped.set_index(["date", "distinct_id"])[[RESOLUTION_Z_FEATURE]]

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        base = UnifiedUserBehaviorCleaner.transform(self, X)
        z_feature = self._resolution_z_feature(X, leave_one_day_out=True)
        result = base.join(z_feature, how="left")
        result[RESOLUTION_Z_FEATURE] = result[RESOLUTION_Z_FEATURE].fillna(0.0)
        return result.reindex(columns=list(self.feature_cols) + [RESOLUTION_Z_FEATURE])

    def transform(self, X):
        base = UnifiedUserBehaviorCleaner.transform(self, X)
        z_feature = self._resolution_z_feature(X, leave_one_day_out=False)
        result = base.join(z_feature, how="left")
        result[RESOLUTION_Z_FEATURE] = result[RESOLUTION_Z_FEATURE].fillna(0.0)
        return result.reindex(columns=list(self.feature_cols) + [RESOLUTION_Z_FEATURE])
