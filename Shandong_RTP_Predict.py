import streamlit as st
import pandas as pd
import numpy as np
import os
import re
from datetime import datetime, timedelta
import holidays
import warnings
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import io

warnings.filterwarnings("ignore")

# ==================== 页面配置 ====================
st.set_page_config(page_title="电价预测系统", layout="wide")
st.title("⚡ 电力市场实时电价预测")
st.markdown("上传历史电价、出力数据及气象数据，训练模型并预测未来电价。")

# ==================== 工具函数 ====================
def parse_datetime_with_24hour(date_str, time_str):
    """处理 24:00 的特殊情况，将其转为次日 00:00"""
    time_str = time_str.strip()
    if time_str == "24:00":
        dt = datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)
        return dt.replace(hour=0, minute=0)
    else:
        return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")

def build_continuous_price_series(price_dict):
    """将每日24点电价插值为连续96点序列"""
    all_times = []
    all_prices = []
    dates_sorted = sorted(price_dict.keys())
    for i, date_str in enumerate(dates_sorted):
        d = datetime.strptime(date_str, "%Y-%m-%d")
        prices = price_dict[date_str]
        if i == 0:
            prev_24 = prices[0]
        else:
            prev_24 = price_dict[dates_sorted[i-1]][-1]
        extended = [prev_24] + prices
        s = pd.Series(extended, index=[d + timedelta(hours=h) for h in range(0, 25)])
        s_15min = s.resample('15min').interpolate(method='linear')
        s_day = s_15min[d:d + timedelta(days=1) - timedelta(minutes=15)]
        all_times.extend(s_day.index)
        all_prices.extend(s_day.values)
    return pd.DataFrame({'实时电价': all_prices}, index=all_times)

def process_weather_data(weather_df, city_weather_map):
    """气象数据24点->96点线性插值"""
    weather_df['record_time'] = pd.to_datetime(weather_df['record_time'])
    result_dfs = []
    for (city, dim), col_name in city_weather_map.items():
        sub = weather_df[(weather_df['city_name'] == city) &
                         (weather_df['weather_dimension'] == dim)].copy()
        if sub.empty:
            continue
        sub.sort_values('record_time', inplace=True)
        sub.set_index('record_time', inplace=True)
        sub_15min = sub['value'].resample('15min').interpolate(method='linear')
        sub_15min = sub_15min.to_frame(name=col_name)
        result_dfs.append(sub_15min)
    if result_dfs:
        weather_15min = pd.concat(result_dfs, axis=1, join='outer')
    else:
        weather_15min = pd.DataFrame()
    return weather_15min

def parse_time_to_slot(time_str):
    """将 'HH:MM' 字符串转换为 0~95 的时刻编号"""
    try:
        return int(time_str)
    except (ValueError, TypeError):
        parts = str(time_str).split(':')
        if len(parts) != 2:
            raise ValueError(f"无法解析的时刻格式: {time_str}")
        hour = int(parts[0])
        minute = int(parts[1])
        return hour * 4 + minute // 15

def extract_date_from_filename(filename):
    """
    从文件名中智能提取日期字符串（YYYY-MM-DD 或 YYYYMMDD）
    返回标准格式 'YYYY-MM-DD'，若提取失败返回 None
    """
    # 匹配 YYYY-MM-DD 格式
    match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
    if match:
        return match.group(1)
    # 匹配 YYYYMMDD 格式
    match = re.search(r'(\d{8})', filename)
    if match:
        date_str = match.group(1)
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    return None

# ==================== 特征名称英文映射（用于可视化）====================
FEATURE_NAME_EN_MAP = {
    '全网负荷': 'Total Load',
    '直调负荷': 'Direct Dispatch Load',
    '联络线受电负荷': 'Tie-line Import',
    '风电': 'Wind Power',
    '光伏': 'Solar Power',
    '地方电厂总加': 'Local Generation',
    '非市场化核电总加': 'Non-market Nuclear',
    '供需比': 'Supply-Demand Ratio',
    '统调负荷占比': 'Dispatch Load Ratio',
    '净负荷': 'Net Load',
    '济南气温': 'Jinan Temp',
    '青岛气温': 'Qingdao Temp',
    '东营风速': 'Dongying Wind Speed',
    '烟台风速': 'Yantai Wind Speed',
    '威海风速': 'Weihai Wind Speed',
    '潍坊风速': 'Weifang Wind Speed',
    '青岛风速': 'Qingdao Wind Speed',
    '是否周末': 'Is Weekend',
    '是否节假日': 'Is Holiday',
    '时刻': 'Time Slot'
}

# ==================== 侧边栏：文件上传与参数设置 ====================
st.sidebar.header("📁 数据上传")

price_files = st.sidebar.file_uploader(
    "历史实时电价文件 (山东_实时出清结果_YYYY-MM-DD.xls)",
    type=["xls"],
    accept_multiple_files=True,
    key="price"
)

load_hist_files = st.sidebar.file_uploader(
    "历史出力文件 (山东_出力-实际【总】_YYYY-MM-DD.xlsx)",
    type=["xlsx"],
    accept_multiple_files=True,
    key="load_hist"
)

load_future_files = st.sidebar.file_uploader(
    "未来出力文件 (预测值，文件名无限制，程序会自动提取日期)",
    type=["xlsx"],
    accept_multiple_files=True,
    key="load_future"
)

weather_file = st.sidebar.file_uploader(
    "气象数据 (SD City.xlsx)",
    type=["xlsx"],
    key="weather"
)

st.sidebar.header("⚙️ 模型参数")

# 验证集天数允许为0
val_days = st.sidebar.number_input(
    "验证集天数（0 = 不使用验证集，全量训练）",
    min_value=0, max_value=30, value=0, step=1
)

# 早停轮数仅在验证集天数>0时生效，但仍显示控件（用户自行理解）
early_stopping_rounds = st.sidebar.number_input(
    "早停轮数（仅当验证集天数>0时生效）",
    min_value=5, max_value=50, value=10, step=5
)

auto_tune = st.sidebar.checkbox("自动调参 (耗时较长)", value=True)

with st.sidebar.expander("手动设置 XGBoost 参数"):
    n_estimators = st.number_input("n_estimators", 50, 1000, 200, step=50)
    max_depth = st.number_input("max_depth", 3, 10, 4, step=1)
    learning_rate = st.number_input("learning_rate", 0.01, 0.3, 0.03, step=0.01)
    subsample = st.slider("subsample", 0.5, 1.0, 0.8, step=0.1)
    colsample_bytree = st.slider("colsample_bytree", 0.5, 1.0, 0.7, step=0.1)
    gamma = st.number_input("gamma", 0.0, 1.0, 0.1, step=0.1)
    reg_alpha = st.number_input("reg_alpha", 0.0, 10.0, 0.1, step=0.1)
    reg_lambda = st.number_input("reg_lambda", 0.5, 10.0, 1.5, step=0.5)

if auto_tune:
    n_iter = st.sidebar.number_input("随机搜索迭代次数", 10, 100, 20, step=5)

# 特征列配置（中文原始名称）
FEATURE_COLS_CN = [
    '全网负荷', '直调负荷', '联络线受电负荷', '风电', '光伏',
    '地方电厂总加', '非市场化核电总加','净负荷',
    '济南气温', '青岛气温', '东营风速', '烟台风速', '威海风速', '潍坊风速', '青岛风速',
    '是否周末', '是否节假日', '时刻'
]

CITY_WEATHER_MAP = {
    ("济南", "气温"): "济南气温",
    ("青岛", "气温"): "青岛气温",
    ("东营", "风速"): "东营风速",
    ("烟台", "风速"): "烟台风速",
    ("威海", "风速"): "威海风速",
    ("潍坊", "风速"): "潍坊风速",
    ("青岛", "风速"): "青岛风速",
}

cn_holidays = holidays.China(years=[2024, 2025, 2026])

# ==================== 主处理按钮 ====================
if st.sidebar.button("🚀 开始训练与预测"):
    if not price_files or not load_hist_files or not load_future_files or not weather_file:
        st.error("请上传所有必需的文件：历史电价、历史出力、未来出力、气象数据。")
    else:
        with st.spinner("正在处理数据，请稍候..."):
            try:
                # -------------------- 1. 读取历史电价数据 --------------------
                price_data = {}
                for f in price_files:
                    match = re.match(r"山东_实时出清结果_(\d{4}-\d{2}-\d{2})\.xls", f.name)
                    if not match:
                        st.warning(f"文件名 {f.name} 不符合电价文件格式，已跳过。")
                        continue
                    date_str = match.group(1)
                    df = pd.read_excel(f)
                    time_col = df.columns[0]
                    price_col = df.columns[1]
                    df['time_str'] = df[time_col].astype(str).str.strip()
                    df['hour'] = df['time_str'].apply(lambda x: int(x.split(':')[0]))
                    df.sort_values('hour', inplace=True)
                    prices = df[price_col].tolist()
                    if len(prices) != 24:
                        st.warning(f"{date_str} 的电价数据不是24个点，跳过。")
                        continue
                    price_data[date_str] = prices

                if not price_data:
                    st.error("没有有效的电价数据。")
                    st.stop()

                price_96 = build_continuous_price_series(price_data)

                # -------------------- 2. 读取出力数据 --------------------
                def read_load_files(files, check_filename=True):
                    """
                    读取出力文件（历史或未来）
                    check_filename=True: 要求文件名符合 '山东_出力-实际【总】_YYYY-MM-DD.xlsx' 格式，并从文件名提取日期
                    check_filename=False: 不校验文件名，智能从文件名提取日期或从时刻列解析完整时间
                    """
                    load_dfs = []
                    for f in files:
                        date_str = None
                        if check_filename:
                            match = re.match(r"山东_出力-实际【总】_(\d{4}-\d{2}-\d{2})\.xlsx", f.name)
                            if not match:
                                st.warning(f"文件名 {f.name} 不符合出力文件格式，已跳过。")
                                continue
                            date_str = match.group(1)

                        df_load = pd.read_excel(f, sheet_name="负荷信息")
                        col_mapping = {}
                        for col in df_load.columns:
                            col_lower = col.strip()
                            if '时刻' in col_lower:
                                col_mapping['时刻'] = col
                            elif '全网负荷' in col_lower:
                                col_mapping['全网负荷'] = col
                            elif '直调负荷' in col_lower:
                                col_mapping['直调负荷'] = col
                            elif '联络线' in col_lower:
                                col_mapping['联络线受电负荷'] = col
                            elif '风电' in col_lower:
                                col_mapping['风电'] = col
                            elif '光伏' in col_lower:
                                col_mapping['光伏'] = col
                            elif '地方电厂' in col_lower:
                                col_mapping['地方电厂总加'] = col
                            elif '非市场化核电' in col_lower:
                                col_mapping['非市场化核电总加'] = col
                        required = ['时刻', '全网负荷', '直调负荷', '联络线受电负荷', '风电', '光伏',
                                    '地方电厂总加', '非市场化核电总加']
                        if not all(k in col_mapping for k in required):
                            st.warning(f"文件 {f.name} 负荷信息列名不匹配，跳过。")
                            continue
                        df_clean = df_load[[col_mapping[k] for k in required]].copy()
                        df_clean.columns = required

                        # 处理未来出力文件：不要求文件名格式
                        if not check_filename:
                            # 1. 优先尝试从文件名智能提取日期
                            extracted_date = extract_date_from_filename(f.name)
                            if extracted_date:
                                date_str = extracted_date
                                df_clean['datetime'] = df_clean['时刻'].apply(
                                    lambda t: parse_datetime_with_24hour(date_str, str(t))
                                )
                            else:
                                # 2. 若文件名无日期，检查时刻列是否包含完整日期时间
                                sample_time = str(df_clean['时刻'].iloc[0])
                                if len(sample_time) > 8 and ('-' in sample_time or '/' in sample_time):
                                    df_clean['datetime'] = pd.to_datetime(df_clean['时刻'])
                                else:
                                    st.error(
                                        f"未来出力文件 {f.name} 无法确定日期。"
                                        "请确保文件名包含日期（如 2026-04-17）或时刻列包含完整日期时间。"
                                    )
                                    continue
                        else:
                            # 历史出力：使用文件名中的日期
                            df_clean['datetime'] = df_clean['时刻'].apply(
                                lambda t: parse_datetime_with_24hour(date_str, t)
                            )

                        df_clean.set_index('datetime', inplace=True)
                        load_dfs.append(df_clean)

                    if not load_dfs:
                        return pd.DataFrame()
                    load_all = pd.concat(load_dfs).sort_index()
                    load_all['供需比'] = (load_all['直调负荷'] - load_all['联络线受电负荷'] - load_all['非市场化核电总加']) / load_all['全网负荷']
                    load_all['统调负荷占比'] = load_all['直调负荷'] / load_all['全网负荷']
                    load_all['净负荷'] = load_all['全网负荷'] - load_all['风电'] - load_all['光伏']
                    return load_all

                # 历史出力：必须校验文件名以获取日期
                hist_load = read_load_files(load_hist_files, check_filename=True)
                # 未来出力：不校验文件名，智能提取日期
                future_load = read_load_files(load_future_files, check_filename=False)

                if hist_load.empty:
                    st.error("历史出力数据为空。")
                    st.stop()
                if future_load.empty:
                    st.error("未来出力数据为空。")
                    st.stop()

                # -------------------- 3. 读取气象数据 --------------------
                weather_raw = pd.read_excel(weather_file)
                weather_96 = process_weather_data(weather_raw, CITY_WEATHER_MAP)

                # -------------------- 4. 合并历史数据 --------------------
                hist_merged = hist_load.join(price_96, how='inner')
                hist_merged = hist_merged.join(weather_96, how='left')
                hist_merged['日期'] = hist_merged.index.date
                hist_merged['时刻'] = hist_merged.index.strftime('%H:%M')

                def get_day_type(dt):
                    if dt.date() in cn_holidays:
                        return '节假日'
                    elif dt.weekday() >= 5:
                        return '周末'
                    else:
                        return '工作日'

                hist_merged['日期类型'] = hist_merged.index.map(get_day_type)
                hist_merged['是否工作日'] = (hist_merged['日期类型'] == '工作日').astype(int)
                hist_merged['是否周末'] = (hist_merged['日期类型'] == '周末').astype(int)
                hist_merged['是否节假日'] = (hist_merged['日期类型'] == '节假日').astype(int)

                hist_merged['时刻'] = hist_merged['时刻'].apply(parse_time_to_slot)
                hist_merged.dropna(subset=['实时电价'] + FEATURE_COLS_CN, inplace=True)

                if hist_merged.empty:
                    st.error("合并后历史数据为空，请检查文件日期是否匹配。")
                    st.stop()

                # -------------------- 5. 构建未来特征数据 --------------------
                future_merged = future_load.copy()
                future_merged = future_merged.join(weather_96, how='left')
                future_merged['日期'] = future_merged.index.date
                future_merged['时刻'] = future_merged.index.strftime('%H:%M')
                future_merged['日期类型'] = future_merged.index.map(get_day_type)
                future_merged['是否工作日'] = (future_merged['日期类型'] == '工作日').astype(int)
                future_merged['是否周末'] = (future_merged['日期类型'] == '周末').astype(int)
                future_merged['是否节假日'] = (future_merged['日期类型'] == '节假日').astype(int)
                future_merged['时刻'] = future_merged['时刻'].apply(parse_time_to_slot)
                future_merged.dropna(subset=FEATURE_COLS_CN, inplace=True)

                if future_merged.empty:
                    st.error("未来特征数据为空，请检查未来出力与气象数据的日期匹配。")
                    st.stop()

                # -------------------- 6. 时序划分训练集与验证集 --------------------
                all_dates = sorted(hist_merged['日期'].unique())
                use_validation = val_days > 0

                if use_validation:
                    if len(all_dates) <= val_days:
                        st.error(f"总历史天数 {len(all_dates)} 不足，无法划分验证集。请减少验证集天数或设为0。")
                        st.stop()

                    train_dates = all_dates[:-val_days]
                    val_dates_list = all_dates[-val_days:]

                    train_mask = hist_merged['日期'].isin(train_dates)
                    val_mask = hist_merged['日期'].isin(val_dates_list)

                    X_train = hist_merged.loc[train_mask, FEATURE_COLS_CN]
                    y_train = hist_merged.loc[train_mask, '实时电价']
                    X_val = hist_merged.loc[val_mask, FEATURE_COLS_CN]
                    y_val = hist_merged.loc[val_mask, '实时电价']

                    st.success(f"数据划分完成！训练集: {len(X_train)} 样本, 验证集: {len(X_val)} 样本, 预测样本: {len(future_merged)}")
                else:
                    # 不使用验证集，全部数据用于训练
                    X_train = hist_merged[FEATURE_COLS_CN]
                    y_train = hist_merged['实时电价']
                    X_val = None
                    y_val = None
                    st.success(f"数据准备完成！训练集: {len(X_train)} 样本, 预测样本: {len(future_merged)}（未使用验证集）")

                X_future = future_merged[FEATURE_COLS_CN]

                # -------------------- 7. 模型训练 --------------------
                if auto_tune:
                    from sklearn.model_selection import RandomizedSearchCV
                    st.info("正在进行自动调参（时序交叉验证），请稍候...")
                    param_dist = {
                        'n_estimators': [100, 200, 300, 400],
                        'max_depth': [3, 4, 5, 6],
                        'learning_rate': [0.01, 0.03, 0.05, 0.1],
                        'subsample': [0.6, 0.7, 0.8, 0.9],
                        'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
                        'gamma': [0, 0.1, 0.2],
                        'reg_alpha': [0, 0.1, 0.5],
                        'reg_lambda': [1, 1.5, 2]
                    }
                    xgb = XGBRegressor(random_state=42, n_jobs=-1)
                    tscv = TimeSeriesSplit(n_splits=3)
                    search = RandomizedSearchCV(
                        xgb, param_dist, n_iter=n_iter, cv=tscv,
                        scoring='neg_mean_absolute_error', random_state=42, n_jobs=-1
                    )
                    search.fit(X_train, y_train)
                    model = search.best_estimator_
                    best_params = search.best_params_
                    st.write("最佳参数：", best_params)
                    best_iteration = model.get_params().get('n_estimators', 200)
                else:
                    model = XGBRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        subsample=subsample,
                        colsample_bytree=colsample_bytree,
                        gamma=gamma,
                        reg_alpha=reg_alpha,
                        reg_lambda=reg_lambda,
                        random_state=42,
                        n_jobs=-1
                    )
                    best_iteration = n_estimators

                # 训练过程（根据是否使用验证集决定是否采用早停）
                early_stopping_used = False
                if use_validation:
                    try:
                        # 尝试带早停的训练
                        model.fit(
                            X_train, y_train,
                            eval_set=[(X_train, y_train), (X_val, y_val)],
                            early_stopping_rounds=early_stopping_rounds,
                            verbose=False
                        )
                        if hasattr(model, 'best_iteration') and model.best_iteration is not None:
                            best_iteration = model.best_iteration
                        early_stopping_used = True
                    except TypeError as e:
                        # 参数不支持，回退到普通训练（但仍使用验证集评估）
                        st.warning(f"当前 XGBoost 版本不支持 early_stopping_rounds 参数，将进行普通训练。错误详情: {e}")
                        model.fit(X_train, y_train)
                        best_iteration = model.get_params().get('n_estimators', n_estimators)
                else:
                    # 无验证集，直接训练
                    model.fit(X_train, y_train)

                if use_validation and early_stopping_used:
                    st.info(f"模型训练完成，早停于第 {best_iteration} 轮")
                else:
                    st.info(f"模型训练完成，使用迭代轮数: {best_iteration}")

                # -------------------- 8. 未来预测 --------------------
                y_future_pred = model.predict(X_future)
                # ==== 限制最低电价为 -80（仅针对未来预测）====
                y_future_pred = np.clip(y_future_pred, -80, None)
                future_merged['预测电价'] = y_future_pred

                # 训练集预测与评估（训练集预测值不限制）
                y_train_pred = model.predict(X_train)
                train_mae = mean_absolute_error(y_train, y_train_pred)
                train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                train_r2 = r2_score(y_train, y_train_pred)

                # 验证集评估（如果存在）
                val_mae = val_rmse = val_r2 = None
                y_val_pred = None
                if use_validation:
                    y_val_pred = model.predict(X_val)
                    # ==== 限制验证集预测值最低为 -80 ====
                    y_val_pred = np.clip(y_val_pred, -80, None)
                    val_mae = mean_absolute_error(y_val, y_val_pred)
                    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
                    val_r2 = r2_score(y_val, y_val_pred)

                # -------------------- 9. 可视化（升级为3x2布局） --------------------
                st.header("📊 模型拟合效果")

                # 显示训练集指标（无 MAPE）
                col1, col2, col3 = st.columns(3)
                col1.metric("训练集 MAE", f"{train_mae:.2f} 元/MWh")
                col2.metric("训练集 RMSE", f"{train_rmse:.2f} 元/MWh")
                col3.metric("训练集 R²", f"{train_r2:.4f}")

                # 如果有验证集，显示验证集指标（无 MAPE）
                if use_validation:
                    col5, col6, col7 = st.columns(3)
                    col5.metric("验证集 MAE", f"{val_mae:.2f} 元/MWh")
                    col6.metric("验证集 RMSE", f"{val_rmse:.2f} 元/MWh")
                    col7.metric("验证集 R²", f"{val_r2:.4f}")

                    if val_mae > train_mae * 1.3:
                        st.warning("⚠️ 验证集误差明显高于训练集，可能存在过拟合。建议增加正则化或减少模型复杂度。")
                    else:
                        st.success("✅ 验证集误差与训练集接近，模型泛化能力良好。")
                else:
                    st.info("ℹ️ 未使用验证集，无法评估泛化能力。")

                # 创建3行2列的子图布局
                fig = plt.figure(figsize=(16, 14))
                # 使用 GridSpec 灵活控制子图大小
                from matplotlib.gridspec import GridSpec
                gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

                # 子图1：训练+验证集真实vs预测散点图
                ax1 = fig.add_subplot(gs[0, 0])
                ax1.scatter(y_train, y_train_pred, alpha=0.3, label='Train', edgecolors='k', linewidth=0.5)
                if use_validation:
                    ax1.scatter(y_val, y_val_pred, alpha=0.5, label='Validation', edgecolors='k', linewidth=0.5)
                min_val = min(y_train.min(), y_train_pred.min())
                max_val = max(y_train.max(), y_train_pred.max())
                if use_validation:
                    min_val = min(min_val, y_val.min(), y_val_pred.min())
                    max_val = max(max_val, y_val.max(), y_val_pred.max())
                ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
                ax1.set_xlabel("Actual Price (Yuan/MWh)")
                ax1.set_ylabel("Predicted Price (Yuan/MWh)")
                title = "Actual vs Predicted (Train"
                if use_validation:
                    title += " + Validation)"
                else:
                    title += ")"
                ax1.set_title(title)
                ax1.legend()
                ax1.grid(True, linestyle='--', alpha=0.6)

                # 子图2：特征重要性
                ax2 = fig.add_subplot(gs[0, 1])
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                top_n = min(10, len(FEATURE_COLS_CN))
                top_features_cn = [FEATURE_COLS_CN[i] for i in indices[:top_n]]
                top_features_en = [FEATURE_NAME_EN_MAP.get(f, f) for f in top_features_cn]
                top_imp = importances[indices[:top_n]]
                ax2.barh(range(top_n), top_imp[::-1], align='center')
                ax2.set_yticks(range(top_n))
                ax2.set_yticklabels(top_features_en[::-1])
                ax2.set_xlabel("Importance")
                ax2.set_title("Feature Importance (Top 10)")
                ax2.grid(True, linestyle='--', alpha=0.6, axis='x')

                # 子图3：验证集时序对比（如果存在验证集）
                ax3 = fig.add_subplot(gs[1, 0])
                if use_validation:
                    val_times = hist_merged.loc[val_mask].index
                    ax3.plot(val_times, y_val.values, label='Actual', color='blue', alpha=0.7, linewidth=1)
                    ax3.plot(val_times, y_val_pred, label='Predicted', color='red', alpha=0.7, linewidth=1)
                    ax3.set_xlabel("Time")
                    ax3.set_ylabel("Price (Yuan/MWh)")
                    ax3.set_title(f"Validation Set Time Series (Last {val_days} day(s))")
                    ax3.legend()
                    ax3.grid(True, linestyle='--', alpha=0.6)
                    ax3.tick_params(axis='x', rotation=45)
                else:
                    ax3.text(0.5, 0.5, "No validation set selected.\nSet val_days > 0 to view.",
                             ha='center', va='center', transform=ax3.transAxes, fontsize=12)
                    ax3.set_title("Validation Time Series (Disabled)")

                # 子图4：未来预测曲线
                ax4 = fig.add_subplot(gs[1, 1])
                future_times = future_merged.index
                ax4.plot(future_times, y_future_pred, color='green', linewidth=1)
                ax4.set_xlabel("Time")
                ax4.set_ylabel("Predicted Price (Yuan/MWh)")
                ax4.set_title("Future Price Prediction")
                ax4.grid(True, linestyle='--', alpha=0.6)
                ax4.tick_params(axis='x', rotation=45)

                # 子图5：预测值分布直方图
                ax5 = fig.add_subplot(gs[2, 0])
                ax5.hist(y_future_pred, bins=30, color='skyblue', edgecolor='black')
                ax5.set_xlabel("Predicted Price (Yuan/MWh)")
                ax5.set_ylabel("Frequency")
                ax5.set_title("Distribution of Predicted Prices")
                ax5.grid(True, linestyle='--', alpha=0.6)

                # 子图6：验证集残差分布（如果存在验证集）
                ax6 = fig.add_subplot(gs[2, 1])
                if use_validation:
                    residuals = y_val - y_val_pred
                    ax6.hist(residuals, bins=30, color='salmon', edgecolor='black')
                    ax6.axvline(x=0, color='blue', linestyle='--', linewidth=2)
                    ax6.set_xlabel("Residual (Actual - Predicted)")
                    ax6.set_ylabel("Frequency")
                    ax6.set_title("Validation Residuals Distribution")
                    ax6.grid(True, linestyle='--', alpha=0.6)
                else:
                    ax6.text(0.5, 0.5, "No validation set selected.",
                             ha='center', va='center', transform=ax6.transAxes, fontsize=12)
                    ax6.set_title("Residuals Distribution (Disabled)")

                plt.tight_layout()
                st.pyplot(fig)

                # -------------------- 10. 准备下载数据 --------------------
                st.header("📥 下载预测结果")

                future_merged['分钟'] = future_merged.index.minute
                future_merged['预测电价_24点'] = np.where(
                    future_merged['分钟'] == 0,
                    future_merged['预测电价'],
                    np.nan
                )

                output_df = future_merged[[
                    '日期', '全网负荷', '直调负荷', '风电', '光伏',
                    '预测电价', '预测电价_24点'
                ]].copy()
                output_df['时刻'] = future_merged.index.strftime('%H:%M')
                output_df.reset_index(drop=True, inplace=True)

                cols_order = ['日期', '时刻', '全网负荷', '直调负荷', '风电', '光伏',
                              '预测电价', '预测电价_24点']
                output_df = output_df[cols_order]

                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    output_df.to_excel(writer, index=False, sheet_name='预测结果')
                buffer.seek(0)

                st.download_button(
                    label="⬇️ 下载预测结果 (Excel)",
                    data=buffer,
                    file_name="未来电价预测结果.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

                st.subheader("预测数据预览（前100行）")
                st.dataframe(output_df.head(100))

            except Exception as e:
                st.error(f"处理过程中出现错误：{str(e)}")
                st.exception(e)
else:
    st.info("👈 请在左侧上传文件并点击按钮开始分析。")
