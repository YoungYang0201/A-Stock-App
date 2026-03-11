import sqlite3
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_option_menu import option_menu

try:
    import akshare as ak
except ImportError:
    ak = None


# =========================
# 基础配置
# =========================
st.set_page_config(page_title="A股持仓辅助决策系统", layout="wide")
DB_PATH = "a_stock_advisor.db"

MODE_PROFILES = {
    "保守": {
        "name": "conservative",
        "rsi_overheat": 68,
        "rsi_add_max": 58,
        "ma_break_tolerance": 0.010,
        "buy_zone_width": 0.015,
        "reduce_zone_width": 0.020,
        "market_risk_penalty": 1.20,
        "allow_early_reduce": True,
        "allow_aggressive_add": False,
    },
    "平衡": {
        "name": "balanced",
        "rsi_overheat": 72,
        "rsi_add_max": 62,
        "ma_break_tolerance": 0.015,
        "buy_zone_width": 0.020,
        "reduce_zone_width": 0.025,
        "market_risk_penalty": 1.00,
        "allow_early_reduce": True,
        "allow_aggressive_add": True,
    },
    "进取": {
        "name": "aggressive",
        "rsi_overheat": 76,
        "rsi_add_max": 68,
        "ma_break_tolerance": 0.025,
        "buy_zone_width": 0.025,
        "reduce_zone_width": 0.030,
        "market_risk_penalty": 0.80,
        "allow_early_reduce": False,
        "allow_aggressive_add": True,
    },
}

SIGNALS = {
    "HOLD": "继续持有",
    "HOLD_NO_CHASE": "持有，不追高",
    "WATCH_BUY_DIP": "回踩观察低吸",
    "ADD_IN_BATCHES": "可分批加仓",
    "REDUCE_NEAR_RESISTANCE": "接近压力位，考虑减仓",
    "REDUCE_ON_WEAKNESS": "趋势转弱，谨慎减仓",
    "RISK_CONTROL": "跌破防守位，控制风险",
    "WAIT_AND_SEE": "暂不参与 / 继续观察",
}

INDEX_MAP = {
    "上证指数": "sh000001",
    "深证成指": "sz399001",
    "创业板指": "sz399006",
}


# =========================
# UI 美化
# =========================
st.markdown("""
<style>
.main {
    background: #f5f7fb;
}
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 1.5rem;
}
div[data-testid="stMetric"] {
    background: white;
    border-radius: 14px;
    padding: 14px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    border: 1px solid rgba(0,0,0,0.04);
}
.stDataFrame, .stTable {
    background: white;
    border-radius: 12px;
}
.custom-card {
    background: white;
    border-radius: 14px;
    padding: 16px 18px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    border: 1px solid rgba(0,0,0,0.04);
    margin-bottom: 14px;
}
.signal-good {
    color: #0f9d58;
    font-weight: 700;
}
.signal-mid {
    color: #d97706;
    font-weight: 700;
}
.signal-bad {
    color: #dc2626;
    font-weight: 700;
}
.small-note {
    color: #6b7280;
    font-size: 0.92rem;
}
</style>
""", unsafe_allow_html=True)


# =========================
# 数据库
# =========================
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS holdings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        stock_code TEXT NOT NULL,
        stock_name TEXT,
        cost_price REAL NOT NULL,
        quantity INTEGER NOT NULL,
        first_buy_date TEXT,
        position_pct REAL,
        allow_add_position INTEGER DEFAULT 1,
        custom_stop_loss REAL,
        notes TEXT,
        created_at TEXT,
        updated_at TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS watchlist (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        stock_code TEXT NOT NULL,
        stock_name TEXT,
        notes TEXT,
        created_at TEXT,
        updated_at TEXT
    )
    """)

    conn.commit()
    conn.close()


def add_holding(stock_code, stock_name, cost_price, quantity, first_buy_date,
                position_pct, allow_add_position, custom_stop_loss, notes):
    conn = get_conn()
    cur = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cur.execute("""
        INSERT INTO holdings
        (stock_code, stock_name, cost_price, quantity, first_buy_date, position_pct,
         allow_add_position, custom_stop_loss, notes, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        stock_code, stock_name, cost_price, quantity, first_buy_date, position_pct,
        1 if allow_add_position else 0, custom_stop_loss, notes, now, now
    ))
    conn.commit()
    conn.close()


def delete_holding(record_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM holdings WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()


def get_holdings():
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM holdings ORDER BY id DESC", conn)
    conn.close()
    return df


def add_watchlist(stock_code, stock_name, notes):
    conn = get_conn()
    cur = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cur.execute("""
        INSERT INTO watchlist
        (stock_code, stock_name, notes, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
    """, (stock_code, stock_name, notes, now, now))
    conn.commit()
    conn.close()


def delete_watchlist(record_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM watchlist WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()


def get_watchlist():
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM watchlist ORDER BY id DESC", conn)
    conn.close()
    return df


# =========================
# 工具函数
# =========================
def normalize_stock_code(code: str) -> str:
    code = str(code).strip()
    if code.startswith(("sh", "sz", "bj")):
        code = code[2:]
    return code.zfill(6)


def safe_float(v, default=np.nan):
    try:
        return float(v)
    except Exception:
        return default


def signal_css_class(signal_text: str) -> str:
    if ("控制风险" in signal_text) or ("减仓" in signal_text) or ("转弱" in signal_text):
        return "signal-bad"
    if ("不追高" in signal_text) or ("观察" in signal_text):
        return "signal-mid"
    return "signal-good"

@st.cache_data(ttl=3600)
def get_stock_list() -> pd.DataFrame:
    if ak is None:
        return pd.DataFrame(columns=["code", "name"])
    try:
        df = ak.stock_info_a_code_name()
        if df is None or df.empty:
            return pd.DataFrame(columns=["code", "name"])
        df["code"] = df["code"].astype(str).str.zfill(6)
        return df
    except Exception:
        return pd.DataFrame(columns=["code", "name"])


def get_stock_name_by_code(stock_code: str) -> str:
    code = normalize_stock_code(stock_code)
    stock_list = get_stock_list()
    if stock_list.empty:
        return code
    match = stock_list[stock_list["code"] == code]
    if not match.empty:
        return str(match.iloc[0]["name"])
    return code


@st.cache_data(ttl=1800)
def get_stock_history(stock_code: str, days: int = 400) -> pd.DataFrame:
    if ak is None:
        raise ImportError("未安装 akshare，请先执行：python3 -m pip install akshare")

    code = normalize_stock_code(stock_code)
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=days * 2)).strftime("%Y%m%d")

    df = ak.stock_zh_a_hist(
        symbol=code,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust="qfq"
    )
    if df is None or df.empty:
        raise ValueError(f"未获取到股票 {code} 的数据")

    rename_map = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "涨跌幅": "pct_chg",
    }
    df = df.rename(columns=rename_map)
    keep_cols = [c for c in rename_map.values() if c in df.columns]
    df = df[keep_cols].copy()

    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "close", "high", "low", "volume", "amount", "pct_chg"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
    return df.tail(days).copy()


# =========================
# 股票名称识别
# =========================
@st.cache_data(ttl=3600)
def get_stock_list() -> pd.DataFrame:
    if ak is None:
        return pd.DataFrame(columns=["code", "name"])
    try:
        df = ak.stock_info_a_code_name()
        if df is None or df.empty:
            return pd.DataFrame(columns=["code", "name"])
        df["code"] = df["code"].astype(str).str.zfill(6)
        return df
    except Exception:
        return pd.DataFrame(columns=["code", "name"])


def get_stock_name_by_code(stock_code: str) -> str:
    code = normalize_stock_code(stock_code)
    stock_list = get_stock_list()
    if stock_list.empty:
        return code
    match = stock_list[stock_list["code"] == code]
    if not match.empty:
        return str(match.iloc[0]["name"])
    return code


# =========================
# 数据抓取
# =========================
@st.cache_data(ttl=1800)
def get_stock_history(stock_code: str, days: int = 400) -> pd.DataFrame:
    if ak is None:
        raise ImportError("未安装 akshare，请先执行：python3 -m pip install akshare")

    code = normalize_stock_code(stock_code)
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=days * 2)).strftime("%Y%m%d")

    df = ak.stock_zh_a_hist(
        symbol=code,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust="qfq"
    )

    if df is None or df.empty:
        raise ValueError(f"未获取到股票 {code} 的数据")

    rename_map = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "涨跌幅": "pct_chg",
    }

    df = df.rename(columns=rename_map)
    keep_cols = [c for c in rename_map.values() if c in df.columns]
    df = df[keep_cols].copy()

    df["date"] = pd.to_datetime(df["date"])

    for col in ["open", "close", "high", "low", "volume", "amount", "pct_chg"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
    return df.tail(days).copy()

@st.cache_data(ttl=1800)
def get_index_history(index_symbol: str, days: int = 250) -> pd.DataFrame:
    """
    使用东方财富指数历史行情接口，兼容 sh000001 / sz399001 / sz399006
    """
    if ak is None:
        return pd.DataFrame()

    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=days * 2)).strftime("%Y%m%d")

    try:
        df = ak.stock_zh_index_daily_em(
            symbol=index_symbol,
            start_date=start_date,
            end_date=end_date
        )
    except Exception as e:
        st.warning(f"指数接口调用失败：{index_symbol} - {e}")
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()


    rename_map = {
        "date": "date",
        "open": "open",
        "close": "close",
        "high": "high",
        "low": "low",
        "volume": "volume",
        "amount": "amount",
        "pct_chg": "pct_chg",
        "涨跌幅": "pct_chg",
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
    }

    df = df.rename(columns=rename_map)

    required_cols = ["date", "open", "close", "high", "low"]
    for col in required_cols:
        if col not in df.columns:
            return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "close", "high", "low", "volume", "amount", "pct_chg"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
    return df.tail(days).copy()
 


# =========================
# 指标计算
# =========================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for win in [5, 10, 20, 60]:
        df[f"ma{win}"] = df["close"].rolling(win).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi14"] = 100 - (100 / (1 + rs))
    df["rsi14"] = df["rsi14"].fillna(50)

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd_dif"] = ema12 - ema26
    df["macd_dea"] = df["macd_dif"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = (df["macd_dif"] - df["macd_dea"]) * 2

    if "volume" in df.columns:
        df["vol_ma5"] = df["volume"].rolling(5).mean()
        df["vol_ma10"] = df["volume"].rolling(10).mean()
    else:
        df["vol_ma5"] = np.nan
        df["vol_ma10"] = np.nan

    return df


# =========================
# 大盘环境
# =========================
def analyze_market_environment() -> Dict:
    scores = []
    details = []

    for name, symbol in INDEX_MAP.items():
        df = get_index_history(symbol)
        if df.empty:
            continue
        df = add_indicators(df)
        last = df.iloc[-1]

        score = 50
        if last["close"] > last["ma20"]:
            score += 10
        if last["close"] > last["ma60"]:
            score += 10
        if last["ma20"] > last["ma60"]:
            score += 10
        if last["macd_dif"] > last["macd_dea"]:
            score += 10
        if 45 <= last["rsi14"] <= 72:
            score += 10

        scores.append(score)
        details.append((name, score))

    if not scores:
        return {
            "market_score": 50,
            "risk_level": "中",
            "short_trend": "中性",
            "medium_trend": "中性",
            "summary": "指数接口当前不可用，系统已按中性环境处理。",
            "details": [],
        }

    market_score = float(np.mean(scores))

    if market_score >= 75:
        risk_level = "低"
        short_trend = "偏强"
        medium_trend = "偏强"
        summary = "大盘环境整体偏强，对趋势持仓更友好。"
    elif market_score >= 60:
        risk_level = "中"
        short_trend = "中性偏强"
        medium_trend = "中性"
        summary = "大盘环境中性偏稳，可以做结构化跟踪。"
    elif market_score >= 45:
        risk_level = "中"
        short_trend = "中性"
        medium_trend = "中性偏弱"
        summary = "大盘环境一般，建议控制追高与激进加仓。"
    else:
        risk_level = "高"
        short_trend = "偏弱"
        medium_trend = "偏弱"
        summary = "大盘环境偏弱，优先控制风险，减少激进操作。"

    return {
        "market_score": round(market_score, 1),
        "risk_level": risk_level,
        "short_trend": short_trend,
        "medium_trend": medium_trend,
        "summary": summary,
        "details": details,
    }


# =========================
# 区间计算
# =========================
def calculate_price_zones(df: pd.DataFrame, profile: Dict, holding: Optional[Dict] = None) -> Dict:
    last = df.iloc[-1]
    recent_20_low = df["low"].tail(20).min()
    recent_20_high = df["high"].tail(20).max()
    ma20 = safe_float(last["ma20"], last["close"])
    ma60 = safe_float(last["ma60"], last["close"])

    buy_center = min(ma20, last["close"])
    reduce_center = recent_20_high

    buy_width = profile["buy_zone_width"]
    reduce_width = profile["reduce_zone_width"]

    defense_price = recent_20_low
    invalidation_price = min(recent_20_low, ma60)

    if holding and holding.get("custom_stop_loss") is not None and not pd.isna(holding.get("custom_stop_loss")):
        defense_price = float(holding["custom_stop_loss"])

    return {
        "buy_zone_low": round(buy_center * (1 - buy_width), 2),
        "buy_zone_high": round(buy_center * (1 + buy_width), 2),
        "reduce_zone_low": round(reduce_center * (1 - reduce_width), 2),
        "reduce_zone_high": round(reduce_center * (1 + reduce_width), 2),
        "defense_price": round(defense_price, 2),
        "invalidation_price": round(invalidation_price, 2),
    }


# =========================
# 建议引擎
# =========================
def classify_trend(last: pd.Series) -> Tuple[str, str]:
    short_trend = "中性"
    medium_trend = "中性"

    if last["close"] > last["ma20"] and last["ma20"] > last["ma60"]:
        short_trend = "偏强"
        medium_trend = "偏强"
    elif last["close"] < last["ma20"] and last["ma20"] < last["ma60"]:
        short_trend = "偏弱"
        medium_trend = "偏弱"
    elif last["close"] > last["ma20"]:
        short_trend = "中性偏强"
        medium_trend = "中性"
    elif last["close"] < last["ma20"]:
        short_trend = "中性偏弱"
        medium_trend = "中性"

    return short_trend, medium_trend


def evaluate_stock(df: pd.DataFrame, market_env: Dict, profile: Dict,
                   holding: Optional[Dict] = None) -> Dict:
    last = df.iloc[-1]

    short_trend, medium_trend = classify_trend(last)
    zones = calculate_price_zones(df, profile, holding)

    close = float(last["close"])
    ma20 = safe_float(last["ma20"], close)
    ma60 = safe_float(last["ma60"], close)
    rsi = safe_float(last["rsi14"], 50)
    dif = safe_float(last["macd_dif"], 0)
    dea = safe_float(last["macd_dea"], 0)
    pct_chg = safe_float(last["pct_chg"], 0)

    risk_score = 0
    if close < ma20:
        risk_score += 1
    if close < ma60:
        risk_score += 1
    if rsi > profile["rsi_overheat"]:
        risk_score += 1
    if dif < dea:
        risk_score += 1
    if market_env["risk_level"] == "高":
        risk_score += 1

    if risk_score >= 4:
        risk_level = "高"
    elif risk_score >= 2:
        risk_level = "中"
    else:
        risk_level = "低"

    near_resistance = close >= zones["reduce_zone_low"]
    near_buy_zone = zones["buy_zone_low"] <= close <= zones["buy_zone_high"]
    broken_defense = close < zones["defense_price"]
    broken_ma60 = close < ma60 * (1 - profile["ma_break_tolerance"])
    weakened = (close < ma20 and dif < dea) or pct_chg < -3
    overheat = rsi >= profile["rsi_overheat"]
    trend_good = close > ma20 and ma20 > ma60 and dif >= dea
    add_allowed = True if holding is None else bool(holding.get("allow_add_position", 1))
    market_weak = market_env["market_score"] < 50

    if broken_defense or broken_ma60:
        main_signal = SIGNALS["RISK_CONTROL"]
        secondary_signal = "已跌破防守位或中期结构转坏"
    elif weakened and medium_trend in ["中性", "偏弱"]:
        main_signal = SIGNALS["REDUCE_ON_WEAKNESS"]
        secondary_signal = "短中期趋势出现转弱迹象"
    elif near_resistance and (overheat or profile["allow_early_reduce"]):
        main_signal = SIGNALS["REDUCE_NEAR_RESISTANCE"]
        secondary_signal = "接近压力区，适合考虑分批减仓"
    elif trend_good and overheat:
        main_signal = SIGNALS["HOLD_NO_CHASE"]
        secondary_signal = "趋势仍在，但短线偏热，不宜追高"
    elif trend_good and not overheat:
        main_signal = SIGNALS["HOLD"]
        secondary_signal = "趋势结构良好，可继续持有"
    elif trend_good and near_buy_zone and add_allowed and not market_weak:
        main_signal = SIGNALS["ADD_IN_BATCHES"]
        secondary_signal = "若支撑区企稳，可考虑小幅分批加仓"
    elif near_buy_zone and add_allowed and rsi <= profile["rsi_add_max"] and not market_weak:
        main_signal = SIGNALS["WATCH_BUY_DIP"]
        secondary_signal = "接近回踩观察区，等待确认更稳"
    else:
        main_signal = SIGNALS["WAIT_AND_SEE"]
        secondary_signal = "当前性价比一般，建议继续观察"

    if holding is not None:
        cost_price = safe_float(holding.get("cost_price"), np.nan)
        if not np.isnan(cost_price):
            pnl_pct = (close / cost_price - 1) * 100
            if pnl_pct > 20 and main_signal == SIGNALS["HOLD"]:
                main_signal = SIGNALS["HOLD_NO_CHASE"]
                secondary_signal = "已有较好浮盈，持有即可，不必继续追高"

    return {
        "current_price": round(close, 2),
        "change_pct": round(pct_chg, 2),
        "short_trend": short_trend,
        "medium_trend": medium_trend,
        "risk_level": risk_level,
        "main_signal": main_signal,
        "secondary_signal": secondary_signal,
        **zones,
    }


def build_explanation(result: Dict, holding: Optional[Dict] = None) -> str:
    base = (
        f"当前短期趋势为“{result['short_trend']}”，中期趋势为“{result['medium_trend']}”，"
        f"风险等级为“{result['risk_level']}”，主建议为“{result['main_signal']}”。"
    )
    zone_part = (
        f"观察低吸区参考 {result['buy_zone_low']} - {result['buy_zone_high']}，"
        f"分批减仓区参考 {result['reduce_zone_low']} - {result['reduce_zone_high']}，"
        f"防守位参考 {result['defense_price']}。"
    )
    return base + result["secondary_signal"] + "。" + zone_part


# =========================
# 图表
# =========================
def plot_price_chart(df: pd.DataFrame, stock_code: str, stock_name: str):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df["date"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="K线"
    ))

    for ma, name in [("ma5", "MA5"), ("ma10", "MA10"), ("ma20", "MA20"), ("ma60", "MA60")]:
        fig.add_trace(go.Scatter(x=df["date"], y=df[ma], mode="lines", name=name))

    fig.update_layout(
        title=f"{stock_name}（{stock_code}）价格走势",
        xaxis_title="日期",
        yaxis_title="价格",
        height=520,
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_rsi_macd(df: pd.DataFrame):
    col1, col2 = st.columns(2)

    with col1:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df["date"], y=df["rsi14"], mode="lines", name="RSI"))
        fig_rsi.add_hline(y=70, line_dash="dash")
        fig_rsi.add_hline(y=30, line_dash="dash")
        fig_rsi.update_layout(title="RSI 指标", height=300)
        st.plotly_chart(fig_rsi, use_container_width=True)

    with col2:
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df["date"], y=df["macd_dif"], mode="lines", name="DIF"))
        fig_macd.add_trace(go.Scatter(x=df["date"], y=df["macd_dea"], mode="lines", name="DEA"))
        fig_macd.add_trace(go.Bar(x=df["date"], y=df["macd_hist"], name="MACD柱"))
        fig_macd.update_layout(title="MACD 指标", height=300)
        st.plotly_chart(fig_macd, use_container_width=True)


# =========================
# 页面工具
# =========================
def analyze_one_stock(stock_code: str, mode_name: str, holding: Optional[Dict] = None):
    profile = MODE_PROFILES[mode_name]
    stock_name = get_stock_name_by_code(stock_code)
    df = get_stock_history(stock_code)
    df = add_indicators(df)
    market_env = analyze_market_environment()
    result = evaluate_stock(df, market_env, profile, holding=holding)
    explanation = build_explanation(result, holding=holding)
    return stock_name, df, market_env, result, explanation


def render_market_summary(mode_name: str):
    st.subheader("大盘环境")
    market_env = analyze_market_environment()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("市场评分", market_env["market_score"])
    c2.metric("风险等级", market_env["risk_level"])
    c3.metric("短期趋势", market_env["short_trend"])
    c4.metric("中期趋势", market_env["medium_trend"])

    st.info(f"当前模式：{mode_name} ｜ {market_env['summary']}")

    if market_env["details"]:
        detail_df = pd.DataFrame(market_env["details"], columns=["指数名称", "评分"])
        st.dataframe(detail_df, use_container_width=True, hide_index=True)

    return market_env


def render_home(mode_name: str):
    st.title("📊 A股持仓辅助决策系统")
    market_env = render_market_summary(mode_name)

    holdings = get_holdings()
    watchlist = get_watchlist()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("持仓数量", len(holdings))
    c2.metric("自选数量", len(watchlist))
    c3.metric("当前模式", mode_name)
    c4.metric("市场风险", market_env["risk_level"])

    st.markdown("### 今日重点提醒")

    if holdings.empty:
        st.info("当前还没有持仓记录。")
        return

    alerts = []
    for _, row in holdings.iterrows():
        try:
            holding_dict = row.to_dict()
            stock_name, df, _, result, _ = analyze_one_stock(row["stock_code"], mode_name, holding=holding_dict)
            alerts.append({
                "股票名称": stock_name,
                "股票代码": row["stock_code"],
                "主建议": result["main_signal"],
                "风险等级": result["risk_level"],
                "说明": result["secondary_signal"],
            })
        except Exception as e:
            alerts.append({
                "股票名称": row["stock_name"] if row["stock_name"] else row["stock_code"],
                "股票代码": row["stock_code"],
                "主建议": "数据获取失败",
                "风险等级": "-",
                "说明": str(e),
            })

    for item in alerts[:8]:
        css = signal_css_class(item["主建议"])
        st.markdown(
            f"""
            <div class="custom-card">
                <div><strong>{item['股票名称']}</strong>（{item['股票代码']}）</div>
                <div class="{css}" style="margin-top:8px;">{item['主建议']}</div>
                <div style="margin-top:6px;">风险等级：{item['风险等级']}</div>
                <div class="small-note" style="margin-top:6px;">{item['说明']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


def render_portfolio(mode_name: str):
    st.title("💼 持仓总览")
    holdings = get_holdings()

    if holdings.empty:
        st.warning("当前还没有持仓记录。")
        return

    rows = []
    for _, row in holdings.iterrows():
        try:
            holding_dict = row.to_dict()
            stock_name, df, _, result, _ = analyze_one_stock(row["stock_code"], mode_name, holding=holding_dict)
            current_price = result["current_price"]
            pnl_amt = (current_price - row["cost_price"]) * row["quantity"]
            pnl_pct = (current_price / row["cost_price"] - 1) * 100

            rows.append({
                "ID": row["id"],
                "股票名称": stock_name,
                "股票代码": row["stock_code"],
                "成本价": round(row["cost_price"], 2),
                "现价": current_price,
                "持仓股数": int(row["quantity"]),
                "浮盈亏金额": round(pnl_amt, 2),
                "浮盈亏比例%": round(pnl_pct, 2),
                "主建议": result["main_signal"],
                "风险等级": result["risk_level"],
                "短期趋势": result["short_trend"],
                "中期趋势": result["medium_trend"],
                "观察低吸区": f"{result['buy_zone_low']} - {result['buy_zone_high']}",
                "分批减仓区": f"{result['reduce_zone_low']} - {result['reduce_zone_high']}",
                "防守位": result["defense_price"],
            })
        except Exception as e:
            rows.append({
                "ID": row["id"],
                "股票名称": row["stock_name"] if row["stock_name"] else row["stock_code"],
                "股票代码": row["stock_code"],
                "成本价": round(row["cost_price"], 2),
                "现价": "获取失败",
                "持仓股数": int(row["quantity"]),
                "浮盈亏金额": "-",
                "浮盈亏比例%": "-",
                "主建议": f"错误: {e}",
                "风险等级": "-",
                "短期趋势": "-",
                "中期趋势": "-",
                "观察低吸区": "-",
                "分批减仓区": "-",
                "防守位": "-",
            })

    result_df = pd.DataFrame(rows)
    st.dataframe(result_df, use_container_width=True, hide_index=True)

    st.markdown("### 删除持仓")
    delete_id = st.number_input("输入要删除的持仓 ID", min_value=1, step=1, key="delete_holding_id")
    if st.button("删除该持仓"):
        delete_holding(delete_id)
        st.success("✅ 已成功删除该持仓")
        st.rerun()


def render_watchlist(mode_name: str):
    st.title("⭐ 自选观察池")
    watchlist = get_watchlist()

    if watchlist.empty:
        st.warning("当前还没有自选股。")
    else:
        rows = []
        for _, row in watchlist.iterrows():
            try:
                stock_name, df, _, result, _ = analyze_one_stock(row["stock_code"], mode_name, holding=None)
                rows.append({
                    "ID": row["id"],
                    "股票名称": stock_name,
                    "股票代码": row["stock_code"],
                    "现价": result["current_price"],
                    "主建议": result["main_signal"],
                    "风险等级": result["risk_level"],
                    "短期趋势": result["short_trend"],
                    "中期趋势": result["medium_trend"],
                    "观察低吸区": f"{result['buy_zone_low']} - {result['buy_zone_high']}",
                    "分批减仓区": f"{result['reduce_zone_low']} - {result['reduce_zone_high']}",
                })
            except Exception as e:
                rows.append({
                    "ID": row["id"],
                    "股票名称": row["stock_name"] if row["stock_name"] else row["stock_code"],
                    "股票代码": row["stock_code"],
                    "现价": "获取失败",
                    "主建议": f"错误: {e}",
                    "风险等级": "-",
                    "短期趋势": "-",
                    "中期趋势": "-",
                    "观察低吸区": "-",
                    "分批减仓区": "-",
                })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("### 删除自选股")
    delete_id = st.number_input("输入要删除的自选股 ID", min_value=1, step=1, key="delete_watch_id")
    if st.button("删除该自选股"):
        delete_watchlist(delete_id)
        st.success("✅ 已成功删除该自选股")
        st.rerun()


def render_single_stock_analysis(mode_name: str):
    st.title("🔎 单股分析")
    stock_code = st.text_input("输入 A 股代码（如 600519 / 000001 / 300750）", value="600519").strip()

    if stock_code:
        stock_name = get_stock_name_by_code(stock_code)
        st.success(f"已识别股票：{stock_name}（{normalize_stock_code(stock_code)}）")

    if st.button("开始分析"):
        try:
            stock_name, df, market_env, result, explanation = analyze_one_stock(stock_code, mode_name, holding=None)

            c1, c2, c3 = st.columns(3)
            c1.metric("📈 当前价格", result["current_price"])
            c2.metric("📊 当日涨跌幅%", result["change_pct"])
            c3.metric("⚠ 风险等级", result["risk_level"])

            css = signal_css_class(result["main_signal"])
            st.markdown(
             f"""
             <div class="custom-card">
              <div><strong>🧠 系统建议</strong></div>
              <div class="{css}" style="margin-top:10px; font-size: 1.1rem;">{result['main_signal']}</div>
              <div class="small-note" style="margin-top:8px;">{result['secondary_signal']}</div>
             </div>
             """,
             unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="custom-card">
                    <div><strong>分析结论</strong></div>
                    <div style="margin-top:8px;">{explanation}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            z1, z2, z3, z4 = st.columns(4)
            z1.metric("观察低吸区", f"{result['buy_zone_low']} - {result['buy_zone_high']}")
            z2.metric("分批减仓区", f"{result['reduce_zone_low']} - {result['reduce_zone_high']}")
            z3.metric("防守位", result["defense_price"])
            z4.metric("趋势失效位", result["invalidation_price"])

            plot_price_chart(df, normalize_stock_code(stock_code), stock_name)
            plot_rsi_macd(df)

            st.markdown("### 最近 10 个交易日行情")
            display_cols = [
                "date", "open", "high", "low", "close", "pct_chg",
                "ma20", "ma60", "rsi14", "macd_dif", "macd_dea"
            ]
            show_df = df[display_cols].tail(10).copy()
            show_df["date"] = show_df["date"].dt.strftime("%Y-%m-%d")

            show_df = show_df.rename(columns={
                "date": "日期",
                "open": "开盘价",
                "high": "最高价",
                "low": "最低价",
                "close": "收盘价",
                "pct_chg": "涨跌幅%",
                "ma20": "20日均线",
                "ma60": "60日均线",
                "rsi14": "RSI指标",
                "macd_dif": "MACD-DIF",
                "macd_dea": "MACD-DEA",
            })

            st.dataframe(show_df, use_container_width=True, hide_index=True)

            st.markdown("### 当前市场环境")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("市场评分", market_env["market_score"])
            m2.metric("市场风险", market_env["risk_level"])
            m3.metric("短期环境", market_env["short_trend"])
            m4.metric("中期环境", market_env["medium_trend"])
            st.info(market_env["summary"])

        except Exception as e:
            st.error(f"分析失败：{e}")


def render_add_forms():
    st.title("➕ 添加持仓 / 自选股")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 添加持仓")
        with st.form("holding_form"):
            stock_code = st.text_input("股票代码", value="600519").strip()
            detected_name = ""
            if stock_code:
                detected_name = get_stock_name_by_code(stock_code)

            st.text_input("自动识别股票名称", value=detected_name, disabled=True)

            cost_price = st.number_input("持仓成本", min_value=0.0, value=100.0, step=0.01)
            quantity = st.number_input("持仓股数", min_value=1, value=100, step=1)
            first_buy_date = st.date_input("首次建仓日期", value=datetime.now().date())
            position_pct = st.number_input("仓位占比%（可选）", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
            allow_add_position = st.checkbox("允许系统给出加仓建议", value=True)
            custom_stop_loss = st.number_input("自定义防守位（可选，0 表示不填）", min_value=0.0, value=0.0, step=0.01)
            notes = st.text_area("备注", value="")

            submitted = st.form_submit_button("添加持仓")
            if submitted:
                stock_code = normalize_stock_code(stock_code)
                stock_name = get_stock_name_by_code(stock_code)
                add_holding(
                    stock_code=stock_code,
                    stock_name=stock_name,
                    cost_price=cost_price,
                    quantity=int(quantity),
                    first_buy_date=str(first_buy_date),
                    position_pct=position_pct,
                    allow_add_position=allow_add_position,
                    custom_stop_loss=None if custom_stop_loss == 0 else custom_stop_loss,
                    notes=notes.strip(),
                )
                st.toast("添加成功 ✅")

    with col2:
        st.markdown("### 添加自选股")
        with st.form("watchlist_form"):
            stock_code = st.text_input("自选股代码", value="300750").strip()
            detected_name = ""
            if stock_code:
                detected_name = get_stock_name_by_code(stock_code)

            st.text_input("自动识别股票名称", value=detected_name, disabled=True)
            notes = st.text_area("自选备注", value="")
            submitted = st.form_submit_button("添加自选股")
            if submitted:
                stock_code = normalize_stock_code(stock_code)
                stock_name = get_stock_name_by_code(stock_code)
                add_watchlist(
                    stock_code=stock_code,
                    stock_name=stock_name,
                    notes=notes.strip(),
                )
                st.toast("添加成功 ✅")


# =========================
# 主应用
# =========================
def main():
    init_db()

    if "success_message" not in st.session_state:
        st.session_state.success_message = ""

    if st.session_state.success_message:
        st.success(st.session_state.success_message)
        st.session_state.success_message = ""

    if ak is None:
        st.error("当前环境没有安装 akshare。请先执行：python3 -m pip install akshare")
        st.stop()

    with st.sidebar:
        st.markdown("## 📘 系统导航")
        selected = option_menu(
            menu_title=None,
            options=["首页", "添加持仓/自选", "持仓总览", "自选观察池", "单股分析"],
            icons=["house", "plus-circle", "briefcase", "star", "search"],
            default_index=0,
        )

        st.markdown("---")
        mode_name = st.selectbox("选择分析模式", ["保守", "平衡", "进取"], index=1)
        st.caption("三种模式会影响建议的保守程度。")
        st.markdown("---")
        st.write("建议优先级：")
        st.caption("风险控制 > 趋势转弱 > 压力减仓 > 不追高 > 持有 > 加仓 > 低吸 > 观察")

    if selected == "首页":
        render_home(mode_name)
    elif selected == "添加持仓/自选":
        render_add_forms()
    elif selected == "持仓总览":
        render_portfolio(mode_name)
    elif selected == "自选观察池":
        render_watchlist(mode_name)
    elif selected == "单股分析":
        render_single_stock_analysis(mode_name)


if __name__ == "__main__":
    main()
