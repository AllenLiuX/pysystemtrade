"""
AkshareClient - A股数据 API 客户端 (akshare 备份数据源)

通过 akshare 库（新浪数据源）获取 A股行情数据。
无需 API token，免费使用。

接口:
    - stock_zh_a_daily (sina): 日线行情 (支持 qfq/hfq 复权)
    - fund_etf_hist_sina (sina): ETF 日线 (不复权)
    - fund_etf_hist_em (east money): ETF 日线 (支持 qfq/hfq 复权)
    - stock_zh_a_minute (sina): 分钟行情 (不复权)

符号格式转换:
    Tushare: 600000.SH / 000001.SZ
    Sina:    sh600000     / sz000001
"""

import time
import logging
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Literal

logger = logging.getLogger(__name__)

RATE_LIMIT_SLEEP = 0.3
MAX_RETRIES = 5
RETRY_BACKOFF = 2.0

AdjustType = Literal["", "qfq", "hfq"]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_AH_MAPPING_FILE = _REPO_ROOT / "data" / "astock" / "csvconfig" / "ah_mapping.csv"

# ── 符号转换 ──────────────────────────────────────────────────


def to_sina_symbol(ts_code: str) -> str:
    """
    Tushare 格式 → Sina 格式

    600000.SH → sh600000
    000001.SZ → sz000001
    688981.SH → sh688981
    159919.SZ → sz159919
    """
    if "." in ts_code:
        code, exchange = ts_code.split(".")
        return f"{exchange.lower()}{code}"
    return ts_code


def from_sina_symbol(sina_symbol: str) -> str:
    """
    Sina 格式 → Tushare 格式

    sh600000 → 600000.SH
    sz000001 → 000001.SZ
    """
    exchange = sina_symbol[:2].upper()
    code = sina_symbol[2:]
    return f"{code}.{exchange}"


# ── 数据标准化 ──────────────────────────────────────────────────


def _normalize_sina_daily(df: pd.DataFrame, ts_code: str) -> pd.DataFrame:
    """
    新浪日线 → 标准 Tushare 格式

    输入: date, open, high, low, close, volume, amount
    输出: ts_code, trade_date, open, high, low, close, vol, amount
    """
    if df.empty:
        return df
    df = df.copy()
    df["ts_code"] = ts_code
    df["trade_date"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["vol"] = df["volume"] / 100
    cols = ["ts_code", "trade_date", "open", "high", "low", "close",
            "vol", "amount"]
    return df[[c for c in cols if c in df.columns]]


def _normalize_sina_minutes(df: pd.DataFrame, ts_code: str) -> pd.DataFrame:
    """
    新浪分钟线 → 标准 Tushare 格式

    输入: day, open, high, low, close, volume, amount
    输出: ts_code, trade_time, open, high, low, close, vol, amount
    """
    if df.empty:
        return df
    df = df.copy()
    df["ts_code"] = ts_code
    df["trade_time"] = pd.to_datetime(df["day"])
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["vol"] = df["volume"] / 100
    cols = ["ts_code", "trade_time", "open", "high", "low", "close",
            "vol", "amount"]
    return df[[c for c in cols if c in df.columns]]


def _normalize_sina_fund(df: pd.DataFrame, ts_code: str) -> pd.DataFrame:
    """
    新浪ETF日线 → 标准 Tushare 格式
    """
    if df.empty:
        return df
    df = df.copy()
    df["ts_code"] = ts_code
    df["trade_date"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["vol"] = df["volume"] / 100
    cols = ["ts_code", "trade_date", "open", "high", "low", "close",
            "vol", "amount"]
    return df[[c for c in cols if c in df.columns]]


def _normalize_em_fund(df: pd.DataFrame, ts_code: str) -> pd.DataFrame:
    """
    东方财富ETF日线 → 标准 Tushare 格式
    """
    if df.empty:
        return df
    df = df.copy()
    df["ts_code"] = ts_code
    df["trade_date"] = pd.to_datetime(df["日期"]).dt.strftime("%Y%m%d")
    for col in ["开盘", "最高", "最低", "收盘", "成交量", "成交额"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    rename_map = {
        "开盘": "open", "最高": "high", "最低": "low",
        "收盘": "close", "成交量": "volume", "成交额": "amount",
    }
    df = df.rename(columns=rename_map)
    df["vol"] = df["volume"] / 100
    cols = ["ts_code", "trade_date", "open", "high", "low", "close",
            "vol", "amount"]
    return df[[c for c in cols if c in df.columns]]


def _normalize_hk_daily(df: pd.DataFrame, ts_code: str) -> pd.DataFrame:
    """
    H股日线 (stock_hk_daily) → 标准 Tushare 格式

    输入: date, open, high, low, close, volume, amount
    输出: ts_code, trade_date, open, high, low, close, vol, amount
    """
    if df.empty:
        return df
    df = df.copy()
    df["ts_code"] = ts_code
    df["trade_date"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["vol"] = df["volume"] / 100
    cols = ["ts_code", "trade_date", "open", "high", "low", "close",
            "vol", "amount"]
    return df[[c for c in cols if c in df.columns]]


class AkshareClient:
    """
    akshare A股数据客户端

    封装 akshare 调用，提供与 XiximiaoClient 相同的接口。
    使用新浪数据源。

    Args:
        adjust: 复权方式 — "" (不复权), "qfq" (前复权), "hfq" (后复权)
                日线支持全部三种；分钟线始终不复权（新浪限制）。
        rate_limit: 调用间隔（秒）
    """

    FREQ_MAP = {
        "1min": "1",
        "5min": "5",
        "15min": "15",
        "30min": "30",
        "60min": "60",
    }

    def __init__(
        self,
        adjust: AdjustType = "",
        rate_limit: float = RATE_LIMIT_SLEEP,
    ):
        self._adjust = adjust
        self._rate_limit = rate_limit
        self._last_call_time = 0.0

    @property
    def adjust(self) -> AdjustType:
        return self._adjust

    def _throttle(self):
        elapsed = time.time() - self._last_call_time
        if elapsed < self._rate_limit:
            time.sleep(self._rate_limit - elapsed)
        self._last_call_time = time.time()

    def _call_with_retry(self, fn, **kwargs) -> Optional[pd.DataFrame]:
        for attempt in range(MAX_RETRIES):
            try:
                self._throttle()
                df = fn(**kwargs)
                if df is not None and not df.empty:
                    return df
                return None
            except Exception as e:
                wait = RETRY_BACKOFF ** (attempt + 1)
                logger.warning(
                    "akshare call failed (attempt %d/%d): %s — retrying in %.0fs",
                    attempt + 1,
                    MAX_RETRIES,
                    str(e)[:120],
                    wait,
                )
                time.sleep(wait)
        logger.error("akshare call failed after %d retries", MAX_RETRIES)
        return None

    # ── daily ──────────────────────────────────────────────────

    def fetch_daily(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        获取日线数据（新浪，支持复权）

        Args:
            ts_code:    '600000.SH'
            start_date: '20240101'
            end_date:   '20241231'

        Returns:
            DataFrame with columns: ts_code, trade_date, open, high, low, close,
                                    vol, amount
        """
        sina_symbol = to_sina_symbol(ts_code)

        df = self._call_with_retry(
            self._sina_daily,
            symbol=sina_symbol,
            start_date=start_date,
            end_date=end_date,
            adjust=self._adjust,
        )
        if df is not None and not df.empty:
            return _normalize_sina_daily(df, ts_code)

        return pd.DataFrame()

    def fetch_fund_daily(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        获取基金日线数据（ETF）

        不复权时使用新浪（快）；复权时回退到东方财富。

        Args:
            ts_code:    '510300.SH'
            start_date: '20240101'
            end_date:   '20241231'

        Returns:
            DataFrame with columns: ts_code, trade_date, open, high, low, close,
                                    vol, amount
        """
        if self._adjust:
            return self._fetch_fund_daily_adjusted(ts_code, start_date, end_date)

        sina_symbol = to_sina_symbol(ts_code)

        df = self._call_with_retry(
            self._sina_fund,
            symbol=sina_symbol,
        )
        if df is not None and not df.empty:
            result = _normalize_sina_fund(df, ts_code)
            result = result[
                (result["trade_date"] >= start_date)
                & (result["trade_date"] <= end_date)
            ]
            return result

        return pd.DataFrame()

    def _fetch_fund_daily_adjusted(
        self, ts_code: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """ETF日线复权 — 使用东方财富"""
        code = ts_code.split(".")[0] if "." in ts_code else ts_code

        df = self._call_with_retry(
            self._em_fund,
            symbol=code,
            start_date=start_date,
            end_date=end_date,
            adjust=self._adjust,
        )
        if df is not None and not df.empty:
            return _normalize_em_fund(df, ts_code)

        return pd.DataFrame()

    # ── minutes ────────────────────────────────────────────────

    def fetch_minutes(
        self,
        ts_code: str,
        freq: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        获取分钟线数据（新浪，不复权）

        注: 新浪分钟线不支持复权（qfq 返回 NaN），始终返回不复权数据。

        Args:
            ts_code:    '600000.SH'
            freq:       '1min' / '5min' / '15min' / '30min' / '60min'
            start_date: '2025-03-17 09:00:00'
            end_date:   '2025-03-17 15:00:00'

        Returns:
            DataFrame with columns: ts_code, trade_time, open, high, low, close,
                                    vol, amount
        """
        if self._adjust:
            logger.warning(
                "Minute bars do not support adjustment (adjust='%s' ignored). "
                "Sina minute qfq returns NaN for OHLC.",
                self._adjust,
            )

        sina_symbol = to_sina_symbol(ts_code)
        period = self.FREQ_MAP.get(freq, freq)

        df = self._call_with_retry(
            self._sina_minute,
            symbol=sina_symbol,
            period=period,
        )
        if df is not None and not df.empty:
            result = _normalize_sina_minutes(df, ts_code)
            result["trade_time"] = pd.to_datetime(result["trade_time"])
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            result = result[
                (result["trade_time"] >= start_dt)
                & (result["trade_time"] <= end_dt)
            ]
            result["trade_time"] = result["trade_time"].dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            return result

        return pd.DataFrame()

    # ── 底层 akshare 调用 ─────────────────────────────────────

    @staticmethod
    def _sina_daily(
        symbol: str, start_date: str, end_date: str, adjust: str
    ) -> pd.DataFrame:
        import akshare as ak
        return ak.stock_zh_a_daily(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )

    @staticmethod
    def _sina_fund(symbol: str) -> pd.DataFrame:
        import akshare as ak
        return ak.fund_etf_hist_sina(symbol=symbol)

    @staticmethod
    def _sina_minute(symbol: str, period: str) -> pd.DataFrame:
        import akshare as ak
        return ak.stock_zh_a_minute(
            symbol=symbol,
            period=period,
            adjust="",
        )

    @staticmethod
    def _em_fund(
        symbol: str, start_date: str, end_date: str, adjust: str
    ) -> pd.DataFrame:
        import akshare as ak
        return ak.fund_etf_hist_em(
            symbol=symbol,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )

    @staticmethod
    def _hk_daily(symbol: str, adjust: str) -> pd.DataFrame:
        import akshare as ak
        return ak.stock_hk_daily(symbol=symbol, adjust=adjust)

    # ── AH mapping ─────────────────────────────────────────────

    @staticmethod
    def load_ah_mapping() -> pd.DataFrame:
        """
        Load A+H dual-listed company mapping.

        Returns:
            DataFrame with columns: a_code, h_code, a_name, h_name
        """
        if not _AH_MAPPING_FILE.exists():
            logger.warning("AH mapping file not found: %s", _AH_MAPPING_FILE)
            return pd.DataFrame()
        df = pd.read_csv(_AH_MAPPING_FILE, dtype={"h_code": str})
        df["h_code"] = df["h_code"].str.zfill(5)
        return df

    @classmethod
    def get_h_code(cls, a_code: str) -> Optional[str]:
        """Get H-share code for an A-share code."""
        mapping = cls.load_ah_mapping()
        if mapping.empty:
            return None
        match = mapping[mapping["a_code"] == a_code]
        if match.empty:
            return None
        return match.iloc[0]["h_code"]

    @classmethod
    def get_a_code(cls, h_code: str) -> Optional[str]:
        """Get A-share code for an H-share code."""
        mapping = cls.load_ah_mapping()
        if mapping.empty:
            return None
        match = mapping[mapping["h_code"] == h_code]
        if match.empty:
            return None
        return match.iloc[0]["a_code"]

    @classmethod
    def get_ah_pairs(cls) -> list:
        """Get all A+H pairs as list of (a_code, h_code) tuples."""
        mapping = cls.load_ah_mapping()
        if mapping.empty:
            return []
        return list(zip(mapping["a_code"], mapping["h_code"]))

    # ── H-share daily ──────────────────────────────────────────

    def fetch_hk_daily(
        self,
        h_code: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        获取H股日线数据（stock_hk_daily，英文列名，支持复权）

        Args:
            h_code:     '02318' (without exchange prefix)
            start_date: '20240101'
            end_date:   '20241231'

        Returns:
            DataFrame with columns: ts_code, trade_date, open, high, low, close,
                                    vol, amount
        """
        ts_code = f"{h_code}.HK"

        df = self._call_with_retry(
            self._hk_daily,
            symbol=h_code,
            adjust=self._adjust,
        )
        if df is not None and not df.empty:
            result = _normalize_hk_daily(df, ts_code)
            result = result[
                (result["trade_date"] >= start_date)
                & (result["trade_date"] <= end_date)
            ]
            return result

        return pd.DataFrame()

    def fetch_hk_daily_range(
        self,
        h_code: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """便捷方法：datetime 入参"""
        start_str = start.strftime("%Y%m%d")
        end_str = end.strftime("%Y%m%d")
        return self.fetch_hk_daily(h_code=h_code, start_date=start_str, end_date=end_str)

    def fetch_ah_pair(
        self,
        a_code: str,
        start_date: str,
        end_date: str,
    ) -> dict:
        """
        获取A+H双市场日线数据

        Args:
            a_code:     '601318.SH'
            start_date: '20240101'
            end_date:   '20241231'

        Returns:
            {'a': df_a, 'h': df_h} — both with standard OHLCV columns
        """
        h_code = self.get_h_code(a_code)
        if not h_code:
            logger.warning("No H-share mapping for %s", a_code)
            return {"a": pd.DataFrame(), "h": pd.DataFrame()}

        df_a = self.fetch_daily(a_code, start_date, end_date)
        df_h = self.fetch_hk_daily(h_code, start_date, end_date)

        return {"a": df_a, "h": df_h}

    # ── batch helpers ──────────────────────────────────────────

    def fetch_daily_range(
        self,
        ts_code: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """便捷方法：datetime 入参"""
        start_str = start.strftime("%Y%m%d")
        end_str = end.strftime("%Y%m%d")
        return self.fetch_daily(ts_code=ts_code, start_date=start_str, end_date=end_str)

    def fetch_fund_daily_range(
        self,
        ts_code: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Convenience method: datetime args for fund_daily (ETFs)."""
        start_str = start.strftime("%Y%m%d")
        end_str = end.strftime("%Y%m%d")
        return self.fetch_fund_daily(
            ts_code=ts_code, start_date=start_str, end_date=end_str
        )

    def fetch_minutes_range(
        self,
        ts_code: str,
        freq: str,
        start: datetime,
        end: datetime,
        chunk_days: int = 5,
    ) -> pd.DataFrame:
        """
        按天窗口分批拉取分钟数据，避免单次请求行数溢出。
        chunk_days: 每批覆盖的天数（默认5天）
        """
        all_chunks = []
        chunk_start = start

        while chunk_start < end:
            chunk_end = min(chunk_start + timedelta(days=chunk_days), end)
            s = chunk_start.strftime("%Y-%m-%d %H:%M:%S")
            e = chunk_end.strftime("%Y-%m-%d %H:%M:%S")

            df = self.fetch_minutes(ts_code=ts_code, freq=freq, start_date=s, end_date=e)
            if not df.empty:
                all_chunks.append(df)

            chunk_start = chunk_end

        if not all_chunks:
            return pd.DataFrame()

        result = pd.concat(all_chunks, ignore_index=True)
        result = result.drop_duplicates(subset=["trade_time"], keep="last")
        result = result.sort_values("trade_time").reset_index(drop=True)
        return result
