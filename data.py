"""複数銘柄データ取得(Dukascopy, 長期履歴対応): 日足・1時間足(→4時間足にリサンプル)。ローカルCSVキャッシュ付き"""
import os
import datetime
import dukascopy_python
from dukascopy_python import instruments as dki
import pandas as pd

TICKERS = {
    "AUDJPY": dki.INSTRUMENT_FX_CROSSES_AUD_JPY,
    "USDJPY": dki.INSTRUMENT_FX_MAJORS_USD_JPY,
    "EURUSD": dki.INSTRUMENT_FX_MAJORS_EUR_USD,
    "GBPJPY": dki.INSTRUMENT_FX_CROSSES_GBP_JPY,
    "AUDUSD": dki.INSTRUMENT_FX_MAJORS_AUD_USD,
    "NZDUSD": dki.INSTRUMENT_FX_MAJORS_NZD_USD,
}

HIST_START = datetime.datetime(2019, 1, 1)
CACHE_DIR = "cache_duka"


def _cache_path(name, interval):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{name}_{interval}.csv")


def _std_ohlc(d):
    d = d.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
    d.index = pd.to_datetime(d.index).tz_localize(None)
    d.index.name = None
    return d[["Open", "High", "Low", "Close", "Volume"]].dropna()


def fetch_daily(name, refresh=False, end=None):
    path = _cache_path(name, "1d")
    if not refresh and os.path.exists(path):
        return pd.read_csv(path, index_col=0, parse_dates=True)
    end = end or datetime.datetime.now()
    d = dukascopy_python.fetch(TICKERS[name], dukascopy_python.INTERVAL_DAY_1,
                                dukascopy_python.OFFER_SIDE_BID, HIST_START, end)
    d = _std_ohlc(d)
    d.to_csv(path)
    return d


def fetch_1h(name, refresh=False, end=None):
    path = _cache_path(name, "1h")
    if not refresh and os.path.exists(path):
        return pd.read_csv(path, index_col=0, parse_dates=True)
    end = end or datetime.datetime.now()
    d = dukascopy_python.fetch(TICKERS[name], dukascopy_python.INTERVAL_HOUR_1,
                                dukascopy_python.OFFER_SIDE_BID, HIST_START, end)
    d = _std_ohlc(d)
    d.to_csv(path)
    return d


def resample_4h(df_1h):
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    d4 = df_1h.resample("4h", origin="start_day").agg(agg).dropna()
    return d4


def load_all(names=None, refresh=False):
    names = names or list(TICKERS.keys())
    out = {}
    for n in names:
        daily = fetch_daily(n, refresh=refresh)
        h1 = fetch_1h(n, refresh=refresh)
        h4 = resample_4h(h1)
        out[n] = (daily, h1, h4)
        print(f"{n}: daily={daily.shape[0]} 1h={h1.shape[0]} 4h={h4.shape[0]} "
              f"range={h1.index.min().date()}~{h1.index.max().date()}")
    return out


if __name__ == "__main__":
    load_all(refresh=True)
