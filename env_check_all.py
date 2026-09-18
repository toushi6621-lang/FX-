"""
FX15組み合わせ+貴金属+原油+株価指数に「環境認識3ステップ」を適用。
直近450日分のみ取得(環境認識には十分、フル7年分より高速)。
"""
import datetime
import numpy as np
import pandas as pd
import dukascopy_python
from dukascopy_python import instruments as dki
from engine import (build_dataset, compute_hit_arrays, compute_ma_reversal_triggers,
                     compute_ma_fan_triggers, detect_fractal_neckline)

CODES = {
    "EURUSD": dki.INSTRUMENT_FX_MAJORS_EUR_USD,
    "GBPUSD": dki.INSTRUMENT_FX_MAJORS_GBP_USD,
    "USDCHF": dki.INSTRUMENT_FX_MAJORS_USD_CHF,
    "AUDUSD": dki.INSTRUMENT_FX_MAJORS_AUD_USD,
    "NZDUSD": dki.INSTRUMENT_FX_MAJORS_NZD_USD,
    "EURGBP": dki.INSTRUMENT_FX_CROSSES_EUR_GBP,
    "EURAUD": dki.INSTRUMENT_FX_CROSSES_EUR_AUD,
    "EURNZD": dki.INSTRUMENT_FX_CROSSES_EUR_NZD,
    "EURCHF": dki.INSTRUMENT_FX_CROSSES_EUR_CHF,
    "GBPAUD": dki.INSTRUMENT_FX_CROSSES_GBP_AUD,
    "GBPNZD": dki.INSTRUMENT_FX_CROSSES_GBP_NZD,
    "GBPCHF": dki.INSTRUMENT_FX_CROSSES_GBP_CHF,
    "AUDNZD": dki.INSTRUMENT_FX_CROSSES_AUD_NZD,
    "AUDCHF": dki.INSTRUMENT_FX_CROSSES_AUD_CHF,
    "NZDCHF": dki.INSTRUMENT_FX_CROSSES_NZD_CHF,
    "USDCAD": dki.INSTRUMENT_FX_MAJORS_USD_CAD,
    "EURCAD": dki.INSTRUMENT_FX_CROSSES_EUR_CAD,
    "GBPCAD": dki.INSTRUMENT_FX_CROSSES_GBP_CAD,
    "AUDCAD": dki.INSTRUMENT_FX_CROSSES_AUD_CAD,
    "NZDCAD": dki.INSTRUMENT_FX_CROSSES_NZD_CAD,
    "CADCHF": dki.INSTRUMENT_FX_CROSSES_CAD_CHF,
    "GOLD": dki.INSTRUMENT_FX_METALS_XAU_USD,
    "SILVER": dki.INSTRUMENT_FX_METALS_XAG_USD,
    "WTI原油": dki.INSTRUMENT_CMD_ENERGY_E_LIGHT,
    "日経225": dki.INSTRUMENT_IDX_ASIA_E_N225JAP,
    "NASDAQ100": dki.INSTRUMENT_IDX_AMERICA_E_NQ_100,
}

DAYS = 450


def fetch(code, interval, days=DAYS):
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=days)
    d = dukascopy_python.fetch(code, interval, dukascopy_python.OFFER_SIDE_BID, start, end)
    d = d.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
    idx = pd.to_datetime(d.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    d.index = pd.DatetimeIndex(idx.values.astype("datetime64[ns]"))
    d.index.name = None
    return d[["Open", "High", "Low", "Close", "Volume"]].dropna()


def check(name, code):
    try:
        h1 = fetch(code, dukascopy_python.INTERVAL_HOUR_1)
        daily = fetch(code, dukascopy_python.INTERVAL_DAY_1)
        if len(h1) < 250 or len(daily) < 60:
            return {"instrument": name, "error": "データ不足"}

        ds = build_dataset(name, daily=daily, h1=h1)
        d4, h4, h1b = ds["daily"], ds["h4"], ds["h1"]

        drow = d4.iloc[-1]
        trend = drow["trend"]

        tol = 0.002
        near_level, near_ema80, in_fib = compute_hit_arrays(ds, tol)
        i = len(h4) - 1
        close = h4["Close"].iloc[-1]
        ema80 = h4["ema80"].iloc[-1]
        hits = []
        if near_level[i]:
            hits.append("水平線")
        if near_ema80[i]:
            hits.append("4H-EMA80")
        if in_fib[i]:
            hits.append("フィボ")
        dist_to_ema80_pct = (close - ema80) / ema80 * 100

        dist_arr = h1b["ma_dist_pct"].to_numpy()
        streak, armed = 0, False
        for d in dist_arr[-500:]:
            if np.isnan(d):
                streak = 0
                continue
            if abs(d) <= 0.002:
                streak += 1
                if streak >= 3:
                    armed = True
            else:
                streak = 0
                if armed and abs(d) >= 0.005:
                    armed = False
        rev_state = "ARMED" if armed else ("収束中" if streak > 0 else "平常")

        spread = h4["ma_spread_pct"].iloc[-1] * 100
        order = h4["ma_order"].iloc[-1]

        # 日足MAと4時間足MAの乖離(大きすぎる時は押し目買い等をNGとする彼のツイートGd96_zXa0AIsEY_に基づく)
        d_ema20 = h4["d_ema20"].iloc[-1]
        daily_4h_dev_pct = abs(close - d_ema20) / close * 100 if not np.isnan(d_ema20) else None
        overextended = daily_4h_dev_pct is not None and daily_4h_dev_pct >= 1.0

        fn = detect_fractal_neckline(ds)

        return {
            "instrument": name,
            "Step1_日足trend": trend,
            "Step2_節目一致": ",".join(hits) if hits else "なし",
            "4H-EMA80乖離%": round(float(dist_to_ema80_pct), 3),
            "収束拡散状態": rev_state,
            "4Hリボン幅%": round(float(spread), 3),
            "リボン並び": order,
            "日足-4H MA乖離%": round(float(daily_4h_dev_pct), 3) if daily_4h_dev_pct is not None else None,
            "過熱警告(NG)": overextended,
            "フラクタルNL": fn.get("pattern", "none"),
            "フラクタルNL発火": fn.get("broken", None),
            "フラクタルNL_RR": fn.get("est_RR", None),
            "終値": round(float(close), 5),
        }
    except Exception as e:
        return {"instrument": name, "error": str(e)}


if __name__ == "__main__":
    rows = [check(n, c) for n, c in CODES.items()]
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.max_rows", 30)
    print(df.to_string(index=False))
    df.to_csv("env_check_all_result.csv", index=False)
