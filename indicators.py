"""インジケーター・スイングポイント・節目(水平線)・フィボナッチ計算"""
import numpy as np
import pandas as pd


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def atr(df, period=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def swing_points(df, k=3):
    """fractal方式: 前後k本より高い/安い足をスイング高値/安値とする"""
    high, low = df["High"], df["Low"]
    is_high = pd.Series(True, index=df.index)
    is_low = pd.Series(True, index=df.index)
    for i in range(1, k + 1):
        is_high &= (high > high.shift(i)) & (high > high.shift(-i))
        is_low &= (low < low.shift(i)) & (low < low.shift(-i))
    return is_high.fillna(False), is_low.fillna(False)


def horizontal_levels(df, k=3, lookback=250, tol_pct=0.0015, min_touches=2):
    """
    直近lookback本のスイング高安値をクラスタリングし、
    tol_pct以内に min_touches 回以上反応した価格帯を「有効な節目」として返す。
    戻り値: 各バー時点で有効な節目レベルのリスト(過去データのみ使用=先読みなし)
    """
    is_high, is_low = swing_points(df, k=k)
    swing_prices = df["High"][is_high].tolist() + df["Low"][is_low].tolist()
    swing_idx = df.index[is_high].tolist() + df.index[is_low].tolist()
    pts = sorted(zip(swing_idx, swing_prices))

    levels_over_time = {}
    active_swings = []  # (time, price)
    for t in df.index:
        # このバーまでに確定したスイングだけ使う(未来データ禁止)
        while pts and pts[0][0] <= t:
            active_swings.append(pts.pop(0))
        recent = [p for (ti, p) in active_swings[-lookback:]]
        levels = []
        used = [False] * len(recent)
        for i, p in enumerate(recent):
            if used[i]:
                continue
            cluster = [p]
            used[i] = True
            for j in range(i + 1, len(recent)):
                if used[j]:
                    continue
                if abs(recent[j] - p) / p <= tol_pct:
                    cluster.append(recent[j])
                    used[j] = True
            if len(cluster) >= min_touches:
                levels.append(float(np.mean(cluster)))
        levels_over_time[t] = levels
    return levels_over_time


def last_swing_leg(df, k=3, lookback=60):
    """直近の主要スイングレッグ(高値-安値 or 安値-高値)を各バー時点で返す(先読みなし)"""
    is_high, is_low = swing_points(df, k=k)
    swing_idx = df.index[is_high | is_low]
    swing_type = np.where(is_high[swing_idx], "H", "L")
    swings = list(zip(swing_idx, swing_type, df.loc[swing_idx, "High"].where(is_high[swing_idx], df.loc[swing_idx, "Low"])))

    result = {}
    confirmed = []
    si = 0
    swings_sorted = sorted(swings, key=lambda x: x[0])
    for t in df.index:
        while si < len(swings_sorted) and swings_sorted[si][0] <= t:
            confirmed.append(swings_sorted[si])
            si += 1
        recent = confirmed[-lookback:]
        leg = None
        for i in range(len(recent) - 1, 0, -1):
            if recent[i][1] != recent[i - 1][1]:
                leg = (recent[i - 1], recent[i])  # (start, end)
                break
        result[t] = leg
    return result


def fib_zone(leg):
    """legから38.2%-61.8%の価格帯を返す (low, high)"""
    if leg is None:
        return None
    (_, _, p0), (_, _, p1) = leg
    lo, hi = min(p0, p1), max(p0, p1)
    rng = hi - lo
    if p1 > p0:  # 安値→高値 (上昇レッグ) => 押し目ゾーンは上から38.2-61.8%戻し
        f382 = hi - rng * 0.382
        f618 = hi - rng * 0.618
    else:  # 高値→安値 (下降レッグ) => 戻りゾーン
        f382 = lo + rng * 0.382
        f618 = lo + rng * 0.618
    return (min(f382, f618), max(f382, f618))
