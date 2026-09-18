"""
パラメータ化されたバックテストエンジン。
levels(節目)の計算はtol_pctごとにキャッシュし、それ以外のパラメータ変更では再計算しない。
"""
import numpy as np
import pandas as pd
from data import fetch_daily, fetch_1h, resample_4h
from indicators import ema, atr, horizontal_levels, last_swing_leg, fib_zone, swing_points

_LEVELS_CACHE = {}
_HITS_CACHE = {}


def build_dataset(name, daily=None, h1=None):
    daily = fetch_daily(name) if daily is None else daily
    h1 = fetch_1h(name) if h1 is None else h1
    h4 = resample_4h(h1)

    daily = daily.copy()
    daily["ema20"] = ema(daily["Close"], 20)
    daily["ema50"] = ema(daily["Close"], 50)
    daily["atr14"] = atr(daily, 14)
    daily["ema20_dev"] = (daily["Close"] - daily["ema20"]).abs() / daily["Close"]
    daily["trend"] = np.where(daily["ema20"] > daily["ema50"], "up",
                       np.where(daily["ema20"] < daily["ema50"], "down", "flat"))

    h4 = h4.copy()
    h4["ema20"] = ema(h4["Close"], 20)
    h4["ema80"] = ema(h4["Close"], 80)
    h4["ema200"] = ema(h4["Close"], 200)
    h4["atr14"] = atr(h4, 14)
    with np.errstate(invalid="ignore"):
        stack = np.stack([h4["ema20"], h4["ema80"], h4["ema200"]], axis=1)
        h4["ma_spread_pct"] = (stack.max(axis=1) - stack.min(axis=1)) / h4["Close"]
        h4["ma_order"] = np.where((h4["ema20"] > h4["ema80"]) & (h4["ema80"] > h4["ema200"]), "bull",
                          np.where((h4["ema20"] < h4["ema80"]) & (h4["ema80"] < h4["ema200"]), "bear", "mixed"))

    h1 = h1.copy()
    h1["atr14"] = atr(h1, 14)
    h1["ema20"] = ema(h1["Close"], 20)  # 「1HMA」

    legs = last_swing_leg(h4, k=3, lookback=60)

    # 4HMA(ema80)を1時間足に先読みなしでブロードキャスト(確定済みの4H足のみ使用)
    h4_for_h1 = h4[["ema80"]].copy()
    h4_for_h1.index = h4_for_h1.index + pd.Timedelta(hours=4)  # その4H足が閉じた後から使用可能
    h4_for_h1.index.name = "ts"
    h1_idx_named = h1.index.rename("ts")
    left1 = h1.copy()
    left1.index = h1_idx_named
    merged1 = pd.merge_asof(
        left1.reset_index(), h4_for_h1.reset_index(), on="ts", direction="backward"
    ).set_index("ts")
    h1["ema80_4h"] = merged1["ema80"]
    h1["ma_dist_pct"] = (h1["ema20"] - h1["ema80_4h"]) / h1["ema80_4h"]

    # 先読み禁止で日足trend/ema20/dev をh4に asof結合 (前日終値時点の確定値のみ使用)
    daily_shift = daily.copy()
    daily_shift.index = daily_shift.index  # 日足行 = その日の終値確定 (翌日から使用可能)
    daily_for_merge = daily_shift[["trend", "ema20", "ema20_dev"]].copy()
    daily_for_merge = daily_for_merge.rename(columns={"ema20": "d_ema20_src"})
    daily_for_merge.index = daily_for_merge.index + pd.Timedelta(days=1)  # 翌日0時から有効
    daily_for_merge.index.name = "ts"
    h4_idx = h4.index.rename("ts")
    left = h4.copy()
    left.index = h4_idx
    merged = pd.merge_asof(
        left.reset_index(),
        daily_for_merge.reset_index(),
        on="ts", direction="backward"
    ).set_index("ts")
    h4["d_trend"] = merged["trend"]
    h4["d_ema20"] = merged["d_ema20_src"]
    h4["d_ema20_dev"] = merged["ema20_dev"]

    left1b = h1.copy()
    left1b.index = h1_idx_named
    merged1b = pd.merge_asof(
        left1b.reset_index(), daily_for_merge.reset_index(), on="ts", direction="backward"
    ).set_index("ts")
    h1["d_trend"] = merged1b["trend"]  # h1側もema20列名衝突を避けるためd_ema20_srcを使用

    h1_index = h1.index
    h4_index = h4.index

    return {"name": name, "daily": daily, "h1": h1, "h4": h4, "legs": legs,
            "h1_index": h1_index, "h4_index": h4_index}


def get_levels(ds, tol_pct, lookback=250, min_touches=2, k=3):
    key = (ds["name"], tol_pct, lookback, min_touches, k)
    if key not in _LEVELS_CACHE:
        _LEVELS_CACHE[key] = horizontal_levels(ds["h4"], k=k, lookback=lookback,
                                                tol_pct=tol_pct, min_touches=min_touches)
    return _LEVELS_CACHE[key]


DEFAULT_PARAMS = dict(
    tol_pct=0.0015,
    rr_target=2.0,
    stop_atr_mult=0.25,
    trigger_window=6,
    min_hits=2,          # filtered: 2, unfiltered: 1
    trend_mode="filtered",  # filtered / unfiltered / random
    ma_convergence_block=False,   # NYダウ型: 4H ema80と日足ema20が接近しすぎのゾーンでのブレイク追随を禁止
    ma_convergence_tol=0.002,
    breakeven_at_r=None,   # 例: 1.0 => +1Rで建値ストップに移動 (決済ライン移動の再現)
    exit_mode="fixed_rr",  # fixed_rr / partial_be
    entry_style="confirm",  # confirm(1H反転確認を待つ) / immediate(4H節目到達で即エントリー)
    body_mult=1.0,          # confirm用: 反転足の実体をATRの何倍以上要求するか(小さいほど早いエントリー)
    extreme_lookback=2,     # confirm用: 直近何本の高安値更新を要求するか(小さいほど早いエントリー)
    partial_r=1.2,          # partial_be: この倍率で半分利確
    partial_fraction=0.5,   # partial_be: 利確する割合
    converge_tol=0.001,     # ma_reversal: 1HMAと4HMAが「収束」とみなす乖離率
    diverge_tol=0.003,      # ma_reversal: 収束後、この乖離率を超えたら「拡散」=トリガー
    converge_lookback=5,    # ma_reversal: 収束状態を何本分維持している必要があるか
    require_daily_align=False,  # ma_reversal: 日足トレンド方向との一致を要求するか(パターン①相当)
    fan_converge_tol=0.003,  # ma_fan: MA20/80/200の最大乖離幅がこれ以下なら「圧縮」とみなす
    fan_diverge_tol=0.01,    # ma_fan: 圧縮後、この乖離幅を超え、かつ正しい並び順になったらトリガー
    max_hold_bars=90,
    date_start=None,
    date_end=None,
)


def compute_hit_arrays(ds, tol_pct):
    """近接判定を全バー分ベクトル化して事前計算 (tol_pctごとにキャッシュ)"""
    key = (ds["name"], tol_pct)
    if key in _HITS_CACHE:
        return _HITS_CACHE[key]

    h4 = ds["h4"]
    legs = ds["legs"]
    levels = get_levels(ds, tol_pct)
    close_arr = h4["Close"].to_numpy()
    ema80_arr = h4["ema80"].to_numpy()

    near_level = np.zeros(len(h4), dtype=bool)
    for i, (t, close) in enumerate(zip(h4.index, close_arr)):
        lv = levels.get(t)
        if lv:
            near_level[i] = any(abs(close - l) / close <= tol_pct for l in lv)

    with np.errstate(invalid="ignore"):
        near_ema80 = np.abs(close_arr - ema80_arr) / close_arr <= tol_pct
    near_ema80 = np.nan_to_num(near_ema80, nan=False).astype(bool)

    in_fib = np.zeros(len(h4), dtype=bool)
    for i, (t, close) in enumerate(zip(h4.index, close_arr)):
        fz = fib_zone(legs.get(t))
        if fz and fz[0] <= close <= fz[1]:
            in_fib[i] = True

    out = (near_level, near_ema80, in_fib)
    _HITS_CACHE[key] = out
    return out


_MA_REV_CACHE = {}


def compute_ma_reversal_triggers(ds, converge_tol, diverge_tol, converge_lookback):
    """
    「4HMAに対して1HMAが収束→拡散するタイミング」をトレンド転換の機械的トリガーとして検出。
    1) 直近converge_lookback本、1HMAと4HMAの乖離率が converge_tol 以内(=収束/レンジ)
    2) その直後に乖離率が diverge_tol を超えた(=拡散/方向が出た) 瞬間をトリガーとする
    """
    key = (ds["name"], converge_tol, diverge_tol, converge_lookback)
    if key in _MA_REV_CACHE:
        return _MA_REV_CACHE[key]

    dist = ds["h1"]["ma_dist_pct"].to_numpy()
    n = len(dist)
    valid = ~np.isnan(dist)

    # 状態機械: converge_lookback本連続で|dist|<=converge_tolになったら「armed」。
    # armed中に|dist|>=diverge_tolに達した瞬間がトリガー(その後armed解除、再収束を待つ)。
    trigger = np.zeros(n, dtype=bool)
    direction = np.full(n, "", dtype=object)
    streak = 0
    armed = False
    for i in range(n):
        if not valid[i]:
            streak = 0
            continue
        d = dist[i]
        if abs(d) <= converge_tol:
            streak += 1
            if streak >= converge_lookback:
                armed = True
        else:
            streak = 0
            if armed and abs(d) >= diverge_tol:
                trigger[i] = True
                direction[i] = "buy" if d > 0 else "sell"
                armed = False

    out = (trigger, direction)
    _MA_REV_CACHE[key] = out
    return out


def run_ma_reversal(ds, params=None):
    p = {**DEFAULT_PARAMS, **(params or {})}
    h1 = ds["h1"]
    h1_index = ds["h1_index"]

    trigger, direction_arr = compute_ma_reversal_triggers(
        ds, p["converge_tol"], p["diverge_tol"], p["converge_lookback"])

    close_arr = h1["Close"].to_numpy()
    atr_arr = h1["atr14"].to_numpy()
    trend_arr = h1["d_trend"].to_numpy()
    low_arr = h1["Low"].to_numpy()
    high_arr = h1["High"].to_numpy()

    trades = []
    cooldown_until = None

    start_i = 90
    end_i = len(h1) - 1
    if p["date_start"]:
        start_i = max(start_i, h1_index.searchsorted(pd.Timestamp(p["date_start"])))
    if p["date_end"]:
        end_i = min(end_i, h1_index.searchsorted(pd.Timestamp(p["date_end"]), side="right"))

    for i in range(start_i, end_i):
        if not trigger[i]:
            continue
        t = h1_index[i]
        if cooldown_until is not None and t <= cooldown_until:
            continue
        direction = direction_arr[i]

        if p["require_daily_align"]:
            trend = trend_arr[i]
            if trend != direction:
                continue

        entry = close_arr[i]
        a1 = atr_arr[i]
        if np.isnan(a1) or a1 == 0:
            continue
        lo = low_arr[max(0, i - 2):i + 1].min()
        hi = high_arr[max(0, i - 2):i + 1].max()
        if direction == "buy":
            stop = min(lo, entry - a1) - p["stop_atr_mult"] * a1
            risk = entry - stop
        else:
            stop = max(hi, entry + a1) + p["stop_atr_mult"] * a1
            risk = stop - entry
        if risk <= 0:
            continue

        if p["exit_mode"] == "partial_be":
            result = simulate_trade_partial(h1, t, entry, stop, direction, p["max_hold_bars"],
                                             p["partial_r"], p["partial_fraction"], h1_index)
            if result is None:
                continue
            exit_ts, exit_price, r, partial_taken = result
        else:
            target = entry + p["rr_target"] * risk if direction == "buy" else entry - p["rr_target"] * risk
            result = simulate_trade(h1, t, entry, stop, target, direction, p["max_hold_bars"],
                                     p["breakeven_at_r"], h1_index)
            if result is None:
                continue
            exit_ts, exit_price, r = result
            partial_taken = False

        trades.append({
            "instrument": ds["name"], "setup_bar": t, "entry_ts": t, "exit_ts": exit_ts,
            "direction": direction, "trend": trend_arr[i], "hits": "ma_reversal", "R": r,
            "partial_taken": partial_taken, "entry": entry, "stop": stop, "risk": risk,
        })
        cooldown_until = exit_ts

    return pd.DataFrame(trades)


_MA_FAN_CACHE = {}


def compute_ma_fan_triggers(ds, converge_tol, diverge_tol, converge_lookback):
    """
    4時間足チャート上のMA20/MA80/MA200の「収束→扇状に開く(ファン)」形を検出。
    1) 直近converge_lookback本、3本の最大乖離幅(ma_spread_pct)がconverge_tol以内(=リボン圧縮)
    2) その後、spread_pctがdiverge_tolを超え、かつ3本が正しい順番(bull/bear)に並んだ瞬間をトリガーとする
    """
    key = (ds["name"], converge_tol, diverge_tol, converge_lookback)
    if key in _MA_FAN_CACHE:
        return _MA_FAN_CACHE[key]

    h4 = ds["h4"]
    spread = h4["ma_spread_pct"].to_numpy()
    order = h4["ma_order"].to_numpy()
    n = len(spread)
    valid = ~np.isnan(spread)

    trigger = np.zeros(n, dtype=bool)
    direction = np.full(n, "", dtype=object)
    streak = 0
    armed = False
    for i in range(n):
        if not valid[i]:
            streak = 0
            continue
        s = spread[i]
        if s <= converge_tol:
            streak += 1
            if streak >= converge_lookback:
                armed = True
        else:
            streak = 0
            if armed and s >= diverge_tol and order[i] in ("bull", "bear"):
                trigger[i] = True
                direction[i] = "buy" if order[i] == "bull" else "sell"
                armed = False

    out = (trigger, direction)
    _MA_FAN_CACHE[key] = out
    return out


def run_ma_fan(ds, params=None):
    p = {**DEFAULT_PARAMS, **(params or {})}
    h4 = ds["h4"]
    h1_index = ds["h1_index"]
    h4_index = ds["h4_index"]

    trigger, direction_arr = compute_ma_fan_triggers(
        ds, p["fan_converge_tol"], p["fan_diverge_tol"], p["converge_lookback"])

    trades = []
    cooldown_until = None

    start_i = 210  # ema200のウォームアップ
    end_i = len(h4) - 1
    if p["date_start"]:
        start_i = max(start_i, h4_index.searchsorted(pd.Timestamp(p["date_start"])))
    if p["date_end"]:
        end_i = min(end_i, h4_index.searchsorted(pd.Timestamp(p["date_end"]), side="right"))

    for i in range(start_i, end_i):
        if not trigger[i]:
            continue
        t = h4_index[i]
        if cooldown_until is not None and t <= cooldown_until:
            continue
        direction = direction_arr[i]

        trig_ts, trig_bar = find_trigger(ds["h1"], t, direction, p["trigger_window"], h1_index,
                                          p["body_mult"], p["extreme_lookback"])
        if trig_ts is None:
            continue

        entry = trig_bar["Close"]
        a1 = trig_bar["atr14"]
        if np.isnan(a1) or a1 == 0:
            continue
        pos0 = h1_index.searchsorted(trig_ts, side="left")
        recent = ds["h1"].iloc[max(0, pos0 - 2):pos0 + 1]
        if len(recent) == 0:
            continue
        if direction == "buy":
            stop = min(recent["Low"].min(), entry - a1) - p["stop_atr_mult"] * a1
            risk = entry - stop
        else:
            stop = max(recent["High"].max(), entry + a1) + p["stop_atr_mult"] * a1
            risk = stop - entry
        if risk <= 0:
            continue

        if p["exit_mode"] == "partial_be":
            result = simulate_trade_partial(ds["h1"], trig_ts, entry, stop, direction, p["max_hold_bars"],
                                             p["partial_r"], p["partial_fraction"], h1_index)
            if result is None:
                continue
            exit_ts, exit_price, r, partial_taken = result
        else:
            target = entry + p["rr_target"] * risk if direction == "buy" else entry - p["rr_target"] * risk
            result = simulate_trade(ds["h1"], trig_ts, entry, stop, target, direction, p["max_hold_bars"],
                                     p["breakeven_at_r"], h1_index)
            if result is None:
                continue
            exit_ts, exit_price, r = result
            partial_taken = False

        trades.append({
            "instrument": ds["name"], "setup_bar": t, "entry_ts": trig_ts, "exit_ts": exit_ts,
            "direction": direction, "trend": None, "hits": "ma_fan", "R": r,
            "partial_taken": partial_taken, "entry": entry, "stop": stop, "risk": risk,
        })
        cooldown_until = exit_ts

    return pd.DataFrame(trades)


def find_trigger(h1, start_ts, direction, window, h1_index=None, body_mult=1.0, extreme_lookback=2):
    idx = h1_index if h1_index is not None else h1.index
    pos = idx.searchsorted(start_ts, side="right")
    sub = h1.iloc[pos:pos + window]
    for j in range(len(sub)):
        bar = sub.iloc[j]
        a = bar["atr14"]
        if np.isnan(a) or a == 0:
            continue
        body = bar["Close"] - bar["Open"]
        if direction == "buy" and body >= body_mult * a:
            lows = sub["Low"].iloc[max(0, j - extreme_lookback):j + 1]
            if bar["Low"] <= lows.min() + 1e-9:
                return sub.index[j], bar
        if direction == "sell" and -body >= body_mult * a:
            highs = sub["High"].iloc[max(0, j - extreme_lookback):j + 1]
            if bar["High"] >= highs.max() - 1e-9:
                return sub.index[j], bar
    return None, None


def simulate_trade(h1, entry_ts, entry_price, stop, target, direction, max_hold, breakeven_at_r=None, h1_index=None):
    idx = h1_index if h1_index is not None else h1.index
    pos = idx.searchsorted(entry_ts, side="right")
    fwd = h1.iloc[pos:pos + max_hold]
    risk = abs(entry_price - stop)
    cur_stop = stop
    moved_be = False
    for ts, bar in fwd.iterrows():
        if breakeven_at_r is not None and not moved_be:
            if direction == "buy" and bar["High"] >= entry_price + breakeven_at_r * risk:
                cur_stop = max(cur_stop, entry_price)
                moved_be = True
            elif direction == "sell" and bar["Low"] <= entry_price - breakeven_at_r * risk:
                cur_stop = min(cur_stop, entry_price)
                moved_be = True
        if direction == "buy":
            if bar["Low"] <= cur_stop:
                r = (cur_stop - entry_price) / risk
                return ts, cur_stop, r
            if bar["High"] >= target:
                r = (target - entry_price) / risk
                return ts, target, r
        else:
            if bar["High"] >= cur_stop:
                r = (entry_price - cur_stop) / risk
                return ts, cur_stop, r
            if bar["Low"] <= target:
                r = (entry_price - target) / risk
                return ts, target, r
    if len(fwd) == 0:
        return None
    last = fwd.iloc[-1]
    if direction == "buy":
        r = (last["Close"] - entry_price) / risk
    else:
        r = (entry_price - last["Close"]) / risk
    return fwd.index[-1], last["Close"], r


def simulate_trade_partial(h1, entry_ts, entry_price, stop, direction, max_hold,
                            partial_r=1.2, partial_fraction=0.5, h1_index=None):
    """
    1:partial_r まで伸びたら partial_fraction を利確し、残りは建値ストップに引き上げて
    伸ばせるだけ伸ばす(上限ターゲットなし、建値ストップ or タイムアウトで手仕舞い)。
    """
    idx = h1_index if h1_index is not None else h1.index
    pos = idx.searchsorted(entry_ts, side="right")
    fwd = h1.iloc[pos:pos + max_hold]
    risk = abs(entry_price - stop)
    cur_stop = stop
    partial_taken = False
    remaining_frac = 1.0
    realized = 0.0
    partial_target = entry_price + partial_r * risk if direction == "buy" else entry_price - partial_r * risk

    last_ts, last_price = entry_ts, entry_price
    for ts, bar in fwd.iterrows():
        last_ts, last_price = ts, bar["Close"]
        if direction == "buy":
            if bar["Low"] <= cur_stop:
                realized += remaining_frac * (cur_stop - entry_price) / risk
                return ts, cur_stop, realized, partial_taken
            if not partial_taken and bar["High"] >= partial_target:
                realized += partial_fraction * partial_r
                remaining_frac = 1 - partial_fraction
                cur_stop = entry_price
                partial_taken = True
        else:
            if bar["High"] >= cur_stop:
                realized += remaining_frac * (entry_price - cur_stop) / risk
                return ts, cur_stop, realized, partial_taken
            if not partial_taken and bar["Low"] <= partial_target:
                realized += partial_fraction * partial_r
                remaining_frac = 1 - partial_fraction
                cur_stop = entry_price
                partial_taken = True

    if len(fwd) == 0:
        return None
    if direction == "buy":
        r_leg = (last_price - entry_price) / risk
    else:
        r_leg = (entry_price - last_price) / risk
    realized += remaining_frac * r_leg
    return last_ts, last_price, realized, partial_taken


def run(ds, params=None, seed=42):
    p = {**DEFAULT_PARAMS, **(params or {})}
    if p["trend_mode"] == "ma_reversal":
        return run_ma_reversal(ds, p)
    if p["trend_mode"] == "ma_fan":
        return run_ma_fan(ds, p)
    h1, h4, legs = ds["h1"], ds["h4"], ds["legs"]
    h1_index = ds["h1_index"]

    near_level, near_ema80, in_fib = compute_hit_arrays(ds, p["tol_pct"])

    trades = []
    cooldown_until = None
    rng = np.random.default_rng(seed)

    close_arr = h4["Close"].to_numpy()
    ema80_arr = h4["ema80"].to_numpy()
    trend_arr = h4["d_trend"].to_numpy()
    ema20_arr = h4["d_ema20"].to_numpy()
    idx_arr = h4.index

    start_i = 90
    end_i = len(h4) - 1
    if p["date_start"]:
        start_i = max(start_i, idx_arr.searchsorted(pd.Timestamp(p["date_start"])))
    if p["date_end"]:
        end_i = min(end_i, idx_arr.searchsorted(pd.Timestamp(p["date_end"]), side="right"))

    for i in range(start_i, end_i):
        t = idx_arr[i]
        if cooldown_until is not None and t <= cooldown_until:
            continue
        trend = trend_arr[i]
        if trend is None or (isinstance(trend, float) and np.isnan(trend)):
            continue
        close_i = close_arr[i]
        ema80_i = ema80_arr[i]

        if p["trend_mode"] == "random":
            direction = "buy" if rng.random() < 0.5 else "sell"
            hits = ["random"]
        elif p["trend_mode"] == "trend_only":
            # 節目(水平線/MA/フィボ)を一切見ない。トレンド方向だけで機械的にエントリー。
            if trend == "flat":
                continue
            direction = "buy" if trend == "up" else "sell"
            hits = ["trend_only"]
        else:
            hits = []
            if near_level[i]:
                hits.append("horizontal")
            if near_ema80[i]:
                hits.append("ema80")
            if in_fib[i]:
                hits.append("fib")
            if p["trend_mode"] == "filtered":
                if trend == "flat" or len(hits) < p["min_hits"]:
                    continue
                direction = "buy" if trend == "up" else "sell"
            else:  # unfiltered
                if len(hits) < 1:
                    continue
                direction = "sell" if close_i < ema80_i else "buy"

            if p["ma_convergence_block"]:
                # 実際の彼のツイート(Gd96_zXa0AIsEY_)によれば、日足MAと4時間足MAが
                # "大幅に乖離"している時こそ、乖離を埋める(=収束する)反対方向の動きが
                # 出やすいため、その状態での順張り(押し目買い等)は避けるべき、という内容。
                # つまり乖離が小さい時ではなく大きい時にブロックするのが正しい。
                ema20_i = ema20_arr[i]
                if np.isnan(ema20_i):
                    continue
                dev = abs(ema80_i - ema20_i) / close_i
                if dev >= p["ma_convergence_tol"]:
                    continue

        if p["entry_style"] == "immediate":
            # 1時間足の反転確認を待たず、4時間足が節目に到達した瞬間にそのバーの終値で即エントリー
            trig_ts = t
            entry = close_i
            a1 = h4["atr14"].to_numpy()[i]
            if np.isnan(a1) or a1 == 0:
                continue
            h4_recent = h4.iloc[max(0, i - 2):i + 1]
            recent_lows = h4_recent["Low"]
            recent_highs = h4_recent["High"]
        else:
            trig_ts, trig_bar = find_trigger(ds["h1"], t, direction, p["trigger_window"], h1_index,
                                              p["body_mult"], p["extreme_lookback"])
            if trig_ts is None:
                continue
            entry = trig_bar["Close"]
            a1 = trig_bar["atr14"]
            if np.isnan(a1) or a1 == 0:
                continue
            pos0 = h1_index.searchsorted(trig_ts, side="left")
            recent = ds["h1"].iloc[max(0, pos0 - 2):pos0 + 1]
            recent_lows = recent["Low"]
            recent_highs = recent["High"]
            if len(recent) == 0:
                continue
        if direction == "buy":
            stop = min(recent_lows.min(), entry - a1) - p["stop_atr_mult"] * a1
            risk = entry - stop
        else:
            stop = max(recent_highs.max(), entry + a1) + p["stop_atr_mult"] * a1
            risk = stop - entry
        if risk <= 0:
            continue

        partial_taken = False
        if p["exit_mode"] == "partial_be":
            result = simulate_trade_partial(ds["h1"], trig_ts, entry, stop, direction,
                                             p["max_hold_bars"], p["partial_r"],
                                             p["partial_fraction"], h1_index)
            if result is None:
                continue
            exit_ts, exit_price, r, partial_taken = result
        else:
            target = entry + p["rr_target"] * risk if direction == "buy" else entry - p["rr_target"] * risk
            result = simulate_trade(ds["h1"], trig_ts, entry, stop, target, direction,
                                     p["max_hold_bars"], p["breakeven_at_r"], h1_index)
            if result is None:
                continue
            exit_ts, exit_price, r = result

        trades.append({
            "instrument": ds["name"], "setup_bar": t, "entry_ts": trig_ts, "exit_ts": exit_ts,
            "direction": direction, "trend": trend, "hits": ",".join(hits), "R": r,
            "partial_taken": partial_taken,
            "entry": entry, "stop": stop, "risk": risk,
        })
        cooldown_until = exit_ts

    return pd.DataFrame(trades)


def summarize(df):
    if len(df) == 0:
        return dict(n=0, win_rate=np.nan, avg_r=np.nan, pf=np.nan, dd=np.nan, total_R=0.0, se=np.nan, tstat=np.nan)
    n = len(df)
    win_rate = (df["R"] > 0).mean()
    avg_r = df["R"].mean()
    std_r = df["R"].std(ddof=1) if n > 1 else np.nan
    se = std_r / np.sqrt(n) if n > 1 else np.nan
    tstat = avg_r / se if se and se > 0 else np.nan
    equity = df["R"].cumsum()
    dd = (equity - equity.cummax()).min()
    gross_win = df.loc[df["R"] > 0, "R"].sum()
    gross_loss = -df.loc[df["R"] < 0, "R"].sum()
    pf = gross_win / gross_loss if gross_loss > 0 else np.inf
    return dict(n=n, win_rate=win_rate, avg_r=avg_r, pf=pf, dd=dd, total_R=equity.iloc[-1] if n else 0.0,
                se=se, tstat=tstat)


def detect_fractal_neckline(ds, upper_tol_pct=0.002, lower_k=2, lookback_bars=60):
    """
    Xの投稿(status 2041350258689552886)で説明されているフラクタル・ネックライン手法の現状検出。

    上位足(4H)で「ダブルトップ/高値切り下げ」(売り側)または「ダブルボトム/安値切り上げ」
    (買い側)の"形成中"を検出し、その頂点/底値付近の下位足(1H)で小さいネックラインが
    あるかどうか、そのネックラインを既に割っている(=シグナル点灯)かをチェックする。

    彼の主張: 上位足のネックライン(例のE安値)までは最低限リワードが見込め、
    下位足の小さいスイングを損切り幅にできるため、理論上おおよそ1:3程度のRRになりやすい。
    """
    h4 = ds["h4"]
    h1 = ds["h1"]
    is_high4, is_low4 = swing_points(h4, k=3)
    highs4 = h4.loc[is_high4, "High"]
    lows4 = h4.loc[is_low4, "Low"]

    result = {"pattern": "none"}
    recency_cutoff = h4.index[-1] - pd.Timedelta(hours=4 * 30)  # 直近30本(=5日)以内の頂点/底のみ「現在進行形」とみなす

    # --- 売り側: 上位足の高値切り下げ/ダブルトップ ---
    if len(highs4) >= 2 and highs4.index[-1] >= recency_cutoff:
        last_h, prev_h = highs4.iloc[-1], highs4.iloc[-2]
        last_h_t = highs4.index[-1]
        is_topping = last_h <= prev_h * (1 + upper_tol_pct * 2)
        # 直近上位足安値(=上位足ネックライン、最低限のリワード目標)
        recent_low4 = lows4[lows4.index < last_h_t]
        upper_neckline = recent_low4.iloc[-1] if len(recent_low4) else None

        # 三尊/ダブルトップの「型」判定(X投稿 status 2056267605250589170系):
        # 右肩の押し安値(L(n-1)=upper_neckline)が、その前の押し安値(L(n-2))を
        # 既に割っているか(=タイプ②/優位性高)、まだ割っていないか(=タイプ①/優位性低)
        pattern_quality = None
        if upper_neckline is not None:
            earlier_low4 = lows4[lows4.index < recent_low4.index[-1]] if len(recent_low4) else pd.Series(dtype=float)
            if len(earlier_low4):
                prior_pullback_low = earlier_low4.iloc[-1]
                pattern_quality = "②高優位性(押し安値を既に割っている)" if upper_neckline < prior_pullback_low \
                    else "①低優位性(押し安値をまだ割っていない)"

        if is_topping and upper_neckline is not None:
            h1_seg = h1[h1.index >= last_h_t - pd.Timedelta(hours=lookback_bars)]
            if len(h1_seg) >= 10:
                ih1, il1 = swing_points(h1_seg, k=lower_k)
                mini_highs = h1_seg.loc[ih1, "High"]
                mini_lows = h1_seg.loc[il1, "Low"]
                if len(mini_highs) >= 1 and len(mini_lows) >= 1:
                    mini_peak_t = mini_highs.index[-1]
                    mini_neckline_series = mini_lows[mini_lows.index < mini_peak_t]
                    if len(mini_neckline_series):
                        mini_neckline = mini_neckline_series.iloc[-1]
                        cur_price = h1["Close"].iloc[-1]
                        broken = cur_price < mini_neckline
                        risk = mini_highs.iloc[-1] - mini_neckline
                        reward = mini_neckline - upper_neckline
                        rr = reward / risk if risk > 0 else np.nan
                        result = {
                            "pattern": "fractal_topping(sell)",
                            "upper_swing_high": float(last_h), "upper_neckline(target)": float(upper_neckline),
                            "mini_neckline(trigger)": float(mini_neckline), "mini_swing_high(stop側)": float(mini_highs.iloc[-1]),
                            "current_price": float(cur_price), "broken": bool(broken),
                            "est_RR": round(float(rr), 2) if not np.isnan(rr) else None,
                            "pattern_quality": pattern_quality,
                        }
                        valid_rr = (not np.isnan(rr)) and rr > 0
                        if broken and valid_rr:
                            return result
                        if not (valid_rr):
                            result = {"pattern": "none"}

    # --- 買い側: 上位足の安値切り上げ/ダブルボトム ---
    if len(lows4) >= 2 and lows4.index[-1] >= recency_cutoff:
        last_l, prev_l = lows4.iloc[-1], lows4.iloc[-2]
        last_l_t = lows4.index[-1]
        is_bottoming = last_l >= prev_l * (1 - upper_tol_pct * 2)
        recent_high4 = highs4[highs4.index < last_l_t]
        upper_neckline = recent_high4.iloc[-1] if len(recent_high4) else None

        pattern_quality = None
        if upper_neckline is not None:
            earlier_high4 = highs4[highs4.index < recent_high4.index[-1]] if len(recent_high4) else pd.Series(dtype=float)
            if len(earlier_high4):
                prior_rally_high = earlier_high4.iloc[-1]
                pattern_quality = "②高優位性(戻り高値を既に超えている)" if upper_neckline > prior_rally_high \
                    else "①低優位性(戻り高値をまだ超えていない)"

        if is_bottoming and upper_neckline is not None:
            h1_seg = h1[h1.index >= last_l_t - pd.Timedelta(hours=lookback_bars)]
            if len(h1_seg) >= 10:
                ih1, il1 = swing_points(h1_seg, k=lower_k)
                mini_highs = h1_seg.loc[ih1, "High"]
                mini_lows = h1_seg.loc[il1, "Low"]
                if len(mini_lows) >= 1 and len(mini_highs) >= 1:
                    mini_trough_t = mini_lows.index[-1]
                    mini_neckline_series = mini_highs[mini_highs.index < mini_trough_t]
                    if len(mini_neckline_series):
                        mini_neckline = mini_neckline_series.iloc[-1]
                        cur_price = h1["Close"].iloc[-1]
                        broken = cur_price > mini_neckline
                        risk = mini_neckline - mini_lows.iloc[-1]
                        reward = upper_neckline - mini_neckline
                        rr = reward / risk if risk > 0 else np.nan
                        buy_result = {
                            "pattern": "fractal_bottoming(buy)",
                            "upper_swing_low": float(last_l), "upper_neckline(target)": float(upper_neckline),
                            "mini_neckline(trigger)": float(mini_neckline), "mini_swing_low(stop側)": float(mini_lows.iloc[-1]),
                            "current_price": float(cur_price), "broken": bool(broken),
                            "est_RR": round(float(rr), 2) if not np.isnan(rr) else None,
                            "pattern_quality": pattern_quality,
                        }
                        valid_rr = (not np.isnan(rr)) and rr > 0
                        if broken and valid_rr:
                            return buy_result
                        if not broken and valid_rr and result["pattern"] == "none":
                            result = buy_result

    return result


_DAILY_LEVELS_CACHE = {}


def compute_daily_levels(ds, tol_pct=0.003, lookback=100, min_touches=2):
    """日足レベルの水平線(抵抗勢力の把握用)。horizontal_levelsを日足に適用するだけ。"""
    key = (ds["name"], tol_pct, lookback, min_touches)
    if key not in _DAILY_LEVELS_CACHE:
        _DAILY_LEVELS_CACHE[key] = horizontal_levels(ds["daily"], k=3, lookback=lookback,
                                                       tol_pct=tol_pct, min_touches=min_touches)
    return _DAILY_LEVELS_CACHE[key]


def check_five_points(ds, direction=None, tol_pct=0.0015, target_atr_mult=8.0):
    """
    「負けにくいエントリー根拠5選」(X投稿 status 2093185466938130897)を現状の市場にそのまま適用する。
    ①4時間足安値切り上げ(高値切り下げ)ポイント内での1時間足のトレンド転換
    ②4時間足MAに対する1時間足MAの収束→拡散ポイント
    ③直近安値(高値)が水平ラインに支えられている
    ④直近安値(高値)が4時間足MAに支えられている
    ⑤近くに(日足レベルの)抵抗勢力がいない
    """
    daily, h4, h1 = ds["daily"], ds["h4"], ds["h1"]
    trend = daily["trend"].iloc[-1]
    if direction is None:
        if trend == "up":
            direction = "buy"
        elif trend == "down":
            direction = "sell"
        else:
            return {"score": 0, "max": 5, "direction": None, "detail": {}, "note": "日足トレンドがflatのため判定不可"}

    close = float(h4["Close"].iloc[-1])
    atr14_h4 = float(h4["atr14"].iloc[-1])

    # ①4H高安値の切り上げ/切り下げ構造 + 1Hの直近での転換(1HMAが4HMAに対し正しい側へ動き出しているか)
    is_high4, is_low4 = swing_points(h4, k=3)
    swings4 = h4.loc[is_low4, "Low"] if direction == "buy" else h4.loc[is_high4, "High"]
    cond1_structure = False
    if len(swings4) >= 2:
        cond1_structure = (swings4.iloc[-1] > swings4.iloc[-2]) if direction == "buy" \
            else (swings4.iloc[-1] < swings4.iloc[-2])

    dist_arr = h1["ma_dist_pct"].to_numpy()
    recent_dist = dist_arr[-5:]
    valid_recent = recent_dist[~np.isnan(recent_dist)]
    cond1_1h_turn = False
    if len(valid_recent) >= 2:
        cond1_1h_turn = bool(valid_recent[-1] > 0 and valid_recent[-1] > valid_recent[0]) if direction == "buy" \
            else bool(valid_recent[-1] < 0 and valid_recent[-1] < valid_recent[0])
    cond1 = bool(cond1_structure and cond1_1h_turn)

    # ②4HMAに対する1HMAの収束→拡散(直近20本(1H)以内に発火したか)
    trig, trig_dir = compute_ma_reversal_triggers(ds, 0.002, 0.005, 3)
    want_dir = "buy" if direction == "buy" else "sell"
    recent_trig = trig[-20:]
    recent_trig_dir = trig_dir[-20:]
    cond2 = bool(np.any(recent_trig & (recent_trig_dir == want_dir)))

    # ③④節目サポート(直近4H足が水平線/4H-EMA80に接触しているか)
    near_level, near_ema80, in_fib = compute_hit_arrays(ds, tol_pct)
    cond3 = bool(near_level[-1])
    cond4 = bool(near_ema80[-1])

    # ⑤近くに日足レベルの抵抗勢力がいない
    d_levels_map = compute_daily_levels(ds)
    last_daily_t = daily.index[-1]
    d_levels = d_levels_map.get(last_daily_t, [])
    target_dist = atr14_h4 * target_atr_mult
    if direction == "buy":
        target_price = close + target_dist
        blocking = [round(float(lv), 5) for lv in d_levels if close < lv <= target_price]
    else:
        target_price = close - target_dist
        blocking = [round(float(lv), 5) for lv in d_levels if target_price <= lv < close]
    cond5 = len(blocking) == 0

    score = int(cond1) + int(cond2) + int(cond3) + int(cond4) + int(cond5)
    return {
        "score": score, "max": 5, "direction": direction,
        "detail": {
            "①4H構造+1H転換": cond1,
            "②MA収束拡散": cond2,
            "③水平線サポート": cond3,
            "④4H-MAサポート": cond4,
            "⑤抵抗勢力なし": cond5,
        },
        "blocking_daily_levels": blocking,
    }


def check_gap_fill_trade(ds, direction=None, min_daily_dist_atr=3.0, min_4h80_dist_atr=2.0,
                          target_atr_mult=6.0, wave_lookback=15, max_wave_atr=3.0):
    """
    「日足MAまでの乖離埋めトレードNGパターン4選」(X投稿 status 2050830092549722127)による
    MA乖離埋め(平均回帰)トレードの妥当性判定。過熱警告が出た時に「本当にリバウンドを狙えるか」を判定する。

    direction未指定なら現在の乖離方向から自動判定(価格が両MAより上=sell、下=buy)。
    ①②日足MA・4H-EMA80までの距離(ATR倍数)が十分あるか
    ③日足レベルの強いライン(節目)がターゲットまでの間にないか
    ④直近の値動き(第1波)がすでに大きく伸びていないか
    """
    daily, h4 = ds["daily"], ds["h4"]
    close = float(h4["Close"].iloc[-1])
    d_ema20 = float(daily["ema20"].iloc[-1])
    ema80_4h = float(h4["ema80"].iloc[-1])
    atr14_h4 = float(h4["atr14"].iloc[-1])
    if atr14_h4 <= 0 or np.isnan(atr14_h4):
        return {"valid": False, "direction": None, "reason": "ATR計算不可"}

    if direction is None:
        if close > d_ema20 and close > ema80_4h:
            direction = "sell"
        elif close < d_ema20 and close < ema80_4h:
            direction = "buy"
        else:
            return {"valid": False, "direction": None, "reason": "MAが入り組んでおり乖離方向が不明瞭"}

    dist_daily = abs(close - d_ema20)
    dist_4h80 = abs(close - ema80_4h)
    cond_daily_room = dist_daily >= min_daily_dist_atr * atr14_h4
    cond_4h80_room = dist_4h80 >= min_4h80_dist_atr * atr14_h4

    d_levels_map = compute_daily_levels(ds)
    last_daily_t = daily.index[-1]
    d_levels = d_levels_map.get(last_daily_t, [])
    target_dist = atr14_h4 * target_atr_mult
    if direction == "sell":
        target_price = close - target_dist
        blocking = [round(float(lv), 5) for lv in d_levels if target_price <= lv < close]
    else:
        target_price = close + target_dist
        blocking = [round(float(lv), 5) for lv in d_levels if close < lv <= target_price]
    cond_no_daily_line = len(blocking) == 0

    recent_close = h4["Close"].iloc[-wave_lookback:]
    wave_size = float(recent_close.max() - close) if direction == "sell" else float(close - recent_close.min())
    cond_not_extended = wave_size <= max_wave_atr * atr14_h4

    ng_patterns = []
    if not cond_daily_room:
        ng_patterns.append("①日足MAまでの距離不足")
    if not cond_4h80_room:
        ng_patterns.append("②4H-EMA80までの距離不足")
    if not cond_no_daily_line:
        ng_patterns.append("③日足レベルの強いラインあり")
    if not cond_not_extended:
        ng_patterns.append("④第1波がすでに大幅に伸びている")

    return {
        "valid": len(ng_patterns) == 0, "direction": direction,
        "dist_daily_atr": round(dist_daily / atr14_h4, 2), "dist_4h80_atr": round(dist_4h80 / atr14_h4, 2),
        "wave_size_atr": round(wave_size / atr14_h4, 2),
        "ng_patterns": ng_patterns, "blocking_daily_levels": blocking,
    }
