"""
26銘柄のシグナル状態を記録し、前回チェックとの差分(新規発火)を検出する。
mode=summary: 常に全26銘柄の詳細(トレンド・5点スコア・ARMED・フラクタルNL・過熱・乖離埋め)を出力。
              エラーが出た銘柄も隠さず明示する(黙って握りつぶさない)。
mode=diff   : 前回保存した状態と比較し、新規に発火したものだけ出力(KAIRIポイントの即時通知用)
"""
import sys
import json
import os
from env_check_all import CODES, check

STATE_PATH = "signal_state.json"


def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_state(state):
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2, default=str)


def current_signals():
    """戻り値: (rows, errors) — rows=正常取得できた銘柄の状態辞書、errors=失敗した銘柄名→エラー文"""
    rows = {}
    errors = {}
    for name, code in CODES.items():
        r = check(name, code)
        if "error" in r:
            errors[name] = r["error"]
            continue
        rows[name] = {
            "armed": r.get("収束拡散状態") == "ARMED",
            "fractal_broken": bool(r.get("フラクタルNL発火")),
            "fractal_pattern": r.get("フラクタルNL"),
            "fractal_rr": r.get("フラクタルNL_RR"),
            "fractal_quality": r.get("フラクタルNL_型"),
            "overheat": bool(r.get("過熱警告(NG)")),
            "trend": r.get("Step1_日足trend"),
            "close": r.get("終値"),
            "five_score": r.get("5点スコア"),
            "five_dir": r.get("5点方向"),
            "gapfill_valid": bool(r.get("乖離埋め妥当")),
            "gapfill_dir": r.get("乖離埋め方向"),
        }
    return rows, errors


def _five_score_num(s):
    """'4/5' -> 4 のように数値部分だけ取り出す"""
    if not s:
        return None
    try:
        return int(str(s).split("/")[0])
    except (ValueError, IndexError):
        return None


def _instrument_line(name, s):
    flags = []
    if s["armed"]:
        flags.append("ARMED")
    if s["fractal_broken"]:
        flags.append(f"フラクタルNL発火({s['fractal_pattern']},RR={s['fractal_rr']},{s['fractal_quality']})")
    if s["gapfill_valid"]:
        flags.append(f"乖離埋め候補({s['gapfill_dir']})")
    elif s["overheat"]:
        flags.append("過熱警告")
    five_n = _five_score_num(s.get("five_score"))
    if five_n is not None:
        flags.append(f"5点スコア{s['five_score']}({s['five_dir']})")
    flags_str = "/".join(flags) if flags else "平常"
    return f"{name}[{s.get('trend')}]: {flags_str}"


def summarize_all(rows, errors):
    """毎回、全銘柄の詳細を必ず出力する(フィルタして省略しない)。エラーも明示する。"""
    lines = [f"=== 全{len(CODES)}銘柄中 {len(rows)}銘柄 分析成功 / {len(errors)}銘柄 エラー ==="]
    for name in CODES:
        if name in rows:
            lines.append(_instrument_line(name, rows[name]))
        elif name in errors:
            lines.append(f"{name}: [エラー] {errors[name]}")
    return "\n".join(lines)


def diff_new_triggers(prev, cur):
    """前回False→今回Trueに変わった項目だけを新規シグナルとして返す"""
    new_events = []
    for name, s in cur.items():
        p = prev.get(name, {})
        if s["armed"] and not p.get("armed", False):
            new_events.append(f"{name}: 新規ARMED(収束完了・拡散待ち)")
        if s["fractal_broken"] and not p.get("fractal_broken", False):
            new_events.append(f"{name}: フラクタルNL新規発火 {s['fractal_pattern']} RR={s['fractal_rr']} {s['fractal_quality']}")
        cur_five = _five_score_num(s.get("five_score"))
        prev_five = _five_score_num(p.get("five_score"))
        if cur_five is not None and cur_five >= 4 and (prev_five is None or prev_five < 4):
            new_events.append(f"{name}: 5点根拠が{s['five_score']}に到達({s['five_dir']}) — 押し目買い/戻り売りの高確度シグナル")
        if s["gapfill_valid"] and not p.get("gapfill_valid", False):
            new_events.append(f"{name}: 乖離埋めトレード新規候補({s['gapfill_dir']}) — MA乖離埋めNG4パターンをクリア")
    return new_events


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "summary"
    prev = load_state()
    cur, errors = current_signals()

    if mode == "summary":
        print(summarize_all(cur, errors))
    else:
        events = diff_new_triggers(prev, cur)
        if errors:
            events.append(f"[警告] {len(errors)}銘柄でエラー: {', '.join(errors.keys())}")
        if events:
            print("NEW_SIGNALS:\n" + "\n".join(events))
        else:
            print("NO_NEW_SIGNALS")

    save_state(cur)
