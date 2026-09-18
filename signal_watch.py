"""
26銘柄のシグナル状態を記録し、前回チェックとの差分(新規発火)を検出する。
mode=summary: 常に現在の全体サマリーを出力(1時間毎の全体通知用)
mode=diff   : 前回保存した状態と比較し、新規に発火したものだけ出力(KAIRIポイントの即時通知用)
"""
import sys
import json
import os
import pandas as pd
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
    rows = {}
    for name, code in CODES.items():
        r = check(name, code)
        if "error" in r:
            continue
        rows[name] = {
            "armed": r.get("収束拡散状態") == "ARMED",
            "fractal_broken": bool(r.get("フラクタルNL発火")),
            "fractal_pattern": r.get("フラクタルNL"),
            "fractal_rr": r.get("フラクタルNL_RR"),
            "overheat": bool(r.get("過熱警告(NG)")),
            "trend": r.get("Step1_日足trend"),
            "close": r.get("終値"),
        }
    return rows


def summarize_all(cur):
    lines = []
    for name, s in cur.items():
        flags = []
        if s["armed"]:
            flags.append("ARMED")
        if s["fractal_broken"]:
            flags.append(f"フラクタルNL発火({s['fractal_pattern']},RR={s['fractal_rr']})")
        if s["overheat"]:
            flags.append("過熱警告")
        if flags:
            lines.append(f"{name}: {'/'.join(flags)}")
    if not lines:
        return "現在アクティブなシグナルなし"
    return " | ".join(lines)


def diff_new_triggers(prev, cur):
    """前回False→今回Trueに変わった項目だけを新規シグナルとして返す"""
    new_events = []
    for name, s in cur.items():
        p = prev.get(name, {})
        if s["armed"] and not p.get("armed", False):
            new_events.append(f"{name}: 新規ARMED(収束完了・拡散待ち)")
        if s["fractal_broken"] and not p.get("fractal_broken", False):
            new_events.append(f"{name}: フラクタルNL新規発火 {s['fractal_pattern']} RR={s['fractal_rr']}")
    return new_events


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "summary"
    prev = load_state()
    cur = current_signals()

    if mode == "summary":
        print(summarize_all(cur))
    else:
        events = diff_new_triggers(prev, cur)
        if events:
            print("NEW_SIGNALS:\n" + "\n".join(events))
        else:
            print("NO_NEW_SIGNALS")

    save_state(cur)
