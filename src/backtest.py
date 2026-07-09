"""
backtest.py  ―― 的中率と回収率の検証

「当てる」だけでなく「儲かるか(回収率100%超えか)」を測るのが本番。
ここでは2つの賭け方を比較する:
  A) 本命作戦  : 各レースで単勝オッズが一番低い馬に賭ける
  B) AI作戦    : 各レースでAIの3着内確率が一番高い馬に賭ける
各レース単勝100円固定。1着なら 100 x 単勝オッズ が払い戻し。
"""

import pandas as pd

STAKE = 100  # 1レースあたりの賭け金(円)


def _evaluate(test: pd.DataFrame, pick_col: str, ascending: bool) -> dict:
    """pick_col が最小(ascending=True)/最大(False)の馬を各レースで1頭選ぶ。"""
    idx = (test.sort_values(pick_col, ascending=ascending)
               .groupby("race_id").head(1).index)
    picks = test.loc[idx]
    n = len(picks)
    payout = (picks["won"] * STAKE * picks["tansho_odds"]).sum()
    spent = n * STAKE
    return {
        "レース数": n,
        "単勝的中率": picks["won"].mean(),       # 選んだ馬が1着だった割合
        "複勝的中率": picks["is_top3"].mean(),   # 選んだ馬が3着内だった割合
        "回収率": payout / spent,                # 払戻 / 賭け金 (1.0=トントン)
    }


def run(test: pd.DataFrame) -> pd.DataFrame:
    fav = _evaluate(test, "tansho_odds", ascending=True)    # 本命作戦
    ai = _evaluate(test, "pred_top3", ascending=False)      # AI作戦
    out = pd.DataFrame({"本命作戦(一番人気)": fav, "AI作戦(予測トップ)": ai}).T
    out["単勝的中率"] = (out["単勝的中率"] * 100).round(1).astype(str) + "%"
    out["複勝的中率"] = (out["複勝的中率"] * 100).round(1).astype(str) + "%"
    out["回収率"] = (out["回収率"] * 100).round(1).astype(str) + "%"
    out["レース数"] = out["レース数"].astype(int)
    return out
