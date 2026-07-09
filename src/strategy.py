"""
strategy.py  ―― 回収率を上げるための「賭け方」の実験

ポイント: 全レース・全頭を機械的に買うと、控除率20%のぶん回収率は構造的に負ける。
そこで「期待値が高い馬だけ」「自信のあるレースだけ」に絞って買うと、買い点数は減るが
回収率(=儲け効率)が上がりうる。当てる力(モデル)より、この『どこで張るか』が本番。

  期待値 EV = 予測勝率 × 単勝オッズ
    EV > 1.0 なら「理論上プラス」。ただし予測勝率が当たっている前提。

3つの賭け方を比較する:
  S0: 全レースで本命(AI予測トップ)を機械的に単勝買い   … 基準
  S1: 期待値しきい値で選ぶ (EV >= ev_min の馬だけ買う)
  S2: 自信のあるレースだけ (本命の3着内確率 >= conf_min のレースのみ本命を買う)
"""

import numpy as np
import pandas as pd

import generate_data
import pipeline

STAKE = 100


def _recovery(picks: pd.DataFrame) -> dict:
    if len(picks) == 0:
        return {"買い点数": 0, "平均オッズ": float("nan"),
                "的中率": float("nan"), "回収率": float("nan")}
    payout = (picks["won"] * STAKE * picks["tansho_odds"]).sum()
    return {
        "買い点数": len(picks),
        "平均オッズ": picks["tansho_odds"].mean(),
        "的中率": picks["won"].mean(),
        "回収率": payout / (len(picks) * STAKE),
    }


def build_predictions():
    """テスト区間に『3着内確率』と『勝率』の両方を付けて返す。"""
    df = pipeline.add_features(generate_data.make_dataset(3000))
    train, test = pipeline.time_split(df, test_frac=0.25)
    test = test.copy()
    test["pred_top3"] = pipeline.fit_predict_prob(train, test, "is_top3")
    test["raw_win"] = pipeline.fit_predict_prob(train, test, "won")
    # 勝率はレース内で合計1になるよう正規化(1着は1頭だけ → 確率として素直)
    race_sum = test.groupby("race_id")["raw_win"].transform("sum")
    test["pred_win"] = test["raw_win"] / race_sum
    test["EV"] = test["pred_win"] * test["tansho_odds"]   # 期待値 = 勝率 × オッズ
    return test


def kelly_sim(test: pd.DataFrame, ev_min: float = 1.1, odds_cap: float = 20.0,
              frac: float = 0.25, cap: float = 0.05, bankroll0: float = 10000.0) -> dict:
    """ケリー基準で賭け金を決めて資金推移をシミュレート。
    f* = (p*b - (1-p)) / b   (b=オッズ-1, p=予測勝率)。
    分数ケリー(frac)と上限(cap)で現実的に抑える。時系列順に1点ずつ賭ける。
    ※ edgeが本当にプラスなら資金は増え、マイナスなら減る。賭け方は edge を作らない。"""
    picks = test[(test["EV"] >= ev_min) & (test["tansho_odds"] <= odds_cap)].copy()
    picks = picks.sort_values(["date", "race_id"]) if "date" in picks else picks
    bk = peak = bankroll0
    maxdd = 0.0
    for r in picks.itertuples():
        p, o = r.pred_win, r.tansho_odds
        b = o - 1.0
        f = (p * b - (1 - p)) / b                 # フルケリー
        f = min(max(f, 0.0), cap / frac) * frac   # 0未満は賭けない・上限・分数ケリー
        stake = bk * f
        bk += stake * b if r.won else -stake
        peak = max(peak, bk)
        maxdd = max(maxdd, (peak - bk) / peak if peak > 0 else 0)
        if bk <= 0:
            bk = 0
            break
    return {"買い点数": len(picks), "開始資金": bankroll0, "最終資金": round(bk),
            "増減": f"{(bk/bankroll0-1)*100:+.1f}%", "最大ドローダウン": f"{maxdd*100:.1f}%"}


def add_ev(test: pd.DataFrame) -> pd.DataFrame:
    """学習済みの予測がある test に勝率の正規化と期待値(EV)を付ける。"""
    test = test.copy()
    race_sum = test.groupby("race_id")["raw_win"].transform("sum")
    test["pred_win"] = test["raw_win"] / race_sum
    test["EV"] = test["pred_win"] * test["tansho_odds"]
    return test


def compare_strategies(test: pd.DataFrame) -> pd.DataFrame:
    """S0/S1/S1'/S2 の賭け方を比較した表を返す(合成・実データ共通で使う)。"""
    s0 = test.sort_values("pred_top3", ascending=False).groupby("race_id").head(1)
    rows = {"S0 全レース本命買い": _recovery(s0)}
    for ev_min in [1.1, 1.3]:
        rows[f"S1 期待値>={ev_min:.1f} (全オッズ)"] = _recovery(test[test["EV"] >= ev_min])
    capped = test[(test["EV"] >= 1.1) & (test["tansho_odds"] <= 20)]
    rows["S1' 期待値>=1.1 かつ オッズ<=20倍"] = _recovery(capped)
    for conf in [0.6, 0.7]:
        conf_races = test.groupby("race_id")["pred_top3"].transform("max") >= conf
        picks = (test[conf_races].sort_values("pred_top3", ascending=False)
                 .groupby("race_id").head(1))
        rows[f"S2 本命の3着内率>={conf:.0%}のレースのみ"] = _recovery(picks)

    out = pd.DataFrame(rows).T
    out["買い点数"] = out["買い点数"].astype(int)
    out["平均オッズ"] = out["平均オッズ"].round(1)
    out["的中率"] = (out["的中率"] * 100).round(1).astype(str) + "%"
    out["回収率"] = (out["回収率"] * 100).round(1).astype(str) + "%"
    return out


def main():
    print("=" * 60)
    print(" 回収率を上げる賭け方の実験 (合成データ)")
    print("=" * 60)
    test = build_predictions()
    n_races = test.race_id.nunique()
    print(f"検証 {n_races} レース / 出走 {len(test)} 頭\n")

    print(compare_strategies(test).to_string())

    print("\n" + "-" * 60)
    print("読み解き (正直な結論):")
    print("  ・S0/S2 が約80%=控除率ぶんきっちり負ける。『本命をただ買う』『自信ある")
    print("    レースを選ぶ』だけでは100%は超えない。市場(オッズ)はそれだけ正確。")
    print("  ・100%を超える唯一の道は『自分の予測勝率 > オッズが示す勝率』の馬を買う")
    print("    こと(S1の期待値買い)。ただし対象は人気薄に偏り、買い点数も少なく、")
    print("    回収率は大きくブレる(平均オッズと的中率を見れば穴狙いの賭けと分かる)。")
    print("  ・つまり鍵は『当てる』より『市場が間違えている所を見つける』。これは難しく、")
    print("    実データの市場はこの合成データよりさらに効率的で、もっと厳しい。")
    print("-" * 60)


if __name__ == "__main__":
    main()
