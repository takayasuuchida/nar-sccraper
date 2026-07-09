"""
exotic_ev.py  ―― 3連複で「市場の甘い所」を期待値で攻められるか検証

前提: 先に load_kaggle.py で data/races_kaggle.csv を作っておくこと。
使い方:
  python src\\exotic_ev.py "C:\\...\\keiba_data\\19860105-20210731_odds.csv"

考え方:
  ・勝率モデル → ハービル法で「3頭が3着内に入る確率(3連複の的中確率)」を推定
  ・オッズ(市場)の勝率からも同じ確率を出す → 市場の見込み
  ・モデル確率 ÷ 市場確率 = 妙味。市場が過小評価(妙味>1)の買い目だけ買う
  ・実際の3連複払戻(odds.csv)で回収率を検証
  ※ 各組み合わせの“事前オッズ”は持っていないので、市場確率はオッズ由来の勝率から
    ハービルで近似する。当たった時の払戻だけは実データを使う(正確)。
"""

import sys
import os
import itertools
import pandas as pd
import numpy as np

import pipeline

TOP_K = 8          # 各レースで上位何頭から3連複を組むか(計算量を抑える)
STAKE = 100


def trio_prob(pwin) -> float:
    """ハービル法: 勝率pwin(3頭)から『この3頭が1-2-3着を占める確率』(順不同)。"""
    s = 0.0
    for x, y, z in itertools.permutations(pwin):
        d1, d2 = 1 - x, 1 - x - y
        if d1 > 0 and d2 > 0:
            s += x * (y / d1) * (z / d2)
    return s


def load_trifecta_results(path: str) -> dict:
    """odds.csv から レースID -> (結果の3頭set, 3連複払戻[円/100円]) を作る。"""
    cols = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns.tolist()

    def f(name):
        return next((c for c in cols if name in c), None)
    need = {"rid": f("レースID"),
            "c1": f("三連複1_組合せ1"), "c2": f("三連複1_組合せ2"),
            "c3": f("三連複1_組合せ3"), "pay": f("三連複1_オッズ")}
    od = pd.read_csv(path, usecols=[v for v in need.values() if v], encoding="utf-8-sig")
    od = od.rename(columns={v: k for k, v in need.items() if v})
    res = {}
    for r in od.itertuples():
        try:
            trio = frozenset(int(x) for x in (r.c1, r.c2, r.c3))
            pay = float(r.pay)
        except (ValueError, TypeError):
            continue
        if len(trio) == 3 and pay > 0:
            res[str(r.rid)] = (trio, pay)
    return res


def main():
    if len(sys.argv) < 2:
        print('使い方: python src\\exotic_ev.py "....odds.csv"')
        return
    if not os.path.exists("data/races_kaggle.csv"):
        print("❌ data/races_kaggle.csv がありません。先に load_kaggle.py を実行してください。")
        return

    print("=" * 60)
    print(" 3連複で市場の隙を攻める (本物データ)")
    print("=" * 60)

    races = pd.read_csv("data/races_kaggle.csv")
    feat = pipeline.add_features(races)
    train, test = pipeline.time_split(feat, test_frac=0.2)
    test = test.copy()
    print("勝率モデルを学習中...")
    test["raw_win"] = pipeline.fit_predict_prob(train, test, "won")
    test["pred_win"] = test["raw_win"] / test.groupby("race_id")["raw_win"].transform("sum")
    inv = 1.0 / test["tansho_odds"]
    test["mkt_win"] = inv / test.groupby("race_id")["tansho_odds"].transform(lambda s: (1.0 / s).sum())

    print("3連複の払戻データを読み込み中...")
    payouts = load_trifecta_results(sys.argv[1])
    print(f"  払戻データ {len(payouts):,}レースぶん")

    # 賭け方: T0=最有力1点 / T1=妙味しきい値で複数点
    strat = {"T0 最有力の3連複1点": {"thr": None},
             "T1 妙味>=2.0": {"thr": 2.0}, "T1 妙味>=3.0": {"thr": 3.0}}
    acc = {k: {"bets": 0, "hits": 0, "payout": 0.0} for k in strat}

    races_used = 0
    for rid, g in test.groupby("race_id"):
        rid = str(rid)
        if rid not in payouts:
            continue
        result_trio, pay = payouts[rid]
        g = g.sort_values("pred_win", ascending=False).head(TOP_K)
        umab = g["umaban"].astype(int).tolist()
        pmod = dict(zip(umab, g["pred_win"]))
        pmkt = dict(zip(umab, g["mkt_win"]))
        races_used += 1

        combos = []
        for trio in itertools.combinations(umab, 3):
            pm = trio_prob([pmod[h] for h in trio])
            pk = trio_prob([pmkt[h] for h in trio])
            value = pm / pk if pk > 0 else 0
            combos.append((frozenset(trio), pm, value))

        for name, cfg in strat.items():
            if cfg["thr"] is None:                       # 最有力1点
                picks = [max(combos, key=lambda c: c[1])[0]]
            else:                                        # 妙味しきい値
                picks = [c[0] for c in combos if c[2] >= cfg["thr"]]
            acc[name]["bets"] += len(picks)
            if result_trio in picks:
                acc[name]["hits"] += 1
                acc[name]["payout"] += pay

    print(f"  検証 {races_used:,} レース\n")
    rows = {}
    for name, a in acc.items():
        n = a["bets"]
        rows[name] = {
            "買い点数": n,
            "的中数": a["hits"],
            "的中率": f"{(a['hits']/n*100):.2f}%" if n else "—",
            "回収率": f"{(a['payout']/(n*STAKE)*100):.1f}%" if n else "—",
        }
    print(pd.DataFrame(rows).T.to_string())

    print("\n" + "-" * 60)
    print("読み解き:")
    print("  ・3連複の控除率は約27.5%(単勝より重い)。100%超えはさらに難しい。")
    print("  ・『最有力1点』は本命寄り=低配当で回収率は伸びにくい。")
    print("  ・『妙味買い』が100%を超えれば、市場が間違えてる所を突けた証拠。")
    print("    ただし当たりは稀で高分散。安定して勝てるかは別問題。")
    print("-" * 60)


if __name__ == "__main__":
    main()
