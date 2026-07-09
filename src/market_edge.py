"""
market_edge.py  ―― 「プラスEVは本当にどこかに存在するか」を実データで検証

『買い方の工夫で勝てる』という主張を数学で検証する。モデル不要、純粋な集計。

  ① オッズ帯ごとの回収率
     → 本命・大穴バイアスの実測。100%を超える帯が実在するか?
  ② リベート(賭け額還元)の損益分岐
     → 何%還元されれば回収率が100%を超えるか?(プロが使う構造的な edge)

前提: data/races_kaggle.csv (load_kaggle.py で作成済み)
使い方: python src\\market_edge.py
"""

import os
import sys
import numpy as np
import pandas as pd

BINS = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0, 1e9]
LABELS = ["1.0-1.5", "1.5-2.0", "2.0-3.0", "3.0-5.0", "5.0-10",
          "10-20", "20-50", "50-100", "100+"]

TRACK_NAMES = {30: "門別", 35: "盛岡", 36: "水沢", 42: "浦和", 43: "船橋",
               44: "大井", 45: "川崎", 46: "金沢", 47: "笠松", 48: "名古屋",
               50: "園田", 51: "姫路", 54: "高知", 55: "佐賀"}


def _by_track(df: pd.DataFrame):
    """race_id の5〜6桁目(場コード)で場ごとに『全頭/1番人気』回収率を横並び比較。
    複数場のデータが混ざっているときだけ意味があるので、2場以上で表示する。"""
    rid = df["race_id"].astype("int64").astype(str).str.zfill(12)
    df = df.assign(track=rid.str[4:6].astype(int))
    if df["track"].nunique() < 2:
        return
    rows = []
    for tcode, g in df.groupby("track"):
        overall = (g["won"] * g["tansho_odds"]).sum() / len(g)
        fav = g[g["tansho_odds"] == g.groupby("race_id")["tansho_odds"].transform("min")]
        fav_rec = (fav["won"] * fav["tansho_odds"]).sum() / len(fav)
        rows.append({
            "場": f"{tcode:02d}{TRACK_NAMES.get(tcode, '')}",
            "レース数": g["race_id"].nunique(),
            "全頭回収率": f"{overall*100:.1f}%",
            "1番人気回収率": f"{fav_rec*100:.1f}%",
            "必要リベート": f"{max(0, 1 - fav_rec)*100:.1f}%",
        })
    out = pd.DataFrame(rows).set_index("場")
    print("\n" + "-" * 64)
    print("③ 場ごとの横並び比較(薄い市場ほど歪みは残るか?):")
    print(out.to_string())
    print("  ・『1番人気回収率』が場によってどれだけ違うかに注目。")
    print("    どの場でも95%前後なら、それは偶然でなく“構造”。")
    print("  ・必要リベートが小さい場ほど、現実的にプラスへ反転させやすい。")


def main():
    # 既定は中央(Kaggle)。引数でNAR等の別CSVを指定可能。
    path = sys.argv[1] if len(sys.argv) > 1 else "data/races_kaggle.csv"
    if not os.path.exists(path):
        print(f"❌ {path} がありません。load_kaggle.py か scrape で作ってください。")
        return
    df = pd.read_csv(path)
    print("=" * 64)
    print(" プラスEVは存在するか? 単勝オッズ帯ごとの回収率 (本物データ)")
    print("=" * 64)
    print(f"対象: {len(df):,}頭 / {df.race_id.nunique():,}レース\n")

    df["bin"] = pd.cut(df["tansho_odds"], bins=BINS, labels=LABELS, right=False)
    df["payout"] = df["won"] * df["tansho_odds"]    # 単勝100円→払戻=100*オッズ(1着のみ)
    g = df.groupby("bin", observed=True)
    tbl = pd.DataFrame({
        "頭数": g.size(),
        "勝率": (g["won"].mean() * 100).round(1),
        "平均オッズ": g["tansho_odds"].mean().round(1),
        "回収率": (g["payout"].sum() / g.size() * 100).round(1),
    })
    tbl["勝率"] = tbl["勝率"].astype(str) + "%"
    tbl["回収率"] = tbl["回収率"].astype(str) + "%"
    print(tbl.to_string())

    # 全体回収率(全頭ベタ買い)と人気別
    overall = (df["won"] * df["tansho_odds"]).sum() / len(df)
    fav = df[df["tansho_odds"] == df.groupby("race_id")["tansho_odds"].transform("min")]
    fav_rec = (fav["won"] * fav["tansho_odds"]).sum() / len(fav)

    print("\n" + "-" * 64)
    print("② リベート(賭け額還元)の損益分岐:")
    print(f"   全頭ベタ買いの回収率   : {overall*100:.1f}%  → 必要リベート {max(0,1-overall)*100:.1f}%")
    print(f"   1番人気だけ買う回収率 : {fav_rec*100:.1f}%  → 必要リベート {max(0,1-fav_rec)*100:.1f}%")
    print("   ※ リベートr%が賭け額に還元されると、実質回収率 = 上記 + r%。")
    print("     これを100%超えにできる r が『構造的にプラスへ反転する』条件。")

    print("\n" + "-" * 64)
    print("読み解き:")
    best = tbl["回収率"].str.rstrip("%").astype(float).max()
    print(f"  ・最も回収率が高いオッズ帯でも {best:.1f}%。")
    if best >= 100:
        print("    → 100%超えの帯が実在！ ただし分散・サンプル数・再現性を厳しく疑うこと。")
    else:
        print("    → どの帯も100%未満。単勝の市場は控除率ぶん、ほぼ綺麗に効率的。")
    print("  ・『本命ほど回収率が高い(大穴ほど低い)』傾向があれば、それが本命・大穴バイアス。")
    print("  ・現実的なプラス化の最短は『当てる工夫』より『リベート(控除率削減)』という構造。")
    print("-" * 64)

    # ③ 場ごとの横並び(複数場が混ざっているときだけ)
    try:
        _by_track(df)
    except Exception as e:
        print(f"(場別比較はスキップ: {e})")


if __name__ == "__main__":
    main()
