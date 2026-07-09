"""
strategy_real.py  ―― 本物のJRAデータで「市場の隙」を期待値で攻められるか検証

前提: 先に load_kaggle.py を実行して data/races_kaggle.csv を作っておくこと。
  python src\\load_kaggle.py "....race_result.csv" 2015
  python src\\strategy_real.py

合成データ版 strategy.py と同じS0/S1/S1'/S2を、本物データで回す。
"""

import os
import pandas as pd

import pipeline
import strategy

DATA = "data/races_kaggle.csv"


def main():
    if not os.path.exists(DATA):
        print(f"❌ {DATA} がありません。先に load_kaggle.py を実行してください:")
        print('   python src\\load_kaggle.py "....race_result.csv" 2015')
        return

    print("=" * 60)
    print(" 回収率を上げる賭け方の実験 (本物のJRAデータ)")
    print("=" * 60)
    df = pd.read_csv(DATA)
    print(f"読み込み: {len(df):,}行 / {df.race_id.nunique():,}レース")

    feat = pipeline.add_features(df)
    train, test = pipeline.time_split(feat, test_frac=0.2)
    test = test.copy()
    print("学習中(3着内モデル＋勝率モデル)...")
    test["pred_top3"] = pipeline.fit_predict_prob(train, test, "is_top3")
    test["raw_win"] = pipeline.fit_predict_prob(train, test, "won")
    test = strategy.add_ev(test)
    print(f"検証 {test.race_id.nunique():,} レース / 出走 {len(test):,} 頭\n")

    print(strategy.compare_strategies(test).to_string())

    # --- ケリー基準で賭け金を最適化して資金推移を見る ---
    print("\n【ケリー基準】期待値>=1.1 かつ オッズ<=20倍 の馬に、分数ケリーで賭けた場合")
    k = strategy.kelly_sim(test, ev_min=1.1, odds_cap=20.0, frac=0.25, cap=0.05)
    for key, val in k.items():
        print(f"    {key}: {val}")
    print("  ※ 賭け方(ケリー)は edge を作らない。edgeがプラスなら増え、マイナスなら減るだけ。")

    print("\n" + "-" * 60)
    print("読み解き (本物データの正直な結論):")
    print("  ・S0/S2(本命買い・自信選択)は約80%。控除率ぶん、やはり負ける。")
    print("  ・S1(期待値買い)が100%を超えるかが全て。超えても買い点数は少なく")
    print("    平均オッズが高い=穴狙いの高分散。安定して勝てる根拠にはなりにくい。")
    print("  ・公開データを市場も見ている以上、市場の隙は小さい。これが現実。")
    print("-" * 60)


if __name__ == "__main__":
    main()
