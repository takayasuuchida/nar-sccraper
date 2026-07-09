"""
run_all.py  ―― これ1本で全工程を実行

  python src/run_all.py

データ生成 -> 特徴量 -> 学習(アンサンブル) -> 予測 -> 回収率の検証 まで通します。
"""

import generate_data
import pipeline
import backtest


def main():
    print("=" * 60)
    print(" 競馬予想AI ミニ版  (合成データでパイプライン全体を実演)")
    print("=" * 60)

    print("\n[1/4] データ生成中 ...")
    df = generate_data.make_dataset(n_races=3000)
    print(f"      {len(df):,}行 / {df.race_id.nunique():,}レース")

    print("\n[2/4] 特徴量づくり ...")
    df = pipeline.add_features(df)
    print(f"      使用特徴量 {len(pipeline.FEATURES)}個: {', '.join(pipeline.FEATURES)}")

    print("\n[3/4] 学習(LightGBM + ロジスティック回帰のアンサンブル) ...")
    train, test = pipeline.time_split(df, test_frac=0.25)
    test, auc, importance = pipeline.train_and_predict(train, test)
    print(f"      学習 {train.race_id.nunique()}レース / 検証 {test.race_id.nunique()}レース")
    print(f"      AUC(3着内の見分け力, 0.5=でたらめ 1.0=完璧): {auc:.3f}")
    print("      重要な特徴量 上位5:")
    for name, val in importance.head(5).items():
        print(f"        - {name}: {int(val)}")

    print("\n[4/4] バックテスト(未来レースで的中率・回収率を検証) ...")
    result = backtest.run(test)
    print()
    print(result.to_string())

    print("\n" + "-" * 60)
    print("読み解き:")
    print("  ・回収率100%超えが「儲かる」ライン。控除率20%があるので普通は届かない。")
    print("  ・本命作戦もAI作戦も回収率は80%前後に張り付くはず。")
    print("    => これが『当てる』と『儲ける』が別物である理由。市場(オッズ)は手強い。")
    print("  ・次の一歩: 実データに差し替え / 馬場・血統・調教など特徴量を増やす /")
    print("    『勝てるレースだけ賭ける』など賭け方を工夫する。")
    print("-" * 60)


if __name__ == "__main__":
    main()
