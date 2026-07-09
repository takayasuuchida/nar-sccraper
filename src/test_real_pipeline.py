"""
test_real_pipeline.py  ―― 実データ経路の動作確認(オフライン)

netkeiba から取得した「生データ(日本語見出し)」を模した表を作り、
  scrape_netkeiba.to_pipeline_format() で整形 -> pipeline で学習・検証
までが通ることを確認する。実際のスクレイピングはしない(ネット不要)。
"""

import numpy as np
import pandas as pd

import scrape_netkeiba
import pipeline
import backtest

RNG = np.random.default_rng(7)
HORSES = [f"テスト馬{i:02d}" for i in range(60)]
JOCKEYS = [f"騎手{i:02d}" for i in range(12)]


def fake_raw_race(race_id: str) -> pd.DataFrame:
    """netkeiba 結果ページ相当の生データ(日本語見出し)を1レース分でっち上げる。"""
    n = int(RNG.integers(8, 15))
    ability = RNG.normal(0, 1, n)
    perf = ability + RNG.normal(0, 0.8, n)
    finish = perf.argsort()[::-1].argsort() + 1
    odds = np.clip(0.8 / (np.exp(3 * (ability + RNG.normal(0, .4, n))) /
                          np.exp(3 * (ability + RNG.normal(0, .4, n))).sum()), 1, 999).round(1)
    sex = RNG.choice(["牡", "牝", "セ"], n)
    age = RNG.integers(3, 8, n)
    return pd.DataFrame({
        "着 順": finish, "枠 番": (np.arange(n) % 8) + 1, "馬 番": np.arange(1, n + 1),
        "馬名": RNG.choice(HORSES, n, replace=False),
        "性齢": [f"{s}{a}" for s, a in zip(sex, age)],
        "斤量": RNG.normal(55, 1.5, n).round(1),
        "騎手": RNG.choice(JOCKEYS, n),
        "単勝": odds,
        "馬体重": [f"{w}({d:+d})" for w, d in
                 zip(RNG.integers(440, 520, n), RNG.integers(-8, 9, n))],
        "race_id": race_id,
        "distance": int(RNG.choice([1200, 1600, 2000])),
    })


def main():
    print("=== 実データ経路の動作確認(オフライン・netkeiba形式の模擬データ) ===\n")

    # 1) 「生データ」を多数レース分つくる(本番では scrape_races() の戻り値に相当)
    race_ids = [f"2021050304{r:02d}" for r in range(1, 13)] + \
               [f"2021050305{r:02d}" for r in range(1, 13)] + \
               [f"2021060101{r:02d}" for r in range(1, 13)]
    raw = pd.concat([fake_raw_race(rid) for rid in race_ids * 12], ignore_index=True)
    print(f"生データ: {len(raw)}行 / {raw.race_id.nunique()}レース  見出し={list(raw.columns)[:6]}...")

    # 2) 整形(ここが本物データ取り込みの肝。日本語見出し -> 数値テーブル)
    data = scrape_netkeiba.to_pipeline_format(raw)
    print(f"整形後 : {len(data)}行  列={list(data.columns)}")
    print("  prev_finish / jockey_win_rate を取得データから自動生成できているか:",
          {"prev_finish": data.prev_finish.notna().all(),
           "jockey_win_rate": data.jockey_win_rate.notna().all()})

    # 3) いつものパイプラインに流す(合成データと全く同じ関数が使える)
    df = pipeline.add_features(data)
    train, test = pipeline.time_split(df, test_frac=0.3)
    test, auc, _ = pipeline.train_and_predict(train, test)
    print(f"\n学習OK  AUC={auc:.3f}  (検証 {test.race_id.nunique()}レース)")
    print(backtest.run(test).to_string())

    print("\n✅ 実データ経路は通りました。あとは fake_raw_race を")
    print("   scrape_netkeiba.scrape_races(実在レースID) に置き換えるだけ(要・自分のPC)。")


if __name__ == "__main__":
    main()
