"""
predict_demo.py  ―― 【デモ専用】「明日のレース予想カード」の出力見本

⚠⚠ 重要 ⚠⚠
  ここで出る馬名・予想はすべて『架空のダミーデータ』です。
  実在の馬・レースとは一切関係ありません。これで馬券を買わないでください。
  目的は「本物の実データにつないだら、予想結果がどんな形で出るか」を見ること。

使い方:
  python3 src/predict_demo.py
"""

import numpy as np
import pandas as pd
import generate_data
import pipeline

RNG = np.random.default_rng(2025)
TAKEOUT = 0.20

# 架空の馬名・騎手名(実在とは無関係のダミー)
HORSE_NAMES = [
    "ミライノオト", "ソラカケル", "ハルカナタ", "ギンガノツバサ", "シズカナウミ",
    "カゼノオウジ", "ホシノカケラ", "アケボノマル", "ユメミドリ", "トキノナミダ",
    "コトブキロード", "ライトニングQ", "ネバーギブアップ", "サクラビヨリ", "ツキノヒカリ",
    "アラシノヨル",
]
JOCKEYS = ["田中", "佐藤", "鈴木", "山本", "中村", "小林", "加藤", "吉田",
           "渡辺", "伊藤", "高橋", "松本", "井上", "木村", "林", "清水"]


def make_demo_race(n_horses: int = 12) -> pd.DataFrame:
    """明日走る“架空の1レース”を作る（着順=結果はまだ無い、という想定）。"""
    n = n_horses
    ability = RNG.normal(0, 1, n)
    kinryo = RNG.normal(55, 1.5, n).round(1)
    bataiju = RNG.normal(480, 22, n).round().astype(int)
    jockey_win_rate = np.clip(0.12 + 0.05 * (0.3 * ability + 0.7 * RNG.normal(0, 1, n)),
                              0.01, 0.30).round(3)
    prev_finish = np.clip(np.round(8 - 3 * ability + RNG.normal(0, 2, n)), 1, 18).astype(int)
    market_score = ability + RNG.normal(0, 0.4, n)
    p_market = np.exp(3.0 * market_score)
    p_market = p_market / p_market.sum()
    odds = np.clip((1 - TAKEOUT) / p_market, 1.0, 999).round(1)

    names = RNG.choice(HORSE_NAMES, size=n, replace=False)
    jockeys = RNG.choice(JOCKEYS, size=n, replace=False)

    df = pd.DataFrame(dict(
        race_id=900001, distance=1600, field_size=n,
        waku=(np.arange(n) % 8) + 1, umaban=np.arange(1, n + 1),
        sex=RNG.integers(0, 3, n), age=RNG.integers(3, 8, n),
        kinryo=kinryo, bataiju=bataiju, jockey_win_rate=jockey_win_rate,
        prev_finish=prev_finish, tansho_odds=odds,
        # ↓ 結果はまだ無いのでダミー（学習には使わない）
        finish=0, won=0, is_top3=0,
        umamei=names, kishu=jockeys,
    ))
    return df


def main():
    print("=" * 64)
    print(" 【デモ】明日のレース予想カード")
    print(" ※ 馬名・数値はすべて架空のダミーです。実在レースではありません。")
    print("=" * 64)

    # 1) 学習（架空の過去データ3000レースでモデルを作る）
    train = pipeline.add_features(generate_data.make_dataset(3000))

    # 2) 明日の“架空レース”を用意して予想
    demo = make_demo_race(12)
    demo_feat = pipeline.add_features(demo)
    pred, _, _ = pipeline.train_and_predict(train, demo_feat)

    # 3) 予想カードを整形
    card = pred.copy()
    card["人気"] = card["tansho_odds"].rank(method="min").astype(int)
    card["3着内確率"] = (card["pred_top3"] * 100).round(1)
    card["AI評価順"] = card["pred_top3"].rank(ascending=False, method="min").astype(int)
    # 妙味: AI評価が人気より上なら「狙い目」
    card["妙味"] = np.where(card["AI評価順"] < card["人気"] - 1, "★狙い目",
                    np.where(card["AI評価順"] > card["人気"] + 1, "△危険", ""))

    card = card.sort_values("pred_top3", ascending=False)
    view = card[["AI評価順", "umaban", "umamei", "kishu", "tansho_odds",
                 "人気", "3着内確率", "妙味"]].rename(columns={
        "AI評価順": "印", "umaban": "馬番", "umamei": "馬名",
        "kishu": "騎手", "tansho_odds": "単勝オッズ"})

    marks = {1: "◎本命", 2: "○対抗", 3: "▲単穴", 4: "△連下", 5: "△連下"}
    view["印"] = view["印"].map(lambda r: marks.get(r, "  注"))

    print()
    print(view.to_string(index=False))

    top = card.iloc[0]
    print("\n" + "-" * 64)
    print(f"AIの本命: {top['umamei']}（{int(top['umaban'])}番） "
          f"3着内確率 {top['pred_top3']*100:.1f}% / 単勝{top['tansho_odds']}倍")
    box = card.sort_values('pred_top3', ascending=False).head(3)
    print("3連複で買うなら軸候補:", " - ".join(f"{int(r.umaban)}{r.umamei}" for r in box.itertuples()))
    print("-" * 64)
    print("⚠ これは架空データのデモです。実際の馬券購入には使えません。")
    print("  本物にするには src/scrape_netkeiba.py で実データを取り込む必要があります。")


if __name__ == "__main__":
    main()
