"""
generate_data.py  ―― 競馬データの「合成（シミュレーション）」生成

実データ(netkeiba等)をスクレイピングする代わりに、競馬の仕組みを真似た
リアルなダミーデータを作ります。これで誰でもオフラインでパイプライン全体を
動かせます。実データに差し替えるときは scrape_netkeiba.py を参照。

ポイント:
- 各馬には「本当の強さ ability」という“見えない値”がある。
- レース当日の着順は ability + 当日のブレ(運) で決まる。
- 我々が予測に使えるのは、強さを“間接的にしか映さない”特徴量だけ
  (前走着順・騎手成績・斤量・馬体重・単勝オッズ…)。
- 単勝オッズには市場(みんなの予想)の知恵が入っているが、控除率(胴元の取り分)が
  約20%乗っているので、オッズ通りに賭けると回収率は構造的に約80%になる。
  ここが「当てる」と「儲ける」が別物である理由。
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)   # 再現性のため乱数を固定
TAKEOUT = 0.20                    # 単勝の控除率(胴元の取り分) ≒ 20%


def make_dataset(n_races: int = 3000) -> pd.DataFrame:
    rows = []
    for race_id in range(n_races):
        n = int(RNG.integers(8, 17))          # 出走頭数 8〜16
        distance = int(RNG.choice([1200, 1400, 1600, 1800, 2000, 2400]))

        # --- 各馬の「本当の強さ」(我々には直接見えない) ---
        ability = RNG.normal(0, 1, n)

        # --- レース当日の着順を決める実力発揮値 (運=ノイズが大きい) ---
        kinryo = RNG.normal(55, 1.5, n).round(1)              # 斤量
        bataiju = RNG.normal(480, 22, n).round().astype(int)  # 馬体重
        jockey_win_rate = np.clip(0.12 + 0.05 * (0.3 * ability + 0.7 * RNG.normal(0, 1, n)),
                                  0.01, 0.30).round(3)         # 騎手の勝率
        # 各馬の「実力ベース値」(運を除いた期待パフォーマンス)
        strength = (ability
                    + 0.20 * (jockey_win_rate - 0.12) * 10
                    - 0.02 * (kinryo - 55))

        # --- 当日の着順: 実力 + 大きな運。1回ぶんを実レース結果とする ---
        NOISE = 0.8
        perf = strength + RNG.normal(0, NOISE, n)
        finish = perf.argsort()[::-1].argsort() + 1           # 着順(1が1着)

        # --- 本当の勝率をモンテカルロで実測(同じ運の分布で多数回シミュ) ---
        # こうして作ったオッズは"本当の勝率"を反映する=効率的な市場になる。
        sims = strength[:, None] + RNG.normal(0, NOISE, (n, 4000))
        winners = sims.argmax(axis=0)
        p_win_true = np.bincount(winners, minlength=n) / sims.shape[1]
        p_win_true = np.clip(p_win_true, 1e-4, None)
        # 市場はほぼ正確だが完全ではない(わずかな値ぶれを乗せる)
        p_market = p_win_true * RNG.lognormal(0, 0.07, n)
        p_market = p_market / p_market.sum()
        odds = np.clip((1 - TAKEOUT) / p_market, 1.0, 999).round(1)

        # --- 予測に使える特徴量 (どれも強さを“ノイズ越しに”映す) ---
        prev_finish = np.clip(np.round(8 - 3 * ability + RNG.normal(0, 2, n)), 1, 18).astype(int)
        waku = (np.arange(n) % 8) + 1                          # 枠番
        umaban = np.arange(1, n + 1)                           # 馬番
        sex = RNG.integers(0, 3, n)                            # 0:牡 1:牝 2:セ
        age = RNG.integers(3, 8, n)                            # 馬齢

        for i in range(n):
            rows.append(dict(
                race_id=race_id, distance=distance, field_size=n,
                waku=int(waku[i]), umaban=int(umaban[i]),
                sex=int(sex[i]), age=int(age[i]),
                kinryo=float(kinryo[i]), bataiju=int(bataiju[i]),
                jockey_win_rate=float(jockey_win_rate[i]),
                prev_finish=int(prev_finish[i]),
                tansho_odds=float(odds[i]),
                finish=int(finish[i]),
                won=int(finish[i] == 1),         # 1着か
                is_top3=int(finish[i] <= 3),     # 3着内か (今回の予測ターゲット)
            ))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = make_dataset()
    out = "data/races.csv"
    df.to_csv(out, index=False)
    print(f"生成完了: {len(df):,}行 / {df.race_id.nunique():,}レース -> {out}")
    print(df.head(8).to_string(index=False))
