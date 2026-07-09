"""
load_kaggle.py  ―― Kaggle「JRA Horse Racing Dataset」を読み込んで学習・検証まで一気に走らせる

使い方(自分のPCで):
  python src\\load_kaggle.py "C:\\Users\\PC_User\\Downloads\\keiba_data\\19860105-20210731_race_result.csv"
  python src\\load_kaggle.py "....csv" 2015      ← 2015年以降だけ使う(任意。既定2010)

race_result.csv の日本語見出しを、pipeline.py が使う数値テーブルに変換し、
前走着順・騎手勝率を履歴から自動生成して、AUC・回収率まで出す。
変換後データは data/races_kaggle.csv に保存(再利用用)。
"""

import sys
import os
import numpy as np
import pandas as pd

import pipeline
import backtest

SEX_MAP = {"牡": 0, "牝": 1, "セ": 2, "せん": 2, "騙": 2}


def _open_columns(path):
    """文字コードを判定しつつヘッダー(列名)だけ読む。"""
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            cols = pd.read_csv(path, nrows=0, encoding=enc).columns.tolist()
            return [str(c).strip() for c in cols], enc
        except (UnicodeDecodeError, LookupError):
            continue
    raise RuntimeError("文字コードを判定できませんでした")


def _find(cols, *cands, avoid=("ID", "注記", "増減", "区分2")):
    """見出しを選ぶ。①完全一致 → ②avoid語を含まない部分一致 → ③部分一致 の順。
    『馬番』が『レース馬番ID』に、『着順』が『着順注記』に誤マッチするのを防ぐ。"""
    for cand in cands:                                  # ① 完全一致を最優先
        for c in cols:
            if c == cand:
                return c
    for cand in cands:                                  # ② avoid語を避けた部分一致
        for c in cols:
            if cand in c and not any(a in c for a in avoid):
                return c
    for cand in cands:                                  # ③ 最後の手段
        for c in cols:
            if cand in c:
                return c
    return None


def load(path: str, min_year: int = 2010) -> pd.DataFrame:
    cols, enc = _open_columns(path)
    print(f"  文字コード: {enc} / 全{len(cols)}列を確認")

    col = dict(
        race_id=_find(cols, "レースID"), date=_find(cols, "レース日付"),
        distance=_find(cols, "距離"), finish=_find(cols, "着順"),
        waku=_find(cols, "枠番"), umaban=_find(cols, "馬番"),
        umamei=_find(cols, "馬名"), sex=_find(cols, "性別"),
        age=_find(cols, "馬齢"), kinryo=_find(cols, "斤量"),
        kishu=_find(cols, "騎手"), odds=_find(cols, "単勝"),
        ninki=_find(cols, "人気"), bataiju=_find(cols, "馬体重"),
        shogai=_find(cols, "障害区分"),
        # --- Tier A 用に追加で拾う列 ---
        baba=_find(cols, "馬場状態1", "馬場状態", "馬場"),
        track=_find(cols, "芝・ダート区分", "芝・ダート"),
        bataiju_diff=_find(cols, "場体重増減", "馬体重増減", avoid=()),
        agari=_find(cols, "上り", "上がり"),
    )
    # どの列をどう拾ったか丸見えにする(診断)
    print("  列マッピング:")
    for k, v in col.items():
        print(f"    {k:12s} <- {v}")
    missing = [k for k, v in col.items() if v is None and k != "shogai"]
    if missing:
        print(f"  ⚠ 見つからない列: {missing}  (この列名が原因かも)")

    use = [v for v in col.values() if v]
    print("  必要列を読み込み中(471MBなら数十秒)...")
    raw = pd.read_csv(path, usecols=use, encoding=enc, low_memory=False)
    raw = raw.rename(columns={v: k for k, v in col.items() if v})
    print(f"  読み込み: {len(raw):,}行")

    # 障害(ジャンプ)レースは除外して平地だけに。
    # ただし『障害区分』が平地を空欄でなく値(例:平地)で持つデータもあるので、
    # 値の中に「障害」を含む行だけを落とす(空欄や"平地"は残す)。
    if col["shogai"]:
        s = raw["shogai"].astype(str)
        raw = raw[~s.str.contains("障害", na=False)]
        print(f"  平地のみ抽出後: {len(raw):,}行")

    df = pd.DataFrame()
    df["race_id"] = raw["race_id"].astype(str)
    df["date"] = pd.to_datetime(raw["date"], errors="coerce")
    df["distance"] = pd.to_numeric(raw["distance"], errors="coerce")
    df["finish"] = pd.to_numeric(raw["finish"], errors="coerce")
    df["waku"] = pd.to_numeric(raw["waku"], errors="coerce")
    df["umaban"] = pd.to_numeric(raw["umaban"], errors="coerce")
    df["umamei"] = raw["umamei"].astype(str)
    df["kishu"] = raw["kishu"].astype(str)
    df["age"] = pd.to_numeric(raw["age"], errors="coerce")
    df["kinryo"] = pd.to_numeric(raw["kinryo"], errors="coerce")
    df["tansho_odds"] = pd.to_numeric(raw["odds"], errors="coerce")
    df["bataiju"] = pd.to_numeric(raw["bataiju"], errors="coerce")
    df["sex"] = raw["sex"].astype(str).str[0].map(SEX_MAP).fillna(0).astype(int)

    # --- Tier A: レース前に分かる情報(リークではない) ---
    if col["baba"]:                                    # 馬場状態 良0/稍重1/重2/不良3
        bmap = {"良": 0, "稍重": 1, "稍": 1, "重": 2, "不良": 3, "不": 3}
        df["baba"] = raw["baba"].astype(str).str.strip().map(
            lambda x: next((v for k, v in bmap.items() if x.startswith(k)), 0))
    if col["track"]:                                   # 芝0 / ダート1
        df["track"] = raw["track"].astype(str).str.contains("ダ").astype(int)
    if col["bataiju_diff"]:                             # 馬体重増減(発走前に発表)
        df["bataiju_diff"] = pd.to_numeric(raw["bataiju_diff"], errors="coerce").fillna(0)
    if col["agari"]:                                   # 上り3F(※"過去走の"だけ後で使う)
        df["_agari"] = pd.to_numeric(raw["agari"], errors="coerce")

    # 年でフィルタ(古すぎる年はルールも違うので既定2010年以降)
    if df["date"].notna().any():
        df = df[df["date"].dt.year >= min_year]
        print(f"  {min_year}年以降に絞り込み: {len(df):,}行")
    else:
        print("  ⚠ 日付が全て解析できず。年フィルタをスキップします(日付列を確認)")
    # 着順・オッズが無い行(中止・除外など)は落とす
    before = len(df)
    df = df.dropna(subset=["finish", "umaban", "tansho_odds", "distance",
                           "bataiju", "age", "kinryo"])
    df = df[df["tansho_odds"] > 0]
    print(f"  欠損行を除外: {before:,} -> {len(df):,}行")
    df["finish"] = df["finish"].astype(int)

    # 時系列順に並べる(リーク防止の土台)
    df = df.sort_values(["date", "race_id"]).reset_index(drop=True)
    df["field_size"] = df.groupby("race_id")["umaban"].transform("count")

    # 前走着順: 馬ごとに時系列で1つ前の着順。初出走は中央値で埋める。
    df["prev_finish"] = df.groupby("umamei")["finish"].shift(1)
    df["prev_finish"] = df["prev_finish"].fillna(df["finish"].median())

    df["won"] = (df["finish"] == 1).astype(int)
    df["is_top3"] = (df["finish"] <= 3).astype(int)

    # ===== Tier A: 過去走だけから作る特徴量(リーク厳禁) =====
    # cumsum - 当該値 = "その馬の過去だけ"の合計。/過去レース数 で平均に。
    horse = df["umamei"]
    n_prev = df.groupby(horse).cumcount()              # その馬の過去レース数
    safe = n_prev.replace(0, np.nan)
    df["past_top3_rate"] = ((df.groupby(horse)["is_top3"].cumsum() - df["is_top3"]) / safe)
    df["past_avg_finish"] = ((df.groupby(horse)["finish"].cumsum() - df["finish"]) / safe)
    df["past_top3_rate"] = df["past_top3_rate"].fillna(df["is_top3"].mean())
    df["past_avg_finish"] = df["past_avg_finish"].fillna(df["finish"].median())

    # 休養明け: 前走からの日数
    prev_date = df.groupby(horse)["date"].shift(1)
    df["days_since_last"] = (df["date"] - prev_date).dt.days
    df["days_since_last"] = df["days_since_last"].fillna(df["days_since_last"].median())

    # 同距離の過去複勝率(距離適性)
    hd = [horse, df["distance"]]
    n_prev_d = df.groupby(hd).cumcount().replace(0, np.nan)
    df["same_dist_top3"] = ((df.groupby(hd)["is_top3"].cumsum() - df["is_top3"]) / n_prev_d)
    df["same_dist_top3"] = df["same_dist_top3"].fillna(df["past_top3_rate"])

    # 騎手の勝率・複勝率: 『過去の騎乗だけ』で計算(リーク防止)。
    # 以前は全期間平均で軽いリークがあった。過去のみにすると正直な(やや厳しい)数字になる。
    jk = df.groupby("kishu")
    jn = jk.cumcount().replace(0, np.nan)
    df["jockey_win_rate"] = ((jk["won"].cumsum() - df["won"]) / jn).fillna(df["won"].mean()).round(3)
    df["jockey_top3_rate"] = ((jk["is_top3"].cumsum() - df["is_top3"]) / jn).fillna(df["is_top3"].mean()).round(3)

    # 過去走の上がり3F平均(末脚の速さ。"過去の"値だけを使う)
    if "_agari" in df:
        af = df["_agari"].fillna(0.0)
        valid = df["_agari"].notna().astype(int)
        csum = af.groupby(horse).cumsum() - af
        ccnt = (valid.groupby(horse).cumsum() - valid).replace(0, np.nan)
        df["past_avg_agari"] = (csum / ccnt).fillna(df["_agari"].median())
        df = df.drop(columns=["_agari"])

    # 1頭立て等の極端なレースは除外
    df = df[df["field_size"] >= 5].reset_index(drop=True)
    return df


def main():
    if len(sys.argv) < 2:
        print("使い方: python src\\load_kaggle.py <race_result.csvのパス> [開始年]")
        sys.exit(1)
    path = sys.argv[1]
    min_year = int(sys.argv[2]) if len(sys.argv) > 2 else 2010

    print("=" * 60)
    print(" Kaggle 実データで競馬AIを学習・検証")
    print("=" * 60)
    print(f"\n[1/4] 読み込み: {path}  ({min_year}年以降)")
    df = load(path, min_year)
    if len(df) == 0:
        print("\n❌ 使えるデータが0行でした。上の『列マッピング』『各行カウント』を見て、")
        print("   どの段階で消えたか確認してください(列名の取り違えが多い)。")
        print("   その出力を貼ってくれれば直します。")
        sys.exit(1)
    print(f"      使用: {len(df):,}行 / {df.race_id.nunique():,}レース "
          f"({df.date.min().date()}〜{df.date.max().date()})")

    os.makedirs("data", exist_ok=True)            # data フォルダが無ければ作る
    out = "data/races_kaggle.csv"
    df.to_csv(out, index=False)
    print(f"      変換データを保存: {out}")

    print("\n[2/4] 特徴量づくり ...")
    feat = pipeline.add_features(df)
    used = pipeline.model_features(feat)
    tier_a = [f for f in pipeline.EXTENDED_FEATURES if f in used]
    print(f"      使用特徴量 {len(used)}個 (うちTier A {len(tier_a)}個: {', '.join(tier_a)})")

    print("[3/4] 学習(LightGBM + ロジスティック回帰のアンサンブル) ...")
    train, test = pipeline.time_split(feat, test_frac=0.2)
    test, auc, importance = pipeline.train_and_predict(train, test)
    print(f"      学習 {train.race_id.nunique():,}R / 検証 {test.race_id.nunique():,}R")
    print(f"      AUC(3着内の見分け力): {auc:.3f}")
    print("      重要な特徴量 上位5:")
    for name, val in importance.head(5).items():
        print(f"        - {name}: {int(val)}")

    print("\n[4/4] バックテスト(未来レースで的中率・回収率) ...")
    print(backtest.run(test).to_string())
    print("\n" + "-" * 60)
    print("これは“本物のJRAデータ”での結果です。回収率80%前後の壁を確認しつつ、")
    print("DATA_GUIDE.md の Tier A 特徴量を足して伸びるか試していく段階。")
    print("-" * 60)


if __name__ == "__main__":
    main()
