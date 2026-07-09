"""
pipeline.py  ―― 特徴量づくり / 学習(アンサンブル) / 予測

出品者がやっている "LightGBM + もう1モデル のアンサンブル" を再現します。
ここでは LightGBM + ロジスティック回帰 の2モデルの確率を平均します。
ターゲットは「3着内に入るか(is_top3)」の確率。
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


BASE_FEATURES = [
    "distance", "field_size", "waku", "umaban", "sex", "age",
    "kinryo", "bataiju", "jockey_win_rate", "prev_finish", "tansho_odds",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """レース内での“相対的な強さ”を表す特徴量を足す(ここが腕の見せ所)。"""
    df = df.copy()
    df["log_odds"] = np.log(df["tansho_odds"])
    # レース内でオッズ・前走着順が何番人気/何番手かを順位特徴に
    df["odds_rank"] = df.groupby("race_id")["tansho_odds"].rank(method="min")
    df["prev_rank"] = df.groupby("race_id")["prev_finish"].rank(method="min")
    # 斤量・馬体重がレース平均からどれだけ離れているか
    df["kinryo_rel"] = df["kinryo"] - df.groupby("race_id")["kinryo"].transform("mean")
    df["bataiju_rel"] = df["bataiju"] - df.groupby("race_id")["bataiju"].transform("mean")
    # 市場が見積もる勝率(オッズの逆数をレース内で正規化)
    inv = 1.0 / df["tansho_odds"]
    df["mkt_winprob"] = inv / df.groupby("race_id")["tansho_odds"].transform(lambda s: (1.0 / s).sum())
    return df


FEATURES = BASE_FEATURES + [
    "log_odds", "odds_rank", "prev_rank", "kinryo_rel", "bataiju_rel", "mkt_winprob",
]

# Tier A 特徴量(実データにある時だけ自動で使う。合成データには無いので無視される)
EXTENDED_FEATURES = [
    "baba", "track", "bataiju_diff", "days_since_last",
    "past_top3_rate", "past_avg_finish", "past_avg_agari",
    "same_dist_top3", "jockey_top3_rate",
]


def model_features(df: pd.DataFrame) -> list:
    """その DataFrame に実在する特徴量だけを返す。
    → 合成データ(基本のみ)でも実データ(Tier A入り)でも同じ関数で動く。"""
    return [f for f in FEATURES + EXTENDED_FEATURES if f in df.columns]


def time_split(df: pd.DataFrame, test_frac: float = 0.25):
    """レース(=時間順)で前半=学習 / 後半=検証 に分ける。
    未来のレースで検証するのが鉄則(リーク防止)。
    race_id は数値(合成データ)でも文字列(netkeibaのID)でも動くようにする。"""
    races = sorted(df["race_id"].unique())        # 昇順=時系列順
    cut = int(len(races) * (1 - test_frac))
    train_ids = set(races[:cut])
    train = df[df["race_id"].isin(train_ids)].copy()
    test = df[~df["race_id"].isin(train_ids)].copy()
    return train, test


def fit_predict_prob(train: pd.DataFrame, test: pd.DataFrame, target: str) -> np.ndarray:
    """指定ターゲット(is_top3 や won)の確率を、LightGBM+ロジ回帰のアンサンブルで予測。
    回収率戦略(strategy.py)から勝率モデルを作るのにも使う。"""
    feats = model_features(train)
    y_tr = train[target].values
    X_tr, X_te = train[feats], test[feats]

    gbm = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.03, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=40,
        random_state=42, verbose=-1,
    )
    gbm.fit(X_tr, y_tr)
    p_gbm = gbm.predict_proba(X_te)[:, 1]

    scaler = StandardScaler().fit(X_tr)
    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(scaler.transform(X_tr), y_tr)
    p_lr = lr.predict_proba(scaler.transform(X_te))[:, 1]

    return 0.5 * p_gbm + 0.5 * p_lr


def train_and_predict(train: pd.DataFrame, test: pd.DataFrame):
    # --- モデル1: LightGBM (非線形・特徴量の相互作用に強い) ---
    # --- モデル2: ロジスティック回帰 (線形・安定) ---
    # --- アンサンブル: 2モデルの平均 ---
    p_ens = fit_predict_prob(train, test, "is_top3")

    test = test.copy()
    test["pred_top3"] = p_ens
    # 結果が未確定のレース(明日の予想等)はAUCを計算できないのでスキップ
    auc = (roc_auc_score(test["is_top3"], p_ens)
           if test["is_top3"].nunique() > 1 else float("nan"))
    # 特徴量の重要度は参考用に再学習1本ぶんから取得
    feats = model_features(train)
    gbm = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.03, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=40,
        random_state=42, verbose=-1,
    ).fit(train[feats], train["is_top3"])
    importance = pd.Series(gbm.feature_importances_, index=feats).sort_values(ascending=False)
    return test, auc, importance
