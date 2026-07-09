"""
scrape_netkeiba.py  ―― netkeiba から本物のレース結果を取得する(実装版)

⚠ 実行前に必ず守ること:
  1. 利用規約 / robots.txt を確認する(このコードは robots.txt を自動チェックする)。
  2. アクセス間隔を必ず空ける(既定 SLEEP=1.5秒)。短くしない。
  3. 取得データは個人の学習利用の範囲で。再配布しない。
  4. 大量取得はしない。まずは数十レースで試す。

★ このクラウド実行環境では netkeiba がegress許可リスト外でブロックされます。
   その場合は「自分のPC」で実行してください(ローカルには制限がありません)。
   動作確認だけなら src/test_real_pipeline.py がオフラインで通ります。

取得 → 整形すると、pipeline.py がそのまま使える列になります:
  race_id, date, distance, field_size, waku, umaban, sex, age, kinryo,
  bataiju, jockey_win_rate, prev_finish, tansho_odds, finish, won, is_top3
"""

import io
import time
import urllib.robotparser as robotparser

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE = "https://db.netkeiba.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (keiba-ai learning project)"}
SLEEP = 1.5            # アクセス間隔(秒)。絶対に短くしない。
SEX_MAP = {"牡": 0, "牝": 1, "セ": 2, "せん": 2}


# ---------------------------------------------------------------- 取得部
def _allowed_by_robots(url: str) -> bool:
    rp = robotparser.RobotFileParser()
    rp.set_url(f"{BASE}/robots.txt")
    try:
        rp.read()
    except Exception:
        return False   # robots が読めない時は安全側で取得しない
    return rp.can_fetch(HEADERS["User-Agent"], url)


def fetch_race(race_id: str) -> pd.DataFrame:
    """1レースの結果ページを取得し、生データ(日本語見出し)の表を返す。
    race_id 例: '202105030411' (年・場・回・日・R)"""
    url = f"{BASE}/race/{race_id}/"
    if not _allowed_by_robots(url):
        raise PermissionError(f"robots.txt により取得不可: {url}")

    res = requests.get(url, headers=HEADERS, timeout=15)
    res.encoding = "euc-jp"                       # netkeiba は EUC-JP
    html = res.text

    tables = pd.read_html(io.StringIO(html))      # 結果表を抽出
    result = max(tables, key=len)                 # 一番行数の多い表が出走表
    result.columns = [str(c).replace(" ", "") for c in result.columns]

    soup = BeautifulSoup(html, "lxml")
    # レースタイトルから距離(例: 芝右1600m)を取る
    dist = np.nan
    diary = soup.select_one("diary_snap_cut span, .data_intro span")
    text = (diary.get_text() if diary else "") + soup.get_text()[:400]
    import re
    m = re.search(r"(\d{3,4})m", text)
    if m:
        dist = int(m.group(1))

    result["race_id"] = race_id
    result["distance"] = dist
    return result


def scrape_races(race_ids: list[str]) -> pd.DataFrame:
    """複数レースを順に取得(アクセス間隔を空けながら)。"""
    frames = []
    for i, rid in enumerate(race_ids, 1):
        try:
            frames.append(fetch_race(rid))
            print(f"  [{i}/{len(race_ids)}] {rid} 取得OK")
        except Exception as e:
            print(f"  [{i}/{len(race_ids)}] {rid} スキップ: {e}")
        time.sleep(SLEEP)                          # ← サーバ負荷軽減。消さない。
    if not frames:
        raise RuntimeError("1件も取得できませんでした")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------- 整形部
def _pick(cols, *cands):
    """日本語見出しの揺れを吸収して列名を選ぶ。"""
    for cand in cands:
        for c in cols:
            if cand in c:
                return c
    return None


def to_pipeline_format(raw: pd.DataFrame) -> pd.DataFrame:
    """生データ(日本語見出し) -> pipeline.py が使える数値テーブルに変換。
    prev_finish(前走着順) と jockey_win_rate(騎手勝率) は取得済みデータから自分で作る。"""
    raw = raw.copy()
    raw.columns = [str(x).replace(" ", "").replace("　", "") for x in raw.columns]
    c = raw.columns
    col = dict(
        finish=_pick(c, "着順"), waku=_pick(c, "枠番"), umaban=_pick(c, "馬番"),
        umamei=_pick(c, "馬名"), seirei=_pick(c, "性齢"), kinryo=_pick(c, "斤量"),
        kishu=_pick(c, "騎手"), odds=_pick(c, "単勝"), bataiju=_pick(c, "馬体重"),
    )
    df = pd.DataFrame()
    df["race_id"] = raw["race_id"]
    df["distance"] = raw["distance"]
    # 着順: 中止/除外など数値でない行は落とす
    df["finish"] = pd.to_numeric(raw[col["finish"]], errors="coerce")
    df["waku"] = pd.to_numeric(raw[col["waku"]], errors="coerce")
    df["umaban"] = pd.to_numeric(raw[col["umaban"]], errors="coerce")
    df["umamei"] = raw[col["umamei"]].astype(str)
    df["kishu"] = raw[col["kishu"]].astype(str)
    df["kinryo"] = pd.to_numeric(raw[col["kinryo"]], errors="coerce")
    df["tansho_odds"] = pd.to_numeric(raw[col["odds"]], errors="coerce")
    # 性齢 "牡3" -> sex=0, age=3
    seirei = raw[col["seirei"]].astype(str)
    df["sex"] = seirei.str[0].map(SEX_MAP).fillna(0).astype(int)
    df["age"] = pd.to_numeric(seirei.str.extract(r"(\d+)")[0], errors="coerce")
    # 馬体重 "470(+2)" -> 470
    df["bataiju"] = pd.to_numeric(
        raw[col["bataiju"]].astype(str).str.extract(r"(\d+)")[0], errors="coerce")

    df = df.dropna(subset=["finish", "umaban"]).copy()
    df["finish"] = df["finish"].astype(int)

    # レース順(race_id は年→場→…の昇順なので時系列代わりに使える)
    df = df.sort_values("race_id").reset_index(drop=True)

    # field_size(出走頭数)
    df["field_size"] = df.groupby("race_id")["umaban"].transform("count")

    # 前走着順: 同じ馬を時系列で並べて1つ前の着順を引く。初出走は中央値で埋める。
    df["prev_finish"] = df.groupby("umamei")["finish"].shift(1)
    df["prev_finish"] = df["prev_finish"].fillna(df["finish"].median())

    # 騎手勝率: 取得データ内での勝率(=1着回数/騎乗回数)。
    #   ※簡易版。厳密にはリーク防止で「過去だけ」で計算するのが理想。
    win = (df["finish"] == 1).astype(int)
    df["jockey_win_rate"] = df.groupby("kishu")["finish"].transform(
        lambda s: (s == 1).mean()).round(3)

    df["won"] = (df["finish"] == 1).astype(int)
    df["is_top3"] = (df["finish"] <= 3).astype(int)

    # 欠損を最終的に落とす
    need = ["distance", "kinryo", "tansho_odds", "bataiju", "age"]
    df = df.dropna(subset=need).reset_index(drop=True)
    return df


if __name__ == "__main__":
    # 使用例(自分のPCで): 取得したいレースIDを並べる
    race_ids = [
        "202105030411", "202105030412",   # ← 実在のレースIDに置き換える
    ]
    print("netkeiba から取得します(アクセス間隔 %.1f秒)..." % SLEEP)
    raw = scrape_races(race_ids)
    data = to_pipeline_format(raw)
    data.to_csv("data/races.csv", index=False)
    print(f"\n整形完了: {len(data)}行 -> data/races.csv")
    print(data.head().to_string(index=False))
    print("\n次: python3 src/run_all.py を実データ版に向けて調整して実行")
