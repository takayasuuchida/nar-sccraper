"""
scrape_nar.py  ―― 地方競馬(NAR)のレースを db.netkeiba.com から取得

『薄い市場ほど歪み(プラスEV)が残る』仮説(E)を検証するためのNARデータ収集。
JRA用 scrape_netkeiba.py の整形(to_pipeline_format)を再利用する。

⚠ 必ず守ること(JRA版と同じ):
  ・利用規約 / robots.txt を確認(本コードも robots を1回チェック)。
  ・アクセス間隔(SLEEP=1.5秒)を短くしない。大量取得しない。個人の学習利用のみ。

★ この環境では netkeiba がブロックされるので、実行は『自分のPC』で。
   netkeiba に繋がらない場合はネットワーク制限が原因(ローカルなら通る)。

使い方(自分のPCで):
  python src\\scrape_nar.py 2023 44      ← 2023年・大井(44)を収集
  # 主要場コード: 門別30 盛岡35 水沢36 浦和42 船橋43 大井44 川崎45
  #               金沢46 笠松47 名古屋48 園田50 姫路51 高知54 佐賀55
取得 → data/races_nar.csv に保存 → 検証:
  python src\\market_edge.py data\\races_nar.csv
"""

import io
import os
import re
import sys
import time
import urllib.robotparser as robotparser

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

import scrape_netkeiba as sk      # BASE, HEADERS, SLEEP, to_pipeline_format を再利用

MAX_RACE = 12                     # 1開催日あたり最大レース数


def _robots_ok() -> bool:
    rp = robotparser.RobotFileParser()
    rp.set_url(f"{sk.BASE}/robots.txt")
    try:
        rp.read()
    except Exception:
        return False
    return rp.can_fetch(sk.HEADERS["User-Agent"], f"{sk.BASE}/race/")


def fetch_one(race_id: str):
    """1レースの結果表を取得。無ければ None。
    どんな例外も飲み込んで None を返す(1ページの失敗で年間収集を落とさない)。
    pandas.read_html がページ内の日付などを誤って月解釈し
    calendar.IllegalMonthError を投げるケースがあるため、ここで握りつぶす。"""
    try:
        url = f"{sk.BASE}/race/{race_id}/"
        res = requests.get(url, headers=sk.HEADERS, timeout=15)
        if res.status_code != 200:
            return None
        res.encoding = "euc-jp"
        html = res.text
        if "着" not in html:                       # 結果表が無いページ
            return None
        tables = pd.read_html(io.StringIO(html))
        result = max(tables, key=len)
        result.columns = [str(c).replace(" ", "") for c in result.columns]
        if not any("着順" in c for c in result.columns):
            return None
        m = re.search(r"(\d{3,4})m", BeautifulSoup(html, "lxml").get_text()[:600])
        result["race_id"] = race_id
        result["distance"] = int(m.group(1)) if m else np.nan
        return result
    except Exception as e:                          # IllegalMonthError 等もここで止める
        print(f"      ! {race_id} skip: {type(e).__name__}: {e}")
        return None


def collect(year: int, track: int, max_kai: int = 20, max_day: int = 14):
    """track の年間レースを収集。
    開催回(kai)はトラックによって 1 から始まらない(大井は8回〜など)ため、
    『先頭の空き回はスキップして探し続け』『データが出た後に空き回が3回続いたら終了』する。
    """
    frames, got, tried = [], 0, 0
    found_any = False                          # 一度でもデータを拾ったか
    empty_kai_streak = 0                        # データ発見後の連続空き回
    for kai in range(1, max_kai + 1):
        empty_days = 0
        kai_hits = 0
        for day in range(1, max_day + 1):
            day_hits = 0
            for race in range(1, MAX_RACE + 1):
                rid = f"{year}{track:02d}{kai:02d}{day:02d}{race:02d}"
                tried += 1
                df = fetch_one(rid)
                time.sleep(sk.SLEEP)           # ← サーバ負荷軽減。消さない。
                if df is None:
                    if race == 1:              # race1が無い=この開催日は無い
                        break
                    continue
                frames.append(df)
                got += 1
                day_hits += 1
            if day_hits == 0:
                empty_days += 1
            else:
                empty_days = 0
                kai_hits += day_hits
                print(f"    {year}/{track:02d} {kai}回{day}日: {day_hits}R 取得 (累計{got}R)")
            if empty_days >= 2:                # 開催日が尽きた
                break
        # 開催回(kai)単位の打ち切り判定: 先頭の空き回は飛ばし、
        # データが出始めた後に空き回が3回続いたら年間収集を終える。
        if kai_hits > 0:
            found_any = True
            empty_kai_streak = 0
        elif found_any:
            empty_kai_streak += 1
            if empty_kai_streak >= 3:
                break
    print(f"  試行 {tried} / 取得 {got}R")
    return pd.concat(frames, ignore_index=True) if frames else None


TRACK_NAMES = {30: "門別", 35: "盛岡", 36: "水沢", 42: "浦和", 43: "船橋",
               44: "大井", 45: "川崎", 46: "金沢", 47: "笠松", 48: "名古屋",
               50: "園田", 51: "姫路", 54: "高知", 55: "佐賀"}


def _rebuild_combined() -> str:
    """data/races_nar_*.csv を全部つないで data/races_nar.csv を作り直す。
    年・場ごとの shard を貯めていけば、複数年/複数場が自動で1本に統合される。
    """
    import glob
    shards = sorted(glob.glob("data/races_nar_*.csv"))
    if not shards:
        return ""
    allf = [pd.read_csv(p) for p in shards]
    combined = pd.concat(allf, ignore_index=True).drop_duplicates(
        subset=["race_id", "umaban"]
    )
    out = "data/races_nar.csv"
    combined.to_csv(out, index=False)
    return out


def main():
    if len(sys.argv) < 3:
        print("使い方: python src\\scrape_nar.py <年> <場コード...>")
        print("  例(1場) : python src\\scrape_nar.py 2023 44")
        print("  例(4場) : python src\\scrape_nar.py 2023 44 43 45 42  ← 南関東4場")
        return
    year = int(sys.argv[1])
    tracks = [int(t) for t in sys.argv[2:]]

    print("=" * 60)
    names = "・".join(f"{t:02d}{TRACK_NAMES.get(t, '')}" for t in tracks)
    print(f" NAR収集: {year}年  対象{len(tracks)}場 [{names}]")
    print("=" * 60)
    if not _robots_ok():
        print("❌ robots.txt により取得不可。中止します。")
        return
    print("収集開始(アクセス間隔 %.1f秒)。netkeibaに繋がらない場合はローカルで実行を。\n" % sk.SLEEP)

    os.makedirs("data", exist_ok=True)
    summary = []
    for track in tracks:
        nm = TRACK_NAMES.get(track, "")
        print(f"\n――― {year}年 {track:02d}{nm} ―――")
        try:
            raw = collect(year, track)
            if raw is None:
                print(f"  {track:02d}{nm}: 1件も取得できず(年・場コードを確認)")
                summary.append((track, nm, 0, 0))
                continue
            data = sk.to_pipeline_format(raw)
            shard = f"data/races_nar_{track:02d}_{year}.csv"
            data.to_csv(shard, index=False)        # 場ごとに即保存(落ちても残る)
            n_races = data.race_id.nunique()
            print(f"  {track:02d}{nm}: {len(data):,}行 / {n_races:,}レース -> {shard}")
            summary.append((track, nm, len(data), n_races))
        except Exception as e:                      # 1場失敗しても次の場へ進む
            print(f"  {track:02d}{nm}: 整形中にエラー → スキップ ({type(e).__name__}: {e})")
            summary.append((track, nm, 0, 0))
            continue

    out = _rebuild_combined()
    print("\n" + "=" * 60)
    print(" 収集サマリ")
    print("=" * 60)
    for track, nm, rows, races in summary:
        print(f"  {track:02d}{nm:<4}: {races:>4}レース / {rows:>6}行")
    if out:
        comb = pd.read_csv(out)
        print(f"\n統合: {len(comb):,}行 / {comb.race_id.nunique():,}レース -> {out}")
        print("横並び検証: python src\\market_edge.py data\\races_nar.csv")
    else:
        print("\n保存できたデータがありませんでした。")


if __name__ == "__main__":
    main()
