import os
import json
import math
import random
import requests
from datetime import datetime
from zoneinfo import ZoneInfo

TZ = ZoneInfo("America/New_York")

SYMBOLS = [
    "AAPL","MSFT","GOOGL","AMZN","TSLA","NVDA","META","BABA","TSM","AMD","NFLX","INTC",
    "PDD","ORCL","ADBE","SHOP","CRM","COIN","PLTR","ASML","AVGO","JPM",
]

REASONS = [
    "新产品周期带来利润改善",
    "技术面突破关键阻力位",
    "机构上调盈利预期",
    "估值回到历史区间低位",
    "行业景气度持续上行",
    "现金流稳健且回购力度增加",
    "政策利好推动需求增长",
    "基本面改善叠加催化事件",
    "订单能见度提升",
    "市场情绪回暖",
]

RECOMMENDERS = ["晨星策略组","蓝海研究院","趋势量化团队","北极星投研","红杉资本顾问","聚焦财报分析"]

def stooq_symbol(sym: str) -> str:
    return sym.lower() + ".us"

def fetch_daily_candles(sym: str):
    url = f"https://stooq.com/q/d/l/?s={stooq_symbol(sym)}&i=d"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    lines = r.text.strip().splitlines()
    if len(lines) < 3:
        return []

    out = []
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) < 5:
            continue
        d, o, h, l, c = parts[:5]
        try:
            o, h, l, c = float(o), float(h), float(l), float(c)
        except:
            continue
        if not d or any(map(lambda x: math.isnan(x), [o,h,l,c])):
            continue
        out.append({"time": d, "open": o, "high": h, "low": l, "close": c})

    out.sort(key=lambda x: x["time"])
    return out

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def main():
    now = datetime.now(TZ)
    last_updated = now.strftime("%Y/%m/%d %H:%M")

    ensure_dir("data")
    ensure_dir("data/candles")

    rows = []

    for sym in SYMBOLS:
        candles = fetch_daily_candles(sym)
        current_price = candles[-1]["close"] if candles else None

        # 这里的“推荐指数/目标价/理由/推荐人”先用示例生成
        score = random.randint(60, 95)
        if current_price:
            target = round(current_price * random.uniform(1.08, 1.35), 2)
        else:
            target = round(random.uniform(80, 320), 2)

        reason = random.choice(REASONS)
        recommender = random.choice(RECOMMENDERS)

        # 写 candles 文件（包含现价 + 全部日K）
        with open(f"data/candles/{sym}.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "symbol": sym,
                    "currentPrice": current_price,
                    "candles": candles,
                    "lastUpdated": last_updated,
                },
                f,
                ensure_ascii=False,
            )

        rows.append(
            {
                "symbol": sym,
                "score": score,
                "currentPrice": current_price,
                "targetPrice": target,
                "reason": reason,
                "recommender": recommender,
            }
        )

    # 只展示前 20 条：按 score 排序
    rows.sort(key=lambda x: x["score"], reverse=True)
    rows = rows[:20]

    with open("data/quotes.json", "w", encoding="utf-8") as f:
        json.dump({"lastUpdated": last_updated, "rows": rows}, f, ensure_ascii=False)

if __name__ == "__main__":
    main()