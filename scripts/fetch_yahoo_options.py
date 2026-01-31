import os
import json
import math
import csv
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import requests


TZ = ZoneInfo("America/New_York")
START_HOUR = 8   # 8:00 AM ET
END_HOUR = 20    # 8:00 PM ET (20:00)

# ✅ 你先改这个：想扫哪些股票
DEFAULT_TICKERS = ["ZETA", "HROW"]

# 目标到期天数（类似你截图“30天”）
TARGET_DTE = int(os.getenv("TARGET_DTE", "30"))
DTE_WINDOW = int(os.getenv("DTE_WINDOW", "7"))  # 允许 ±7 天
MAX_EXPIRATIONS_PER_TICKER = int(os.getenv("MAX_EXPS", "2"))
MAX_ROWS_PER_TICKER = int(os.getenv("MAX_ROWS_PER_TICKER", "80"))

OUT_JSON = os.getenv("OUT_JSON", "docs/data/options.json")
OUT_CSV = os.getenv("OUT_CSV", "docs/data/options.csv")

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (compatible; options-scanner/1.0; +https://github.com/)",
    "Accept": "application/json",
})

YAHOO_OPTIONS_URL = "https://query2.finance.yahoo.com/v7/finance/options/{ticker}"


def norm_cdf(x: float) -> float:
    # 标准正态分布 CDF
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def prob_itm_put(spot: float, strike: float, iv: float, dte: int) -> float:
    """
    用一个简单的 Black-Scholes 风格近似（假设 r=0）估算 Put 到期 ITM 概率
    P(S_T <= K)
    - 只是测试阶段的近似指标，不是交易建议
    """
    if spot <= 0 or strike <= 0 or iv <= 0 or dte <= 0:
        return None
    T = dte / 365.0
    vol_sqrt_t = iv * math.sqrt(T)
    if vol_sqrt_t <= 0:
        return None

    # z = (ln(K/S) + 0.5*σ^2*T) / (σ*sqrt(T))
    z = (math.log(strike / spot) + 0.5 * (iv * iv) * T) / vol_sqrt_t
    p = norm_cdf(z)

    # clamp
    return max(0.0, min(1.0, p))


def annualized_yield_bid(bid: float, strike: float, dte: int) -> float:
    # Cash-secured put: premium / collateral * (365/dte)
    if bid is None or strike is None or dte is None:
        return None
    if bid <= 0 or strike <= 0 or dte <= 0:
        return None
    return (bid / strike) * (365.0 / dte)


def safe_float(x):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def ensure_dir(path: str):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)


def within_et_window(now_et: datetime) -> bool:
    # 只跑 8:00-20:00（含 8:00，不含 20:00 后的分钟）
    return (now_et.hour >= START_HOUR) and (now_et.hour < END_HOUR)


def fetch_json(url: str, params=None, timeout=15):
    r = SESSION.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def pick_expirations(exp_ts_list, now_utc: datetime):
    """
    选择最接近 TARGET_DTE 的 1~2 个到期日（在窗口内）
    """
    picked = []
    for ts in exp_ts_list:
        exp_dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
        dte = (exp_dt.date() - now_utc.date()).days
        if dte <= 0:
            continue
        if abs(dte - TARGET_DTE) <= DTE_WINDOW:
            picked.append((abs(dte - TARGET_DTE), dte, int(ts), exp_dt))

    picked.sort(key=lambda x: x[0])  # 离 TARGET_DTE 最近
    return picked[:MAX_EXPIRATIONS_PER_TICKER]


def main():
    now_utc = datetime.now(timezone.utc)
    now_et = now_utc.astimezone(TZ)

    # 不在时间范围就直接退出（保持免费但不刷夜里）
    force = os.getenv("FORCE_RUN", "0") == "1"
    if (not force) and (not within_et_window(now_et)):
        print(f"Skip: outside ET window. Now ET = {now_et.isoformat()}")
        return


    tickers_env = os.getenv("TICKERS", "")
    tickers = [t.strip().upper() for t in tickers_env.split(",") if t.strip()] or DEFAULT_TICKERS

    all_rows = []
    errors = []

    for ticker in tickers:
        try:
            base_url = YAHOO_OPTIONS_URL.format(ticker=ticker)
            data0 = fetch_json(base_url)

            result = (data0.get("optionChain", {}) or {}).get("result", []) or []
            if not result:
                errors.append({"ticker": ticker, "error": "no result"})
                continue

            r0 = result[0]
            quote = r0.get("quote", {}) or {}
            spot = safe_float(quote.get("regularMarketPrice")) or safe_float(quote.get("postMarketPrice")) or safe_float(quote.get("previousClose"))

            exp_list = r0.get("expirationDates", []) or []
            if not exp_list or not spot:
                errors.append({"ticker": ticker, "error": "missing expirations or spot"})
                continue

            picked_exps = pick_expirations(exp_list, now_utc)
            if not picked_exps:
                # 没找到窗口内到期日，就选最近的一个
                ts = int(exp_list[0])
                exp_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                dte = (exp_dt.date() - now_utc.date()).days
                picked_exps = [(999, dte, ts, exp_dt)]

            for _, dte, exp_ts, exp_dt in picked_exps:
                data = fetch_json(base_url, params={"date": exp_ts})
                result2 = (data.get("optionChain", {}) or {}).get("result", []) or []
                if not result2:
                    continue
                r = result2[0]
                opts = (r.get("options", []) or [{}])[0]
                puts = opts.get("puts", []) or []

                count_rows = 0
                for p in puts:
                    strike = safe_float(p.get("strike"))
                    bid = safe_float(p.get("bid"))
                    ask = safe_float(p.get("ask"))
                    iv = safe_float(p.get("impliedVolatility"))  # 通常是小数，如 0.85

                    if strike is None or strike <= 0:
                        continue

                    # 过滤：太离谱的深度ITM/OTM先不展示（你可删掉）
                    price_diff = (spot - strike) / spot if spot else None
                    if price_diff is not None and price_diff < -0.50:
                        continue

                    # premium 用 bid 更保守（卖出能拿到的）
                    premium = (bid * 100.0) if (bid is not None and bid > 0) else None
                    apr = annualized_yield_bid(bid, strike, dte) if bid else None
                    p_itm = prob_itm_put(spot, strike, iv, dte) if iv else None

                    # 简单评分：年化 * (1-ITM概率)
                    score = None
                    if apr is not None and p_itm is not None:
                        score = apr * (1.0 - p_itm) * 100.0

                    all_rows.append({
                        "ticker": ticker,
                        "expiration": exp_dt.date().isoformat(),
                        "dte": dte,
                        "spot": spot,

                        "strike": strike,
                        "bid": bid,
                        "ask": ask,
                        "premium_$": premium,  # 1张合约
                        "price_diff_%": (price_diff * 100.0) if price_diff is not None else None,

                        "iv": iv,
                        "prob_itm_%": (p_itm * 100.0) if p_itm is not None else None,

                        "apr_%": (apr * 100.0) if apr is not None else None,
                        "cash_collateral_$": strike * 100.0,  # cash-secured

                        "score": score,
                    })

                    count_rows += 1
                    if count_rows >= MAX_ROWS_PER_TICKER:
                        break

        except Exception as e:
            errors.append({"ticker": ticker, "error": str(e)})

        # 小睡一下，别太猛（更稳定）
        time.sleep(0.8)

    # 默认排序：score 从高到低
    def sort_key(x):
        s = x.get("score")
        return (-s) if isinstance(s, (int, float)) else float("inf")

    all_rows.sort(key=sort_key)

    payload = {
        "generated_at_utc": now_utc.isoformat(),
        "generated_at_et": now_et.isoformat(),
        "target_dte": TARGET_DTE,
        "dte_window": DTE_WINDOW,
        "tickers": tickers,
        "rows": all_rows,
        "errors": errors,
        "note": "Yahoo data is unofficial; for testing only. Prob/score are rough estimates.",
    }

    ensure_dir(OUT_JSON)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # 同时输出 CSV，方便你导入别的地方
    ensure_dir(OUT_CSV)
    fieldnames = [
        "ticker","expiration","dte","spot",
        "strike","bid","ask","premium_$","price_diff_%",
        "iv","prob_itm_%","apr_%","cash_collateral_$","score"
    ]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k) for k in fieldnames})

    print(f"✅ Wrote {OUT_JSON} and {OUT_CSV}")
    if errors:
        print("⚠️ Errors:", errors)


if __name__ == "__main__":
    main()
