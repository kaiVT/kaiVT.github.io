import os
import json
import math
import csv
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import requests


# =========================
# Config
# =========================
TZ = ZoneInfo("America/New_York")

START_HOUR = int(os.getenv("START_HOUR", "8"))   # 8:00 AM ET
END_HOUR   = int(os.getenv("END_HOUR", "20"))    # 8:00 PM ET (20:00)

FORCE_RUN = os.getenv("FORCE_RUN", "0") == "1"

DEFAULT_TICKERS = ["ZETA", "HROW"]  # ✅ 你可以直接改这里

TARGET_DTE = int(os.getenv("TARGET_DTE", "30"))          # 目标到期天数
DTE_WINDOW = int(os.getenv("DTE_WINDOW", "7"))           # 允许偏差 ±N 天（如果抓不到，先调大到 30~45）
MAX_EXPIRATIONS_PER_TICKER = int(os.getenv("MAX_EXPS", "2"))
MAX_ROWS_PER_TICKER = int(os.getenv("MAX_ROWS_PER_TICKER", "80"))

SLEEP_BETWEEN_TICKERS = float(os.getenv("SLEEP_TICKER", "0.8"))
SLEEP_BETWEEN_CALLS = float(os.getenv("SLEEP_CALL", "0.2"))

OUT_JSON = os.getenv("OUT_JSON", "docs/data/options.json")
OUT_CSV = os.getenv("OUT_CSV", "docs/data/options.csv")

# Yahoo URLs (unofficial)
YAHOO_OPTIONS_PAGE = "https://finance.yahoo.com/quote/{ticker}/options?p={ticker}"

YAHOO_OPTIONS_APIS = [
    "https://query2.finance.yahoo.com/v7/finance/options/{ticker}",
    "https://query1.finance.yahoo.com/v7/finance/options/{ticker}",
]

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
})


# =========================
# Helpers
# =========================
def ensure_dir(path: str):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)


def within_et_window(now_et: datetime) -> bool:
    return (now_et.hour >= START_HOUR) and (now_et.hour < END_HOUR)


def safe_float(x):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def prob_itm_put(spot: float, strike: float, iv: float, dte: int) -> float | None:
    """
    粗略估算 Put 到期变成 ITM 的概率：P(S_T <= K)
    - 仅用于排序/筛选（测试阶段），不是交易建议
    """
    if spot is None or strike is None or iv is None or dte is None:
        return None
    if spot <= 0 or strike <= 0 or iv <= 0 or dte <= 0:
        return None

    T = dte / 365.0
    vol_sqrt_t = iv * math.sqrt(T)
    if vol_sqrt_t <= 0:
        return None

    z = (math.log(strike / spot) + 0.5 * (iv * iv) * T) / vol_sqrt_t
    p = norm_cdf(z)
    return max(0.0, min(1.0, p))


def annualized_yield_bid(bid: float, strike: float, dte: int) -> float | None:
    """
    Cash-secured put: (权利金/保证金) * (365/dte)
    bid/strike * (365/dte)
    """
    if bid is None or strike is None or dte is None:
        return None
    if bid <= 0 or strike <= 0 or dte <= 0:
        return None
    return (bid / strike) * (365.0 / dte)


def prime_yahoo_session(ticker: str):
    """
    关键：先访问网页，拿 cookie/session，否则 GitHub Actions 上经常 401/403。
    """
    url = YAHOO_OPTIONS_PAGE.format(ticker=ticker)
    try:
        SESSION.get(
            "https://finance.yahoo.com/",
            timeout=15,
            headers={"Referer": "https://finance.yahoo.com/"},
        )
        time.sleep(SLEEP_BETWEEN_CALLS)
        SESSION.get(
            url,
            timeout=15,
            headers={"Referer": "https://finance.yahoo.com/"},
        )
        time.sleep(SLEEP_BETWEEN_CALLS)
    except Exception:
        # prime 失败也不一定致命，继续尝试 API
        pass


def fetch_yahoo_options_json(ticker: str, params=None, timeout=20):
    """
    依次尝试 query2 / query1。
    若 401/403：重新 prime 一次 session 再重试。
    """
    last_err = None
    referer = YAHOO_OPTIONS_PAGE.format(ticker=ticker)

    for api in YAHOO_OPTIONS_APIS:
        url = api.format(ticker=ticker)

        # 每个域名最多尝试 2 次：第一次直接请求；如果 401/403，prime 后再来一次
        for attempt in range(2):
            try:
                r = SESSION.get(
                    url,
                    params=params,
                    timeout=timeout,
                    headers={"Referer": referer},
                )

                if r.status_code in (401, 403):
                    last_err = f"{r.status_code} blocked: {url}"
                    if attempt == 0:
                        prime_yahoo_session(ticker)
                        continue
                    else:
                        break

                r.raise_for_status()
                return r.json()

            except Exception as e:
                last_err = str(e)
                if attempt == 0:
                    prime_yahoo_session(ticker)
                time.sleep(SLEEP_BETWEEN_CALLS)

    raise RuntimeError(f"Yahoo options API blocked/failed. Last error: {last_err}")


def pick_expirations(exp_ts_list, now_utc: datetime):
    """
    选最接近 TARGET_DTE 的到期日（允许 ±DTE_WINDOW），最多 MAX_EXPIRATIONS_PER_TICKER 个。
    """
    picked = []
    for ts in exp_ts_list:
        exp_dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
        dte = (exp_dt.date() - now_utc.date()).days
        if dte <= 0:
            continue

        if abs(dte - TARGET_DTE) <= DTE_WINDOW:
            picked.append((abs(dte - TARGET_DTE), dte, int(ts), exp_dt))

    picked.sort(key=lambda x: x[0])
    return picked[:MAX_EXPIRATIONS_PER_TICKER]


# =========================
# Main
# =========================
def main():
    now_utc = datetime.now(timezone.utc)
    now_et = now_utc.astimezone(TZ)

    if (not FORCE_RUN) and (not within_et_window(now_et)):
        print(f"Skip: outside ET window. Now ET = {now_et.isoformat()}")
        return

    tickers_env = os.getenv("TICKERS", "")
    tickers = [t.strip().upper() for t in tickers_env.split(",") if t.strip()] or DEFAULT_TICKERS

    all_rows = []
    errors = []

    for ticker in tickers:
        try:
            prime_yahoo_session(ticker)

            # 先拿 expirations + spot
            data0 = fetch_yahoo_options_json(ticker)
            result = (data0.get("optionChain", {}) or {}).get("result", []) or []
            if not result:
                errors.append({"ticker": ticker, "error": "no result from yahoo"})
                continue

            r0 = result[0]
            quote = r0.get("quote", {}) or {}
            spot = (
                safe_float(quote.get("regularMarketPrice"))
                or safe_float(quote.get("postMarketPrice"))
                or safe_float(quote.get("previousClose"))
            )

            exp_list = r0.get("expirationDates", []) or []
            if not exp_list:
                errors.append({"ticker": ticker, "error": "missing expirationDates"})
                continue
            if not spot:
                errors.append({"ticker": ticker, "error": "missing spot price"})
                continue

            picked_exps = pick_expirations(exp_list, now_utc)

            # 如果窗口内找不到，就退而求其次：选最近的 1 个到期日（保证有数据先）
            if not picked_exps:
                ts = int(exp_list[0])
                exp_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                dte = (exp_dt.date() - now_utc.date()).days
                picked_exps = [(999, dte, ts, exp_dt)]

            rows_for_ticker = 0

            for _, dte, exp_ts, exp_dt in picked_exps:
                data = fetch_yahoo_options_json(ticker, params={"date": exp_ts})
                result2 = (data.get("optionChain", {}) or {}).get("result", []) or []
                if not result2:
                    continue

                r = result2[0]
                opts = (r.get("options", []) or [{}])[0]
                puts = opts.get("puts", []) or []

                for p in puts:
                    strike = safe_float(p.get("strike"))
                    bid = safe_float(p.get("bid"))
                    ask = safe_float(p.get("ask"))
                    iv = safe_float(p.get("impliedVolatility"))  # 小数，如 0.85
                    oi = p.get("openInterest")
                    vol = p.get("volume")

                    if strike is None or strike <= 0:
                        continue

                    # 价格差：现价-行权价（越大越 OTM）
                    price_diff = (spot - strike) / spot if spot else None
                    price_diff_pct = (price_diff * 100.0) if price_diff is not None else None

                    # premium：用 bid 更保守（你卖出更接近能拿到的）
                    premium = (bid * 100.0) if (bid is not None and bid > 0) else None

                    apr = annualized_yield_bid(bid, strike, dte) if bid else None
                    apr_pct = (apr * 100.0) if apr is not None else None

                    p_itm = prob_itm_put(spot, strike, iv, dte) if iv else None
                    p_itm_pct = (p_itm * 100.0) if p_itm is not None else None

                    # 简单评分：APR * (1-ITM概率) * 100（只是排序用）
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
                        "premium_$": premium,
                        "price_diff_%": price_diff_pct,

                        "iv": iv,
                        "prob_itm_%": p_itm_pct,

                        "apr_%": apr_pct,
                        "cash_collateral_$": strike * 100.0,  # cash-secured 保守保证金

                        "open_interest": oi,
                        "volume": vol,

                        "score": score,
                    })

                    rows_for_ticker += 1
                    if rows_for_ticker >= MAX_ROWS_PER_TICKER:
                        break

                if rows_for_ticker >= MAX_ROWS_PER_TICKER:
                    break

                time.sleep(SLEEP_BETWEEN_CALLS)

        except Exception as e:
            errors.append({"ticker": ticker, "error": str(e)})

        time.sleep(SLEEP_BETWEEN_TICKERS)

    # 排序：score 高 -> 低
    def sort_key(x):
        s = x.get("score")
        return -(s if isinstance(s, (int, float)) else -1e18)

    all_rows.sort(key=sort_key)

    payload = {
        "generated_at_utc": now_utc.isoformat(),
        "generated_at_et": now_et.isoformat(),
        "tickers": tickers,
        "target_dte": TARGET_DTE,
        "dte_window": DTE_WINDOW,
        "rows": all_rows,
        "errors": errors,
        "note": "Yahoo data is unofficial; for testing only. Prob/score are rough estimates.",
    }

    ensure_dir(OUT_JSON)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    ensure_dir(OUT_CSV)
    fieldnames = [
        "ticker","expiration","dte","spot",
        "strike","bid","ask","premium_$","price_diff_%",
        "iv","prob_itm_%","apr_%","cash_collateral_$",
        "open_interest","volume","score"
    ]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k) for k in fieldnames})

    print(f"✅ Wrote: {OUT_JSON} ({len(all_rows)} rows)")
    print(f"✅ Wrote: {OUT_CSV}")
    if errors:
        print("⚠️ Errors (first 5):", errors[:5])


if __name__ == "__main__":
    main()
