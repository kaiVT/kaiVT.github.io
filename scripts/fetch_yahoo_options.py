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
START_HOUR = int(os.getenv("START_HOUR", "8"))
END_HOUR = int(os.getenv("END_HOUR", "20"))
FORCE_RUN = os.getenv("FORCE_RUN", "0") == "1"

DEFAULT_TICKERS = ["ZETA", "HROW"]  # 你也可以在 workflow 用 env: TICKERS 覆盖
TARGET_DTE = int(os.getenv("TARGET_DTE", "30"))
DTE_WINDOW = int(os.getenv("DTE_WINDOW", "45"))
MAX_EXPIRATIONS_PER_TICKER = int(os.getenv("MAX_EXPS", "2"))
MAX_ROWS_PER_TICKER = int(os.getenv("MAX_ROWS_PER_TICKER", "80"))

# 关键：每次 run 更新几个 ticker（建议 1，避免 429）
MAX_TICKERS_PER_RUN = int(os.getenv("MAX_TICKERS_PER_RUN", "1"))

# 请求间隔（放慢一点更稳）
SLEEP_CALL = float(os.getenv("SLEEP_CALL", "0.6"))
SLEEP_TICKER = float(os.getenv("SLEEP_TICKER", "1.2"))

OUT_JSON = os.getenv("OUT_JSON", "docs/data/options.json")
OUT_CSV = os.getenv("OUT_CSV", "docs/data/options.csv")

# Yahoo (unofficial)
YAHOO_OPTIONS_PAGE = "https://finance.yahoo.com/quote/{ticker}/options?p={ticker}"
YAHOO_APIS = [
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


def prob_itm_put(spot: float, strike: float, iv: float, dte: int):
    # 粗略估计：仅用于排序/筛选
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


def annualized_yield_bid(bid: float, strike: float, dte: int):
    if bid is None or strike is None or dte is None:
        return None
    if bid <= 0 or strike <= 0 or dte <= 0:
        return None
    return (bid / strike) * (365.0 / dte)


def load_previous_payload():
    if not os.path.exists(OUT_JSON):
        return None
    try:
        with open(OUT_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def prime_yahoo(ticker: str):
    # 先访问主页 + options 页面，拿 cookie/session
    try:
        SESSION.get("https://finance.yahoo.com/", timeout=15,
                    headers={"Referer": "https://finance.yahoo.com/"})
        time.sleep(SLEEP_CALL)
        SESSION.get(YAHOO_OPTIONS_PAGE.format(ticker=ticker), timeout=15,
                    headers={"Referer": "https://finance.yahoo.com/"})
        time.sleep(SLEEP_CALL)
    except Exception:
        pass


def request_with_backoff(url: str, params: dict | None, referer: str, max_attempts=6):
    """
    处理 429：指数退避；优先用 Retry-After
    """
    last_err = None
    for attempt in range(max_attempts):
        try:
            r = SESSION.get(url, params=params, timeout=25, headers={"Referer": referer})

            # 429: Too Many Requests
            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    wait = int(retry_after)
                else:
                    wait = min(2 ** attempt, 60)  # 1,2,4,8,16,32,60
                last_err = f"429 Too Many Requests (wait {wait}s)"
                time.sleep(wait)
                continue

            # 401/403: session/cookie 不够，交给上层 prime 再试
            if r.status_code in (401, 403):
                last_err = f"{r.status_code} blocked"
                raise RuntimeError(last_err)

            r.raise_for_status()
            return r.json()

        except Exception as e:
            last_err = str(e)
            # 小等一下再继续
            time.sleep(min(1.5 * (attempt + 1), 10))

    raise RuntimeError(last_err or "request failed")


def fetch_yahoo_options_json(ticker: str, params=None):
    referer = YAHOO_OPTIONS_PAGE.format(ticker=ticker)

    # 每个域名尝试：先请求；若 401/403，prime 后再请求；若 429，会自动 backoff
    for api in YAHOO_APIS:
        url = api.format(ticker=ticker)

        # 第一次尝试
        try:
            return request_with_backoff(url, params, referer)
        except Exception:
            # prime 后第二次尝试
            prime_yahoo(ticker)
            try:
                return request_with_backoff(url, params, referer)
            except Exception as e2:
                last = str(e2)
                continue

    raise RuntimeError(f"Yahoo blocked/failed for {ticker}")


def pick_expirations(exp_ts_list, now_utc: datetime):
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


def choose_tickers_for_this_run(tickers: list[str]) -> list[str]:
    """
    每次只更新一部分 ticker，避免 429。
    用 GITHUB_RUN_NUMBER 做轮换（没有就用时间戳）
    """
    if MAX_TICKERS_PER_RUN >= len(tickers):
        return tickers

    run_no = os.getenv("GITHUB_RUN_NUMBER")
    if run_no and run_no.isdigit():
        idx = int(run_no) % len(tickers)
    else:
        idx = int(time.time() // 300) % len(tickers)  # 每5分钟变化一次

    chosen = []
    for i in range(MAX_TICKERS_PER_RUN):
        chosen.append(tickers[(idx + i) % len(tickers)])
    return chosen


def merge_rows(old_rows: list[dict], new_rows: list[dict], updated_tickers: set[str]) -> list[dict]:
    # 移除被更新 ticker 的旧行，然后加新行
    kept = [r for r in (old_rows or []) if r.get("ticker") not in updated_tickers]
    return kept + (new_rows or [])


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

    prev = load_previous_payload()
    prev_rows = (prev or {}).get("rows", []) if prev else []
    prev_errors = (prev or {}).get("errors", []) if prev else []

    chosen = choose_tickers_for_this_run(tickers)
    print(f"Chosen tickers this run: {chosen}")

    new_rows_all = []
    errors = []

    updated_tickers = set()

    for ticker in chosen:
        try:
            prime_yahoo(ticker)

            data0 = fetch_yahoo_options_json(ticker)
            result = (data0.get("optionChain", {}) or {}).get("result", []) or []
            if not result:
                raise RuntimeError("no result")

            r0 = result[0]
            quote = r0.get("quote", {}) or {}
            spot = (
                safe_float(quote.get("regularMarketPrice"))
                or safe_float(quote.get("postMarketPrice"))
                or safe_float(quote.get("previousClose"))
            )
            exp_list = r0.get("expirationDates", []) or []
            if not spot:
                raise RuntimeError("missing spot")
            if not exp_list:
                raise RuntimeError("missing expirationDates")

            picked_exps = pick_expirations(exp_list, now_utc)

            # 如果没挑到（窗口内没有），用最近一个先保证有数据
            if not picked_exps:
                ts = int(exp_list[0])
                exp_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                dte = (exp_dt.date() - now_utc.date()).days
                picked_exps = [(999, dte, ts, exp_dt)]

            rows_for_ticker = []
            rows_count = 0

            # data0 默认已经包含“某个到期日”的 options（通常是最近到期）
            # 如果我们选中的到期日就是 data0 的那个，就不需要再打一枪 date=xx
            def extract_chain_payload(chain_json):
                res = (chain_json.get("optionChain", {}) or {}).get("result", []) or []
                if not res:
                    return None
                rr = res[0]
                opts = (rr.get("options", []) or [{}])[0]
                return rr, opts

            rr0, opts0 = extract_chain_payload(data0)
            default_exp_ts = None
            if opts0 and "expirationDate" in opts0:
                default_exp_ts = opts0.get("expirationDate")

            for _, dte, exp_ts, exp_dt in picked_exps:
                if default_exp_ts and int(default_exp_ts) == int(exp_ts):
                    rr, opts = rr0, opts0
                else:
                    chain = fetch_yahoo_options_json(ticker, params={"date": exp_ts})
                    rr, opts = extract_chain_payload(chain)
                    time.sleep(SLEEP_CALL)

                puts = (opts.get("puts", []) or []) if opts else []
                for p in puts:
                    strike = safe_float(p.get("strike"))
                    bid = safe_float(p.get("bid"))
                    ask = safe_float(p.get("ask"))
                    iv = safe_float(p.get("impliedVolatility"))
                    oi = p.get("openInterest")
                    vol = p.get("volume")

                    if strike is None or strike <= 0:
                        continue

                    price_diff = (spot - strike) / spot if spot else None
                    price_diff_pct = (price_diff * 100.0) if price_diff is not None else None

                    premium = (bid * 100.0) if (bid is not None and bid > 0) else None

                    apr = annualized_yield_bid(bid, strike, dte) if bid else None
                    apr_pct = (apr * 100.0) if apr is not None else None

                    p_itm = prob_itm_put(spot, strike, iv, dte) if iv else None
                    p_itm_pct = (p_itm * 100.0) if p_itm is not None else None

                    score = None
                    if apr is not None and p_itm is not None:
                        score = apr * (1.0 - p_itm) * 100.0

                    rows_for_ticker.append({
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
                        "cash_collateral_$": strike * 100.0,

                        "open_interest": oi,
                        "volume": vol,

                        "score": score,
                    })

                    rows_count += 1
                    if rows_count >= MAX_ROWS_PER_TICKER:
                        break

                if rows_count >= MAX_ROWS_PER_TICKER:
                    break

            # 排序一下（score 大->小）
            rows_for_ticker.sort(key=lambda x: (x.get("score") is None, -(x.get("score") or -1e18)))

            new_rows_all.extend(rows_for_ticker)
            updated_tickers.add(ticker)

        except Exception as e:
            errors.append({"ticker": ticker, "error": str(e)})

        time.sleep(SLEEP_TICKER)

    # 如果这次全失败：保留旧 rows 不动，避免页面变空
    if not updated_tickers:
        final_rows = prev_rows
        final_errors = errors or prev_errors
        print("All tickers failed this run; keeping previous rows.")
    else:
        final_rows = merge_rows(prev_rows, new_rows_all, updated_tickers)
        final_errors = errors  # 只展示本次错误更清晰

    payload = {
        "generated_at_utc": now_utc.isoformat(),
        "generated_at_et": now_et.isoformat(),
        "tickers": tickers,
        "tickers_updated_this_run": sorted(list(updated_tickers)),
        "target_dte": TARGET_DTE,
        "dte_window": DTE_WINDOW,
        "rows": final_rows,
        "errors": final_errors,
        "note": "Yahoo data is unofficial; GitHub Actions IPs may get rate-limited. Script keeps last good data.",
    }

    ensure_dir(OUT_JSON)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # CSV（输出当前页面会展示的 rows）
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
        for r in final_rows:
            w.writerow({k: r.get(k) for k in fieldnames})

    print(f"✅ Wrote {OUT_JSON} | rows={len(final_rows)} | updated={sorted(list(updated_tickers))}")
    if final_errors:
        print("⚠️ Errors:", final_errors[:5])


if __name__ == "__main__":
    main()
