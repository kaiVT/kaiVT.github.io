#!/usr/bin/env python3
"""Write prices.json for the dashboard fallback (no API key)."""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

WATCHLIST = [
    "ASTS",
    "APP",
    "CRDO",
    "SPY",
    "TSLA",
    "NCLD",
    "AVGO",
    "VST",
    "AAOI",
    "CRCL",
    "RKLB",
    "DRAM",
]

UA = "Mozilla/5.0 (compatible; desk-dashboard/1.0)"
OUT = Path(__file__).resolve().parents[1] / "docs" / "desk-dashboard" / "prices.json"


def parse_num(value):
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = re.sub(r"[%$,+]", "", str(value)).replace(",", "").strip()
    try:
        return float(text)
    except ValueError:
        return None


def fetch(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=25) as resp:
        raw = resp.read()
        ctype = resp.headers.get("Content-Type", "")
        text = raw.decode("utf-8")
        if "json" in ctype or text[:1] in "{[":
            return json.loads(text)
        return text


def from_cnbc():
    symbols = urllib.parse.quote("|".join(WATCHLIST), safe="")
    url = (
        "https://quote.cnbc.com/quote-html-webservice/restQuote/symbolType/symbol"
        f"?symbols={symbols}&requestMethod=quick&exthrs=1&noform=1&partnerId=2&output=json"
    )
    data = fetch(url)
    rows = data["FormattedQuoteResult"]["FormattedQuote"]
    if not isinstance(rows, list):
        rows = [rows]
    quotes = {}
    market_status = ""
    for row in rows:
        if row.get("code") not in (0, "0", None):
            continue
        ext = row.get("ExtendedMktQuote") or {}
        quotes[row["symbol"]] = {
            "price": parse_num(row.get("last")),
            "change": parse_num(row.get("change")),
            "changePercent": parse_num(row.get("change_pct")),
            "previousClose": parse_num(row.get("previous_day_closing")),
            "marketStatus": row.get("curmktstatus") or "",
            "extended": {
                "price": parse_num(ext.get("last")),
                "change": parse_num(ext.get("change")),
                "changePercent": parse_num(ext.get("change_pct")),
            },
        }
        market_status = row.get("curmktstatus") or market_status
    if len(quotes) < 3:
        raise RuntimeError(f"CNBC returned too few quotes: {list(quotes)}")
    return quotes, market_status, "CNBC"


def from_yahoo():
    url = (
        "https://query1.finance.yahoo.com/v8/finance/spark?"
        + urllib.parse.urlencode({"symbols": ",".join(WATCHLIST), "range": "5d", "interval": "1d"})
    )
    data = fetch(url)
    quotes = {}
    for sym, row in data.items():
        if not isinstance(row, dict):
            continue
        closes = [c for c in (row.get("close") or []) if c is not None]
        price = closes[-1] if closes else None
        prev = row.get("chartPreviousClose")
        change = row.get("fulldayChange")
        change_pct = row.get("fulldayChangePercent")
        if change is None and price is not None and prev:
            change = price - prev
        if change_pct is None and change is not None and prev:
            change_pct = (change / prev) * 100
        quotes[sym] = {
            "price": parse_num(price),
            "change": parse_num(change),
            "changePercent": parse_num(change_pct),
            "previousClose": parse_num(prev),
            "marketStatus": "",
            "extended": {"price": None, "change": None, "changePercent": None},
        }
    if len(quotes) < 3:
        raise RuntimeError(f"Yahoo returned too few quotes: {list(quotes)}")
    return quotes, "", "Yahoo"


def main() -> None:
    errors = []
    quotes = market_status = source = None
    for loader in (from_cnbc, from_yahoo):
        try:
            quotes, market_status, source = loader()
            break
        except Exception as exc:
            errors.append(f"{loader.__name__}: {exc}")
    if not quotes:
        raise SystemExit("All price sources failed:\n" + "\n".join(errors))
    now = datetime.now(ZoneInfo("America/New_York"))
    payload = {
        "updated": now.isoformat(),
        "source": source,
        "marketStatus": market_status,
        "quotes": quotes,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT} ({len(quotes)} tickers from {source})")


if __name__ == "__main__":
    main()
