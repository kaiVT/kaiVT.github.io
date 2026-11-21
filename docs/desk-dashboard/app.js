// 自选股列表 —— 想加或删直接改这里。
// 页面只展示 ticker 和市场价格，不会显示持股数量或市值。
const WATCHLIST = [
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
];

const TZ = "America/New_York";
const WEATHER_URL =
  "https://api.open-meteo.com/v1/forecast?latitude=38.9586&longitude=-77.3570&current=temperature_2m,weather_code,apparent_temperature,relative_humidity_2m,wind_speed_10m&daily=temperature_2m_max,temperature_2m_min,weather_code&temperature_unit=fahrenheit&wind_speed_unit=mph&timezone=America%2FNew_York";

const WEEKDAYS = ["周日", "周一", "周二", "周三", "周四", "周五", "周六"];
const MONTHS = ["1月", "2月", "3月", "4月", "5月", "6月", "7月", "8月", "9月", "10月", "11月", "12月"];

const WX = {
  0: ["sun", "晴"],
  1: ["cloud-sun", "大部晴朗"],
  2: ["cloud-sun", "局部多云"],
  3: ["cloud", "多云"],
  45: ["cloud-fog", "雾"],
  48: ["cloud-fog", "雾"],
  51: ["cloud-drizzle", "毛毛雨"],
  53: ["cloud-drizzle", "毛毛雨"],
  55: ["cloud-drizzle", "毛毛雨"],
  56: ["cloud-drizzle", "冻毛毛雨"],
  57: ["cloud-drizzle", "冻毛毛雨"],
  61: ["cloud-rain", "小雨"],
  63: ["cloud-rain", "中雨"],
  65: ["cloud-rain", "大雨"],
  66: ["cloud-rain", "冻雨"],
  67: ["cloud-rain", "冻雨"],
  71: ["cloud-snow", "小雪"],
  73: ["cloud-snow", "中雪"],
  75: ["cloud-snow", "大雪"],
  77: ["cloud-snow", "雪粒"],
  80: ["cloud-rain", "阵雨"],
  81: ["cloud-rain", "阵雨"],
  82: ["cloud-rain", "强阵雨"],
  85: ["cloud-snow", "阵雪"],
  86: ["cloud-snow", "阵雪"],
  95: ["cloud-lightning", "雷雨"],
  96: ["cloud-lightning", "雷雨夹冰雹"],
  99: ["cloud-lightning", "强雷雨"],
};


const ICONS = {
  sun: '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"/></svg>',
  "cloud-sun": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 2v2M4.93 4.93l1.41 1.41M2 12h2"/><path d="M10 9a4 4 0 0 1 7.46 1.5"/><path d="M17.5 19H9a7 7 0 1 1 6.71-9h1.79a4.5 4.5 0 1 1 0 9Z"/></svg>',
  cloud: '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M17.5 19H9a7 7 0 1 1 6.71-9h1.79a4.5 4.5 0 1 1 0 9Z"/></svg>',
  "cloud-fog": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 14h14M4 18h10M4 22h14"/><path d="M17.5 8H16a5 5 0 0 0-9.8 1A4 4 0 0 0 4 16"/></svg>',
  "cloud-drizzle": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 14.9A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.2"/><path d="M8 19v1M8 14v1M16 19v1M16 14v1M12 21v1M12 16v1"/></svg>',
  "cloud-rain": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 14.9A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.2"/><path d="M16 14v6M8 14v6M12 16v6"/></svg>',
  "cloud-snow": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 14.9A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.2"/><path d="M8 15h.01M8 19h.01M12 17h.01M12 21h.01M16 15h.01M16 19h.01"/></svg>',
  "cloud-lightning": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M6 16.326A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 .88 8.89"/><path d="M13 12l-3 7h4l-3 7"/></svg>',
};



const els = {
  clock: document.getElementById("clock"),
  seconds: document.getElementById("seconds"),
  ampm: document.getElementById("ampm"),
  date: document.getElementById("date"),
  status: document.getElementById("status"),
  wxIcon: document.getElementById("wxIcon"),
  wxTemp: document.getElementById("wxTemp"),
  wxCond: document.getElementById("wxCond"),
  wxMeta: document.getElementById("wxMeta"),
  stocks: document.getElementById("stocks"),
  quoteSrc: document.getElementById("quoteSrc"),
  error: document.getElementById("error"),
  installHint: document.getElementById("installHint"),
  sessionLamp: document.getElementById("sessionLamp"),
};

let lastQuotesAt = 0;
let lastQuoteSource = "";

if (window.navigator.standalone || window.matchMedia("(display-mode: standalone)").matches) {
  document.body.classList.add("standalone");
}

function partsNow() {
  const fmt = new Intl.DateTimeFormat("en-US", {
    timeZone: TZ,
    weekday: "short",
    month: "numeric",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
    hour12: true,
  });
  const bag = {};
  for (const p of fmt.formatToParts(new Date())) {
    if (p.type !== "literal") bag[p.type] = p.value;
  }
  return bag;
}

function weekdayLabel(en) {
  const map = { Sun: 0, Mon: 1, Tue: 2, Wed: 3, Thu: 4, Fri: 5, Sat: 6 };
  return WEEKDAYS[map[en]] || en;
}

function tickClock() {
  const p = partsNow();
  els.clock.textContent = `${p.hour}:${p.minute}`;
  els.seconds.textContent = p.second;
  els.ampm.textContent = p.dayPeriod === "AM" ? "AM" : "PM";
  els.date.textContent = `${weekdayLabel(p.weekday)} · ${MONTHS[Number(p.month) - 1]}${Number(p.day)}日`;
}

function formatAgo(ts) {
  if (!ts) return "";
  const sec = Math.max(0, Math.round((Date.now() - ts) / 1000));
  if (sec < 15) return "刚刚更新";
  if (sec < 60) return `${sec} 秒前`;
  const min = Math.round(sec / 60);
  if (min < 60) return `${min} 分钟前`;
  return `${Math.round(min / 60)} 小时前`;
}

function marketLabel(status) {
  const s = (status || "").toUpperCase();
  if (s.includes("PRE")) return "盘前";
  if (s.includes("POST") || s.includes("AFTER") || s.includes("EXT")) return "盘后";
  if (s.includes("REG") || s.includes("OPEN")) return "开盘中";
  if (s.includes("CLOSE") || s.includes("HOLIDAY")) return "休市";
  return "";
}

function updateStatus(marketStatus) {
  const mkt = marketLabel(marketStatus);
  const ago = formatAgo(lastQuotesAt);
  const bits = ["ET"];
  if (mkt) bits.push(mkt);
  if (ago) bits.push(ago);
  els.status.textContent = bits.join(" · ");
  if (els.sessionLamp) {
    els.sessionLamp.classList.toggle("open", mkt === "开盘中");
  }
  const src = lastQuoteSource || "";
  els.quoteSrc.textContent = src ? `今日 · ${src}` : "今日";
}

function parseNum(value) {
  if (value == null || value === "") return null;
  if (typeof value === "number") return Number.isFinite(value) ? value : null;
  const n = parseFloat(String(value).replace(/[%$,+]/g, "").replace(/,/g, "").trim());
  return Number.isFinite(n) ? n : null;
}

function formatPrice(n) {
  if (n == null) return "—";
  const abs = Math.abs(n);
  const digits = abs >= 1000 ? 2 : abs < 1 ? 4 : 2;
  return n.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

function formatPct(n) {
  if (n == null) return "—";
  const sign = n > 0 ? "+" : n < 0 ? "" : "";
  return `${sign}${n.toFixed(2)}%`;
}

function chgClass(n) {
  if (n == null || n === 0) return "flat";
  return n > 0 ? "up" : "down";
}

function renderStocks(quotes) {
  const html = WATCHLIST.map((ticker) => {
    const q = quotes[ticker];
    if (!q) {
      return `<article class="stock" data-ticker="${ticker}">
        <div class="ticker">${ticker}</div>
        <div class="price">—</div>
        <div class="chg flat">—</div>
      </article>`;
    }
    const ext = q.extended;
    const showExt =
      ext &&
      ext.price != null &&
      (q.marketStatus || "").toUpperCase().match(/PRE|POST|AFTER|EXT/);
    const extHtml = showExt
      ? `<div class="ext">${marketLabel(q.marketStatus)} ${formatPrice(ext.price)}
           <span class="${chgClass(ext.changePercent)}">${formatPct(ext.changePercent)}</span>
         </div>`
      : "";
    return `<article class="stock" data-ticker="${ticker}">
      <div class="ticker">${ticker}</div>
      <div class="price">${formatPrice(q.price)}</div>
      <div class="chg ${chgClass(q.changePercent)}">${formatPct(q.changePercent)}</div>
      ${extHtml}
    </article>`;
  }).join("");
  els.stocks.innerHTML = html;
}

function skeleton() {
  renderStocks({});
}

function cacheQuotes(quotes, source) {
  try {
    localStorage.setItem(
      "desk-dashboard-quotes",
      JSON.stringify({ quotes, source, at: Date.now() })
    );
  } catch {
    /* ignore quota */
  }
}

function loadCachedQuotes() {
  try {
    const raw = localStorage.getItem("desk-dashboard-quotes");
    if (!raw) return null;
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function cnbcUrl() {
  const symbols = WATCHLIST.join("|");
  return (
    "https://quote.cnbc.com/quote-html-webservice/restQuote/symbolType/symbol" +
    `?symbols=${encodeURIComponent(symbols)}&requestMethod=quick&exthrs=1&noform=1&partnerId=2&output=json`
  );
}

function normalizeCnbc(data) {
  let rows = data?.FormattedQuoteResult?.FormattedQuote;
  if (!rows) throw new Error("CNBC 返回格式不对");
  if (!Array.isArray(rows)) rows = [rows];
  const quotes = {};
  let marketStatus = "";
  for (const row of rows) {
    if (row.code && Number(row.code) !== 0) continue;
    const ext = row.ExtendedMktQuote || {};
    quotes[row.symbol] = {
      price: parseNum(row.last),
      change: parseNum(row.change),
      changePercent: parseNum(row.change_pct),
      previousClose: parseNum(row.previous_day_closing),
      marketStatus: row.curmktstatus || "",
      extended: {
        price: parseNum(ext.last),
        change: parseNum(ext.change),
        changePercent: parseNum(ext.change_pct),
      },
    };
    marketStatus = row.curmktstatus || marketStatus;
  }
  return { quotes, marketStatus };
}

function normalizeFallback(data) {
  const quotes = data.quotes || {};
  return {
    quotes,
    marketStatus: data.marketStatus || "",
    source: data.source || "backup JSON",
  };
}

async function fetchCnbc() {
  const res = await fetch(cnbcUrl(), { cache: "no-store" });
  if (!res.ok) throw new Error(`CNBC HTTP ${res.status}`);
  return normalizeCnbc(await res.json());
}

async function fetchBackupJson() {
  const res = await fetch("./prices.json", { cache: "no-store" });
  if (!res.ok) throw new Error(`prices.json HTTP ${res.status}`);
  const data = await res.json();
  return { ...normalizeFallback(data), updated: data.updated };
}

async function refreshQuotes() {
  els.error.textContent = "";
  try {
    const data = await fetchCnbc();
    lastQuotesAt = Date.now();
    lastQuoteSource = "CNBC";
    renderStocks(data.quotes);
    updateStatus(data.marketStatus);
    cacheQuotes(data.quotes, "CNBC");
    return;
  } catch (err) {
    console.warn("CNBC failed, trying prices.json", err);
  }
  try {
    const data = await fetchBackupJson();
    lastQuotesAt = Date.parse(data.updated) || Date.now();
    lastQuoteSource = data.source ? `${data.source} 备份` : "备份 JSON";
    renderStocks(data.quotes);
    updateStatus(data.marketStatus);
    cacheQuotes(data.quotes, lastQuoteSource);
  } catch (err) {
    const cached = loadCachedQuotes();
    if (cached?.quotes) {
      lastQuotesAt = cached.at || 0;
      lastQuoteSource = (cached.source || "缓存") + "（离线）";
      renderStocks(cached.quotes);
      updateStatus("");
    }
    els.error.textContent = "行情暂时拉不到。过一会儿会自动再试，页面不用一直开着。";
    console.warn(err);
  }
}

async function refreshWeather() {
  try {
    const res = await fetch(WEATHER_URL, { cache: "no-store" });
    if (!res.ok) throw new Error(`weather HTTP ${res.status}`);
    const data = await res.json();
    const cur = data.current || {};
    const daily = data.daily || {};
    const code = cur.weather_code;
    const [icon, label] = WX[code] || ["cloud", "天气"];
    const temp = Math.round(cur.temperature_2m);
    const feel = Math.round(cur.apparent_temperature);
    const hi = daily.temperature_2m_max ? Math.round(daily.temperature_2m_max[0]) : "—";
    const lo = daily.temperature_2m_min ? Math.round(daily.temperature_2m_min[0]) : "—";
    const rh = cur.relative_humidity_2m;
    if (els.wxIcon) els.wxIcon.innerHTML = ICONS[icon] || ICONS.cloud;
    els.wxTemp.textContent = `${temp}°`;
    if (els.wxCond) els.wxCond.textContent = label;
    els.wxMeta.textContent = `Reston, VA · 体感 ${feel}° · 今天 ${hi}° / ${lo}° · 湿度 ${rh}%`;
  } catch (err) {
    els.wxMeta.textContent = "天气暂时拉不到，待会再试";
    console.warn(err);
  }
}

skeleton();
const cached = loadCachedQuotes();
if (cached?.quotes) {
  lastQuotesAt = cached.at || 0;
  lastQuoteSource = cached.source || "";
  renderStocks(cached.quotes);
  updateStatus("");
}

tickClock();
setInterval(tickClock, 1000);
setInterval(() => updateStatus(), 15000);

refreshQuotes();
refreshWeather();
setInterval(refreshQuotes, 60 * 1000);
setInterval(refreshWeather, 20 * 60 * 1000);

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("./sw.js").catch(() => {});
  });
}
