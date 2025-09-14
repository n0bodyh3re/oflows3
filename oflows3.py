#!/usr/bin/env python3
"""
Orderflow Engine for Binance (Spot or USDⓈ-M Futures) + Pretty Log + Imbalance Scoring
--------------------------------------------------------------------------------------
Fitur inti (tetap ada, tidak dihapus):
- Delta volume & delta percentage (per bar dan rolling window)
- Volume Profile & POC (dari traded ticks; opsi step bin)
- Liquidity clusters dari orderbook depth (threshold abs/rel)
- Wick detection (upper/lower) dari kline
- Peak volume & peak imbalance (z-score)
- Trapped trader heuristics (lookahead adverse excursion)
- Liquidations (Futures) + mark price (opsional)
- Candlestick pattern detector (shooting_star, hammer, engulfing)
- CSV output per bar + heartbeat JSON tiap N event

Tambahan (merge):
- Pretty CLI alert (--pretty-log) + session info (ASIA/EUROPE/US)
- Footprint-lite imbalance per price (buy vs sell) dari aggTrade (bin by price step)
- Scoring & tiering (Good/Strong/Very Strong) + filter --min-score

NEW (merge tambahan, tanpa menghapus fitur lama):
- BUY/SELL/NEUTRAL signal di pretty-log dan CSV (berdasarkan trapped/delta/pattern/liquidity)
- 15+ advanced orderflow features
- Enhanced pretty logs dengan strength indicator
"""

import asyncio
import json
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, Deque, List, Optional, Tuple
import numpy as np

import pandas as pd
import websockets
import requests, os


# ----------------------------- Discord Alert ----------------------------- #

DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK")  # Render env variable

def send_discord(msg: str):
    if not DISCORD_WEBHOOK:
        return
    try:
        requests.post(DISCORD_WEBHOOK, json={"content": msg})
    except Exception as e:
        print("[ERR] Discord send fail:", e)


# ----------------------------- Configuration ----------------------------- #

@dataclass
class EngineConfig:
    symbol: str = "btcusdt"                 # lowercase per Binance stream names
    interval: str = "1m"                    # kline interval (e.g., 1m, 5m, 15m)
    futures: bool = True                    # if True, use USDⓈ-M futures endpoints & extra streams
    depth_levels: int = 50                  # depth snapshot/updates to track per side
    depth_update: str = "100ms"             # depth stream speed: 100ms or 1000ms
    poc_from: str = "trades"                # "trades" or "depth" (POC calculation base)
    vp_price_step: Optional[float] = None   # price bin size for volume profile; if None, use raw price
    liquidity_abs_threshold: float = 50.0   # absolute threshold (in base asset) to flag a level as "thick"
    liquidity_rel_threshold: float = 4.0    # multiple of rolling median level size to flag as thick
    delta_rolling_secs: int = 30            # rolling window for delta spikes (seconds)
    peak_zscore: float = 3.0                # z-score cutoff to mark a peak in volume/imbalance
    trapped_lookahead_secs: int = 20        # lookahead window to test adverse excursion
    trapped_ae_threshold: float = 0.08      # adverse excursion threshold (%) to label trapped
    trapped_min_imbalance: float = 0.6      # min |delta%| (0..1) to consider one-sided aggression
    wick_min_ratio: float = 0.35            # min wick/body ratio to count as a "long wick"
    session_reset: str = "day"              # "day" or "bar" -> when to reset volume profile/POC
    print_every_n_events: int = 200         # throttle console prints
    write_csv: bool = True
    out_prefix: str = "oflow_"              # CSV outputs prefix
    pretty_log: bool = False                # pretty alert-style CLI output toggle
    # NEW (imbalance + scoring):
    imb_price_step: Optional[float] = None  # bin size khusus footprint-lite (default=vp_price_step)
    imb_ratio_threshold: float = 0.7        # ambang (0..1) imbalance buy/sell per level
    near_px_bps: float = 5.0                # radius "dekat harga" (basis point) utk cluster near close
    min_score: int = 60                     # pretty-log hanya tampil jika score >= min_score
    # Advanced features
    cum_delta_window: int = 300             # window untuk cumulative delta (detik)
    ob_imbalance_levels: int = 20           # levels untuk orderbook imbalance
    vwap_enabled: bool = True               # enable VWAP calculation
    micro_noise_enabled: bool = True        # enable microstructure noise ratio
    hidden_order_threshold: float = 100000  # threshold USD untuk detect hidden orders


# ------------------------------ Data Models ------------------------------ #

@dataclass
class Kline:
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def upper_wick(self) -> float:
        return max(0.0, self.high - max(self.open, self.close))

    @property
    def lower_wick(self) -> float:
        return max(0.0, min(self.open, self.close) - self.low)

    def wick_flags(self, min_ratio: float) -> Tuple[bool, bool]:
        body = max(self.body, 1e-8)
        up = (self.upper_wick / body) >= min_ratio
        lo = (self.lower_wick / body) >= min_ratio
        return up, lo


@dataclass
class Trade:
    ts: int
    price: float
    qty: float
    is_buyer_aggressor: bool  # True if buyer lifted ask (Binance: m==False)


@dataclass
class DepthBook:
    bids: Dict[float, float] = field(default_factory=dict)  # price -> size
    asks: Dict[float, float] = field(default_factory=dict)  # price -> size

    def topn(self, n: int) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        best_bids = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)[:n]
        best_asks = sorted(self.asks.items(), key=lambda x: x[0])[:n]
        return best_bids, best_asks


@dataclass
class CumulativeDelta:
    total_buy: float = 0.0
    total_sell: float = 0.0
    ob_delta: float = 0.0  # orderbook imbalance
    timestamp: int = 0


@dataclass
class AdvancedMetrics:
    cum_delta: CumulativeDelta = field(default_factory=CumulativeDelta)
    vwap: float = 0.0
    ob_imbalance: float = 0.0
    spread: float = 0.0
    spread_pct: float = 0.0
    noise_ratio: float = 0.0
    large_trades: List[Trade] = field(default_factory=list)
    liquidation_metrics: Dict[str, float] = field(default_factory=lambda: {'long': 0.0, 'short': 0.0})
    flow_velocity: float = 0.0
    depth_pressure: List[float] = field(default_factory=list)
    hidden_orders: List[Trade] = field(default_factory=list)
    mark_premium: float = 0.0
    volume_clusters: Dict[float, float] = field(default_factory=lambda: defaultdict(float))
    session_metrics: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'ASIA': {'volume': 0.0, 'volatility': 0.0},
        'EUROPE': {'volume': 0.0, 'volatility': 0.0},
        'US': {'volume': 0.0, 'volatility': 0.0}
    })


# ------------------------------ Engine State ----------------------------- #

@dataclass
class EngineState:
    cfg: EngineConfig
    kline: Optional[Kline] = None

    # rolling windows for delta/volume
    trades_window: Deque[Trade] = field(default_factory=deque)
    window_buy_vol: float = 0.0
    window_sell_vol: float = 0.0

    # per-bar accumulators
    bar_buy_vol: float = 0.0
    bar_sell_vol: float = 0.0
    bar_ticks: List[Trade] = field(default_factory=list)

    # volume profile (price -> traded volume) for POC
    vp_map: Dict[float, float] = field(default_factory=lambda: defaultdict(float))

    # NEW: footprint-lite buy/sell per price-bin
    bs_map: Dict[float, List[float]] = field(default_factory=lambda: defaultdict(lambda: [0.0, 0.0]))
    #              price_bin -> [buy_vol, sell_vol]

    # depth book (limited levels)
    book: DepthBook = field(default_factory=DepthBook)

    # stats series for peak detection
    vol_series: Deque[float] = field(default_factory=lambda: deque(maxlen=300))
    delta_series: Deque[float] = field(default_factory=lambda: deque(maxlen=300))

    # counters
    events: int = 0

    # CSV buffers
    csv_rows: List[Dict] = field(default_factory=list)

    # Advanced metrics
    adv_metrics: AdvancedMetrics = field(default_factory=AdvancedMetrics)
    mark_price: float = 0.0
    last_liquidations: List[Dict] = field(default_factory=list)
    
    # Tambahkan ini untuk inisialisasi atribut yang diperlukan
    def __post_init__(self):
        self.last_trade_price = 0.0
        self.adv_metrics = AdvancedMetrics()

    # per-bar accumulators
    bar_buy_vol: float = 0.0
    bar_sell_vol: float = 0.0
    bar_ticks: List[Trade] = field(default_factory=list)

    # volume profile (price -> traded volume) for POC
    vp_map: Dict[float, float] = field(default_factory=lambda: defaultdict(float))

    # NEW: footprint-lite buy/sell per price-bin
    bs_map: Dict[float, List[float]] = field(default_factory=lambda: defaultdict(lambda: [0.0, 0.0]))
    #              price_bin -> [buy_vol, sell_vol]

    # depth book (limited levels)
    book: DepthBook = field(default_factory=DepthBook)

    # stats series for peak detection
    vol_series: Deque[float] = field(default_factory=lambda: deque(maxlen=300))
    delta_series: Deque[float] = field(default_factory=lambda: deque(maxlen=300))

    # counters
    events: int = 0

    # CSV buffers
    csv_rows: List[Dict] = field(default_factory=list)

    # Advanced metrics
    adv_metrics: AdvancedMetrics = field(default_factory=AdvancedMetrics)
    mark_price: float = 0.0
    last_liquidations: List[Dict] = field(default_factory=list)

    def reset_bar(self):
        self.bar_buy_vol = 0.0
        self.bar_sell_vol = 0.0
        self.bar_ticks.clear()
        self.bs_map.clear()
        if self.cfg.session_reset == "bar":
            self.vp_map.clear()

    def reset_session(self, now_ms: int):
        self.vp_map.clear()
        self.adv_metrics.cum_delta = CumulativeDelta()

    def _price_bin(self, price: float, step: Optional[float]) -> float:
        if not step or step <= 0:
            return price
        return round(math.floor(price / step) * step, ndigits=8)

    def update_volume_profile(self, price: float, qty: float, step: Optional[float] = None):
        key = self._price_bin(price, step)
        self.vp_map[key] += qty

    def update_bs_map(self, price: float, qty: float, is_buy: bool, step: Optional[float]):
        key = self._price_bin(price, step)
        if is_buy:
            self.bs_map[key][0] += qty
        else:
            self.bs_map[key][1] += qty

    def poc(self) -> Optional[Tuple[float, float]]:
        if not self.vp_map:
            return None
        price, vol = max(self.vp_map.items(), key=lambda kv: kv[1])
        return price, vol

    def liquidity_clusters(self, n: int) -> Dict[str, List[Tuple[float, float, str]]]:
        bids, asks = self.book.topn(n)
        def median_level(levels: List[Tuple[float, float]]) -> float:
            if not levels:
                return 0.0
            sizes = sorted([sz for _, sz in levels])
            m = len(sizes)
            return sizes[m//2] if m % 2 == 1 else (sizes[m//2-1] + sizes[m//2])/2
        mb = median_level(bids)
        ma = median_level(asks)

        clusters = {"bids": [], "asks": []}
        for p, s in bids:
            if s >= max(self.cfg.liquidity_abs_threshold, self.cfg.liquidity_rel_threshold * (mb or 1e-9)):
                clusters["bids"].append((p, s, "abs" if s>=self.cfg.liquidity_abs_threshold else "rel"))
        for p, s in asks:
            if s >= max(self.cfg.liquidity_abs_threshold, self.cfg.liquidity_rel_threshold * (ma or 1e-9)):
                clusters["asks"].append((p, s, "abs" if s>=self.cfg.liquidity_abs_threshold else "rel"))
        return clusters

    def delta_and_percent(self, buy: float, sell: float) -> Tuple[float, float]:
        delta = buy - sell
        tot = buy + sell
        perc = (delta / tot) * 100.0 if tot > 0 else 0.0
        return delta, perc

    def append_series(self, vol: float, delta_perc: float):
        self.vol_series.append(vol)
        self.delta_series.append(delta_perc)

    def zscore(self, series: Deque[float], x: float) -> float:
        if len(series) < 20:
            return 0.0
        s = pd.Series(series)
        mu = s.mean()
        sd = s.std(ddof=0) or 1e-9
        return (x - mu) / sd

    # Advanced metrics calculations
    def update_cumulative_delta(self, trade: Trade):
        if trade.is_buyer_aggressor:
            self.adv_metrics.cum_delta.total_buy += trade.qty
        else:
            self.adv_metrics.cum_delta.total_sell += trade.qty
        self.adv_metrics.cum_delta.timestamp = trade.ts

    def calculate_vwap(self) -> float:
        vwap_accumulator = 0.0
        volume_accumulator = 0.0
        for trade in self.trades_window:
            vwap_accumulator += trade.price * trade.qty
            volume_accumulator += trade.qty
        return vwap_accumulator / volume_accumulator if volume_accumulator > 0 else 0.0

    def calculate_orderbook_imbalance(self) -> float:
        bids, asks = self.book.topn(self.cfg.ob_imbalance_levels)
        bid_vol = sum([sz for _, sz in bids])
        ask_vol = sum([sz for _, sz in asks])
        imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol) if (bid_vol + ask_vol) > 0 else 0
        return imbalance

    def calculate_spread(self) -> Tuple[float, float]:
        bids, asks = self.book.topn(1)
        best_bid = bids[0][0] if bids else 0
        best_ask = asks[0][0] if asks else 0
        spread = best_ask - best_bid
        spread_pct = (spread / best_bid * 100) if best_bid > 0 else 0
        return spread, spread_pct

    def calculate_noise_ratio(self) -> float:
        if not self.kline or not self.bar_ticks:
            return 0.0
            
        body_high = max(self.kline.open, self.kline.close)
        body_low = min(self.kline.open, self.kline.close)
        
        outside_volume = 0
        total_volume = 0
        
        for trade in self.bar_ticks:
            total_volume += trade.qty
            if trade.price > body_high or trade.price < body_low:
                outside_volume += trade.qty
                
        return outside_volume / total_volume if total_volume > 0 else 0.0

    def detect_large_trades(self) -> List[Trade]:
        if not self.trades_window:
            return []
        avg_trade_size = np.mean([t.qty * t.price for t in self.trades_window])
        return [t for t in self.trades_window if t.qty * t.price > self.cfg.hidden_order_threshold]

    def calculate_flow_velocity(self) -> float:
        if not self.trades_window:
            return 0.0
        return len(self.trades_window) / self.cfg.delta_rolling_secs

    def calculate_depth_pressure(self) -> List[float]:
        current_price = getattr(self, 'last_trade_price', 0)
        if current_price == 0:
            return []
            
        depth_levels = [0.5, 1.0, 2.0]  # % dari current price
        pressures = []
        bids, asks = self.book.topn(100)  # Use more levels for depth analysis
        
        for level in depth_levels:
            bid_zone = current_price * (1 - level/100)
            ask_zone = current_price * (1 + level/100)
            bid_vol = sum(sz for p, sz in bids if p >= bid_zone)
            ask_vol = sum(sz for p, sz in asks if p <= ask_zone)
            pressure = (bid_vol - ask_vol) / (bid_vol + ask_vol) if (bid_vol + ask_vol) > 0 else 0
            pressures.append(pressure)
        return pressures

    def update_volume_clusters(self):
        for trade in self.bar_ticks:
            price_bin = self._price_bin(trade.price, self.cfg.vp_price_step or 0.5)
            self.adv_metrics.volume_clusters[price_bin] += trade.qty

    def update_session_metrics(self):
        if not self.kline:
            return
            
        ts = pd.to_datetime(self.kline.close_time, unit="ms").tz_localize("UTC")
        session = get_session(ts.hour)
        volatility = (self.kline.high - self.kline.low) / self.kline.low * 100 if self.kline.low > 0 else 0
        
        self.adv_metrics.session_metrics[session]['volume'] += self.kline.volume
        self.adv_metrics.session_metrics[session]['volatility'] = volatility


# ------------------------------- Utilities ------------------------------- #
def human_fmt(num: float) -> str:
    if num >= 1_000_000_000:
        return f"{num/1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num/1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num/1_000:.2f}K"
    else:
        return f"{num:.2f}"

def parse_trade(msg: dict) -> Trade:
    price = float(msg["p"])
    qty = float(msg["q"])
    ts = int(msg["T"])
    is_buyer_maker = bool(msg["m"])
    return Trade(ts=ts, price=price, qty=qty, is_buyer_aggressor=not is_buyer_maker)


def apply_depth_update(book: DepthBook, msg: dict, max_levels: int):
    for side_key, store in (("b", book.bids), ("a", book.asks)):
        updates = msg.get(side_key, [])
        for price_str, qty_str in updates:
            p = float(price_str)
            q = float(qty_str)
            if q == 0.0:
                if p in store:
                    del store[p]
            else:
                store[p] = q
    bids_sorted = dict(sorted(book.bids.items(), key=lambda x: x[0], reverse=True)[:max_levels])
    asks_sorted = dict(sorted(book.asks.items(), key=lambda x: x[0])[:max_levels])
    book.bids = bids_sorted
    book.asks = asks_sorted


# ---------------------------- Heuristic Signals -------------------------- #

def detect_trapped(state: EngineState, last_price: float, now_ms: int) -> Optional[dict]:
    cfg = state.cfg
    cutoff = now_ms - cfg.delta_rolling_secs * 1000
    buy, sell = 0.0, 0.0
    for t in list(state.trades_window):
        if t.ts < cutoff:
            continue
        if t.is_buyer_aggressor:
            buy += t.qty
        else:
            sell += t.qty
    _, dperc = state.delta_and_percent(buy, sell)
    if abs(dperc) < cfg.trapped_min_imbalance * 100.0:  # NOTE: cfg is 0..1, convert to %
        return None

    recent_prices = [tr.price for tr in state.trades_window if tr.ts >= cutoff]
    if len(recent_prices) < 10:
        return None

    if dperc > 0:
        is_extreme = last_price >= max(recent_prices) * 0.999
        direction = "long_trapped"
        ae_check = lambda cur: (last_price - cur) / last_price * 100.0 >= cfg.trapped_ae_threshold
    else:
        is_extreme = last_price <= min(recent_prices) * 1.001
        direction = "short_trapped"
        ae_check = lambda cur: (cur - last_price) / last_price * 100.0 >= cfg.trapped_ae_threshold

    if not is_extreme:
        return None

    async def confirm_trap():
        await asyncio.sleep(cfg.trapped_lookahead_secs)
        cur = getattr(state, "last_trade_price", last_price)
        if ae_check(cur):
            return {
                "type": direction,
                "delta_perc": dperc,
                "entry_price": last_price,
                "confirm_price": cur,
                "adverse_excursion_%": abs(cur - last_price) / last_price * 100.0,
                "ts": int(time.time() * 1000),
            }
        return None

    return {"pending": confirm_trap, "direction": direction, "delta_perc": dperc}


# ------------------------------ Scoring/Tiers ----------------------------- #

def get_session(hour_local: int) -> str:
    if 0 <= hour_local < 8: return "ASIA"
    if 8 <= hour_local < 16: return "EUROPE"
    if 16 <= hour_local < 24: return "US"
    return "OTHER"

def compute_score_and_tier(kline: Kline,
                           dperc: float,
                           peak_vol: bool,
                           peak_delta: bool,
                           pattern: Optional[str],
                           liq_amt: float,
                           st: EngineState) -> Tuple[int, str, Dict[str, int]]:
    """
    Skoring "berat di inti" (bisa di-tune):
    - Trapped (proxy) = |delta%| besar (rolling) & di extreme → +40 (estimasi via dperc saja)
    - Peak volume/delta = +25 (vol) +10 (delta) (maks 30)
    - Pattern (hammer/shooting_star) = +15; engulfing = +8 (ambil terbesar)
    - Liquidity cluster dekat harga (± near_px_bps) = +10
    - Advanced metrics bonus = +15 (maks)
    Bonus ringan:
    - Session EU/US overlap = +5
    """
    score = 0
    breakdown: Dict[str, int] = {}

    # Trapped proxy (pakai dperc bar ini sebagai indikator agresi — cepat & sederhana)
    trapped_pts = 0
    if abs(dperc) >= max(20.0, st.cfg.trapped_min_imbalance * 100.0):
        trapped_pts = 40
        score += trapped_pts
    breakdown["trapped"] = trapped_pts

    # Peak
    peak_pts = 0
    if peak_vol: peak_pts += 25
    if peak_delta: peak_pts += 10
    peak_pts = min(30, peak_pts)
    score += peak_pts
    breakdown["peak"] = peak_pts

    # Pattern
    patt_pts = 0
    if pattern in ("hammer", "shooting_star"):
        patt_pts = 15
    elif pattern in ("bullish_engulfing", "bearish_engulfing"):
        patt_pts = 8
    score += patt_pts
    breakdown["pattern"] = patt_pts

    # Liquidity near price
    liq_pts = 0
    if st.bs_map:
        close = kline.close
        bps = st.cfg.near_px_bps / 10000.0  # 1 bps = 0.01%
        near_low, near_high = close * (1 - bps), close * (1 + bps)
        near_vol = 0.0
        for px, (bv, sv) in st.bs_map.items():
            if near_low <= px <= near_high:
                near_vol += (bv + sv)
        if near_vol > 0:
            liq_pts = 10
            score += liq_pts
    breakdown["liquidity"] = liq_pts

    # Advanced metrics bonus
    adv_pts = 0
    if hasattr(st, 'adv_metrics'):
        # Cumulative delta strength
        if abs(st.adv_metrics.cum_delta.total_buy - st.adv_metrics.cum_delta.total_sell) > st.kline.volume * 0.3:
            adv_pts += 5
        # Orderbook imbalance
        if abs(st.adv_metrics.ob_imbalance) > 0.2:
            adv_pts += 3
        # VWAP deviation
        vwap_dev = abs(kline.close - st.adv_metrics.vwap) / st.adv_metrics.vwap * 100 if st.adv_metrics.vwap > 0 else 0
        if vwap_dev > 0.5:
            adv_pts += 2
        # Low noise ratio (clean moves)
        if st.adv_metrics.noise_ratio < 0.2:
            adv_pts += 2
        # Large trades presence
        if st.adv_metrics.large_trades:
            adv_pts += 3
    adv_pts = min(15, adv_pts)
    score += adv_pts
    breakdown["advanced"] = adv_pts

    # Session bonus
    ts = pd.to_datetime(kline.close_time, unit="ms").tz_localize("UTC").tz_convert("Asia/Jakarta")
    sess = get_session(ts.hour)
    sess_pts = 5 if sess == "EUROPE" or sess == "US" else 0
    score += sess_pts
    breakdown["session"] = sess_pts

    score = min(100, int(score))

    if score <= 40:
        tier = "Ignore"
    elif score <= 70:
        tier = "Good"
    elif score <= 85:
        tier = "Strong"
    else:
        tier = "Very Strong"

    return score, tier, breakdown


# ------------------------------ Direction (NEW) --------------------------- #

def decide_signal(trapped_flag: bool,
                  dperc: float,
                  pattern: Optional[str],
                  clusters: Dict[str, List[Tuple[float, float, str]]],
                  adv_metrics: AdvancedMetrics) -> Tuple[str, str]:
    """
    Konsensus sederhana dengan strength indicator
    Returns: (signal, strength)
    """
    votes = {"BUY": 0, "SELL": 0}

    # Trapped proxy: gunakan tanda arah dari dperc
    if trapped_flag:
        if dperc > 0:
            votes["SELL"] += 2  # long trapped → SELL
        elif dperc < 0:
            votes["BUY"] += 2  # short trapped → BUY

    # Delta directional bias
    if dperc > 10:
        votes["BUY"] += 1
    elif dperc < -10:
        votes["SELL"] += 1

    # Candlestick
    if pattern in ("hammer", "bullish_engulfing"):
        votes["BUY"] += 1
    elif pattern in ("shooting_star", "bearish_engulfing"):
        votes["SELL"] += 1

    # Liquidity clusters
    if clusters.get("asks"):
        votes["SELL"] += 1
    if clusters.get("bids"):
        votes["BUY"] += 1

    # Advanced metrics
    if adv_metrics.ob_imbalance > 0.1:
        votes["BUY"] += 1
    elif adv_metrics.ob_imbalance < -0.1:
        votes["SELL"] += 1

    if adv_metrics.cum_delta.total_buy > adv_metrics.cum_delta.total_sell * 1.5:
        votes["BUY"] += 1
    elif adv_metrics.cum_delta.total_sell > adv_metrics.cum_delta.total_buy * 1.5:
        votes["SELL"] += 1

    # Determine signal and strength
    buy_votes = votes["BUY"]
    sell_votes = votes["SELL"]
    
    if buy_votes > sell_votes:
        strength = "WEAK"
        if buy_votes >= 4:
            strength = "MEDIUM"
        if buy_votes >= 6:
            strength = "STRONG"
        return "BUY", strength
    elif sell_votes > buy_votes:
        strength = "WEAK"
        if sell_votes >= 4:
            strength = "MEDIUM"
        if sell_votes >= 6:
            strength = "STRONG"
        return "SELL", strength
    return "NEUTRAL", "N/A"


def pretty_print(symbol, interval, kline: Kline, poc, liq_amt, delta_perc,
                 trapped, pattern, peak, score=None, tier=None, breakdown=None, 
                 signal: Optional[str] = None, strength: Optional[str] = None,
                 adv_metrics: Optional[AdvancedMetrics] = None):
    ts = pd.to_datetime(kline.close_time, unit="ms").tz_localize("UTC").tz_convert("Asia/Jakarta")
    session = get_session(ts.hour)

    print("\n" + "="*60)
    print(f"#{symbol.upper()} [TF {interval}] - {ts.strftime('%d %b %Y - %H:%M GMT+7')}")
    print(f"Session: {session}")
    print("="*60)

    if score is not None and tier is not None:
        score_color = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
        print(f"{score_color} Score: {score}/100 → {tier}")

    if signal and strength:
        signal_color = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "⚪"
        strength_icon = "💪" if strength == "STRONG" else "👍" if strength == "MEDIUM" else "👎"
        print(f"{signal_color} Signal: {signal} {strength_icon} ({strength})")

    print("\n📊 Core Metrics:")
    print(f"- Trapped Trader: {'✅' if trapped else '❌'} (+{breakdown.get('trapped',0)})")
    print(f"- Pattern: {pattern if pattern else 'None'} (+{breakdown.get('pattern',0)})")
    print(f"- Peak Vol/Delta: {'✅' if peak else '❌'} (+{breakdown.get('peak',0)})")
    print(f"- Liquidity Near: {'✅' if breakdown.get('liquidity',0) > 0 else '❌'} (+{breakdown.get('liquidity',0)})")
    print(f"- Advanced Features: (+{breakdown.get('advanced',0)})")
    print(f"- Session Bonus: (+{breakdown.get('session',0)})")

    if adv_metrics:
        print(f"\n🔍 Advanced Insights:")
        print(f"- Cumulative Δ: {human_fmt(adv_metrics.cum_delta.total_buy - adv_metrics.cum_delta.total_sell)}")
        print(f"- OB Imbalance: {adv_metrics.ob_imbalance:+.3f}")
        print(f"- VWAP Dev: {abs(kline.close - adv_metrics.vwap)/adv_metrics.vwap*100:.2f}%")
        print(f"- Noise Ratio: {adv_metrics.noise_ratio:.3f}")
        print(f"- Large Trades: {len(adv_metrics.large_trades)}")
        print(f"- Flow Velocity: {adv_metrics.flow_velocity:.1f}/s")

    print(f"\n📈 Market Data:")
    if poc:
        print(f"- PoC: {poc[0]} ({human_fmt(poc[1])})")
    print(f"- Wick: H={kline.high} | L={kline.low}")
    print(f"- Liquidity: {human_fmt(liq_amt)}")
    print(f"- Delta: {delta_perc:+.2f}%")
    print("="*60 + "\n")

    # Discord message (Pretty Log style)
    msg = (
        f"============================================================\n"
        f"#{symbol.upper()} [TF {interval}] - {ts.strftime('%d %b %Y - %H:%M GMT+7')}\n"
        f"Session: {session}\n"
        f"============================================================\n"
        f"{'🟢' if signal=='BUY' else '🔴' if signal=='SELL' else '🟡'} "
        f"Score: {score}/100 → {tier}\n"
        f"{'🟢' if signal=='BUY' else '🔴' if signal=='SELL' else '⚪'} "
        f"Signal: {signal} ({strength})\n\n"
        f"📊 Core Metrics:\n"
        f"- Trapped Trader: {'✅' if trapped else '❌'} (+{breakdown.get('trapped',0)})\n"
        f"- Pattern: {pattern} (+{breakdown.get('pattern',0)})\n"
        f"- Peak Vol/Delta: {'✅' if peak else '❌'} (+{breakdown.get('peak',0)})\n"
        f"- Liquidity Near: {'✅' if breakdown.get('liquidity',0)>0 else '❌'} (+{breakdown.get('liquidity',0)})\n"
        f"- Advanced Features: (+{breakdown.get('advanced',0)})\n"
        f"- Session Bonus: (+{breakdown.get('session',0)})\n\n"
        f"🔍 Advanced Insights:\n"
        f"- Cumulative Δ: {cum_delta:+.2f}\n"
        f"- OB Imbalance: {ob_imbalance:+.3f}\n"
        f"- VWAP Dev: {vwap_dev:.2f}%\n"
        f"- Noise Ratio: {noise_ratio:.3f}\n"
        f"- Large Trades: {large_trades}\n"
        f"- Flow Velocity: {flow_velocity:.1f}/s\n\n"
        f"📈 Market Data:\n"
        f"- PoC: {poc[0] if poc else 'N/A'} ({human_fmt(poc[1]) if poc else 'N/A'})\n"
        f"- Wick: H={kline.high} | L={kline.low}\n"
        f"- Liquidity: {human_fmt(liquidity)}\n"
        f"- Delta: {delta_perc:+.2f}%\n"
        f"============================================================"
    )
    send_discord(msg)


# ------------------------------- Main Loop -------------------------------- #

class OrderflowEngine:
    def __init__(self, cfg: EngineConfig):
        self.state = EngineState(cfg=cfg)
        self.pending_tasks: List[asyncio.Task] = []

    def streams(self) -> List[str]:
        s = self.state.cfg.symbol
        streams = [
            f"{s}@aggTrade",
            f"{s}@kline_{self.state.cfg.interval}",
            f"{s}@depth@{self.state.cfg.depth_update}",
        ]
        if self.state.cfg.futures:
            streams += [
                f"{s}@forceOrder",
                f"{s}@markPrice",
            ]
        return streams

    def base_ws(self) -> str:
        return "wss://fstream.binance.com/stream?streams=" if self.state.cfg.futures \
            else "wss://stream.binance.com:9443/stream?streams="

    async def connect(self):
        url = self.base_ws() + "/".join(self.streams())
        async with websockets.connect(url, ping_interval=15, ping_timeout=20) as ws:
            print(f"[WS] Connected: {url}")
            await self.loop(ws)

    async def loop(self, ws):
        st = self.state
        cfg = st.cfg
        last_print = 0

        while True:
            raw = await ws.recv()
            msg = json.loads(raw)
            st.events += 1

            if "stream" not in msg:
                continue
            stream = msg["stream"]
            data = msg["data"]

            now_ms = data.get("E", int(time.time()*1000))

            if stream.endswith("@aggTrade"):
                tr = parse_trade(data)
                st.last_trade_price = tr.price
                st.trades_window.append(tr)
                st.bar_ticks.append(tr)

                # maintain rolling window
                cutoff = tr.ts - cfg.delta_rolling_secs * 1000
                while st.trades_window and st.trades_window[0].ts < cutoff:
                    st.trades_window.popleft()

                # per-bar accumulators
                if tr.is_buyer_aggressor:
                    st.bar_buy_vol += tr.qty
                else:
                    st.bar_sell_vol += tr.qty

                # VP (traded) + footprint-lite bs_map
                st.update_volume_profile(tr.price, tr.qty, cfg.vp_price_step)
                st.update_bs_map(tr.price, tr.qty, tr.is_buyer_aggressor, cfg.imb_price_step or cfg.vp_price_step)

                # Advanced metrics
                st.update_cumulative_delta(tr)
                st.update_volume_clusters()

                # event-driven trapped (optional async confirm)
                trap_evt = detect_trapped(st, tr.price, tr.ts)
                if trap_evt and "pending" in trap_evt:
                    self.pending_tasks.append(asyncio.create_task(trap_evt["pending"]()))

            elif "@kline_" in stream:
                k = data["k"]
                st.kline = Kline(
                    open_time=int(k["t"]),
                    open=float(k["o"]),
                    high=float(k["h"]),
                    low=float(k["l"]),
                    close=float(k["c"]),
                    volume=float(k["v"]),
                    close_time=int(k["T"]),
                )

                if k["x"]:  # kline closed
                    # Calculate all advanced metrics
                    if cfg.vwap_enabled:
                        st.adv_metrics.vwap = st.calculate_vwap()
                    st.adv_metrics.ob_imbalance = st.calculate_orderbook_imbalance()
                    st.adv_metrics.spread, st.adv_metrics.spread_pct = st.calculate_spread()
                    if cfg.micro_noise_enabled:
                        st.adv_metrics.noise_ratio = st.calculate_noise_ratio()
                    st.adv_metrics.large_trades = st.detect_large_trades()
                    st.adv_metrics.flow_velocity = st.calculate_flow_velocity()
                    st.adv_metrics.depth_pressure = st.calculate_depth_pressure()
                    st.update_session_metrics()

                    delta, dperc = st.delta_and_percent(st.bar_buy_vol, st.bar_sell_vol)
                    up, lo = st.kline.wick_flags(cfg.wick_min_ratio)
                    vp = st.poc()
                    poc_price, poc_vol = (vp if vp else (None, None))

                    # peaks
                    st.append_series(st.kline.volume, dperc)
                    v_z = st.zscore(st.vol_series, st.kline.volume)
                    d_z = st.zscore(st.delta_series, dperc)
                    peak_vol = v_z >= cfg.peak_zscore
                    peak_delta = abs(d_z) >= cfg.peak_zscore

                    pattern = detect_candlestick_pattern(st.kline)

                    # imbalance summary near price
                    liq_amt = max(st.vp_map.values()) if st.vp_map else 0.0
                    trapped_flag = abs(dperc) >= max(20.0, cfg.trapped_min_imbalance * 100.0)
                    peak_flag = (peak_vol or peak_delta)

                    # scoring & tier
                    score, tier, breakdown = compute_score_and_tier(
                        st.kline, dperc, peak_vol, peak_delta, pattern, liq_amt, st
                    )

                    # arah sinyal dengan strength indicator
                    clusters = st.liquidity_clusters(n=10)
                    signal, strength = decide_signal(trapped_flag, dperc, pattern, clusters, st.adv_metrics)

                    # Pretty-log dengan filter min_score
                    if cfg.pretty_log and score >= cfg.min_score:
                        pretty_print(cfg.symbol, cfg.interval, st.kline, vp, liq_amt,
                                     dperc, trapped_flag, pattern, peak_flag,
                                     score=score, tier=tier, breakdown=breakdown, 
                                     signal=signal, strength=strength, adv_metrics=st.adv_metrics)

                    # CSV row dengan semua data advanced
                    row = {
                        "bar_time": pd.to_datetime(st.kline.close_time, unit="ms"),
                        "open": st.kline.open, "high": st.kline.high,
                        "low": st.kline.low, "close": st.kline.close,
                        "volume": st.kline.volume,
                        "buy_vol": st.bar_buy_vol, "sell_vol": st.bar_sell_vol,
                        "delta": delta, "delta_perc": dperc,
                        "upper_wick": st.kline.upper_wick, "lower_wick": st.kline.lower_wick,
                        "long_upper_wick": up, "long_lower_wick": lo,
                        "poc_price": poc_price, "poc_vol": poc_vol,
                        "peak_vol": peak_vol, "peak_delta": peak_delta,
                        "pattern": pattern,
                        "score": score, "tier": tier,
                        "signal": signal, "signal_strength": strength,
                        # Advanced metrics
                        "cum_delta_buy": st.adv_metrics.cum_delta.total_buy,
                        "cum_delta_sell": st.adv_metrics.cum_delta.total_sell,
                        "vwap": st.adv_metrics.vwap,
                        "ob_imbalance": st.adv_metrics.ob_imbalance,
                        "spread": st.adv_metrics.spread,
                        "spread_pct": st.adv_metrics.spread_pct,
                        "noise_ratio": st.adv_metrics.noise_ratio,
                        "flow_velocity": st.adv_metrics.flow_velocity,
                        "large_trades": len(st.adv_metrics.large_trades),
                    }
                    st.csv_rows.append(row)

                    if cfg.write_csv and len(st.csv_rows) % 20 == 0:
                        df = pd.DataFrame(st.csv_rows)
                        df.to_csv(f"{cfg.out_prefix}{cfg.symbol}_{cfg.interval}.csv", index=False)

                    st.reset_bar()

            elif stream.endswith("@depth@"+cfg.depth_update):
                apply_depth_update(st.book, data, cfg.depth_levels)

            elif stream.endswith("@forceOrder"):
                o = data.get("o", {})
                side = o.get("S")
                p = float(o.get("ap", 0))
                q = float(o.get("q", 0))
                st.csv_rows.append({
                    "bar_time": pd.to_datetime(now_ms, unit="ms"),
                    "liquidation_side": side,
                    "liq_price": p,
                    "liq_qty": q,
                })

            elif stream.endswith("@markPrice"):
                st.mark_price = float(data.get("p", 0))
                if st.last_trade_price:
                    st.adv_metrics.mark_premium = (st.mark_price - st.last_trade_price) / st.last_trade_price * 100

            # heartbeat JSON tiap N event
            if st.events - last_print >= cfg.print_every_n_events:
                last_print = st.events
                vp = st.poc()
                clusters = st.liquidity_clusters(n=10)
                delta, dperc = st.delta_and_percent(st.bar_buy_vol, st.bar_sell_vol)
                print(json.dumps({
                    "events": st.events,
                    "last_price": getattr(st, "last_trade_price", None),
                    "delta": round(delta, 4), "delta_perc": round(dperc, 2),
                    "poc": vp,
                    "liq_top_bids": clusters["bids"][:3],
                    "liq_top_asks": clusters["asks"][:3],
                    "cum_delta": st.adv_metrics.cum_delta.total_buy - st.adv_metrics.cum_delta.total_sell,
                    "ob_imbalance": st.adv_metrics.ob_imbalance,
                }, ensure_ascii=False))

    def finalize(self):
        cfg = self.state.cfg
        if cfg.write_csv and self.state.csv_rows:
            df = pd.DataFrame(self.state.csv_rows)
            df.to_csv(f"{cfg.out_prefix}{cfg.symbol}_{cfg.interval}.csv", index=False)


# ------------------------------- Entrypoint ------------------------------- #

async def main():
    import argparse
    p = argparse.ArgumentParser(description="Binance Orderflow Engine (Full) + Pretty Log + Imbalance Scoring")
    p.add_argument("--symbol", default="btcusdt", help="symbol lowercase, e.g., btcusdt")
    p.add_argument("--interval", default="1m", help="kline interval, e.g., 1m, 5m, 15m")
    p.add_argument("--spot", action="store_true", help="use Spot streams (default futures)")
    p.add_argument("--depth-levels", type=int, default=50)
    p.add_argument("--depth-update", default="100ms", choices=["100ms", "1000ms"])
    p.add_argument("--poc-from", default="trades", choices=["trades", "depth"])
    p.add_argument("--vp-step", type=float, default=None, help="volume profile price bin size")
    p.add_argument("--liq-abs", type=float, default=50.0, help="absolute liquidity threshold (base qty)")
    p.add_argument("--liq-rel", type=float, default=4.0, help="relative liquidity threshold (x median)")
    p.add_argument("--delta-roll", type=int, default=30, help="delta rolling window (sec)")
    p.add_argument("--peak-z", type=float, default=3.0, help="z-score cutoff for peaks")
    p.add_argument("--trap-look", type=int, default=20, help="trapped lookahead (sec)")
    p.add_argument("--trap-ae", type=float, default=0.08, help="adverse excursion threshold (%)")
    p.add_argument("--trap-imb", type=float, default=0.6, help="min |delta%| (0..1) for one-sided aggression")
    p.add_argument("--wick-ratio", type=float, default=0.35, help="wick/body ratio to flag long wick")
    p.add_argument("--session-reset", default="day", choices=["day", "bar"])
    p.add_argument("--no-csv", action="store_true", help="disable CSV writes")
    # pretty alert + scoring
    p.add_argument("--pretty-log", action="store_true", help="pretty alert-style log output")
    p.add_argument("--imb-step", type=float, default=None, help="footprint-lite bin size (default=vp-step)")
    p.add_argument("--imb-th", type=float, default=0.7, help="imbalance threshold (0..1)")
    p.add_argument("--near-bps", type=float, default=5.0, help="near price radius in bps for liquidity bonus")
    p.add_argument("--min-score", type=int, default=60, help="only print pretty log if score >= min-score")
    # advanced features
    p.add_argument("--cum-delta-window", type=int, default=300, help="cumulative delta window (sec)")
    p.add_argument("--ob-levels", type=int, default=20, help="orderbook imbalance levels")
    p.add_argument("--no-vwap", action="store_true", help="disable VWAP calculation")
    p.add_argument("--no-noise", action="store_true", help="disable noise ratio calculation")
    p.add_argument("--hidden-th", type=float, default=100000, help="hidden order threshold (USD)")
    args = p.parse_args()

    cfg = EngineConfig(
        symbol=args.symbol.lower(),
        interval=args.interval,
        futures=not args.spot,
        depth_levels=args.depth_levels,
        depth_update=args.depth_update,
        poc_from=args.poc_from,
        vp_price_step=args.vp_step,
        liquidity_abs_threshold=args.liq_abs,
        liquidity_rel_threshold=args.liq_rel,
        delta_rolling_secs=args.delta_roll,
        peak_zscore=args.peak_z,
        trapped_lookahead_secs=args.trap_look,
        trapped_ae_threshold=args.trap_ae,
        trapped_min_imbalance=args.trap_imb,
        wick_min_ratio=args.wick_ratio,
        session_reset=args.session_reset,
        write_csv=(not args.no_csv),
        pretty_log=args.pretty_log,
        imb_price_step=args.imb_step,
        imb_ratio_threshold=args.imb_th,
        near_px_bps=args.near_bps,
        min_score=args.min_score,
        cum_delta_window=args.cum_delta_window,
        ob_imbalance_levels=args.ob_levels,
        vwap_enabled=not args.no_vwap,
        micro_noise_enabled=not args.no_noise,
        hidden_order_threshold=args.hidden_th,
    )
    eng = OrderflowEngine(cfg)
    try:
        await eng.connect()
    except KeyboardInterrupt:
        print("Interrupted. Finalizing...")
    finally:
        eng.finalize()


# ------------------------ Candlestick Pattern (orig) ---------------------- #

def detect_candlestick_pattern(kline):
    body = abs(kline.close - kline.open)
    upper = kline.upper_wick
    lower = kline.lower_wick
    full = kline.high - kline.low

    if full == 0:
        return None

    # Shooting Star
    if body < full * 0.25 and upper > body * 2 and kline.close < (kline.open + kline.close)/2:
        return "shooting_star"
    # Hammer
    elif body < full * 0.25 and lower > body * 2 and kline.close > (kline.open + kline.close)/2:
        return "hammer"
    # Bullish Engulfing
    elif kline.close > kline.open and body > full * 0.6:
        return "bullish_engulfing"
    # Bearish Engulfing
    elif kline.close < kline.open and body > full * 0.6:
        return "bearish_engulfing"
    return None


# ------------------------------ Pretty Helpers ---------------------------- #

if __name__ == "__main__":
    asyncio.run(main())
