"""
GARO Algo - Stock Screener Backend
yfinance with retry + backoff, batch download, SSE progress streaming

Install: pip install flask flask-cors yfinance pandas numpy openpyxl gunicorn
Run:     python garo_server.py
"""

from flask import Flask, jsonify, request, send_from_directory, Response, stream_with_context
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
import io
import json
import os
import time

app = Flask(__name__, static_folder='.')
CORS(app, origins='*')

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.ticker_cache')
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_TTL_HOURS = 12
CHUNK_SIZE = 100      # tickers per batch download call
MAX_RETRIES = 3       # retry attempts if rate limited
RETRY_WAIT  = 60      # seconds to wait between retries

MARKET_SUFFIX = {'IN': '.NS', 'US': ''}  # yfinance: NSE India uses .NS


# ── Cache ──────────────────────────────────────────────────────────────────────

def cache_path(ticker):
    safe = ticker.replace('/', '_').replace('\\', '_').replace('^', '_')
    return os.path.join(CACHE_DIR, f'{safe}.csv')

def cache_is_fresh(ticker):
    p = cache_path(ticker)
    if not os.path.exists(p): return False
    return (time.time() - os.path.getmtime(p)) < CACHE_TTL_HOURS * 3600

def cache_read(ticker):
    try:
        return pd.read_csv(cache_path(ticker), parse_dates=['Date'])
    except Exception:
        return None

def cache_write(ticker, df):
    try:
        df.to_csv(cache_path(ticker), index=False)
    except Exception:
        pass


# ── yfinance download with retry ───────────────────────────────────────────────

def download_chunk(symbols, raw_tickers, start, end, yield_fn):
    """
    Download a chunk of tickers via yfinance with retry+backoff.
    yield_fn is called to stream SSE progress updates.
    Returns (stock_data dict, errors dict).
    """
    stock_data = {}
    errors     = {}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if len(symbols) == 1:
                raw_df = yf.download(
                    symbols[0], start=start, end=end,
                    progress=False, auto_adjust=True
                )
                if isinstance(raw_df.columns, pd.MultiIndex):
                    raw_df.columns = raw_df.columns.get_level_values(0)
                if raw_df.empty:
                    errors[raw_tickers[0]] = 'No data returned'
                else:
                    raw_df = raw_df.reset_index()
                    raw_df['Date'] = pd.to_datetime(raw_df['Date']).dt.strftime('%Y-%m-%d')
                    stock_data[raw_tickers[0]] = raw_df
            else:
                batch = yf.download(
                    symbols, start=start, end=end,
                    progress=False, auto_adjust=True,
                    group_by='ticker'
                )
                for sym, raw in zip(symbols, raw_tickers):
                    try:
                        # Extract this ticker's slice
                        if sym in batch.columns.get_level_values(0):
                            df = batch[sym].copy()
                        else:
                            errors[raw] = 'Not in batch result'
                            continue
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = df.columns.get_level_values(0)
                        df = df.dropna(how='all')
                        if df.empty:
                            errors[raw] = 'Empty data'
                            continue
                        df = df.reset_index()
                        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
                        stock_data[raw] = df
                    except Exception as e:
                        errors[raw] = str(e)
            break  # success — exit retry loop

        except Exception as e:
            err_str = str(e)
            if ('RateLimit' in err_str or 'Too Many' in err_str or '429' in err_str) and attempt < MAX_RETRIES:
                wait_msg = f'Rate limited by Yahoo — waiting {RETRY_WAIT}s then retrying (attempt {attempt}/{MAX_RETRIES})…'
                print(f"  ⚠ {wait_msg}")
                yield_fn({'type': 'progress', 'pct': -1, 'msg': wait_msg})
                time.sleep(RETRY_WAIT)
            else:
                for raw in raw_tickers:
                    if raw not in stock_data:
                        errors[raw] = err_str
                break

    return stock_data, errors


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(df):
    df = df.sort_values('Date').copy()
    df['TradingRange'] = (df['High'] - df['Low']) / df['Open'].replace(0, np.nan)
    df['MarketCap']    = df['Close'] * df['Volume']
    return df

def rolling_mean_s(s, w):
    return s.rolling(window=w, min_periods=max(1, w // 2)).mean()

def compute_rsi(series, period=14):
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period-1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period-1, min_periods=period).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_volume_spike(volume_series, window=20):
    avg_vol = volume_series.rolling(window=window, min_periods=max(1, window//2)).mean()
    ratio   = volume_series / avg_vol.replace(0, np.nan)
    return ratio

def rsi_signal(v):
    if v is None or np.isnan(v): return 'N/A'
    if v >= 70: return 'Overbought'
    if v <= 30: return 'Oversold'
    if v >= 55: return 'Bullish'
    if v <= 45: return 'Bearish'
    return 'Neutral'



# -- MACD -------------------------------------------------------------------

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast    = series.ewm(span=fast,   adjust=False, min_periods=fast).mean()
    ema_slow    = series.ewm(span=slow,   adjust=False, min_periods=slow).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


# -- Bollinger Bands --------------------------------------------------------

def compute_bollinger(series, window=20, num_std=2):
    sma   = series.rolling(window=window, min_periods=max(1, window//2)).mean()
    std   = series.rolling(window=window, min_periods=max(1, window//2)).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    pct_b = (series - lower) / (upper - lower).replace(0, np.nan)
    bw    = (upper - lower) / sma.replace(0, np.nan)
    return upper, sma, lower, pct_b, bw


# -- Composite Score --------------------------------------------------------

def compute_composite_score(avg_tr, rsi, vol_ratio, above_dma5, above_dma20,
                             pct_from_high, macd_hist):
    score = 0.0
    # Trading Range 0-20
    if avg_tr is not None and not np.isnan(avg_tr):
        score += min(20, avg_tr * 100 * 4)
    # RSI 0-20
    if rsi is not None and not np.isnan(rsi):
        if 50 <= rsi <= 65:      score += 20
        elif 45 <= rsi < 50:     score += 15
        elif 65 < rsi <= 70:     score += 12
        elif 40 <= rsi < 45:     score += 10
        elif rsi > 70:           score += 5
        elif 30 <= rsi < 40:     score += 8
        else:                    score += 3
    # Volume 0-20
    if vol_ratio is not None and not np.isnan(vol_ratio):
        if vol_ratio >= 3.0:     score += 20
        elif vol_ratio >= 2.0:   score += 16
        elif vol_ratio >= 1.5:   score += 10
        elif vol_ratio >= 1.0:   score += 5
    # DMA alignment 0-15
    if above_dma5 and above_dma20:            score += 15
    elif above_dma5 and not above_dma20:      score += 7
    elif not above_dma5 and above_dma20:      score += 4
    # Proximity to 52w high 0-15
    if pct_from_high is not None and not np.isnan(pct_from_high):
        d = abs(pct_from_high)
        if d <= 2:               score += 15
        elif d <= 5:             score += 12
        elif d <= 10:            score += 8
        elif d <= 20:            score += 4
    # MACD histogram 0-10
    if macd_hist is not None and not np.isnan(macd_hist):
        if macd_hist > 0:        score += 10
        elif macd_hist > -0.5:   score += 4
    return round(min(100, score), 1)


def is_breakout(rsi, vol_ratio, pct_from_high, above_dma5, above_dma20, macd_hist):
    if rsi is None or vol_ratio is None or pct_from_high is None:
        return False
    return (
        abs(pct_from_high) <= 5 and
        vol_ratio >= 1.5 and
        50 <= rsi <= 72 and
        above_dma5 and above_dma20 and
        (macd_hist is None or macd_hist > 0)
    )


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/api/screen', methods=['POST'])
def screen():
    file     = request.files.get('tickerFile')
    market   = request.form.get('market', 'US')
    start    = request.form.get('start')
    end      = request.form.get('end')
    min_tr   = float(request.form.get('minTR', 2)) / 100
    min_mc   = float(request.form.get('minMC', 0.5))
    lookback = int(request.form.get('lookback', 100))

    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    try:
        tickers_df   = pd.read_excel(io.BytesIO(file.read()))
        ticker_col   = next((c for c in tickers_df.columns if c.strip().lower() in ['ticker','symbol','stock']), tickers_df.columns[0])
        industry_col = next((c for c in tickers_df.columns if c.strip().lower() in ['industry','sector','category']), None)
        raw_tickers  = tickers_df[ticker_col].dropna().astype(str).str.strip().tolist()
        industry_map = {}
        if industry_col:
            for _, row in tickers_df.iterrows():
                t = str(row[ticker_col]).strip()
                industry_map[t] = str(row[industry_col]).strip() if pd.notna(row[industry_col]) else 'Unknown'
        else:
            industry_map = {t: 'Unknown' for t in raw_tickers}
    except Exception as e:
        return jsonify({'error': f'Could not read Excel: {e}'}), 400

    suffix = MARKET_SUFFIX.get(market, '')
    # Build yfinance symbols e.g. RELIANCE -> RELIANCE.NS
    symbols = [t + suffix if suffix and not t.endswith(suffix) else t for t in raw_tickers]

    print(f"\n{'='*55}")
    print(f"  {len(raw_tickers)} tickers | market={market} | {start} → {end}")
    print(f"  Filters: TR>={min_tr*100}%  MC>=${min_mc}B  Lookback={lookback}d")
    print(f"{'='*55}")

    def generate():
        events_buf = []

        def sse(data):
            return f"data: {json.dumps(data)}\n\n"

        def yield_progress(data):
            # Used as callback from download_chunk
            events_buf.append(data)

        # ── Split cached vs needs-fetch ───────────────────────────────────────
        cached_data      = {}
        tickers_to_fetch = []
        symbols_to_fetch = []

        for ticker, symbol in zip(raw_tickers, symbols):
            if cache_is_fresh(ticker):
                df = cache_read(ticker)
                if df is not None and not df.empty:
                    cached_data[ticker] = df
                    continue
            tickers_to_fetch.append(ticker)
            symbols_to_fetch.append(symbol)

        total = len(raw_tickers)
        yield sse({'type': 'start', 'total': total,
                   'msg': f'{len(cached_data)} loaded from cache · {len(tickers_to_fetch)} to download…'})

        # ── Download in chunks ────────────────────────────────────────────────
        stock_data   = dict(cached_data)
        all_errors   = {}
        chunks_total = max(1, (len(tickers_to_fetch) + CHUNK_SIZE - 1) // CHUNK_SIZE)

        for chunk_idx in range(0, len(tickers_to_fetch), CHUNK_SIZE):
            chunk_tickers = tickers_to_fetch[chunk_idx : chunk_idx + CHUNK_SIZE]
            chunk_symbols = symbols_to_fetch[chunk_idx : chunk_idx + CHUNK_SIZE]
            chunk_num     = chunk_idx // CHUNK_SIZE + 1

            done_before = len(cached_data) + chunk_idx
            pct = int(done_before / total * 70) + 5
            msg = f'Downloading batch {chunk_num}/{chunks_total} ({len(chunk_tickers)} tickers)…'
            yield sse({'type': 'progress', 'pct': pct, 'done': done_before, 'total': total, 'msg': msg})
            print(f"\n  {msg}")

            # Flush any buffered events from retry callbacks
            for ev in events_buf:
                yield sse(ev)
            events_buf.clear()

            chunk_data, chunk_errors = download_chunk(
                chunk_symbols, chunk_tickers, start, end, yield_progress
            )

            # Flush retry messages
            for ev in events_buf:
                yield sse(ev)
            events_buf.clear()

            stock_data.update(chunk_data)
            all_errors.update(chunk_errors)

            done_after = len(cached_data) + chunk_idx + len(chunk_tickers)
            pct2 = int(done_after / total * 70) + 5
            yield sse({'type': 'progress', 'pct': pct2, 'done': done_after, 'total': total,
                       'msg': f'Batch {chunk_num} done — {len(chunk_data)} ok · {len(chunk_errors)} failed'})

            for t in chunk_data:
                print(f"  ✓ {t}")
            for t, e in chunk_errors.items():
                print(f"  ✗ {t}: {e}")

        yield sse({'type': 'progress', 'pct': 78, 'done': len(stock_data), 'total': total,
                   'msg': f'Downloaded {len(stock_data)} stocks · computing metrics…'})

        # ── Compute metrics & screen ──────────────────────────────────────────
        industry_mc      = {}
        industry_vol     = {}
        total_mc_by_date = {}
        shortlist        = []
        all_dates_set    = set()

        for ticker, df in stock_data.items():
            try:
                df = compute_metrics(df)
                industry = industry_map.get(ticker, 'Unknown')
                df['AvgTR'] = rolling_mean_s(df['TradingRange'], lookback)

                last   = df.dropna(subset=['Close']).iloc[-1]
                avg_tr = float(last['AvgTR'])
                mc_bn  = float(last['MarketCap']) / 1e9

                passes = (not np.isnan(avg_tr)) and (avg_tr >= min_tr) and (mc_bn >= min_mc)

                for _, row in df.iterrows():
                    d   = str(row['Date'])[:10]
                    mc  = float(row['MarketCap']) if not pd.isna(row['MarketCap']) else 0
                    vol = float(row['Volume'])    if not pd.isna(row['Volume'])    else 0
                    all_dates_set.add(d)
                    if industry not in industry_mc:
                        industry_mc[industry]  = {}
                        industry_vol[industry] = {}
                    industry_mc[industry][d]   = industry_mc[industry].get(d, 0)  + mc
                    industry_vol[industry][d]  = industry_vol[industry].get(d, 0) + vol
                    total_mc_by_date[d]        = total_mc_by_date.get(d, 0) + mc

                # RSI-14
                rsi_series  = compute_rsi(df['Close'])
                rsi_val     = float(rsi_series.iloc[-1]) if not rsi_series.empty else None
                rsi_val     = None if (rsi_val is not None and np.isnan(rsi_val)) else rsi_val
                rsi_label   = rsi_signal(rsi_val if rsi_val is not None else float('nan'))

                # Volume spike
                vol_ratio_series = compute_volume_spike(df['Volume'])
                vol_ratio        = float(vol_ratio_series.iloc[-1]) if not vol_ratio_series.empty else None
                vol_ratio        = None if (vol_ratio is not None and np.isnan(vol_ratio)) else vol_ratio
                vol_spike        = bool(vol_ratio >= 2.0) if vol_ratio is not None else False

                # 52-week high/low
                high_52w = float(df['High'].max())
                low_52w  = float(df['Low'].min())
                pct_from_high = round((float(last['Close']) - high_52w) / high_52w * 100, 2) if high_52w else None
                pct_from_low  = round((float(last['Close']) - low_52w)  / low_52w  * 100, 2) if low_52w  else None

                if passes:
                    dma5  = rolling_mean_s(df['Close'], 5).iloc[-1]
                    dma20 = rolling_mean_s(df['Close'], 20).iloc[-1]
                    above5  = bool(float(last['Close']) > float(dma5))  if not np.isnan(float(dma5))  else False
                    above20 = bool(float(last['Close']) > float(dma20)) if not np.isnan(float(dma20)) else False

                    # MACD
                    macd_line, macd_signal, macd_hist_s = compute_macd(df['Close'])
                    macd_val    = float(macd_line.iloc[-1])   if not macd_line.empty   else None
                    macd_sig    = float(macd_signal.iloc[-1]) if not macd_signal.empty else None
                    macd_hist_v = float(macd_hist_s.iloc[-1]) if not macd_hist_s.empty else None
                    macd_val    = None if (macd_val    is not None and np.isnan(macd_val))    else macd_val
                    macd_sig    = None if (macd_sig    is not None and np.isnan(macd_sig))    else macd_sig
                    macd_hist_v = None if (macd_hist_v is not None and np.isnan(macd_hist_v)) else macd_hist_v
                    macd_cross  = (macd_val > macd_sig) if (macd_val is not None and macd_sig is not None) else None

                    # Bollinger Bands
                    bb_upper_s, bb_mid_s, bb_lower_s, pct_b_s, bw_s = compute_bollinger(df['Close'])
                    bb_upper = float(bb_upper_s.iloc[-1]) if not bb_upper_s.empty else None
                    bb_mid   = float(bb_mid_s.iloc[-1])   if not bb_mid_s.empty   else None
                    bb_lower = float(bb_lower_s.iloc[-1]) if not bb_lower_s.empty else None
                    pct_b    = float(pct_b_s.iloc[-1])    if not pct_b_s.empty    else None
                    bb_upper = None if (bb_upper is not None and np.isnan(bb_upper)) else bb_upper
                    bb_mid   = None if (bb_mid   is not None and np.isnan(bb_mid))   else bb_mid
                    bb_lower = None if (bb_lower is not None and np.isnan(bb_lower)) else bb_lower
                    pct_b    = None if (pct_b    is not None and np.isnan(pct_b))    else pct_b

                    # BB position label
                    if pct_b is not None:
                        if pct_b >= 1.0:    bb_pos = 'Above Upper'
                        elif pct_b >= 0.8:  bb_pos = 'Near Upper'
                        elif pct_b <= 0.0:  bb_pos = 'Below Lower'
                        elif pct_b <= 0.2:  bb_pos = 'Near Lower'
                        else:               bb_pos = 'Middle'
                    else:
                        bb_pos = 'N/A'

                    # Composite Score
                    comp_score = compute_composite_score(
                        avg_tr, rsi_val, vol_ratio, above5, above20,
                        pct_from_high, macd_hist_v
                    )

                    # Breakout flag
                    breakout = is_breakout(rsi_val, vol_ratio, pct_from_high,
                                           above5, above20, macd_hist_v)

                    shortlist.append({
                        'ticker':        ticker,
                        'industry':      industry,
                        'avgTR':         round(avg_tr * 100, 2),
                        'marketCap':     round(mc_bn, 3),
                        'latestClose':   round(float(last['Close']), 2),
                        'dma5':          round(float(dma5),  2) if not np.isnan(float(dma5))  else None,
                        'dma20':         round(float(dma20), 2) if not np.isnan(float(dma20)) else None,
                        'aboveDMA5':     above5,
                        'aboveDMA20':    above20,
                        'rsi':           round(rsi_val, 1) if rsi_val is not None else None,
                        'rsiLabel':      rsi_label,
                        'volRatio':      round(vol_ratio, 2) if vol_ratio is not None else None,
                        'volSpike':      vol_spike,
                        'high52w':       round(high_52w, 2),
                        'low52w':        round(low_52w,  2),
                        'pctFromHigh':   pct_from_high,
                        'pctFromLow':    pct_from_low,
                        'macd':          round(macd_val,    4) if macd_val    is not None else None,
                        'macdSignal':    round(macd_sig,    4) if macd_sig    is not None else None,
                        'macdHist':      round(macd_hist_v, 4) if macd_hist_v is not None else None,
                        'macdCross':     macd_cross,
                        'bbUpper':       round(bb_upper, 2) if bb_upper is not None else None,
                        'bbMid':         round(bb_mid,   2) if bb_mid   is not None else None,
                        'bbLower':       round(bb_lower, 2) if bb_lower is not None else None,
                        'pctB':          round(pct_b,    3) if pct_b    is not None else None,
                        'bbPosition':    bb_pos,
                        'compositeScore': comp_score,
                        'breakout':      breakout,
                    })
                    flags = []
                    if breakout:  flags.append("🚀 BREAKOUT")
                    if vol_spike: flags.append("🔥 VOL SPIKE")
                    flag_str = " " + " ".join(flags) if flags else ""
                    print(f"  ✓ PASS {ticker:12s} Score={comp_score:5.1f} TR={avg_tr*100:.1f}% RSI={rsi_val:.0f if rsi_val else 0} MC=${mc_bn:.1f}B [{industry}]{flag_str}")
                else:
                    print(f"  – skip {ticker:12s} TR={avg_tr*100:.2f}% MC=${mc_bn:.2f}B")

            except Exception as e:
                all_errors[ticker] = str(e)
                print(f"  ✗ metric {ticker}: {e}")

        yield sse({'type': 'progress', 'pct': 96, 'msg': 'Finalising…'})

        all_dates  = sorted(all_dates_set)
        industries = list(industry_mc.keys())

        cumulative_growth = {}
        for ind in industries:
            vals = [industry_mc[ind].get(d, 0) for d in all_dates]
            base = next((v for v in vals if v > 0), 1)
            cumulative_growth[ind] = [round(v / base, 6) for v in vals]

        total_mc_series = [total_mc_by_date.get(d, 0) for d in all_dates]
        last_mc    = total_mc_series[-1] if total_mc_series else 0
        prev_mc    = total_mc_series[-2] if len(total_mc_series) > 1 else last_mc
        first_mc   = total_mc_series[0]  if total_mc_series else 0
        mc_chg_1d  = ((last_mc - prev_mc)  / prev_mc  * 100) if prev_mc  else 0
        mc_chg_tot = ((last_mc - first_mc) / first_mc * 100) if first_mc else 0

        print(f"\n{'='*55}")
        print(f"  Done — {len(stock_data)} fetched · {len(shortlist)} passed · {len(all_errors)} errors")
        print(f"{'='*55}\n")

        yield sse({
            'type':             'result',
            'shortlist':        shortlist,
            'industryMC':       industry_mc,
            'industryVol':      industry_vol,
            'totalMCByDate':    total_mc_by_date,
            'cumulativeGrowth': cumulative_growth,
            'allDates':         all_dates,
            'allTickers':       raw_tickers,
            'errors':           list(all_errors.keys()),
            'errorDetail':      all_errors,
            'summary': {
                'totalTickers': len(raw_tickers),
                'fetched':      len(stock_data),
                'failed':       len(all_errors),
                'passed':       len(shortlist),
                'lastMC':       last_mc,
                'mcChg1d':      round(mc_chg_1d,  2),
                'mcChgTotal':   round(mc_chg_tot, 2),
                'industries':   len(industries),
                'dateRange':    [all_dates[0], all_dates[-1]] if all_dates else [start, end],
            }
        })
        yield sse({'type': 'done'})

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
    )


@app.route('/api/clear-cache', methods=['POST'])
def clear_cache():
    cleared = 0
    for f in os.listdir(CACHE_DIR):
        if f.endswith('.csv'):
            os.remove(os.path.join(CACHE_DIR, f))
            cleared += 1
    return jsonify({'cleared': cleared})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    print("\n" + "="*55)
    print("  GARO Algo  |  Stock Screener Backend")
    print("  Data source : Yahoo Finance (yfinance)")
    print("  Batch size  : 100 tickers per call")
    print("  Retry logic : 3 attempts, 60s wait")
    print("  Cache TTL   : 12 hours")
    print(f"  URL         : http://localhost:{port}")
    print("="*55 + "\n")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)


# Render.com requires binding to 0.0.0.0 and using the PORT env variable
# The __main__ block above handles local dev; gunicorn handles Render automatically
