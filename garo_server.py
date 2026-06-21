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
                    shortlist.append({
                        'ticker':        ticker,
                        'industry':      industry,
                        'avgTR':         round(avg_tr * 100, 2),
                        'marketCap':     round(mc_bn, 3),
                        'latestClose':   round(float(last['Close']), 2),
                        'dma5':          round(float(dma5),  2) if not np.isnan(float(dma5))  else None,
                        'dma20':         round(float(dma20), 2) if not np.isnan(float(dma20)) else None,
                        'aboveDMA5':     bool(float(last['Close']) > float(dma5))  if not np.isnan(float(dma5))  else None,
                        'aboveDMA20':    bool(float(last['Close']) > float(dma20)) if not np.isnan(float(dma20)) else None,
                        'rsi':           round(rsi_val, 1) if rsi_val is not None else None,
                        'rsiLabel':      rsi_label,
                        'volRatio':      round(vol_ratio, 2) if vol_ratio is not None else None,
                        'volSpike':      vol_spike,
                        'high52w':       round(high_52w, 2),
                        'low52w':        round(low_52w,  2),
                        'pctFromHigh':   pct_from_high,
                        'pctFromLow':    pct_from_low,
                    })
                    spike_str = " 🔥 VOL SPIKE" if vol_spike else ""
                    print(f"  ✓ PASS {ticker:12s} TR={avg_tr*100:.2f}% MC=${mc_bn:.2f}B RSI={rsi_val:.1f if rsi_val else 'N/A'} [{industry}]{spike_str}")
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
