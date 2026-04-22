from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import time
import threading
import math
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

app = Flask(__name__)
CORS(app)

HEADERS = {'User-Agent': 'Mozilla/5.0'}

ENTRY_TTL     = 6 * 60
STANDINGS_TTL = 5 * 60
CURRENTGW_TTL = 5 * 60
MAX_WORKERS   = 25
SIM_CHUNK     = 10000

_entry_cache     = {}
_standings_cache = {}
_currentgw_cache = {}
_lock            = threading.RLock()
ALL_CHIPS        = {'bboost', '3xc', 'freehit', 'wildcard'}
_rng             = np.random.Generator(np.random.PCG64DXSM())


def get_current_gw():
    with _lock:
        if _currentgw_cache and (time.time() - _currentgw_cache['ts']) < CURRENTGW_TTL:
            return _currentgw_cache['gw']
    try:
        r = requests.get('https://fantasy.premierleague.com/api/bootstrap-static/',
                         headers=HEADERS, timeout=10)
        current = 1
        for event in r.json()['events']:
            if event['finished']:
                current = event['id']
        with _lock:
            _currentgw_cache['gw'] = current
            _currentgw_cache['ts'] = time.time()
        return current
    except Exception:
        with _lock:
            return _currentgw_cache.get('gw', 1)


def fetch_entry(entry_id):
    with _lock:
        cached = _entry_cache.get(entry_id)
        if cached and (time.time() - cached['ts']) < ENTRY_TTL:
            return cached['scores'], cached['chips']
    url  = f'https://fantasy.premierleague.com/api/entry/{entry_id}/history/'
    r    = requests.get(url, headers=HEADERS, timeout=15)
    data = r.json()
    scores     = [gw['points'] for gw in data['current']]
    chips_used = [c['name']    for c in data.get('chips', [])]
    with _lock:
        _entry_cache[entry_id] = {'scores': scores, 'chips': chips_used, 'ts': time.time()}
    return scores, chips_used


def fetch_all_entries(entries):
    results  = {}
    to_fetch = []
    with _lock:
        for e in entries:
            eid    = e['entry']
            cached = _entry_cache.get(eid)
            if cached and (time.time() - cached['ts']) < ENTRY_TTL:
                results[eid] = (cached['scores'], cached['chips'])
            else:
                to_fetch.append(eid)
    if to_fetch:
        print(f'[ENTRY FETCH] {len(to_fetch)} uncached / {len(entries)-len(to_fetch)} cached')
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(to_fetch))) as executor:
            future_map = {executor.submit(fetch_entry, eid): eid for eid in to_fetch}
            for future in as_completed(future_map):
                eid = future_map[future]
                try:
                    results[eid] = future.result()
                except Exception as ex:
                    print(f'[ENTRY ERROR] entry={eid} error={ex}')
                    results[eid] = ([], [])
    else:
        print(f'[ENTRY FETCH] all {len(entries)} from cache')
    return results


def get_league_data(league_id):
    now        = time.time()
    current_gw = get_current_gw()
    with _lock:
        cached = _standings_cache.get(league_id)
    if cached:
        age = now - cached['ts']
        if age < STANDINGS_TTL and cached['gw'] == current_gw:
            print(f'[STANDINGS HIT] league={league_id} age={age:.1f}s')
            return cached['raw'], None

    url = f'https://fantasy.premierleague.com/api/leagues-classic/{league_id}/standings/'
    r   = requests.get(url, headers=HEADERS, timeout=10)
    if r.status_code != 200:
        return None, 'League not found'

    data        = r.json()
    entries     = data['standings']['results']
    league_info = data.get('league', {})
    histories   = fetch_all_entries(entries)

    teams = []
    for e in entries:
        eid                = e['entry']
        scores, chips_used = histories.get(eid, ([], []))
        chips_remaining    = {chip: max(0, 2 - chips_used.count(chip)) for chip in ALL_CHIPS}
        teams.append({
            'name':            e['entry_name'],
            'manager':         e['player_name'],
            'pts':             e['total'],
            'event_total':     e['event_total'],
            'rank':            e['rank'],
            'last_rank':       e['last_rank'],
            'history':         scores,
            'chips_remaining': chips_remaining,
        })

    raw = {
        'league_name':  league_info.get('name', ''),
        'league_start': league_info.get('start_event', 1),
        'teams':        sorted(teams, key=lambda t: t['name'].lower()),
    }
    with _lock:
        _standings_cache[league_id] = {'raw': raw, 'ts': time.time(), 'gw': current_gw}
    return raw, None


# ── SIMULATION ENGINE ─────────────────────────────────────────────────────
# Optimisations applied:
# 1. Sum-of-Gammas identity: Σ(GWS × Gamma(k,θ)) = Gamma(k×GWS, θ)
#    Eliminates the inner GWS loop — one RNG call per team per chunk.
# 2. Chunked: peak RAM ≈ N × CHUNK × 8 bytes (4MB for 50 teams). Never OOMs.
# 3. Ravelled bincount: finish_counts built with one np.bincount instead of N calls.
# 4. argsort(argsort): gives each team's rank without scatter/assignment.
# 5. PCG64DXSM: numpy's fastest bit generator.

def trim_outliers(scores):
    if len(scores) < 4:
        return scores
    arr  = np.array(scores, dtype=float)
    mask = arr <= (arr.mean() + 2 * arr.std())
    trimmed = arr[mask].tolist()
    return trimmed if len(trimmed) >= math.ceil(len(scores) / 2) else scores


def calc_stats(scores):
    if not scores:
        return None
    arr      = np.array(scores, dtype=float)
    mean     = float(arr.mean())
    variance = float(arr.var())
    if variance < 1:
        variance = mean * 0.10
    return {'mean': mean, 'variance': variance}


def build_form_stats(history, season_mean, mode):
    ss = calc_stats(trim_outliers(history)) if history else None
    if not ss:
        return {'mean': season_mean, 'variance': season_mean * 0.25}
    if mode == 'season':
        return ss
    recent = history[-6:]
    rs     = calc_stats(trim_outliers(recent))
    if not rs or len(trim_outliers(recent)) < 3:
        return ss
    if mode == 'form':
        return rs
    return {
        'mean':     0.70 * ss['mean']     + 0.30 * rs['mean'],
        'variance': 0.70 * ss['variance'] + 0.30 * rs['variance'],
    }


def run_simulation(teams, gws_played, league_start, sims, pred_mode, deductions):
    n             = len(teams)
    gws_remaining = max(1, 38 - gws_played)
    lgp           = max(1, gws_played - league_start + 1) if gws_played >= league_start else max(1, gws_played)

    # Apply deductions
    adj_pts = []
    for t in teams:
        pts = t['pts']
        for d in deductions:
            if d['name'] in t['name'].lower():
                pts = max(0, pts - d['pts'])
        adj_pts.append(float(pts))
    base_pts = np.array(adj_pts)

    # Per-team Gamma params — collapse GWS with sum-of-gammas identity
    ks_gws  = np.zeros(n)
    thetas  = np.zeros(n)
    chip_tc = np.zeros(n)   # triple captain mean value (0 = unused)
    chip_bb = np.zeros(n)   # bench boost flag (0 = unused)
    chip_fh = np.zeros(n)   # free hit flat bonus (0 = unused)

    for i, t in enumerate(teams):
        history     = t.get('history', [])
        league_hist = history[league_start - 1:] if history else []
        fs          = build_form_stats(league_hist, adj_pts[i] / lgp, pred_mode)
        theta       = fs['variance'] / fs['mean']
        k           = fs['mean'] / theta
        thetas[i]   = max(theta, 0.01)
        ks_gws[i]   = max(k, 0.01) * gws_remaining  # sum-of-gammas
        chips       = t.get('chips_remaining', {})
        mean        = fs['mean']
        if chips.get('3xc',    0) > 0: chip_tc[i] = mean
        if chips.get('bboost', 0) > 0: chip_bb[i] = 1.0
        if chips.get('freehit',0) > 0: chip_fh[i] = mean * 0.15

    any_chips    = chip_tc.any() or chip_bb.any() or chip_fh.any()
    rank_offsets = (np.arange(n) * n)[np.newaxis, :]  # for ravelled bincount

    wins         = np.zeros(n, dtype=np.int64)
    lasts        = np.zeros(n, dtype=np.int64)
    proj_sum     = np.zeros(n, dtype=np.float64)
    pos_sum      = np.zeros(n, dtype=np.float64)
    finish_counts = np.zeros((n, n), dtype=np.int64)

    done = 0
    while done < sims:
        cs = min(SIM_CHUNK, sims - done)

        # Weekly totals — one gamma call per team (GWS collapsed)
        weekly = np.empty((n, cs))
        for i in range(n):
            weekly[i] = _rng.standard_gamma(ks_gws[i], size=cs) * thetas[i]

        # Chip bonuses
        if any_chips:
            for i in np.where(chip_tc > 0)[0]:
                weekly[i] += chip_tc[i] * (0.8 + _rng.random(cs) * 0.6)
            for i in np.where(chip_bb > 0)[0]:
                weekly[i] += np.maximum(2.0, _rng.normal(18, 8, cs))
            if chip_fh.any():
                weekly += chip_fh[:, np.newaxis]

        ft    = (weekly + base_pts[:, np.newaxis]).T   # (cs, N)
        order = np.argsort(-ft, axis=1)                # (cs, N)

        wins  += np.bincount(order[:, 0],  minlength=n)
        lasts += np.bincount(order[:, -1], minlength=n)
        proj_sum += ft.sum(axis=0)
        pos_sum  += np.argsort(order, axis=1).sum(axis=0)  # argsort(argsort) = 0-indexed ranks
        finish_counts += np.bincount(
            (order + rank_offsets).ravel(), minlength=n * n
        ).reshape(n, n)

        done += cs

    results = []
    for i, t in enumerate(teams):
        results.append({
            'name':            t['name'],
            'manager':         t.get('manager', ''),
            'pts':             int(adj_pts[i]),
            'raw_pts':         t['pts'],
            'mean':            round(adj_pts[i] / lgp, 1),
            'win_prob':        round(float(wins[i])  / sims * 100, 2),
            'last_prob':       round(float(lasts[i]) / sims * 100, 2),
            'win_count':       int(wins[i]),
            'last_count':      int(lasts[i]),
            'projected':       int(round(float(proj_sum[i]) / sims)),
            'avg_pos':         round(float(pos_sum[i]) / sims + 1, 2),
            'finish_counts':   finish_counts[i].tolist(),
            'chips_remaining': t.get('chips_remaining', {}),
        })

    return sorted(results, key=lambda x: x['avg_pos'])


@app.route('/')
@app.route('/health')
def health():
    return 'ok', 200


@app.route('/simulate', methods=['POST'])
def simulate():
    body       = request.get_json(silent=True) or {}
    league_id  = body.get('league_id')
    sims       = min(int(body.get('sims', 100000)), 1000000)
    pred_mode  = body.get('pred_mode', 'blended')
    gws_played = int(body.get('gws_played', 32))
    deductions = body.get('deductions', [])

    if not league_id:
        return jsonify({'error': 'league_id required'}), 400
    if pred_mode not in ('season', 'blended', 'form'):
        return jsonify({'error': 'invalid pred_mode'}), 400

    deductions = [
        {'name': d['name'].lower().strip(), 'pts': int(d['pts'])}
        for d in deductions if d.get('name') and d.get('pts')
    ]

    raw, err = get_league_data(str(league_id))
    if err:
        return jsonify({'error': err}), 404

    if not raw['teams']:
        return jsonify({'error': 'No teams found'}), 404

    t0      = time.time()
    results = run_simulation(raw['teams'], gws_played, raw['league_start'], sims, pred_mode, deductions)
    elapsed = time.time() - t0
    print(f'[SIM] league={league_id} n={len(raw["teams"])} sims={sims} elapsed={elapsed:.3f}s')

    return jsonify({
        'league_name':  raw['league_name'],
        'league_start': raw['league_start'],
        'gws_played':   gws_played,
        'sims':         sims,
        'teams':        results,
    })


@app.route('/standings')
def standings():
    league_id     = request.args.get('league_id')
    force_refresh = request.args.get('refresh') == '1'
    if not league_id:
        return jsonify({'error': 'league_id parameter required'}), 400
    if force_refresh:
        with _lock:
            _standings_cache.pop(league_id, None)
    raw, err = get_league_data(league_id)
    if err:
        return jsonify({'error': err}), 404
    return jsonify(raw)


@app.route('/currentgw')
def currentgw():
    return jsonify({'gw': get_current_gw()})


@app.route('/clearcache')
def clearcache():
    secret    = request.args.get('key')
    league_id = request.args.get('league_id')
    if secret != 'suleiman':
        return jsonify({'error': 'unauthorized'}), 403
    with _lock:
        if league_id:
            removed = _standings_cache.pop(league_id, None)
            msg = f'Cleared standings for {league_id}' if removed else f'No cache for {league_id}'
        else:
            n_l = len(_standings_cache); n_e = len(_entry_cache)
            _standings_cache.clear(); _entry_cache.clear(); _currentgw_cache.clear()
            msg = f'Cleared {n_l} leagues and {n_e} entries'
    print(f'[CACHE CLEARED] {msg}')
    return jsonify({'ok': True, 'message': msg})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    
