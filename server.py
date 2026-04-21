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

# ── CACHE CONFIG ──────────────────────────────────────────────────────────
ENTRY_TTL     = 6 * 60
STANDINGS_TTL = 5 * 60
CURRENTGW_TTL = 5 * 60
MAX_WORKERS   = 25
SIM_CHUNK     = 10000   # sims per chunk — keeps peak RAM ~3MB for 7 teams

_entry_cache     = {}
_standings_cache = {}
_currentgw_cache = {}
_lock            = threading.RLock()

ALL_CHIPS = {'bboost', '3xc', 'freehit', 'wildcard'}


# ── CURRENT GW ────────────────────────────────────────────────────────────

def get_current_gw():
    with _lock:
        if _currentgw_cache and (time.time() - _currentgw_cache['ts']) < CURRENTGW_TTL:
            return _currentgw_cache['gw']
    try:
        r = requests.get(
            'https://fantasy.premierleague.com/api/bootstrap-static/',
            headers=HEADERS, timeout=10
        )
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


# ── FPL DATA FETCHING ─────────────────────────────────────────────────────

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
        print(f'[ENTRY FETCH] {len(to_fetch)} uncached / {len(entries)-len(to_fetch)} from cache')
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

    data      = r.json()
    entries   = data['standings']['results']
    league_info = data.get('league', {})
    histories = fetch_all_entries(entries)

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
# Chunked NumPy: vectorised within each chunk so it's fast, but peak RAM
# stays at ~(chunk_size × N × GWS × 8 bytes) regardless of total sims.
# At SIM_CHUNK=10000, 7 teams, 6 GWs → ~3 MB peak. Safe for Render free tier.

def trim_outliers(scores):
    if len(scores) < 4:
        return scores
    arr  = np.array(scores, dtype=float)
    mean = arr.mean()
    sd   = arr.std()
    mask = arr <= (mean + 2 * sd)
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
    season_stats = calc_stats(trim_outliers(history)) if history else None
    if not season_stats:
        return {'mean': season_mean, 'variance': season_mean * 0.25}
    if mode == 'season':
        return season_stats
    recent       = history[-6:]
    recent_stats = calc_stats(trim_outliers(recent))
    if not recent_stats or len(trim_outliers(recent)) < 3:
        return season_stats
    if mode == 'form':
        return recent_stats
    return {
        'mean':     0.70 * season_stats['mean']     + 0.30 * recent_stats['mean'],
        'variance': 0.70 * season_stats['variance'] + 0.30 * recent_stats['variance'],
    }


def run_simulation(teams, gws_played, league_start, sims, pred_mode, deductions):
    n             = len(teams)
    gws_remaining = max(1, 38 - gws_played)
    league_gws_played = (
        max(1, gws_played - league_start + 1)
        if gws_played >= league_start
        else max(1, gws_played)
    )

    # Apply deductions
    adjusted_pts = []
    for t in teams:
        pts   = t['pts']
        tname = t['name'].lower()
        for d in deductions:
            if d['name'] in tname:
                pts = max(0, pts - d['pts'])
        adjusted_pts.append(float(pts))

    base_pts = np.array(adjusted_pts)

    # Pre-compute Gamma params and form stats per team
    ks         = np.zeros(n)
    thetas     = np.zeros(n)
    stats_list = []
    for i, t in enumerate(teams):
        history     = t.get('history', [])
        league_hist = history[league_start - 1:] if history else []
        season_mean = adjusted_pts[i] / league_gws_played
        fs          = build_form_stats(league_hist, season_mean, pred_mode)
        stats_list.append(fs)
        theta     = fs['variance'] / fs['mean']
        k         = fs['mean'] / theta
        thetas[i] = max(theta, 0.01)
        ks[i]     = max(k, 0.01)

    # Accumulators
    wins         = np.zeros(n, dtype=np.int64)
    lasts        = np.zeros(n, dtype=np.int64)
    proj_sum     = np.zeros(n, dtype=np.float64)
    pos_sum      = np.zeros(n, dtype=np.float64)
    finish_counts = np.zeros((n, n), dtype=np.int64)

    # ── CHUNKED SIMULATION LOOP ───────────────────────────────────────────
    done = 0
    while done < sims:
        cs = min(SIM_CHUNK, sims - done)  # actual chunk size

        # Chip bonuses: (N, cs)
        chip_bonus = np.zeros((n, cs))
        for i, t in enumerate(teams):
            chips = t.get('chips_remaining', {})
            mean  = stats_list[i]['mean']
            if chips.get('3xc', 0) > 0:
                chip_bonus[i] += mean * (0.8 + np.random.random(cs) * 0.6)
            if chips.get('bboost', 0) > 0:
                chip_bonus[i] += np.maximum(2.0, np.random.normal(18, 8, cs))
            if chips.get('freehit', 0) > 0:
                chip_bonus[i] += mean * 0.15

        # Weekly scores: (N, cs, GWS) → sum → (N, cs)
        weekly = np.stack([
            np.random.gamma(ks[i], thetas[i], (cs, gws_remaining))
            for i in range(n)
        ], axis=0)
        weekly_totals = weekly.sum(axis=2)  # (N, cs)

        # Final scores: (N, cs)
        final = base_pts[:, np.newaxis] + weekly_totals + chip_bonus

        # Rank: (cs, N) — argsort descending
        ft    = final.T
        order = np.argsort(-ft, axis=1)  # (cs, N)

        # Update accumulators
        np.add.at(wins,  order[:, 0],  1)
        np.add.at(lasts, order[:, -1], 1)
        proj_sum += ft.sum(axis=0)

        positions = np.empty_like(order)
        positions[np.arange(cs)[:, np.newaxis], order] = np.arange(1, n + 1)[np.newaxis, :]
        pos_sum += positions.sum(axis=0)

        for r in range(n):
            np.add.at(finish_counts[:, r], order[:, r], 1)

        done += cs

    # ── RESULTS ───────────────────────────────────────────────────────────
    results = []
    for i, t in enumerate(teams):
        results.append({
            'name':            t['name'],
            'manager':         t.get('manager', ''),
            'pts':             int(adjusted_pts[i]),
            'raw_pts':         t['pts'],
            'mean':            round(adjusted_pts[i] / league_gws_played, 1),
            'win_prob':        round(float(wins[i])  / sims * 100, 2),
            'last_prob':       round(float(lasts[i]) / sims * 100, 2),
            'win_count':       int(wins[i]),
            'last_count':      int(lasts[i]),
            'projected':       int(round(float(proj_sum[i]) / sims)),
            'avg_pos':         round(float(pos_sum[i]) / sims, 2),
            'finish_counts':   finish_counts[i].tolist(),
            'chips_remaining': t.get('chips_remaining', {}),
        })

    return sorted(results, key=lambda x: x['avg_pos'])


# ── ROUTES ────────────────────────────────────────────────────────────────

@app.route('/health')
def health():
    return 'ok', 200


@app.route('/simulate', methods=['POST'])
def simulate():
    body = request.get_json(silent=True) or {}

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

    teams        = raw['teams']
    league_start = raw['league_start']
    league_name  = raw['league_name']

    if not teams:
        return jsonify({'error': 'No teams found'}), 404

    t0      = time.time()
    results = run_simulation(teams, gws_played, league_start, sims, pred_mode, deductions)
    elapsed = time.time() - t0
    print(f'[SIM] league={league_id} teams={len(teams)} sims={sims} mode={pred_mode} elapsed={elapsed:.3f}s')

    return jsonify({
        'league_name':  league_name,
        'league_start': league_start,
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
            msg = f'Cleared standings for league {league_id}' if removed else f'No cache for {league_id}'
        else:
            n_l = len(_standings_cache); n_e = len(_entry_cache)
            _standings_cache.clear(); _entry_cache.clear(); _currentgw_cache.clear()
            msg = f'Cleared {n_l} leagues and {n_e} entries'
    print(f'[CACHE CLEARED] {msg}')
    return jsonify({'ok': True, 'message': msg})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
