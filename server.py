from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import time
import threading
import math
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

app = Flask(__name__)
CORS(app)

HEADERS = {'User-Agent': 'Mozilla/5.0'}

# ── CACHE CONFIG ──────────────────────────────────────────────────────────
ENTRY_TTL     = 6 * 60
STANDINGS_TTL = 5 * 60
CURRENTGW_TTL = 5 * 60
MAX_WORKERS   = 25

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


# ── FPL DATA FETCHING (with per-entry cache) ──────────────────────────────

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
    """Fetch league standings + all histories. Uses standings cache."""
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

    data         = r.json()
    entries      = data['standings']['results']
    league_info  = data.get('league', {})
    histories    = fetch_all_entries(entries)

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


# ── SIMULATION ENGINE (Python port of the browser JS) ────────────────────
# This is the proprietary logic — it never leaves the server.

def rand_gamma(k, theta):
    """Marsaglia-Tsang gamma sampler. Always positive, right-skewed."""
    if k < 1:
        return rand_gamma(k + 1, theta) * (random.random() ** (1.0 / k))
    d = k - 1.0 / 3.0
    c = 1.0 / math.sqrt(9.0 * d)
    while True:
        x = random.gauss(0, 1)
        v = 1.0 + c * x
        if v <= 0:
            continue
        v = v * v * v
        u = random.random()
        if u < 1.0 - 0.0331 * (x * x) * (x * x):
            return d * v * theta
        if math.log(u) < 0.5 * x * x + d * (1.0 - v + math.log(v)):
            return d * v * theta


def trim_outliers(scores):
    """Remove scores more than 2 SD above mean (chip weeks). Only trims upward."""
    if len(scores) < 4:
        return scores
    mean = sum(scores) / len(scores)
    variance = sum((x - mean) ** 2 for x in scores) / len(scores)
    sd        = math.sqrt(variance)
    threshold = mean + 2 * sd
    trimmed   = [s for s in scores if s <= threshold]
    return trimmed if len(trimmed) >= math.ceil(len(scores) / 2) else scores


def calc_stats(scores):
    if not scores:
        return None
    n        = len(scores)
    mean     = sum(scores) / n
    variance = sum((x - mean) ** 2 for x in scores) / n
    if variance < 1:
        variance = mean * 0.10
    return {'mean': mean, 'variance': variance}


def build_form_stats(history, season_mean, mode):
    """Returns {mean, variance} for the Gamma sampler based on prediction mode."""
    season_stats = None
    if history:
        season_stats = calc_stats(trim_outliers(history[:]))

    if not season_stats:
        return {'mean': season_mean, 'variance': season_mean * 0.25}

    if mode == 'season':
        return season_stats

    recent_raw    = history[-6:]
    recent_stats  = calc_stats(trim_outliers(recent_raw[:]))

    if not recent_stats or len(trim_outliers(recent_raw[:])) < 3:
        return season_stats

    if mode == 'form':
        return recent_stats

    # blended: 70% season + 30% recent
    return {
        'mean':     0.70 * season_stats['mean']     + 0.30 * recent_stats['mean'],
        'variance': 0.70 * season_stats['variance'] + 0.30 * recent_stats['variance'],
    }


def sample_week(stats):
    theta = stats['variance'] / stats['mean']
    k     = stats['mean'] / theta
    if k <= 0 or not math.isfinite(k):
        k = 3.33
    if theta <= 0 or not math.isfinite(theta):
        theta = stats['mean'] * 0.30
    return rand_gamma(k, theta)


def build_chip_bonuses(chips, gws_remaining, team_mean):
    bonuses = [0.0] * gws_remaining
    if not chips or gws_remaining < 1:
        return bonuses

    def pick_gw():
        return random.randint(0, gws_remaining - 1)

    # Triple Captain: adds 80-140% of mean to one GW
    if chips.get('3xc', 0) > 0:
        bonuses[pick_gw()] += team_mean * (0.8 + random.random() * 0.6)

    # Bench Boost: ~18 pts (SD 8) on one GW
    if chips.get('bboost', 0) > 0:
        bb_bonus = max(2.0, random.gauss(18, 8))
        bonuses[pick_gw()] += bb_bonus

    return bonuses


def run_simulation(teams, gws_played, league_start, sims, pred_mode, deductions):
    """
    Core Monte Carlo simulation. Returns result list sorted by avg position.
    deductions: list of {name: str (lowercase), pts: int}
    """
    n            = len(teams)
    gws_remaining = max(1, 38 - gws_played)

    # Apply deductions to current pts
    adjusted_pts = []
    for t in teams:
        pts   = t['pts']
        tname = t['name'].lower()
        for d in deductions:
            if d['name'] in tname:
                pts = max(0, pts - d['pts'])
        adjusted_pts.append(pts)

    # Build per-team form stats
    league_gws_played = max(1, gws_played - league_start + 1) if gws_played >= league_start else max(1, gws_played)

    form_stats = []
    for i, t in enumerate(teams):
        history      = t.get('history', [])
        league_hist  = history[league_start - 1:] if history else []
        season_mean  = adjusted_pts[i] / league_gws_played
        form_stats.append(build_form_stats(league_hist, season_mean, pred_mode))

    chips_arr = [t.get('chips_remaining', {}) for t in teams]

    # Accumulators
    wins         = [0] * n
    lasts        = [0] * n
    proj_sum     = [0.0] * n
    pos_sum      = [0.0] * n
    finish_counts = [[0] * n for _ in range(n)]

    for _ in range(sims):
        scores = []
        for i in range(n):
            sc          = adjusted_pts[i]
            week_scores = [sample_week(form_stats[i]) for _ in range(gws_remaining)]
            chips       = chips_arr[i]

            if chips and (chips.get('3xc', 0) > 0 or chips.get('bboost', 0) > 0 or chips.get('freehit', 0) > 0):
                bonuses = build_chip_bonuses(chips, gws_remaining, form_stats[i]['mean'])
                # Free Hit: replace worst week with ~mean
                if chips.get('freehit', 0) > 0:
                    min_idx = week_scores.index(min(week_scores))
                    week_scores[min_idx] = max(week_scores[min_idx], form_stats[i]['mean'] * 0.9)
                sc += sum(w + b for w, b in zip(week_scores, bonuses))
            else:
                sc += sum(week_scores)

            scores.append(sc)
            proj_sum[i] += sc

        order = sorted(range(n), key=lambda x: scores[x], reverse=True)
        for rank, idx in enumerate(order):
            pos_sum[idx]          += rank + 1
            finish_counts[idx][rank] += 1
        wins[order[0]]   += 1
        lasts[order[-1]] += 1

    results = []
    for i, t in enumerate(teams):
        results.append({
            'name':         t['name'],
            'manager':      t.get('manager', ''),
            'pts':          adjusted_pts[i],
            'raw_pts':      t['pts'],
            'mean':         round(adjusted_pts[i] / league_gws_played, 1),
            'win_prob':     round(wins[i] / sims * 100, 2),
            'last_prob':    round(lasts[i] / sims * 100, 2),
            'win_count':    wins[i],
            'last_count':   lasts[i],
            'projected':    round(proj_sum[i] / sims),
            'avg_pos':      round(pos_sum[i] / sims, 2),
            'finish_counts': finish_counts[i],
            'chips_remaining': t.get('chips_remaining', {}),
        })

    return sorted(results, key=lambda x: x['avg_pos'])


# ── ROUTES ────────────────────────────────────────────────────────────────

@app.route('/health')
def health():
    return 'ok', 200


@app.route('/simulate', methods=['POST'])
def simulate():
    """
    Main endpoint. Client sends league config, server fetches data,
    runs simulation, returns results. No simulation logic ever sent to client.

    Body (JSON):
    {
        "league_id":   2615606,
        "sims":        100000,       // 10000 | 100000 | 1000000
        "pred_mode":   "blended",    // "season" | "blended" | "form"
        "gws_played":  32,           // current GW (client sends from its UI)
        "deductions":  [             // optional
            {"name": "bayern bru", "pts": 82}
        ]
    }
    """
    body = request.get_json(silent=True) or {}

    league_id  = body.get('league_id')
    sims       = min(int(body.get('sims', 100000)), 1000000)  # cap at 1M
    pred_mode  = body.get('pred_mode', 'blended')
    gws_played = int(body.get('gws_played', 32))
    deductions = body.get('deductions', [])

    if not league_id:
        return jsonify({'error': 'league_id required'}), 400
    if pred_mode not in ('season', 'blended', 'form'):
        return jsonify({'error': 'invalid pred_mode'}), 400

    # Normalise deduction names to lowercase for matching
    deductions = [{'name': d['name'].lower().strip(), 'pts': int(d['pts'])} for d in deductions if d.get('name') and d.get('pts')]

    # Fetch league data (uses cache)
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
    print(f'[SIM] league={league_id} teams={len(teams)} sims={sims} mode={pred_mode} elapsed={elapsed:.2f}s')

    return jsonify({
        'league_name':  league_name,
        'league_start': league_start,
        'gws_played':   gws_played,
        'sims':         sims,
        'teams':        results,
    })


@app.route('/standings')
def standings():
    """Still available for the 'Load League' / team preview step."""
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
            n_l = len(_standings_cache)
            n_e = len(_entry_cache)
            _standings_cache.clear()
            _entry_cache.clear()
            _currentgw_cache.clear()
            msg = f'Cleared {n_l} leagues and {n_e} entries'
    print(f'[CACHE CLEARED] {msg}')
    return jsonify({'ok': True, 'message': msg})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    
