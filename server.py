from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

app = Flask(__name__)
CORS(app)

HEADERS = {'User-Agent': 'Mozilla/5.0'}

ENTRY_TTL      = 6 * 60
STANDINGS_TTL  = 5 * 60
CURRENTGW_TTL  = 5 * 60
MAX_WORKERS    = 25

_entry_cache     = {}
_standings_cache = {}
_currentgw_cache = {}
_lock            = threading.RLock()

ALL_CHIPS = {'bboost', '3xc', 'freehit', 'wildcard'}


def get_current_gw():
    with _lock:
        now = time.time()
        if _currentgw_cache and (now - _currentgw_cache['ts']) < CURRENTGW_TTL:
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


def build_standings(league_id):
    url = f'https://fantasy.premierleague.com/api/leagues-classic/{league_id}/standings/'
    r   = requests.get(url, headers=HEADERS, timeout=10)
    if r.status_code != 200:
        return None, r.status_code
    data         = r.json()
    entries      = data['standings']['results']
    league_info  = data.get('league', {})
    league_name  = league_info.get('name', '')
    league_start = league_info.get('start_event', 1)
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
    payload = {
        'league_name':  league_name,
        'league_start': league_start,
        'teams':        sorted(teams, key=lambda t: t['name'].lower()),
    }
    return payload, 200


@app.route('/health')
def health():
    return 'ok', 200


@app.route('/standings')
def standings():
    league_id     = request.args.get('league_id')
    force_refresh = request.args.get('refresh') == '1'
    if not league_id:
        return jsonify({'error': 'league_id parameter required'}), 400
    now        = time.time()
    current_gw = get_current_gw()
    if not force_refresh:
        with _lock:
            cached = _standings_cache.get(league_id)
        if cached:
            age = now - cached['ts']
            if age < STANDINGS_TTL and cached['gw'] == current_gw:
                print(f'[STANDINGS HIT] league={league_id} age={age:.1f}s')
                return jsonify(cached['data'])
    t0              = time.time()
    payload, status = build_standings(league_id)
    elapsed         = time.time() - t0
    print(f'[STANDINGS BUILT] league={league_id} status={status} elapsed={elapsed:.2f}s')
    if status != 200 or payload is None:
        return jsonify({'error': 'League not found'}), 404
    with _lock:
        _standings_cache[league_id] = {'data': payload, 'ts': time.time(), 'gw': current_gw}
    return jsonify(payload)


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
