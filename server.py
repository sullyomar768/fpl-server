from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

app = Flask(__name__)
CORS(app)

HEADERS = {'User-Agent': 'Mozilla/5.0'}

def get_entry_data(entry_id):
    url = f'https://fantasy.premierleague.com/api/entry/{entry_id}/history/'
    r = requests.get(url, headers=HEADERS, timeout=10)
    data = r.json()
    scores = [gw['points'] for gw in data['current']]
    chips_used = [c['name'] for c in data.get('chips', [])]
    return scores, chips_used

@app.route('/health')
def health():
    return 'ok', 200

@app.route('/standings')
def standings():
    league_id = request.args.get('league_id')
    if not league_id:
        return jsonify({'error': 'league_id parameter required'}), 400

    url = f'https://fantasy.premierleague.com/api/leagues-classic/{league_id}/standings/'
    r = requests.get(url, headers=HEADERS, timeout=10)
    if r.status_code != 200:
        return jsonify({'error': 'League not found'}), 404

    data = r.json()
    entries = data['standings']['results']
    league_info = data.get('league', {})
    league_name  = league_info.get('name', '')
    league_start = league_info.get('start_event', 1)

    # Fetch all histories in parallel
    histories = {}
    chips_map = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(get_entry_data, e['entry']): e['entry'] for e in entries}
        for future in as_completed(futures):
            eid = futures[future]
            try:
                scores, chips_used = future.result()
                histories[eid] = scores
                chips_map[eid] = chips_used
            except Exception:
                histories[eid] = []
                chips_map[eid] = []

    ALL_CHIPS = {'bboost', '3xc', 'freehit', 'wildcard'}
    teams = []
    for e in entries:
        eid = e['entry']
        used_list = chips_map.get(eid, [])
        chips_remaining = {chip: max(0, 2 - used_list.count(chip)) for chip in ALL_CHIPS}
        teams.append({
            'name':            e['entry_name'],
            'manager':         e['player_name'],
            'pts':             e['total'],
            'event_total':     e['event_total'],
            'rank':            e['rank'],
            'last_rank':       e['last_rank'],
            'history':         histories.get(eid, []),
            'chips_remaining': chips_remaining,
        })

    return jsonify({
        'league_name':  league_name,
        'league_start': league_start,
        'teams':        sorted(teams, key=lambda t: t['name'].lower()),
    })

@app.route('/currentgw')
def currentgw():
    url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
    r = requests.get(url, headers=HEADERS, timeout=10)
    data = r.json()
    current = 1
    for event in data['events']:
        if event['finished']:
            current = event['id']
    return jsonify({'gw': current})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
