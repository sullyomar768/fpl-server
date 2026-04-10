from flask import Flask, jsonify
from flask_cors import CORS
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
 
app = Flask(__name__)
CORS(app)
 
LEAGUE_ID = 2615606
HEADERS = {'User-Agent': 'Mozilla/5.0'}
 
# Chip names as returned by the FPL API
CHIP_NAMES = {
    'bboost':    'Bench Boost',
    '3xc':       'Triple Captain',
    'freehit':   'Free Hit',
    'wildcard':  'Wildcard',
}
 
def get_entry_data(entry_id):
    """Fetch history (GW scores + chips used) for a single manager."""
    url = f'https://fantasy.premierleague.com/api/entry/{entry_id}/history/'
    r = requests.get(url, headers=HEADERS, timeout=10)
    data = r.json()
    scores = [gw['points'] for gw in data['current']]
    # chips used: list of {name, event} — name is e.g. 'bboost', '3xc'
    chips_used = [c['name'] for c in data.get('chips', [])]
    return scores, chips_used
 
@app.route('/standings')
def standings():
    # Step 1 — fetch league standings
    url = f'https://fantasy.premierleague.com/api/leagues-classic/{LEAGUE_ID}/standings/'
    r = requests.get(url, headers=HEADERS, timeout=10)
    data = r.json()
    entries = data['standings']['results']
 
    # Step 2 — fetch all entry histories in parallel
    histories = {}
    chips_used_map = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_entry = {
            executor.submit(get_entry_data, e['entry']): e['entry']
            for e in entries
        }
        for future in as_completed(future_to_entry):
            entry_id = future_to_entry[future]
            try:
                scores, chips_used = future.result()
                histories[entry_id]       = scores
                chips_used_map[entry_id]  = chips_used
            except Exception:
                histories[entry_id]       = []
                chips_used_map[entry_id]  = []
 
    # All chips available per half-season
    ALL_CHIPS = {'bboost', '3xc', 'freehit', 'wildcard'}
 
    # Step 3 — build response
    teams = []
    for e in entries:
        eid        = e['entry']
        used       = set(chips_used_map.get(eid, []))
        # A chip is available if it has NOT been used at all
        # (2025/26 gives 2 of each chip, so a chip is available if used < 2 times)
        used_list  = chips_used_map.get(eid, [])
        chips_remaining = {}
        for chip in ALL_CHIPS:
            times_used = used_list.count(chip)
            remaining  = max(0, 2 - times_used)
            chips_remaining[chip] = remaining
 
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
 
    return jsonify(sorted(teams, key=lambda t: t['name'].lower()))
 
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
