from flask import Flask, jsonify
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

LEAGUE_ID = 2615606

def get_history(entry_id):
    url = f'https://fantasy.premierleague.com/api/entry/{entry_id}/history/'
    r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
    data = r.json()
    scores = [gw['points'] for gw in data['current']]
    return scores

@app.route('/standings')
def standings():
    url = f'https://fantasy.premierleague.com/api/leagues-classic/{LEAGUE_ID}/standings/'
    r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
    data = r.json()
    teams = []
    for e in data['standings']['results']:
        try:
            history = get_history(e['entry'])
        except:
            history = []
        teams.append({
            'name':        e['entry_name'],
            'manager':     e['player_name'],
            'pts':         e['total'],
            'event_total': e['event_total'],
            'rank':        e['rank'],
            'last_rank':   e['last_rank'],
            'history':     history,
        })
    return jsonify(sorted(teams, key=lambda t: t['name'].lower()))

@app.route('/currentgw')
def currentgw():
    url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
    r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
    data = r.json()
    current = 1
    for event in data['events']:
        if event['finished']:
            current = event['id']
    return jsonify({'gw': current})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
