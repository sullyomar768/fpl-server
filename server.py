from flask import Flask, jsonify
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

LEAGUE_ID = 2615606

@app.route('/standings')
def standings():
    url = f'https://fantasy.premierleague.com/api/leagues-classic/{LEAGUE_ID}/standings/'
    r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    data = r.json()
    teams = []
    for e in data['standings']['results']:
        teams.append({
            'name':    e['entry_name'],
            'manager': e['player_name'],
            'pts':     e['total']
        })
    return jsonify(sorted(teams, key=lambda t: t['name'].lower()))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
