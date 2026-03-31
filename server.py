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
```

Scroll down and click **Commit changes**. Render will automatically redeploy — wait about 2 minutes then check `https://fpl-server-5vy4.onrender.com/standings` in your browser to confirm it's working. The response should now include a `history` array for each team alongside the existing fields.

---

**2 — Update the HTML file**

Download the updated `FPL_League_Simulator.html` from this chat. This version includes:
- Form window slider (3–15 GWs)
- Real variance calculated from each manager's GW history
- Fixed startup order (live data first, localStorage fallback, defaults last)
- Duplicate `/currentgw` call removed
- `restorePreferences()` function so sim/form/GW settings survive between sessions

---

**3 — Rebuild the APK**

On your PC:

Go to your `fpl-app` folder on the Desktop, open the `www` folder inside it. Delete the existing `index.html`. Put the new `FPL_League_Simulator.html` in there and rename it to `index.html`.

Open the `fpl-app` folder, click the address bar, type `cmd`, press Enter. Run:
```
npx cap sync
```

Wait for it to finish, then run:
```
npx cap open android
